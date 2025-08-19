import json
from logging import Logger
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from tensorboardX import SummaryWriter
import torch
from tqdm import trange
from torch.optim.lr_scheduler import ExponentialLR

from .evaluate import evaluate, evaluate_predictions
from .predict import predict
from .train import train
from .loss_functions import get_loss_func
from chemprop.spectra_utils import normalize_spectra, load_phase_mask
from chemprop.args import TrainArgs
from chemprop.constants import MODEL_FILE_NAME
from chemprop.data import get_class_sizes, get_data, MoleculeDataLoader, MoleculeDataset, set_cache_graph, split_data
from chemprop.models import MoleculeModel
from chemprop.nn_utils import param_count, param_count_all
from chemprop.utils import build_optimizer, build_lr_scheduler, load_checkpoint, makedirs, \
    save_checkpoint, save_smiles_splits, load_frzn_model, multitask_mean, load_mpn_model

from .gnt import GnT
from .AdamGnT import AdamGnT
import torch.nn.functional as F

import math

@torch.no_grad()
def compute_dataset_loss(model: MoleculeModel,
                         data_loader: MoleculeDataLoader,
                         loss_func,
                         args: TrainArgs) -> float:
    """Return weighted-average loss over a dataset, matching train() logic."""
    model.eval()
    device = args.device

    epoch_num = 0.0   # sum of (batch_mean_loss * effective_count)
    epoch_den = 0     # sum of effective_count (i.e., mask.sum())

    for batch in data_loader:
        # === 准备 batch（与 train() 对齐） ===
        mol_batch, features_batch, target_batch, mask_batch, atom_descriptors_batch, \
        atom_features_batch, bond_features_batch, data_weights_batch = \
            batch.batch_graph(), batch.features(), batch.targets(), batch.mask(), batch.atom_descriptors(), \
            batch.atom_features(), batch.bond_features(), batch.data_weights()

        mask = torch.tensor(mask_batch, dtype=torch.bool)
        targets = torch.tensor([[0 if x is None else x for x in tb] for tb in target_batch])

        if args.target_weights is not None:
            target_weights = torch.tensor(args.target_weights).unsqueeze(0)
        else:
            target_weights = torch.ones(targets.shape[1]).unsqueeze(0)
        data_weights = torch.tensor(data_weights_batch).unsqueeze(1)

        if args.loss_function == 'bounded_mse':
            lt_target_batch = torch.tensor(batch.lt_targets())
            gt_target_batch = torch.tensor(batch.gt_targets())

        # === 前向 ===
        preds, _ = model(mol_batch, features_batch, atom_descriptors_batch, atom_features_batch, bond_features_batch)

        # === to(device) ===
        mask = mask.to(device)
        targets = targets.to(device)
        target_weights = target_weights.to(device)
        data_weights = data_weights.to(device)
        if args.loss_function == 'bounded_mse':
            lt_target_batch = lt_target_batch.to(device)
            gt_target_batch = gt_target_batch.to(device)

        effective = int(mask.sum().item())
        if effective == 0:
            # 验证/测试集里如果整批都无效，直接报错
            raise ValueError("No valid targets in this batch (mask.sum()==0) during evaluation.")

        # === 与 train() 一致的 loss 分支 ===
        if args.loss_function == 'mcc' and args.dataset_type == 'classification':
            loss = loss_func(preds, targets, data_weights, mask) * target_weights.squeeze(0)
        elif args.loss_function == 'mcc':  # multiclass
            targets = targets.long()
            target_losses = []
            for target_index in range(preds.size(1)):
                target_loss = loss_func(preds[:, target_index, :],
                                        targets[:, target_index],
                                        data_weights,
                                        mask[:, target_index]).unsqueeze(0)
                target_losses.append(target_loss)
            loss = torch.cat(target_losses).to(device) * target_weights.squeeze(0)
        elif args.dataset_type == 'multiclass':
            targets = targets.long()
            if args.loss_function == 'dirichlet':
                loss = loss_func(preds, targets, args.evidential_regularization) * target_weights * data_weights * mask
            else:
                target_losses = []
                for target_index in range(preds.size(1)):
                    target_loss = loss_func(preds[:, target_index, :], targets[:, target_index]).unsqueeze(1)
                    target_losses.append(target_loss)
                loss = torch.cat(target_losses, dim=1).to(device) * target_weights * data_weights * mask
        elif args.dataset_type == 'spectra':
            loss = loss_func(preds, targets, mask) * target_weights * data_weights * mask
        elif args.loss_function == 'bounded_mse':
            loss = loss_func(preds, targets, lt_target_batch, gt_target_batch) * target_weights * data_weights * mask
        elif args.loss_function == 'evidential':
            loss = loss_func(preds, targets, args.evidential_regularization) * target_weights * data_weights * mask
        elif args.loss_function == 'dirichlet':  # classification
            loss = loss_func(preds, targets, args.evidential_regularization) * target_weights * data_weights * mask
        else:
            loss = loss_func(preds, targets) * target_weights * data_weights * mask

        batch_mean_loss = (loss.sum() / mask.sum()).item()

        # === 加权累计（与 train.py 的 epoch 统计口径一致） ===
        epoch_num += batch_mean_loss * effective
        epoch_den += effective

    if epoch_den == 0:
        raise ValueError("No valid targets in this dataset (all mask==0).")
    return epoch_num / epoch_den

def run_training(args: TrainArgs,
                 data: MoleculeDataset,
                 logger: Logger = None) -> Dict[str, List[float]]:
    """
    Loads data, trains a Chemprop model, and returns test scores for the model checkpoint with the highest validation score.

    :param args: A :class:`~chemprop.args.TrainArgs` object containing arguments for
                 loading data and training the Chemprop model.
    :param data: A :class:`~chemprop.data.MoleculeDataset` containing the data.
    :param logger: A logger to record output.
    :return: A dictionary mapping each metric in :code:`args.metrics` to a list of values for each task.

    """
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    # Set pytorch seed for random initial weights
    torch.manual_seed(args.pytorch_seed)

    # Split data
    debug(f'Splitting data with seed {args.seed}')
    if args.separate_test_path:
        test_data = get_data(path=args.separate_test_path,
                             args=args,
                             features_path=args.separate_test_features_path,
                             atom_descriptors_path=args.separate_test_atom_descriptors_path,
                             bond_features_path=args.separate_test_bond_features_path,
                             phase_features_path=args.separate_test_phase_features_path,
                             smiles_columns=args.smiles_columns,
                             loss_function=args.loss_function,
                             logger=logger)
    if args.separate_val_path:
        val_data = get_data(path=args.separate_val_path,
                            args=args,
                            features_path=args.separate_val_features_path,
                            atom_descriptors_path=args.separate_val_atom_descriptors_path,
                            bond_features_path=args.separate_val_bond_features_path,
                            phase_features_path=args.separate_val_phase_features_path,
                            smiles_columns=args.smiles_columns,
                            loss_function=args.loss_function,
                            logger=logger)

    if args.separate_val_path and args.separate_test_path:
        train_data = data
    elif args.separate_val_path:
        train_data, _, test_data = split_data(data=data,
                                              split_type=args.split_type,
                                              sizes=args.split_sizes,
                                              key_molecule_index=args.split_key_molecule,
                                              seed=args.seed,
                                              num_folds=args.num_folds,
                                              args=args,
                                              logger=logger)
    elif args.separate_test_path:
        train_data, val_data, _ = split_data(data=data,
                                             split_type=args.split_type,
                                             sizes=args.split_sizes,
                                             key_molecule_index=args.split_key_molecule,
                                             seed=args.seed,
                                             num_folds=args.num_folds,
                                             args=args,
                                             logger=logger)
    else:
        train_data, val_data, test_data = split_data(data=data,
                                                     split_type=args.split_type,
                                                     sizes=args.split_sizes,
                                                     key_molecule_index=args.split_key_molecule,
                                                     seed=args.seed,
                                                     num_folds=args.num_folds,
                                                     args=args,
                                                     logger=logger)

    if args.dataset_type == 'classification':
        class_sizes = get_class_sizes(data)
        debug('Class sizes')
        for i, task_class_sizes in enumerate(class_sizes):
            debug(f'{args.task_names[i]} '
                  f'{", ".join(f"{cls}: {size * 100:.2f}%" for cls, size in enumerate(task_class_sizes))}')
        train_class_sizes = get_class_sizes(train_data, proportion=False)
        args.train_class_sizes = train_class_sizes

    if args.save_smiles_splits:
        save_smiles_splits(
            data_path=args.data_path,
            save_dir=args.save_dir,
            task_names=args.task_names,
            features_path=args.features_path,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            smiles_columns=args.smiles_columns,
            logger=logger,
        )

    if args.features_scaling:
        features_scaler = train_data.normalize_features(replace_nan_token=0)
        val_data.normalize_features(features_scaler)
        test_data.normalize_features(features_scaler)
    else:
        features_scaler = None

    if args.atom_descriptor_scaling and args.atom_descriptors is not None:
        atom_descriptor_scaler = train_data.normalize_features(
            replace_nan_token=0, scale_atom_descriptors=True)
        val_data.normalize_features(
            atom_descriptor_scaler, scale_atom_descriptors=True)
        test_data.normalize_features(
            atom_descriptor_scaler, scale_atom_descriptors=True)
    else:
        atom_descriptor_scaler = None

    if args.bond_feature_scaling and args.bond_features_size > 0:
        bond_feature_scaler = train_data.normalize_features(
            replace_nan_token=0, scale_bond_features=True)
        val_data.normalize_features(
            bond_feature_scaler, scale_bond_features=True)
        test_data.normalize_features(
            bond_feature_scaler, scale_bond_features=True)
    else:
        bond_feature_scaler = None

    args.train_data_size = len(train_data)

    debug(f'Total size = {len(data):,} | '
          f'train size = {len(train_data):,} | val size = {len(val_data):,} | test size = {len(test_data):,}')

    if len(val_data) == 0:
        raise ValueError('The validation data split is empty. During normal chemprop training (non-sklearn functions), \
            a validation set is required to conduct early stopping according to the selected evaluation metric. This \
            may have occurred because validation data provided with `--separate_val_path` was empty or contained only invalid molecules.')

    if len(test_data) == 0:
        debug('The test data split is empty. This may be either because splitting with no test set was selected, \
            such as with `cv-no-test`, or because test data provided with `--separate_test_path` was empty or contained only invalid molecules. \
            Performance on the test set will not be evaluated and metric scores will return `nan` for each task.')
        empty_test_set = True
    else:
        empty_test_set = False

    # Initialize scaler and scale training targets by subtracting mean and dividing standard deviation (regression only)
    if args.dataset_type == 'regression':
        debug('Fitting scaler')
        scaler = train_data.normalize_targets()
        args.spectra_phase_mask = None
    elif args.dataset_type == 'spectra':
        debug('Normalizing spectra and excluding spectra regions based on phase')
        args.spectra_phase_mask = load_phase_mask(args.spectra_phase_mask_path)
        for dataset in [train_data, test_data, val_data]:
            data_targets = normalize_spectra(
                spectra=dataset.targets(),
                phase_features=dataset.phase_features(),
                phase_mask=args.spectra_phase_mask,
                excluded_sub_value=None,
                threshold=args.spectra_target_floor,
            )
            dataset.set_targets(data_targets)
        scaler = None
    else:
        args.spectra_phase_mask = None
        scaler = None

    # Get loss function
    loss_func = get_loss_func(args)

    # Set up test set evaluation
    test_smiles, test_targets = test_data.smiles(), test_data.targets()
    if args.dataset_type == 'multiclass':
        sum_test_preds = np.zeros(
            (len(test_smiles), args.num_tasks, args.multiclass_num_classes))
    else:
        sum_test_preds = np.zeros((len(test_smiles), args.num_tasks))

    # Automatically determine whether to cache
    if len(data) <= args.cache_cutoff:
        set_cache_graph(True)
        num_workers = 0
    else:
        set_cache_graph(False)
        num_workers = args.num_workers

    # Create data loaders
    train_data_loader = MoleculeDataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        num_workers=num_workers,
        class_balance=args.class_balance,
        shuffle=True,
        seed=args.seed
    )
    val_data_loader = MoleculeDataLoader(
        dataset=val_data,
        batch_size=args.batch_size,
        num_workers=num_workers
    )
    test_data_loader = MoleculeDataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        num_workers=num_workers
    )

    if args.class_balance:
        debug(
            f'With class_balance, effective train size = {train_data_loader.iter_size:,}')

    # record all loss
    all_models_epoch_losses = []
    
    # Train ensemble of models
    for model_idx in range(args.ensemble_size):
        # Tensorboard writer
        save_dir = os.path.join(args.save_dir, f'model_{model_idx}')
        epoch_losses = [] # current epoch loss
        
        makedirs(save_dir)
        try:
            writer = SummaryWriter(log_dir=save_dir)
        except:
            writer = SummaryWriter(logdir=save_dir)

        # Load/build model
        if args.checkpoint_paths is not None:
            debug(
                f'Loading model {model_idx} from {args.checkpoint_paths[model_idx]}')
            model = load_checkpoint(
                args.checkpoint_paths[model_idx], logger=logger)
        else:
            debug(f'Building model {model_idx}')
            model = MoleculeModel(args)

        # Optionally, overwrite weights:
        if args.checkpoint_frzn is not None:
            debug(
                f'Loading and freezing parameters from {args.checkpoint_frzn}.')
            model = load_frzn_model(
                model=model, path=args.checkpoint_frzn, current_args=args, logger=logger)

        if args.mpn_path is not None:
            debug(f'Loading MPN parameters from {args.mpn_path}.')
            model = load_mpn_model(
                model=model, path=args.mpn_path, current_args=args, logger=logger)

        debug(model)

        if args.checkpoint_frzn is not None or args.freeze_mpn:
            debug(f'Number of unfrozen parameters = {param_count(model):,}')
            debug(f'Total number of parameters = {param_count_all(model):,}')
        else:
            debug(f'Number of parameters = {param_count_all(model):,}')

        if args.cuda:
            debug('Moving model to cuda')
        model = model.to(args.device)

        # Ensure that model is saved in correct location for evaluation if 0 epochs
        save_checkpoint(os.path.join(save_dir, MODEL_FILE_NAME), model, scaler,
                        features_scaler, atom_descriptor_scaler, bond_feature_scaler, args)

        # Optimizers
        optimizer = build_optimizer(model, args)

        # Learning rate schedulers
        scheduler = build_lr_scheduler(optimizer, args)
      
        if args.use_gnt:
            model.gnt = GnT(
                net=model.ffn,
                hidden_activation=args.activation.lower(),
                opt=optimizer,
                decay_rate=0.99,
                replacement_rate=0,
                init='kaiming',
                device=args.device,
                maturity_threshold=20,
                util_type='contribution',
                loss_func=loss_func,
                accumulate=False
            )

        # Run training
        best_score = float('inf') if args.minimize_score else -float('inf')
        best_epoch, n_iter = 0, 0
        for epoch in trange(args.epochs):
            debug(f'Epoch {epoch}')
            n_iter, epoch_loss = train(
                model=model,
                data_loader=train_data_loader,
                loss_func=loss_func,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                n_iter=n_iter,
                logger=logger,
                writer=writer
            )
            
            # 计算 validation loss
            val_loss = compute_dataset_loss(model, val_data_loader, loss_func, args)

            # 每个 epoch 计算一次 test loss（如果不担心数据泄漏）
            test_loss = compute_dataset_loss(model, test_data_loader, loss_func, args) if not empty_test_set else float('nan')

            epoch_losses.append({
                "model_idx": model_idx,
                "epoch": int(epoch),
                "train_loss": float(epoch_loss),
                "val_loss": float(val_loss),
                "test_loss": float(test_loss)
            })
            
            if isinstance(scheduler, ExponentialLR):
                scheduler.step()
            val_scores = evaluate(
                model=model,
                data_loader=val_data_loader,
                num_tasks=args.num_tasks,
                metrics=args.metrics,
                dataset_type=args.dataset_type,
                scaler=scaler,
                logger=logger
            )

            for metric, scores in val_scores.items():
                # Average validation score\
                mean_val_score = multitask_mean(scores, metric=metric)
                debug(f'Validation {metric} = {mean_val_score:.6f}')
                writer.add_scalar(
                    f'validation_{metric}', mean_val_score, n_iter)

                if args.show_individual_scores:
                    # Individual validation scores
                    for task_name, val_score in zip(args.task_names, scores):
                        debug(
                            f'Validation {task_name} {metric} = {val_score:.6f}')
                        writer.add_scalar(
                            f'validation_{task_name}_{metric}', val_score, n_iter)

            # Save model checkpoint if improved validation score
            mean_val_score = multitask_mean(
                val_scores[args.metric], metric=args.metric)
            if args.minimize_score and mean_val_score < best_score or \
                    not args.minimize_score and mean_val_score > best_score:
                best_score, best_epoch = mean_val_score, epoch
                save_checkpoint(os.path.join(save_dir, MODEL_FILE_NAME), model, scaler, features_scaler,
                                atom_descriptor_scaler, bond_feature_scaler, args)
            
        # Evaluate on test set using model with best validation score
        info(
            f'Model {model_idx} best validation {args.metric} = {best_score:.6f} on epoch {best_epoch}')
        model = load_checkpoint(os.path.join(
            save_dir, MODEL_FILE_NAME), device=args.device, logger=logger)

        if empty_test_set:
            info(
                f'Model {model_idx} provided with no test set, no metric evaluation will be performed.')
        else:
            test_preds = predict(
                model=model,
                data_loader=test_data_loader,
                scaler=scaler
            )
            test_scores = evaluate_predictions(
                preds=test_preds,
                targets=test_targets,
                num_tasks=args.num_tasks,
                metrics=args.metrics,
                dataset_type=args.dataset_type,
                gt_targets=test_data.gt_targets(),
                lt_targets=test_data.lt_targets(),
                logger=logger
            )

            if len(test_preds) != 0:
                sum_test_preds += np.array(test_preds)

            # Average test score
            for metric, scores in test_scores.items():
                avg_test_score = np.nanmean(scores)
                info(f'Model {model_idx} test {metric} = {avg_test_score:.6f}')
                writer.add_scalar(f'test_{metric}', avg_test_score, 0)

                if args.show_individual_scores and args.dataset_type != 'spectra':
                    # Individual test scores
                    for task_name, test_score in zip(args.task_names, scores):
                        info(
                            f'Model {model_idx} test {task_name} {metric} = {test_score:.6f}')
                        writer.add_scalar(
                            f'test_{task_name}_{metric}', test_score, n_iter)
        writer.close()
        model_loss_df = pd.DataFrame(epoch_losses)
        model_loss_csv = os.path.join(save_dir, f"epoch_withoutGNT_train_loss_seed{args.seed}.csv")
        model_loss_df.to_csv(model_loss_csv, index=False)


    # Evaluate ensemble on test set
    if empty_test_set:
        ensemble_scores = {
            metric: [np.nan for task in args.task_names] for metric in args.metrics
        }
    else:
        avg_test_preds = (sum_test_preds / args.ensemble_size).tolist()

        ensemble_scores = evaluate_predictions(
            preds=avg_test_preds,
            targets=test_targets,
            num_tasks=args.num_tasks,
            metrics=args.metrics,
            dataset_type=args.dataset_type,
            gt_targets=test_data.gt_targets(),
            lt_targets=test_data.lt_targets(),
            logger=logger
        )

    for metric, scores in ensemble_scores.items():
        # Average ensemble score
        mean_ensemble_test_score = multitask_mean(scores, metric=metric)
        info(f'Ensemble test {metric} = {mean_ensemble_test_score:.6f}')

        # Individual ensemble scores
        if args.show_individual_scores:
            for task_name, ensemble_score in zip(args.task_names, scores):
                info(
                    f'Ensemble test {task_name} {metric} = {ensemble_score:.6f}')

    # Save scores
    with open(os.path.join(args.save_dir, 'test_scores.json'), 'w') as f:
        json.dump(ensemble_scores, f, indent=4, sort_keys=True)

    # Optionally save test preds
    if args.save_preds and not empty_test_set:
        test_preds_dataframe = pd.DataFrame(
            data={'smiles': test_data.smiles()})

        for i, task_name in enumerate(args.task_names):
            test_preds_dataframe[task_name] = [pred[i]
                                               for pred in avg_test_preds]
            test_preds_dataframe[task_name + '_true'] = [target[i]
                                                         for target in test_targets]

        test_preds_dataframe.to_csv(os.path.join(
            args.save_dir, 'test_preds.csv'), index=False)
        
    # save loss

    model_loss_df = pd.DataFrame(epoch_losses)
    model_loss_csv = os.path.join(save_dir, f"epoch_withAdam_train_loss_seed{args.seed}.csv")
    model_loss_df.to_csv(model_loss_csv, index=False)

    return ensemble_scores
