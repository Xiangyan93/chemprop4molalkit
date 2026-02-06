#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from typing import List, Literal
from tqdm import trange
from logging import Logger
import numpy as np
import torch
from torch.optim.lr_scheduler import ExponentialLR
from chemprop.data import get_class_sizes, MoleculeDataLoader, get_task_names
from chemprop.data.utils import get_no_scale_indices
from chemprop.utils import build_optimizer, build_lr_scheduler, makedirs, load_mpn_model, save_checkpoint, load_checkpoint
from chemprop.nn_utils import param_count, param_count_all
from chemprop.models import MoleculeModel
from chemprop.train.loss_functions import get_loss_func
from chemprop.train import train
from chemprop.args import TrainArgs, PredictArgs
from chemprop.train.make_predictions import set_features, predict_and_save
from molalkit.data.utils import get_subset_from_idx


class MPNN:
    def __init__(self,
                 # TrainArgs parameters
                 save_dir: str, data_path: str,
                 dataset_type: Literal["regression", "classification", "multiclass", "spectra"],
                 loss_function: Literal["mse", "bounded_mse", "binary_cross_entropy", "cross_entropy", "mcc", "sid",
                                        "wasserstein", "mve", "evidential", "dirichlet"],
                 smiles_columns: List[str] = None, target_columns: List[str] = None,
                 multiclass_num_classes: int = 3,
                 features_columns: List[str] = None,
                 features_generator=None,
                 no_features_scaling: bool = False,
                 features_only: bool = False,
                 features_size: int = 0,
                 epochs: int = 30,
                 depth: int = 3,
                 hidden_size: int = 300,
                 ffn_num_layers: int = 2,
                 ffn_hidden_size: int = None,
                 dropout: float = 0.0,
                 batch_size: int = 50,
                 ensemble_size: int = 1,
                 number_of_molecules: int = 1,
                 mpn_shared: bool = False,
                 atom_messages: bool = False,
                 undirected: bool = False,
                 const_lr: bool = False,
                 weight_decay: float = 0.0,
                 cbp: bool = False,
                 replacement_rate: float = 0.00001,
                 maturity_threshold: int = 100,
                 reinit_weights: Literal['xavier', 'kaiming', 'lecun', 'default'] = 'xavier',
                 decay_rate: float = 0.99,
                 n_jobs: int = 8,
                 class_balance: bool = False,
                 checkpoint_dir: str = None,
                 checkpoint_frzn: str = None,
                 frzn_ffn_layers: int = 0,
                 freeze_first_only: bool = False,
                 mpn_path: str = None,
                 freeze_mpn: bool = False,
                 seed: int = 0,
                 # PredictArgs parameters
                 uncertainty_method: Literal["mve", "ensemble", "evidential_epistemic", "evidential_aleatoric",
                                             "evidential_total", "classification", "dropout", "spectra_roundrobin"] = None,
                 uncertainty_dropout_p: float = 0.1,
                 dropout_sampling_size: int = 10,
                 # other parameters
                 continuous_fit: bool = False,
                 logger: Logger = None,
                 ):
        args = TrainArgs()
        args.save_dir = save_dir
        args.data_path = data_path
        args.dataset_type = dataset_type
        args.loss_function = loss_function
        args.smiles_columns = smiles_columns
        args.target_columns = target_columns
        args.multiclass_num_classes = multiclass_num_classes
        args.features_columns = features_columns
        args.features_generator = features_generator
        args.no_features_scaling = no_features_scaling
        args.features_only = features_only
        args.features_size = features_size
        args.epochs = epochs
        args.depth = depth
        args.hidden_size = hidden_size
        args.ffn_num_layers = ffn_num_layers
        args.ffn_hidden_size = ffn_hidden_size
        args.dropout = dropout
        args.batch_size = batch_size
        args.ensemble_size = ensemble_size
        args.number_of_molecules = number_of_molecules
        args.mpn_shared = mpn_shared
        args.atom_messages = atom_messages
        args.undirected = undirected
        args.const_lr = const_lr
        args.weight_decay = weight_decay
        args.cbp = cbp
        args.replacement_rate = replacement_rate
        args.maturity_threshold = maturity_threshold
        args.reinit_weights = reinit_weights
        args.decay_rate = decay_rate
        args.num_workers = n_jobs
        args.class_balance = class_balance
        args.checkpoint_dir = checkpoint_dir
        args.checkpoint_frzn = checkpoint_frzn
        args.frzn_ffn_layers = frzn_ffn_layers
        args.freeze_first_only = freeze_first_only
        args.mpn_path = mpn_path
        args.freeze_mpn = freeze_mpn
        args.seed = seed
        args.process_args()
        args.task_names = get_task_names(path=args.data_path, smiles_columns=args.smiles_columns,
                                         target_columns=args.target_columns, ignore_columns=args.ignore_columns)
        args._parsed = True
        self.chemprop_train_args = args
        self.continuous_fit = continuous_fit
        self.logger = logger
        args_predict = PredictArgs()
        args_predict.uncertainty_method = uncertainty_method
        args_predict.uncertainty_dropout_p = uncertainty_dropout_p
        args_predict.dropout_sampling_size = dropout_sampling_size
        # args_predict.checkpoint_dir = save_dir
        args_predict.test_path = "fake"
        args_predict.preds_path = "fake"
        # args_predict.process_args()
        args_predict._parsed = True
        args_predict.checkpoint_paths = [None] * args.ensemble_size
        self.chemprop_predict_args = args_predict

    def fit_molalkit(self, train_data, iteration: int = 0):
        if not self.continuous_fit and torch.cuda.is_available():
            torch.cuda.empty_cache()
        args = self.chemprop_train_args
        args.train_data_size = len(train_data)
        logger = self.logger
        if logger is not None:
            debug, info = logger.debug, logger.info
        else:
            debug = info = print

        # Set pytorch seed for random initial weights
        torch.manual_seed(args.pytorch_seed)

        if args.dataset_type == "classification":
            train_class_sizes = get_class_sizes(train_data, proportion=False)
            args.train_class_sizes = train_class_sizes

        if args.features_scaling:
            no_scale_indices = get_no_scale_indices(args, train_data)
            self.features_scaler = train_data.normalize_features(
                replace_nan_token=0, no_scale_indices=no_scale_indices)
        else:
            self.features_scaler = None

        args.train_data_size = len(train_data)

        # Initialize scaler and scale training targets by subtracting mean and dividing standard deviation (
        # regression only)
        if args.dataset_type == "regression":
            debug("Fitting scaler")
            scaler = train_data.normalize_targets()
            args.spectra_phase_mask = None
        else:
            args.spectra_phase_mask = None
            scaler = None

        # Get loss function
        loss_func = get_loss_func(args)

        train_data_loader = MoleculeDataLoader(
            dataset=train_data,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            class_balance=args.class_balance,
            shuffle=True,
            seed=args.seed
        )

        if args.class_balance:
            debug(
                f"With class_balance, effective train size = {train_data_loader.iter_size:,}")

        if self.continuous_fit and hasattr(self, "models"):
            assert len(self.models) == args.ensemble_size
        else:
            self.models = []

        self.scalers = []
        for model_idx in range(args.ensemble_size):
            save_dir = os.path.join(args.save_dir, f"model_{model_idx}")
            makedirs(save_dir)
            writer = None
            if self.continuous_fit and len(self.models) == args.ensemble_size:
                debug(
                    f"Loading model {model_idx} that fitted at previous iteration")
                model = self.models[model_idx]
            else:
                debug(f"Building model {model_idx} from scratch")
                model = MoleculeModel(args)
                if args.cuda:
                    debug("Moving model to cuda")
                model = model.to(args.device)

            if args.mpn_path is not None:
                debug(f"Loading MPN parameters from {args.mpn_path}.")
                model = load_mpn_model(
                    model=model, path=args.mpn_path, current_args=args, logger=logger)

            debug(model)

            if args.freeze_mpn:
                debug(f"Number of unfrozen parameters = {param_count(model):,}")
                debug(f"Total number of parameters = {param_count_all(model):,}")
            else:
                debug(f"Number of parameters = {param_count_all(model):,}")
            # Optimizers
            optimizer = build_optimizer(model, args)

            # Learning rate schedulers
            scheduler = build_lr_scheduler(optimizer, args)

            n_iter = 0
            for epoch in trange(args.epochs):
                debug(f"Epoch {epoch}")
                n_iter = train(
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
                if isinstance(scheduler, ExponentialLR):
                    scheduler.step()
            if len(self.models) < args.ensemble_size:
                assert len(self.models) == model_idx
                self.models.append(model)

            self.scalers.append((scaler, self.features_scaler, None, None))
            # save the model after training
            # save_checkpoint(os.path.join(save_dir, 'model.pth'), model, scaler,
            #                 features_scaler, None, None, args)

    def predict(self, pred_data, batch_size: int = 10000):
        """
        Generate predictions for input data using trained models.
        
        Parameters
        ----------
        pred_data : MoleculeDataset
            Dataset containing molecules to make predictions on. Must be preprocessed
            in the same way as the training data.
        
        batch_size : int, optional (default=100000)
            Number of molecules to process in each batch. Controls memory usage
            during prediction. Larger values process data faster but require more memory.
            
        Returns
        -------
        np.ndarray
            Array of shape (n_molecules,) containing model predictions for each molecule.
        
        np.ndarray
            Array of shape (n_molecules,) containing uncertainty estimates for each prediction.
        """
        args = self.chemprop_predict_args
        train_args = self.chemprop_train_args
        num_tasks = train_args.num_tasks
        task_names = train_args.task_names

        set_features(args, train_args)

        if self.features_scaler is not None:
            pred_data.normalize_features(self.features_scaler)

        # Initialize arrays to store predictions and uncertainties
        all_preds = []
        all_uncs = []
        # Calculate total number of batches
        total_batches = (len(pred_data) + batch_size - 1) // batch_size
        # Process data in chunks of 100,000
        for i in range(total_batches):
            models = (model for model in self.models)
            scalers = (scaler for scaler in self.scalers)

            start = i * batch_size
            end = min((i + 1) * batch_size, len(pred_data))
            test_data = get_subset_from_idx(pred_data, range(start, end))
            test_data_loader = MoleculeDataLoader(
                dataset=test_data,
                batch_size=train_args.batch_size,
                num_workers=train_args.num_workers
            )
            preds, unc = predict_and_save(
                args=args,
                train_args=train_args,
                test_data=test_data,
                task_names=task_names,
                num_tasks=num_tasks,
                test_data_loader=test_data_loader,
                full_data=pred_data,
                full_to_valid_indices={j: j for j in range(len(pred_data))},
                models=models,
                scalers=scalers,
                num_models=len(self.models),
                return_invalid_smiles=False,
                save_results=False
            )
            all_preds.append(preds)
            all_uncs.append(unc)
        all_preds = np.concatenate(all_preds)
        all_uncs = np.concatenate(all_uncs)
        return all_preds, all_uncs

    def predict_uncertainty(self, pred_data):
        if self.chemprop_predict_args.uncertainty_method is None and self.chemprop_train_args.dataset_type == "classification":
            preds = np.array(self.predict(pred_data)[0])  # (n_samples, n_tasks)
            preds = np.clip(preds, 1e-10, 1 - 1e-10)
            # Binary entropy per task: -p*log(p) - (1-p)*log(1-p)
            task_entropies = -(preds * np.log(preds) + (1 - preds) * np.log(1 - preds))
            # Average across tasks for multi-task; collapses to single task entropy for n_tasks=1
            return np.mean(task_entropies, axis=-1)  # (n_samples,)
        else:
            return self.predict(pred_data)[1]

    def predict_value(self, pred_data):
        return self.predict(pred_data)[0]

    def save_checkpoint(self):
        args = self.chemprop_train_args
        for model_idx, model in enumerate(self.models):
            save_dir = os.path.join(args.save_dir, f"model_{model_idx}")
            makedirs(save_dir)
            scaler, features_scaler, _, _ = self.scalers[model_idx]
            save_checkpoint(os.path.join(save_dir, 'model.pth'), model, scaler,
                            features_scaler, None, None, args)

    def load_checkpoint(self):
        args = self.chemprop_train_args
        models = []
        for model_idx, model in enumerate(self.models):
            save_dir = os.path.join(args.save_dir, f"model_{model_idx}")
            model1 = load_checkpoint(os.path.join(save_dir, 'model.pth'), args.device)
            models.append(model1)
        self.models = models
