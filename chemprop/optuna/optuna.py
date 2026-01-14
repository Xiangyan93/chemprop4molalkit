import os
import copy
import torch
import optuna
from optuna.samplers import TPESampler
from chemprop.args import OptunaArgs
from chemprop.data.utils import get_data
from chemprop.models.mpnn import MPNN
from chemprop.optuna.evaluator import Evaluator
from chemprop.data.data import MoleculeDataset


def chemprop_optuna(arguments=None):
    args = OptunaArgs().parse_args(arguments)
    os.makedirs(args.save_dir, exist_ok=True)
    dataset = get_data(path=args.data_path,
                       args=args, n_jobs=args.num_workers)
    if args.separate_val_path is not None:
        dataset_val = get_data(path=args.separate_val_path,
                               args=args, n_jobs=args.num_workers)
        data_train_val = copy.deepcopy(dataset.data + dataset_val.data)
        dataset_train_val = MoleculeDataset(data_train_val)
    if args.separate_test_path is not None:
        dataset_test = get_data(path=args.separate_test_path,
                                args=args, n_jobs=args.num_workers)

    def objective(trial):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        params = {
            'depth': trial.suggest_int('depth', 2, 6, step=1),
            'hidden_size': trial.suggest_int('hidden_size', 300, 2400, step=100),
            'ffn_num_layers': trial.suggest_int('ffn_num_layers', 1, 6, step=1),
            'ffn_hidden_size': trial.suggest_int('ffn_hidden_size', 300, 2400, step=100),
            'dropout': trial.suggest_float('dropout', 0.0, 0.4, step=0.05),
            'weight_decay': trial.suggest_categorical('weight_decay', [0.0, 1e-5, 1e-4, 1e-3]),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256]),
            'epochs': trial.suggest_categorical('epochs', [10, 30, 50, 100, 200]), # 10, 30, 50, 100, 200
            'set1': trial.suggest_categorical('set1', [(True, False), (False, True), (False, False)]),
            # 'atom_messages': trial.suggest_categorical('atom_messages', [True, False]),
            # 'undirected': trial.suggest_categorical('undirected', [True, False])
        }
            # 'ensemble_size': trial.suggest_categorical('ensemble_size', [1, 4, 16]),
        atom_messages, undirected = params['set1']
        model = MPNN(save_dir='%s/trial-%d' % (args.save_dir, trial.number),
                    data_path=args.data_path,
                    dataset_type=args.dataset_type,
                    loss_function=args.loss_function,
                    smiles_columns=args.smiles_columns,
                    target_columns=args.target_columns,
                    multiclass_num_classes=args.multiclass_num_classes,
                    features_generator=args.features_generator,
                    no_features_scaling=args.no_features_scaling,
                    features_only=args.features_only,
                    features_size=dataset.features_size(),
                    epochs=params['epochs'],
                    depth=params['depth'],
                    hidden_size=params['hidden_size'],
                    ffn_num_layers=params['ffn_num_layers'],
                    ffn_hidden_size=params['ffn_hidden_size'],
                    dropout=params['dropout'],
                    weight_decay=params['weight_decay'],
                    batch_size=params['batch_size'],
                    ensemble_size=args.ensemble_size,
                    number_of_molecules=args.number_of_molecules,
                    mpn_shared=args.mpn_shared,
                    atom_messages=atom_messages,
                    undirected=undirected,
                    n_jobs=args.num_workers,
                    class_balance=args.class_balance,
                    checkpoint_dir=args.checkpoint_dir,
                    checkpoint_frzn=args.checkpoint_frzn,
                    frzn_ffn_layers=args.frzn_ffn_layers,
                    freeze_first_only=args.freeze_first_only,
                    mpn_path=args.mpn_path,
                    freeze_mpn=args.freeze_mpn,
                    seed=args.seed,
                    continuous_fit=False,
                    logger=None)
        evaluator = Evaluator(save_dir='%s/trial-%d' % (args.save_dir, trial.number),
                            dataset=dataset,
                            model=model,
                            task_type=args.dataset_type,
                            metrics=args.metrics,
                            cross_validation=args.cross_validation,
                            n_splits=args.n_splits,
                            split_type=args.split_type,
                            split_sizes=args.split_sizes,
                            num_folds=args.num_folds,
                            seed=args.seed)
        if args.separate_val_path is not None:
            if args.separate_test_path is not None:
                evaluator1 = Evaluator(save_dir='%s/trial-%d' % (args.save_dir, trial.number),
                                       dataset=dataset_train_val,
                                       model=model,
                                       task_type=args.dataset_type,
                                       metrics=args.metrics,
                                       cross_validation=args.cross_validation,
                                       n_splits=args.n_splits,
                                       split_type=args.split_type,
                                       split_sizes=args.split_sizes,
                                       num_folds=args.num_folds,
                                       seed=args.seed)
                evaluator1.run_external(dataset_test, name='test')
                evaluator.run_external(dataset_test, name='test_train_only')
            return evaluator.run_external(dataset_val, name='val')
        else:
            if args.separate_test_path is not None:
                evaluator.run_external(dataset_test, name='test')
            return evaluator.run_cross_validation()
        
    study = optuna.create_study(
        study_name="optuna-study",
        sampler=TPESampler(seed=args.seed),
        storage="sqlite:///%s/optuna.db" % args.save_dir,
        load_if_exists=True,
        direction='minimize' if args.dataset_type == 'regression' else 'maximize'
    )
    n_to_run = args.n_trials - len(study.trials)
    if n_to_run > 0:
        study.optimize(objective, n_trials=n_to_run)
