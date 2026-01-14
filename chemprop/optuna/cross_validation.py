from chemprop.args import CrossValidationArgs
from chemprop.data.utils import get_data
from chemprop.models.mpnn import MPNN
from chemprop.optuna.evaluator import Evaluator


def chemprop_cv(arguments=None):
    args = CrossValidationArgs().parse_args(arguments)
    dataset = get_data(path=args.data_path,
                       args=args)
    model = MPNN(save_dir=args.save_dir,
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
                 epochs=args.epochs,
                 depth=args.depth,
                 hidden_size=args.hidden_size,
                 ffn_num_layers=args.ffn_num_layers,
                 ffn_hidden_size=args.ffn_hidden_size,
                 dropout=args.dropout,
                 weight_decay=args.weight_decay,
                 batch_size=args.batch_size,
                 ensemble_size=args.ensemble_size,
                 number_of_molecules=args.number_of_molecules,
                 mpn_shared=args.mpn_shared,
                 atom_messages=args.atom_messages,
                 undirected=args.undirected,
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
    evaluator = Evaluator(save_dir=args.save_dir,
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
    if args.separate_test_path is not None:
        dataset_test = get_data(path=args.separate_test_path,
                                args=args)
        evaluator.run_external(dataset_test, name='test')
    else:
        evaluator.run_cross_validation()
