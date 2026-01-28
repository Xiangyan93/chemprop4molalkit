# Chemprop for MolALKit

A customized version of [Chemprop](https://github.com/chemprop/chemprop) (v1.5.2) optimized for molecular active learning. This package is utilized by [MolALKit](https://github.com/RekerLab/MolALKit).

## Key Modifications

### New CLI Commands

#### `chemprop_cv` - Cross-Validation

Perform k-fold or Monte-Carlo cross-validation with detailed metrics reporting.

```bash
chemprop_cv \
    --data_path data.csv \
    --dataset_type classification \
    --save_dir ./cv_results \
    --cross_validation kFold \
    --n_splits 5 \
    --split_type scaffold_random \
    --metrics auc prc-auc
```

**Arguments:**
- `--cross_validation`: `kFold` or `Monte-Carlo`
- `--n_splits`: Number of folds for k-fold CV (default: 5)
- `--split_type`: `random`, `scaffold_order`, `scaffold_random`, or `stratified`
- `--split_sizes`: Train/test split ratios for Monte-Carlo CV (e.g., `0.8 0.2`)
- `--num_folds`: Number of CV repeats with different seeds
- `--separate_test_path`: Optional external test set for evaluation

**Output:**
- `kFold_metrics.csv` or `Monte-Carlo_metrics.csv`: Metrics for each fold
- `*_prediction.csv`: Predictions for each fold
- `results.log`: Summary of performance metrics

#### `chemprop_optuna` - Hyperparameter Optimization

Bayesian hyperparameter optimization using [Optuna](https://optuna.org/) with TPE sampler.

```bash
chemprop_optuna \
    --data_path data.csv \
    --dataset_type regression \
    --save_dir ./optuna_results \
    --n_trials 100 \
    --cross_validation kFold \
    --n_splits 5
```

**Search Space:**
| Parameter | Range |
|-----------|-------|
| `depth` | 2-6 |
| `hidden_size` | 300-2400 (step 100) |
| `ffn_num_layers` | 1-6 |
| `ffn_hidden_size` | 300-2400 (step 100) |
| `dropout` | 0.0-0.4 (step 0.05) |
| `weight_decay` | 0, 1e-5, 1e-4, 1e-3 |
| `batch_size` | 16, 32, 64, 128, 256 |
| `epochs` | 10, 30, 50, 100, 200 |
| `atom_messages/undirected` | (True,False), (False,True), (False,False) |

**Arguments:**
- `--n_trials`: Number of optimization trials (default: 100)
- `--separate_val_path`: Optional validation set (otherwise uses CV)
- `--separate_test_path`: Optional test set for final evaluation

**Output:**
- `optuna.db`: SQLite database with all trial results (can resume interrupted runs)
- `trial-{n}/`: Directory for each trial with model and metrics

## License

MIT License
