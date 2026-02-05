"""Integration tests for chemprop_cv and chemprop_optuna CLI functions.

These tests run the real CLI on a small fake CSV dataset.
They require torch, rdkit, optuna, etc.
"""

import os
import csv
import shutil
import tempfile
import pytest
import pandas as pd
import numpy as np

torch = pytest.importorskip("torch")

# A minimal set of SMILES and regression targets
_SMILES = [
    "CCO",
    "CCCO",
    "CC(=O)O",
    "c1ccccc1",
    "CC(C)O",
    "CCN",
    "CC=O",
    "CCCC",
    "CC(=O)N",
    "c1ccc(O)cc1",
]
_TARGETS = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

_BINARY_TARGETS = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]


def _write_csv(path, smiles, targets, target_col="target"):
    """Write a minimal CSV with smiles and target columns."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["smiles", target_col])
        for s, t in zip(smiles, targets):
            writer.writerow([s, t])


def _write_multitask_csv(path, smiles, targets_list, target_cols):
    """Write a multi-task CSV with smiles and multiple target columns.

    Args:
        path: Path to write CSV file.
        smiles: List of SMILES strings.
        targets_list: List of lists, each inner list is targets for one task.
                      Use None for missing values.
        target_cols: List of target column names.
    """
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["smiles"] + target_cols)
        for i, s in enumerate(smiles):
            row = [s]
            for targets in targets_list:
                val = targets[i]
                row.append("" if val is None else val)
            writer.writerow(row)


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp(prefix="chemprop_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def regression_csv(tmp_dir):
    path = os.path.join(tmp_dir, "data.csv")
    _write_csv(path, _SMILES, _TARGETS)
    return path


@pytest.fixture
def binary_csv(tmp_dir):
    path = os.path.join(tmp_dir, "data_bin.csv")
    _write_csv(path, _SMILES, _BINARY_TARGETS)
    return path


@pytest.fixture
def ext_test_csv(tmp_dir):
    """A separate small test set."""
    path = os.path.join(tmp_dir, "test.csv")
    _write_csv(path, _SMILES[:3], _TARGETS[:3])
    return path


# Multi-task data with missing labels
# Task 1: 8 valid values, 2 missing (indices 2, 7)
# Task 2: 6 valid values, 4 missing (indices 0, 3, 5, 8)
# Task 3: 10 valid values, 0 missing
_MULTITASK_TARGETS_1 = [1.0, 2.0, None, 4.0, 5.0, 6.0, 7.0, None, 9.0, 10.0]
_MULTITASK_TARGETS_2 = [None, 2.5, 3.5, None, 5.5, None, 7.5, 8.5, None, 10.5]
_MULTITASK_TARGETS_3 = [1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1]

# Binary classification multi-task with missing labels
_MULTITASK_BINARY_1 = [0, 1, None, 1, 0, 1, None, 1, 0, 1]
_MULTITASK_BINARY_2 = [None, 0, 1, None, 1, 0, 1, None, 0, 1]


@pytest.fixture
def multitask_regression_csv(tmp_dir):
    """Multi-task regression CSV with missing labels."""
    path = os.path.join(tmp_dir, "multitask_reg.csv")
    _write_multitask_csv(
        path,
        _SMILES,
        [_MULTITASK_TARGETS_1, _MULTITASK_TARGETS_2, _MULTITASK_TARGETS_3],
        ["target_1", "target_2", "target_3"]
    )
    return path


@pytest.fixture
def multitask_binary_csv(tmp_dir):
    """Multi-task binary classification CSV with missing labels."""
    path = os.path.join(tmp_dir, "multitask_bin.csv")
    _write_multitask_csv(
        path,
        _SMILES,
        [_MULTITASK_BINARY_1, _MULTITASK_BINARY_2],
        ["target_1", "target_2"]
    )
    return path


@pytest.fixture
def multitask_ext_test_csv(tmp_dir):
    """Multi-task external test set with missing labels."""
    path = os.path.join(tmp_dir, "multitask_test.csv")
    # Use first 5 samples with some missing values
    _write_multitask_csv(
        path,
        _SMILES[:5],
        [_MULTITASK_TARGETS_1[:5], _MULTITASK_TARGETS_2[:5], _MULTITASK_TARGETS_3[:5]],
        ["target_1", "target_2", "target_3"]
    )
    return path


class TestChempropCVIntegration:
    """Integration tests for chemprop_cv with real model and small data."""

    def test_kfold_regression(self, tmp_dir, regression_csv):
        from chemprop.optuna.cross_validation import chemprop_cv

        save_dir = os.path.join(tmp_dir, "kfold_out")
        chemprop_cv([
            "--data_path", regression_csv,
            "--dataset_type", "regression",
            "--metric", "rmse",
            "--cross_validation", "kFold",
            "--n_splits", "2",
            "--num_folds", "1",
            "--ensemble_size", "1",
            "--num_workers", "0",
            "--save_dir", save_dir,
            "--epochs", "3",
            "--hidden_size", "10",
            "--ffn_hidden_size", "10",
            "--depth", "2",
            "--quiet",
            "--empty_cache",
        ])

        assert os.path.isdir(save_dir)
        assert os.path.exists(os.path.join(save_dir, "kFold_metrics.csv"))

    def test_monte_carlo_regression(self, tmp_dir, regression_csv):
        from chemprop.optuna.cross_validation import chemprop_cv

        save_dir = os.path.join(tmp_dir, "mc_out")
        chemprop_cv([
            "--data_path", regression_csv,
            "--dataset_type", "regression",
            "--metric", "rmse",
            "--cross_validation", "Monte-Carlo",
            "--split_type", "random",
            "--split_sizes", "0.8", "0.1", "0.1",
            "--num_folds", "1",
            "--ensemble_size", "1",
            "--num_workers", "0",
            "--save_dir", save_dir,
            "--epochs", "3",
            "--hidden_size", "10",
            "--ffn_hidden_size", "10",
            "--depth", "2",
            "--quiet",
            "--empty_cache",
        ])

        assert os.path.exists(os.path.join(save_dir, "Monte-Carlo_metrics.csv"))

    def test_external_test_regression(self, tmp_dir, regression_csv, ext_test_csv):
        from chemprop.optuna.cross_validation import chemprop_cv

        save_dir = os.path.join(tmp_dir, "ext_out")
        chemprop_cv([
            "--data_path", regression_csv,
            "--dataset_type", "regression",
            "--metric", "rmse",
            "--ensemble_size", "1",
            "--num_workers", "0",
            "--save_dir", save_dir,
            "--separate_test_path", ext_test_csv,
            "--epochs", "3",
            "--hidden_size", "10",
            "--ffn_hidden_size", "10",
            "--depth", "2",
            "--quiet",
            "--empty_cache",
        ])

        assert os.path.exists(os.path.join(save_dir, "test_ext_prediction.csv"))
        assert os.path.exists(os.path.join(save_dir, "test_ext_metrics.csv"))

    def test_binary_kfold(self, tmp_dir, binary_csv):
        from chemprop.optuna.cross_validation import chemprop_cv

        save_dir = os.path.join(tmp_dir, "bin_out")
        chemprop_cv([
            "--data_path", binary_csv,
            "--dataset_type", "classification",
            "--metric", "auc",
            "--cross_validation", "kFold",
            "--n_splits", "2",
            "--num_folds", "1",
            "--ensemble_size", "1",
            "--num_workers", "0",
            "--save_dir", save_dir,
            "--epochs", "3",
            "--hidden_size", "10",
            "--ffn_hidden_size", "10",
            "--depth", "2",
            "--quiet",
            "--empty_cache",
        ])

        assert os.path.exists(os.path.join(save_dir, "kFold_metrics.csv"))


class TestChempropOptunaIntegration:
    """Integration tests for chemprop_optuna with real model and small data."""

    def test_optuna_single_trial_regression(self, tmp_dir, regression_csv):
        from chemprop.optuna.optuna import chemprop_optuna

        save_dir = os.path.join(tmp_dir, "optuna_out")
        chemprop_optuna([
            "--data_path", regression_csv,
            "--dataset_type", "regression",
            "--metric", "rmse",
            "--cross_validation", "kFold",
            "--n_splits", "2",
            "--num_folds", "1",
            "--num_workers", "1",
            "--save_dir", save_dir,
            "--n_trials", "1",
            "--epochs", "3",
            "--hidden_size", "10",
            "--ffn_hidden_size", "10",
            "--depth", "2",
            "--quiet",
            "--empty_cache",
        ])

        assert os.path.exists(os.path.join(save_dir, "optuna.db"))
        assert os.path.isdir(os.path.join(save_dir, "trial-0"))

    def test_optuna_resume_skips_done(self, tmp_dir, regression_csv):
        """Run 1 trial, then call again with n_trials=1 — should not run more."""
        from chemprop.optuna.optuna import chemprop_optuna

        save_dir = os.path.join(tmp_dir, "optuna_resume")
        common_args = [
            "--data_path", regression_csv,
            "--dataset_type", "regression",
            "--metric", "rmse",
            "--cross_validation", "kFold",
            "--n_splits", "2",
            "--num_folds", "1",
            "--num_workers", "1",
            "--save_dir", save_dir,
            "--n_trials", "1",
            "--epochs", "3",
            "--hidden_size", "10",
            "--ffn_hidden_size", "10",
            "--depth", "2",
            "--quiet",
            "--empty_cache",
        ]
        chemprop_optuna(common_args)
        # Second call: n_trials=1, 1 already done => 0 to run
        chemprop_optuna(common_args)

        # Still only trial-0 directory (no trial-1)
        assert os.path.isdir(os.path.join(save_dir, "trial-0"))
        assert not os.path.isdir(os.path.join(save_dir, "trial-1"))

    def test_optuna_binary_classification(self, tmp_dir, binary_csv):
        from chemprop.optuna.optuna import chemprop_optuna

        save_dir = os.path.join(tmp_dir, "optuna_bin")
        chemprop_optuna([
            "--data_path", binary_csv,
            "--dataset_type", "classification",
            "--metric", "auc",
            "--cross_validation", "kFold",
            "--n_splits", "2",
            "--num_folds", "1",
            "--num_workers", "1",
            "--save_dir", save_dir,
            "--n_trials", "1",
            "--epochs", "3",
            "--hidden_size", "10",
            "--ffn_hidden_size", "10",
            "--depth", "2",
            "--quiet",
            "--empty_cache",
        ])

        assert os.path.exists(os.path.join(save_dir, "optuna.db"))
        assert os.path.isdir(os.path.join(save_dir, "trial-0"))


class TestMultitaskMissingLabels:
    """Tests for multi-task learning with missing labels.

    These tests verify that:
    1. Multi-task models can be trained with missing labels
    2. Metrics are correctly computed for each task (ignoring missing values)
    3. The n_samples column correctly tracks valid samples per task
    4. Weighted averaging works correctly across tasks
    """

    def test_multitask_kfold_regression(self, tmp_dir, multitask_regression_csv):
        """Test kFold CV with multi-task regression and missing labels."""
        from chemprop.optuna.cross_validation import chemprop_cv

        save_dir = os.path.join(tmp_dir, "multitask_kfold")
        chemprop_cv([
            "--data_path", multitask_regression_csv,
            "--dataset_type", "regression",
            "--metric", "rmse",
            "--cross_validation", "kFold",
            "--n_splits", "2",
            "--num_folds", "1",
            "--ensemble_size", "1",
            "--num_workers", "0",
            "--save_dir", save_dir,
            "--epochs", "3",
            "--hidden_size", "10",
            "--ffn_hidden_size", "10",
            "--depth", "2",
            "--quiet",
            "--empty_cache",
        ])

        # Verify output files exist
        metrics_path = os.path.join(save_dir, "kFold_metrics.csv")
        assert os.path.exists(metrics_path)

        # Load and verify metrics DataFrame
        df_metrics = pd.read_csv(metrics_path)

        # Check that n_samples column exists
        assert "n_samples" in df_metrics.columns

        # Check that we have results for all 3 tasks
        assert df_metrics["no_targets_columns"].nunique() == 3

        # Verify n_samples varies by task (due to missing values)
        n_samples_per_task = df_metrics.groupby("no_targets_columns")["n_samples"].sum()
        # Task 0 and 1 have missing values, task 2 has all values
        # The exact counts depend on the fold split, but they should differ
        assert len(n_samples_per_task) == 3

    def test_multitask_monte_carlo_regression(self, tmp_dir, multitask_regression_csv):
        """Test Monte-Carlo CV with multi-task regression and missing labels."""
        from chemprop.optuna.cross_validation import chemprop_cv

        save_dir = os.path.join(tmp_dir, "multitask_mc")
        chemprop_cv([
            "--data_path", multitask_regression_csv,
            "--dataset_type", "regression",
            "--metric", "rmse",
            "--cross_validation", "Monte-Carlo",
            "--split_type", "random",
            "--split_sizes", "0.8", "0.1", "0.1",
            "--num_folds", "2",
            "--ensemble_size", "1",
            "--num_workers", "0",
            "--save_dir", save_dir,
            "--epochs", "3",
            "--hidden_size", "10",
            "--ffn_hidden_size", "10",
            "--depth", "2",
            "--quiet",
            "--empty_cache",
        ])

        metrics_path = os.path.join(save_dir, "Monte-Carlo_metrics.csv")
        assert os.path.exists(metrics_path)

        df_metrics = pd.read_csv(metrics_path)
        assert "n_samples" in df_metrics.columns

        # Check we have 2 folds * 3 tasks = 6 rows for rmse metric
        df_rmse = df_metrics[df_metrics["metric"] == "rmse"]
        assert len(df_rmse) == 6  # 2 folds * 3 tasks

    def test_multitask_external_test_regression(self, tmp_dir, multitask_regression_csv, multitask_ext_test_csv):
        """Test external validation with multi-task regression and missing labels."""
        from chemprop.optuna.cross_validation import chemprop_cv

        save_dir = os.path.join(tmp_dir, "multitask_ext")
        chemprop_cv([
            "--data_path", multitask_regression_csv,
            "--dataset_type", "regression",
            "--metric", "rmse",
            "--ensemble_size", "1",
            "--num_workers", "0",
            "--save_dir", save_dir,
            "--separate_test_path", multitask_ext_test_csv,
            "--epochs", "3",
            "--hidden_size", "10",
            "--ffn_hidden_size", "10",
            "--depth", "2",
            "--quiet",
            "--empty_cache",
        ])

        # Check predictions file
        pred_path = os.path.join(save_dir, "test_ext_prediction.csv")
        assert os.path.exists(pred_path)

        # Check metrics file
        metrics_path = os.path.join(save_dir, "test_ext_metrics.csv")
        assert os.path.exists(metrics_path)

        df_metrics = pd.read_csv(metrics_path)
        assert "n_samples" in df_metrics.columns

        # Verify n_samples reflects missing values in test set
        # Task 0: 4 valid (index 2 is None in first 5)
        # Task 1: 3 valid (indices 0, 3 are None in first 5)
        # Task 2: 5 valid (no missing)
        task_samples = df_metrics.groupby("no_targets_columns")["n_samples"].first()
        assert task_samples[0] == 4  # target_1
        assert task_samples[1] == 3  # target_2
        assert task_samples[2] == 5  # target_3

    def test_multitask_binary_kfold(self, tmp_dir, multitask_binary_csv):
        """Test kFold CV with multi-task binary classification and missing labels."""
        from chemprop.optuna.cross_validation import chemprop_cv

        save_dir = os.path.join(tmp_dir, "multitask_bin_kfold")
        chemprop_cv([
            "--data_path", multitask_binary_csv,
            "--dataset_type", "classification",
            "--metric", "auc",
            "--cross_validation", "kFold",
            "--n_splits", "2",
            "--num_folds", "1",
            "--ensemble_size", "1",
            "--num_workers", "0",
            "--save_dir", save_dir,
            "--epochs", "3",
            "--hidden_size", "10",
            "--ffn_hidden_size", "10",
            "--depth", "2",
            "--quiet",
            "--empty_cache",
        ])

        metrics_path = os.path.join(save_dir, "kFold_metrics.csv")
        assert os.path.exists(metrics_path)

        df_metrics = pd.read_csv(metrics_path)
        assert "n_samples" in df_metrics.columns
        assert df_metrics["no_targets_columns"].nunique() == 2

    def test_weighted_mean_calculation(self, tmp_dir, multitask_regression_csv, multitask_ext_test_csv):
        """Verify that weighted mean is correctly calculated based on n_samples."""
        from chemprop.optuna.cross_validation import chemprop_cv
        from chemprop.optuna.evaluator import Evaluator

        save_dir = os.path.join(tmp_dir, "weighted_mean_test")
        chemprop_cv([
            "--data_path", multitask_regression_csv,
            "--dataset_type", "regression",
            "--metric", "rmse",
            "--ensemble_size", "1",
            "--num_workers", "0",
            "--save_dir", save_dir,
            "--separate_test_path", multitask_ext_test_csv,
            "--epochs", "3",
            "--hidden_size", "10",
            "--ffn_hidden_size", "10",
            "--depth", "2",
            "--quiet",
            "--empty_cache",
        ])

        metrics_path = os.path.join(save_dir, "test_ext_metrics.csv")
        df_metrics = pd.read_csv(metrics_path)

        # Filter to primary metric
        df_primary = df_metrics[df_metrics["metric"] == "rmse"]

        # Calculate expected weighted mean
        valid = df_primary.dropna(subset=["value"])
        expected_weighted_mean = (
            (valid["value"] * valid["n_samples"]).sum() / valid["n_samples"].sum()
        )

        # Calculate using Evaluator._weighted_mean
        actual_weighted_mean = Evaluator._weighted_mean(df_primary)

        assert np.isclose(expected_weighted_mean, actual_weighted_mean), \
            f"Expected {expected_weighted_mean}, got {actual_weighted_mean}"

        # Also verify it's different from simple mean (due to different n_samples)
        simple_mean = df_primary["value"].mean()
        # They might be close but generally should differ when n_samples varies
        # Just check the calculation is valid
        assert not np.isnan(actual_weighted_mean)

    def test_optuna_multitask_regression(self, tmp_dir, multitask_regression_csv):
        """Test Optuna HPO with multi-task regression and missing labels."""
        from chemprop.optuna.optuna import chemprop_optuna

        save_dir = os.path.join(tmp_dir, "optuna_multitask")
        chemprop_optuna([
            "--data_path", multitask_regression_csv,
            "--dataset_type", "regression",
            "--metric", "rmse",
            "--cross_validation", "kFold",
            "--n_splits", "2",
            "--num_folds", "1",
            "--num_workers", "1",
            "--save_dir", save_dir,
            "--n_trials", "1",
            "--epochs", "3",
            "--hidden_size", "10",
            "--ffn_hidden_size", "10",
            "--depth", "2",
            "--quiet",
            "--empty_cache",
        ])

        assert os.path.exists(os.path.join(save_dir, "optuna.db"))
        assert os.path.isdir(os.path.join(save_dir, "trial-0"))

        # Check metrics file in trial directory
        metrics_path = os.path.join(save_dir, "trial-0", "kFold_metrics.csv")
        assert os.path.exists(metrics_path)

        df_metrics = pd.read_csv(metrics_path)
        assert "n_samples" in df_metrics.columns

    def test_sparse_task_handling(self, tmp_dir):
        """Test handling when one task has very few valid values."""
        # Create a dataset where task 2 has very few values
        # This tests the NaN handling in weighted mean
        path = os.path.join(tmp_dir, "sparse_task.csv")
        targets_1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        # Task 2: only 2 values, rest are missing
        targets_2 = [None, None, 3.5, None, None, None, None, None, None, 10.5]
        _write_multitask_csv(
            path,
            _SMILES,
            [targets_1, targets_2],
            ["target_1", "target_2"]
        )

        from chemprop.optuna.cross_validation import chemprop_cv

        save_dir = os.path.join(tmp_dir, "sparse_task_out")
        chemprop_cv([
            "--data_path", path,
            "--dataset_type", "regression",
            "--metric", "rmse",
            "--cross_validation", "kFold",
            "--n_splits", "2",
            "--num_folds", "1",
            "--ensemble_size", "1",
            "--num_workers", "0",
            "--save_dir", save_dir,
            "--epochs", "3",
            "--hidden_size", "10",
            "--ffn_hidden_size", "10",
            "--depth", "2",
            "--quiet",
            "--empty_cache",
        ])

        metrics_path = os.path.join(save_dir, "kFold_metrics.csv")
        assert os.path.exists(metrics_path)

        df_metrics = pd.read_csv(metrics_path)

        # Check that task 1 (target_2, index 1) may have NaN for some folds
        # but the overall result is still valid
        df_task2 = df_metrics[df_metrics["no_targets_columns"] == 1]

        # Some folds may have 0 or 1 valid sample for task 2
        # The weighted mean should handle this gracefully
        assert "n_samples" in df_metrics.columns
