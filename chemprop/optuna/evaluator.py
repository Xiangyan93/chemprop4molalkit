#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import List, Optional, Literal, Tuple
import os
import math
from tqdm import tqdm
import pandas as pd
import numpy as np
import inspect
from chemprop import data
from rdkit import Chem
from sklearn.model_selection import KFold
from chemprop.data.data import MoleculeDataset
from mgktools.data.split import data_split_index
from mgktools.evaluators.metric import Metric, metric_regression, metric_binary


def get_data_from_index(dataset: MoleculeDataset, index):
    """Get a subset of the dataset based on the provided index.

    Args:
        dataset (MoleculeDataset): The dataset to sample from.
        index (int): The index of the data point to retrieve.

    Returns:
        MoleculeDatapoint: The data point at the specified index.
    """
    subset = [dataset[i] for i in index]
    return MoleculeDataset(subset)


def dataset_split(dataset,
                  split_type: Literal['random', 'scaffold_order', 'scaffold_random', 'init_al', 'stratified'] = None,
                  sizes: List[float] = [0.8, 0.2],
                  seed: int = 0) -> List:
    """ Split the data set into two data sets: training set and test set.

    Parameters
    ----------
    split_type: The algorithm used for data splitting.
    sizes: [float, float].
        If split_type == 'random' or 'scaffold_balanced'.
        sizes are the percentages of molecules in training and test sets.
    seed: int

    Returns
    -------
    [Dataset, Dataset]
    """
    data = []
    if split_type in ['random', 'stratified']:
        mols = None
    else:
        mols = []
        for m in dataset.mols:
            assert len(m) == 1
            mols.append(m[0])
    split_index = data_split_index(n_samples=len(dataset),
                                   mols=mols,
                                   targets=dataset.y,
                                   split_type=split_type,
                                   sizes=sizes,
                                   seed=seed)
    for s_index in split_index:
        data.append(get_data_from_index(dataset, s_index))
    return data


class Evaluator:
    def __init__(self,
                 save_dir: str,
                 dataset: MoleculeDataset,
                 model,
                 task_type: Literal["regression", "classification", "multi-class"],
                 metrics: List[Metric] = None,
                 cross_validation: Literal["kFold", "Monte-Carlo", "no"] = "Monte-Carlo",
                 n_splits: int = None,
                 split_type: Literal["random", "scaffold_order", "scaffold_random"] = None,
                 split_sizes: List[float] = None,
                 num_folds: int = 1,
                 seed: int = 0,
                 verbose: bool = True
                 ):
        """Evaluator object to evaluate the performance of the machine learning model.

        Parameters
        ----------
        save_dir:
            The directory that save all output files.
        dataset:
            The dataset used for cross-validation or as the training data.
        model:
            The machine learning model.
        task_type:
            The type of the task: "regression", "binary", "multi-class".
        metrics:
            The metrics used to evaluate the model.
        cross_validation:
            The type of cross-validation: "kFold", "leave-one-out", "Monte-Carlo", "no".
        n_splits:
            Number of folds. Must be at least 2.
        split_type:
            The type of the data split.
        split_sizes:
            The sizes of the data split.
        num_folds:
            The number of folds for cross-validation.
        seed:
            random seed for cross-validation.
        """
        self.save_dir = save_dir
        if self.write_file:
            if not os.path.exists(self.save_dir):
                os.mkdir(self.save_dir)
            self.logfile = open("%s/results.log" % self.save_dir, "w")
        self.dataset = dataset
        self.model = model
        self.task_type = task_type
        self.cross_validation = cross_validation
        self.n_splits = n_splits
        self.split_type = split_type
        self.split_sizes = split_sizes
        self.metrics = metrics
        self.num_folds = num_folds
        self.seed = seed
        self.verbose = verbose
        self.skip_train = False

    @property
    def write_file(self) -> bool:
        if self.save_dir is None:
            return False
        else:
            return True

    def run_cross_validation(self) -> float:
        if self.cross_validation == "kFold":
            assert self.n_splits is not None, "n_splits must be specified for nfold cross-validation."
            # repeat cross-validation for num_folds times
            df_metrics = pd.DataFrame(columns=["metric", "no_targets_columns", "value", "seed", "split"])
            for i in range(self.num_folds):
                kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed + i)
                kf.get_n_splits(self.dataset.data)
                for i_fold, (train_index, test_index) in enumerate(kf.split(self.dataset.data)):
                    dataset_train = get_data_from_index(self.dataset, train_index)
                    dataset_test = get_data_from_index(self.dataset, test_index)
                    df_predict, df_ = self.evaluate_train_test(dataset_train, dataset_test)
                    df_predict.to_csv("%s/kFold_%d-%d_prediction.csv" % (self.save_dir, i, i_fold), index=False)
                    df_["seed"] = self.seed + i
                    df_["split"] = i_fold
                    df_metrics = pd.concat([df_metrics, df_], ignore_index=True)
            df_metrics.to_csv("%s/kFold_metrics.csv" % self.save_dir, index=False)
            self.log("kFold cross-validation performance:")
            self.log_metrics(df_metrics)
            return df_metrics['value'].mean()
        elif self.cross_validation == "Monte-Carlo":
            assert self.split_type is not None, "split_type must be specified for Monte-Carlo cross-validation."
            assert self.split_sizes is not None, "split_sizes must be specified for Monte-Carlo cross-validation."
            df_metrics = pd.DataFrame(columns=["metric", "no_targets_columns", "value", "seed"])
            for i in range(self.num_folds):
                if len(self.split_sizes) == 2:
                    dataset_train, dataset_test = dataset_split(
                        self.dataset,
                        split_type=self.split_type,
                        sizes=self.split_sizes,
                        seed=self.seed + i)
                # the second part, validation set, is abandoned.
                elif len(self.split_sizes) == 3:
                    dataset_train, _, dataset_test = dataset_split(
                        self.dataset,
                        split_type=self.split_type,
                        sizes=self.split_sizes,
                        seed=self.seed + i)
                else:
                    raise ValueError("split_sizes must be 2 or 3.")
                df_predict, df_ = self.evaluate_train_test(dataset_train, dataset_test)
                df_predict.to_csv("%s/test_%d_prediction.csv" % (self.save_dir, i), index=False)
                df_["seed"] = self.seed + i
                df_metrics = pd.concat([df_metrics, df_], ignore_index=True)
            df_metrics.to_csv("%s/Monte-Carlo_metrics.csv" % self.save_dir, index=False)
            self.log("Monte-Carlo cross-validation performance:")
            self.log_metrics(df_metrics)
            return df_metrics['value'].mean()
        elif self.cross_validation == "no":
            raise ValueError("When set cross_validation to 'no', please use run_external() not eval_cross_validation.")
        else:
            raise ValueError("Unsupported cross-validation method %s." % self.cross_validation)

    def run_external(self, dataset_test: MoleculeDataset, name='test_ext'):
        # assert self.cross_validation == "no", "cross_validation must be 'no' for run_external()."
        df_predict, df_metrics = self.evaluate_train_test(self.dataset, dataset_test, skip_train=self.skip_train)
        self.skip_train = True
        df_predict.to_csv("%s/%s_prediction.csv" % (self.save_dir, name), index=False)
        if df_metrics is not None:
            # Calculate metrics values.
            df_metrics.to_csv("%s/%s_metrics.csv" % (self.save_dir, name), index=False)
            self.log("External test set performance:")
            self.log_metrics(df_metrics)
        return df_metrics['value'].mean()

    def evaluate_train_test(self, dataset_train: MoleculeDataset,
                            dataset_test: MoleculeDataset, skip_train=False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if not skip_train:
            self.model.fit_molalkit(dataset_train)
        y_preds = self.model.predict_value(dataset_test)
        y_true = np.array(dataset_test.raw_targets())
        pred_dict = {}
        for i in range(dataset_train.num_tasks()):
            pred_dict["target_%d" % i] = y_true[:, i]
            pred_dict["predict_%d" % i] = y_preds[:, i] if y_preds.ndim > 1 else y_preds
        df_predict = pd.DataFrame(pred_dict)
        metrics_data = []
        for metric in self.metrics:
            for i in range(dataset_train.num_tasks()):
                v = self.eval_metric(y_true[:, i], 
                                     y_preds[:, i] if y_preds.ndim > 1 else y_preds, 
                                     metric)
                metrics_data.append([metric, i, v])
        df_metrics = pd.DataFrame(metrics_data, columns=["metric", "no_targets_columns", "value"])
        return df_predict, df_metrics

    def eval_metric(self, y, y_pred, metric):
        if y.dtype == object:
            y = np.asarray(y, dtype=float)
        mask = ~np.isnan(y) & ~np.isnan(y_pred)
        y = y[mask]
        y_pred = y_pred[mask]
        if len(y) == 0:
            return float('nan')
        elif self.task_type == "regression":
            return metric_regression(y, y_pred, metric)
        elif self.task_type == "classification":
            return metric_binary(y, y_pred, metric)
        elif self.task_type == "multi-class":
            raise NotImplementedError("multi-class classification is not supported yet.")
            # return metric_multiclass(y, y_pred, metric)

    def log_metrics(self, df_metrics: pd.DataFrame):
        N_targets_columns = df_metrics["no_targets_columns"].max() + 1
        for metric in self.metrics:
            df_ = df_metrics[df_metrics["metric"] == metric]
            assert len(df_) > 0
            if len(df_) == 1:
                self.log(f"Metric({metric}): %.5f" % df_["value"].iloc[0])
            else:
                self.log(f"Metric({metric}): %.5f +/- %.5f" % (df_["value"].mean(), df_["value"].std()))
        for i in range(N_targets_columns):
            df_ = df_metrics[df_metrics["no_targets_columns"] == i]
            assert len(df_) > 0
            if len(df_) == 1:
                self.log(f"Target({i}): %.5f" % df_["value"].iloc[0])
            else:
                self.log(f"Target({i}): %.5f +/- %.5f" % (df_["value"].mean(), df_["value"].std()))
        for i in range(N_targets_columns):
            for metric in self.metrics:
                df_ = df_metrics[(df_metrics["metric"] == metric) & (df_metrics["no_targets_columns"] == i)]
                assert len(df_) > 0
                if len(df_) == 1:
                    self.log(f"Target({i}),Metric({metric}): %.5f" % df_["value"].iloc[0])
                else:
                    self.log(f"Target({i}),Metric({metric}): %.5f +/- %.5f" % (df_["value"].mean(), df_["value"].std()))

    def log(self, info: str):
        if self.verbose:
            if self.write_file:
                self.logfile.write(info + "\n")
            else:
                print(info)
        else:
            pass
    