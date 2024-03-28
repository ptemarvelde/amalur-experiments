from collections import defaultdict
import pandas as pd
import numpy as np

np.random.seed(42)
import matplotlib.pyplot as plt
from src.util import read_data_chars, add_gpu_chars_to_df, read_gpu_chars, read_results, model_operators
import numpy as np

np.seterr(divide="ignore", invalid="ignore")
from typing import Tuple, List
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from loguru import logger
from tqdm import tqdm
tqdm.pandas()
import copy

from memoization import cached
from joblib import hash as hash_pandas


def load_X(metric_file="/home/pepijn/Documents/uni/y5/thesis/amalur/amalur-experiments/amalur-factorization/profiling/operator_metrics.parquet") -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Load data for training the analytical model
    Returns:
        _type_: _description_
    """
    logger.info("Loading data for analytical model")
    result_df = read_results(from_parquet=True)
    result_df.drop(columns=result_df.columns[result_df.isnull().any()], inplace=True)
    # result_df.drop(columns=['mem_mat_read', 'mem_mat_write',
    #     'mem_fac_read', 'mem_fac_write', 'comp_scalar_mat', 'comp_lmm_mat',
    #     'comp_rmm_mat', 'comp_scalar_fac', 'comp_lmm_fac', 'comp_rmm_fac',
    #     'comp_mat_col_major', 'comp_fac_col_major', 'comp_scalar_dense', '13',
    #     '14', 'comp_matrix_dense', 'mem_read_scalar_dense',
    #     'mem_write_scalar_dense', 'mem_read_matrix_dense',
    #     'mem_write_matrix_dense', 'mem_read_rowsum', 'mem_write_rowsum',
    #     'mem_read_colsum', 'mem_write_colsum', '24', '25', 'comp_rowsum',
    #     'comp_colsum', 'comp_mat', 'comp_fac', 'comp_ratio', 'tr', 'fr'], inplace=True)

    operator_metrics_df = pd.read_parquet(
        metric_file
    ).reset_index()
    # operator_metrics_df = add_gpu_chars_to_df(operator_metrics_df, 'gpu')
    operator_metrics_df["math_cost_seconds"] = (
        operator_metrics_df["sm_active_cycles_sum"] / operator_metrics_df["sm_frequency_weighted_mean"]
    )
    operator_metrics_df["dram_bytes_sum"] = (
        operator_metrics_df["dram_bytes_read_sum"] + operator_metrics_df["dram_bytes_write_sum"]
    )
    operator_metrics_df["mem_cost_seconds"] = (
        operator_metrics_df["dram_bytes_sum"] / operator_metrics_df["memory_throughput_byte_weighted_mean"]
    )
    operator_metrics_df["ops_per_second"] = (operator_metrics_df["compute_throughput_weighted_mean"] / 100) * (
        operator_metrics_df["gpu_processing_power_double_precision"] * 1e12
    )
    operator_metrics_df["arithmetic_intensity"] = (
        operator_metrics_df["ops_per_second"] * (operator_metrics_df.duration_sum / 1e9)
    ) / (operator_metrics_df["dram_bytes_sum"])

    col_filtered = result_df.drop(
        columns=set(result_df.columns).intersection(operator_metrics_df.columns) - {"dataset", "operator"}
    )
    df = operator_metrics_df.merge(col_filtered, on=["dataset", "operator"], how="left")
    df = df.groupby(["dataset", "operator", "gpu", "type"]).first().reset_index()

    df["model"] = df["type"]
    df.drop(columns=["type", "index"], inplace=True)

    data_characteristics_cols = col_filtered.select_dtypes(include=[np.number]).columns
    operator_metrics_cols = set(df.select_dtypes(include=np.number)) - set(data_characteristics_cols)
    runtime_targets = ["speedup", "times_mean", "times_std", "time_saved", "materialized_times_mean"]
    dep = [
        x
        for x in operator_metrics_cols
        if not x.startswith("gpu_")
        and not x in runtime_targets
    ]
    indep = [x for x in set(df.select_dtypes(include=np.number)) if x not in {*dep, *runtime_targets} and x not in ["dataset"]]
    logger.info(
        f"Loaded data for analytical model with {len(dep)} dependent variables and {len(indep)} independent variables"
    )

    return df, dep, indep


def predict_linreg_ensemble(linreg_ensemble: dict, df: pd.DataFrame, X, y_col_names, split_by=["model", "operator"]):
    """Predict the results of the analytical model using a linear regression ensemble.


    Args:
        linreg_ensemble (dict): Dict with tuple of type, operator, and y_col as key and a linear regression model as value
        df (pd.DataFrame): full dataframe that is iterated over (used for indexing X and y)
        X : Training data
        y_col_names : Target column names
        split_by (List[str], optional): List of columns used to split. Defaults to ['model', 'operator'].

    Returns:
        pd.DataFrame: Dataframe with the predictions. (multiple columns if y has multiple columns)
    """
    logger.debug("Predicting results of the analytical model using a linear regression ensemble")
    y_preds = []
    assert set(X.index) == set(df.index), "Index of y and df should be the same"
    for split_value_tuple, group_df in df.groupby(split_by):
        preds = []
        idx = group_df.index

        for y_col in y_col_names:
            dict_keys = (*split_value_tuple, y_col) if isinstance(split_by, list) else (split_value_tuple, y_col)
            preds.append(linreg_ensemble[dict_keys].predict(X.loc[idx]))
        df_pred = pd.DataFrame(np.array(preds).T, columns=y_col_names, index=idx)
        y_preds.append(df_pred)
    return pd.concat(y_preds).sort_index()

def predict_single_row_linreg_ensemble(linreg_ensemble, X, y_col_names, split_by=["model", "operator"]):
    preds = []
    split_tuple = list(X[col] for col in split_by)
    X = pd.DataFrame(X).T
    X=X.drop(columns=split_by)
    for y_col in y_col_names:
        dict_keys = (*split_tuple, y_col) if isinstance(split_by, list) else (X[split_by], y_col)
        preds.append(linreg_ensemble[dict_keys].predict(X))
    df_pred = pd.DataFrame(np.array(preds).T, columns=y_col_names, index=X.index)
    return df_pred

class LinRegEnsemble:
    def __init__(
        self,
        df: pd.DataFrame,
        X,
        y: pd.DataFrame,
        clf_func=LinearRegression,
        clf_kwargs={},
        split_by=["model", "operator"],
        rfecv=True,
        target_to_pred_func=None
    ):
        self.linreg_ensemble = create_linreg_ensemble(df, X, y, clf_func, clf_kwargs, split_by, rfecv)
        self.split_by = split_by
        self.y_col_names = y.columns if isinstance(y, pd.DataFrame) else [y.name]
        if target_to_pred_func is None:
            target_to_pred_func = lambda x: x
        self.target_to_pred_func = target_to_pred_func

    def predict(self, df, X):
        if isinstance(X, pd.Series):
            res = predict_single_row_linreg_ensemble(self.linreg_ensemble, X, self.y_col_names, self.split_by)
        else:
            res = predict_linreg_ensemble(self.linreg_ensemble, df, X, self.y_col_names, self.split_by)
        return self.target_to_pred_func(res)


def create_linreg_ensemble(
    df: pd.DataFrame,
    X,
    y: pd.DataFrame,
    clf_func=LinearRegression,
    clf_kwargs={},
    split_by=["model", "operator"],
    rfecv=True,
) -> dict:
    """Create a linear regression ensemble for the analytical model.
        Uses sklearn recursive feature elimination with cross-validation to select features.

    Args:
        df (pd.DataFrame): Dataframe with the data
        features (List[str]): Features for X
        X: X
        y (pd.DataFrame): DataFrame with target (dataframe as we support multiple target vars)
        clf (RegressionModel, optional): regression model. Defaults to LinearRegression().
        split_by (List[str], optional): List of columns to split the dataframe by. Defaults to ['model', 'operator'].
        rfecv (bool, optional): Whether to use recursive feature elimination with cross-validation. Defaults to True.

    Returns:
        dict: Dict with tuple of type, operator, and y_col as key and a linear regression model as value
    """
    logger.info(f"Creating a linear regression ensemble for the analytical model, splitting by {split_by}")
    linreg_ensemble = {}
    for (split_value_tuple), group_df in df.groupby(split_by):
        y_col_names = y.columns if isinstance(y, pd.DataFrame) else [y.name]
        for y_vals in [df[col] for col in y_col_names]:
            kwargs_copy = copy.deepcopy(clf_kwargs)
            min_features_to_select = 5  # Minimum number of features to consider
            if clf_kwargs:
                for key, value in kwargs_copy.items():
                    if callable(value):
                        kwargs_copy[key] = value()
            clf = clf_func(**kwargs_copy)
            if rfecv:
                cv = KFold(5)
                clf = RFECV(
                    estimator=clf,
                    step=1,
                    cv=cv,
                    min_features_to_select=min_features_to_select,
                    n_jobs=10,
                )

            selector = group_df.index
            clf.fit(X.loc[selector], y_vals.loc[selector])
            dict_keys = (
                (*split_value_tuple, y_vals.name) if isinstance(split_by, list) else (split_value_tuple, y_vals.name)
            )
            linreg_ensemble[dict_keys] = clf
    logger.info(f"Created a linear regression ensemble for the analytical model with {len(linreg_ensemble)} models")
    return linreg_ensemble

def custom_hash(self, operator, X):
    return operator + hash_pandas(X)

class ModelCost:
    """
    Calculate the cost of a model using the analytical model.
    """

    def __init__(self, operator_cost_clf, dataset: pd.DataFrame):
        """_summary_ create instance to calculate analytical cost

        Args:
            operator_cost_clf (_type_): Regressor to compute cost with
            dataset (pd.DataFrame): Full dataset (used for indexing when the regressor is an ensemble)
        """
        self.operator_cost_clf: LinRegEnsemble = operator_cost_clf
        self.dataset = dataset

    def calculate_cost(self, model_type, characteristics):
        # TODO check if all characteristics are present
        
        if model_type == "Linear Regression":
            costs = self.linear_regression(characteristics)
        elif model_type == "Gaussian":
            costs = self.gnmf(characteristics)
        elif model_type == "Logistic Regression":
            costs = self.logistic_regression(characteristics)
        elif model_type == "KMeans":
            costs = self.kmeans(characteristics)
        else:
            raise ValueError(f"Model type {model_type} not recognized. only {model_operators} are supported.")

        return {
            **costs,
            "sum": np.sum(list(costs.values()),axis=0)
        }

    @cached(custom_key_maker=custom_hash)
    def operator_cost(self, operator, X) -> Tuple[float, float]:
        """Predict cost for operator for given scenario X
        Returns:
            _type_: Tuple (mat_cost, fact_cost)
        """
        if operator == "elementwise":
            operator = "Left multiply"
        X['operator'] = operator
        X['model'] = "materialized"
        X2  = X.copy()
        X2['model'] = "factorized"
        return np.array([
            self.operator_cost_clf.predict(self.dataset, X),
            self.operator_cost_clf.predict(self.dataset, X2)
        ]).reshape(2,)

    def predict(self, df: pd.DataFrame):
        col = df.progress_apply(lambda x: self.calculate_cost(x["operator"], x.drop(columns=["operator"])), axis=1)
        return col

    # Currently disregard operators that are the same between the fact/mat case
    def linear_regression(self, characteristics):
        # AM = amalurmatrix, fact or mat
        # (X* self.w)                         = AM(rows, cols) * w(cols, 1)                   = fact/mat LMM
        # (X* self.w) - Y                     = prev(rows,1) - Y(rows,1)                      = element wise subtract
        # X.T * (X* self.w - Y)               = AMT(cols, rows) * prev(rows,1)                = fact/mat LMM T
        # gamma * (X.T * (X * self.w - Y))    = w(cols, 1) * prev(cols, 1)                    = elementwise multiplication
        # self.w -= gamma * (X.T * (X * self.w - Y)) = w(cols, 1) - prev(cols, 1)             = elementwise subtract
        costs = defaultdict(lambda : np.array([0., 0.]))
        costs["LMM"] += self.operator_cost("LMM", characteristics)
        costs["LMM T"] += self.operator_cost("LMM T", characteristics)
        # self.operator_cost("elementwise", rows, 1)
        # self.operator_cost("elementwise", cols, 1) * 2
        return costs

    def logistic_regression(self, characteristics):
        # TODO figure out why logreg speedup is twice linreg speedup
        costs = defaultdict(lambda : np.array([0., 0.]))
        costs["LMM"] += self.operator_cost("LMM", characteristics)
        costs["LMM T"] += self.operator_cost("LMM T", characteristics)
        # self.operator_cost("elementwise", rows, 1)
        # self.operator_cost("elementwise", cols, 1) * 2
        return costs
    
    def kmeans(self, characteristics):
        costs = defaultdict(lambda : np.array([0., 0.]))
        costs['exp'] += self.operator_cost("elementwise", characteristics)
        costs['rowSums'] += self.operator_cost("Row summation", characteristics)
        costs['mult'] += self.operator_cost("Right multiply", characteristics)
        costs['MM'] += self.operator_cost("LMM T", characteristics)
        return costs
    
    def gnmf(self, characteristics):
        costs = defaultdict(lambda : np.array([0., 0.]))
        costs['MM'] += self.operator_cost("LMM", characteristics)
        costs['MM'] += self.operator_cost("RMM", characteristics)
        return costs
        
