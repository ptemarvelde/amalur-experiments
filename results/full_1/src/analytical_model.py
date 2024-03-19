import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.util import read_data_chars, add_gpu_chars_to_df, read_gpu_chars, read_results
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from typing import Tuple, List
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from loguru import logger
import copy
    
def load_X() -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Load data for training the analytical model
    Returns:
        _type_: _description_
    """
    logger.info("Loading data for analytical model")
    result_df = read_results(from_parquet=True)
    result_df.drop(columns=result_df.columns[result_df.isnull().any()], inplace=True)
    result_df.drop(columns=['mem_mat_read', 'mem_mat_write',
        'mem_fac_read', 'mem_fac_write', 'comp_scalar_mat', 'comp_lmm_mat',
        'comp_rmm_mat', 'comp_scalar_fac', 'comp_lmm_fac', 'comp_rmm_fac',
        'comp_mat_col_major', 'comp_fac_col_major', 'comp_scalar_dense', '13',
        '14', 'comp_matrix_dense', 'mem_read_scalar_dense',
        'mem_write_scalar_dense', 'mem_read_matrix_dense',
        'mem_write_matrix_dense', 'mem_read_rowsum', 'mem_write_rowsum',
        'mem_read_colsum', 'mem_write_colsum', '24', '25', 'comp_rowsum',
        'comp_colsum', 'comp_mat', 'comp_fac', 'comp_ratio', 'tr', 'fr'], inplace=True)
    
    operator_metrics_df = pd.read_parquet('/home/pepijn/Documents/uni/y5/thesis/amalur/amalur-experiments/amalur-factorization/profiling/operator_metrics.parquet').reset_index()
    operator_metrics_df = add_gpu_chars_to_df(operator_metrics_df, 'gpu')
    operator_metrics_df['math_cost_seconds'] = operator_metrics_df['sm_active_cycles_sum'] / operator_metrics_df['sm_frequency_weighted_mean']
    operator_metrics_df['dram_bytes_sum'] = (operator_metrics_df['dram_bytes_read_sum'] + operator_metrics_df['dram_bytes_write_sum'])
    operator_metrics_df['mem_cost_seconds'] = operator_metrics_df['dram_bytes_sum'] / operator_metrics_df['memory_throughput_byte_weighted_mean']
    
    col_filtered = result_df.drop(columns=set(result_df.columns).intersection(operator_metrics_df.columns) - {'dataset', 'operator'})
    df = operator_metrics_df.merge(col_filtered, on=['dataset', 'operator'], how='left')
    df = df.groupby(['dataset', 'operator', 'gpu', 'type']).first().reset_index()
    data_characteristics_cols = col_filtered.select_dtypes(include=[np.number]).columns
    operator_metrics_cols = set(df.select_dtypes(include=np.number)) - set(data_characteristics_cols)

    dep = [x for x in operator_metrics_cols if not x.startswith("gpu_") and not x in ['speedup', 'times_mean', 'times_std', 'times_saved']]
    indep = [x for x in set(df.select_dtypes(include=np.number)) if x not in dep and x not in ['dataset']]
    logger.info(f"Loaded data for analytical model with {len(dep)} dependent variables and {len(indep)} independent variables")
    df['model'] = df['type']
    df.drop(columns=['type'], inplace=True)
    return df, dep, indep

def predict_linreg_ensemble(linreg_ensemble: dict, df: pd.DataFrame, X, y_col_names, split_by=['model', 'operator']):
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
    logger.info("Predicting results of the analytical model using a linear regression ensemble")
    y_preds = []
    assert set(X.index) == set(df.index), "Index of y and df should be the same"
    for split_value_tuple, group_df in df.groupby(split_by):
        preds = []
        idx = group_df.index
        
        for y_col in y_col_names:
            dict_keys = (*split_value_tuple, y_col) if isinstance(split_by, list) else (split_value_tuple, y_col)
            preds.append( linreg_ensemble[dict_keys].predict(X.loc[idx]))
        df_pred = pd.DataFrame(np.array(preds).T, columns=y_col_names, index=idx)
        y_preds.append(df_pred)
    return pd.concat(y_preds).sort_index()


def create_linreg_ensemble(df: pd.DataFrame, X, y: pd.DataFrame, clf_func=LinearRegression, clf_kwargs=None, split_by=['model', 'operator'], rfecv=True) -> dict:
    """ Create a linear regression ensemble for the analytical model.
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
            dict_keys = (*split_value_tuple, y_vals.name) if isinstance(split_by, list) else (split_value_tuple, y_vals.name)
            linreg_ensemble[dict_keys] = clf
    logger.info(f"Created a linear regression ensemble for the analytical model with {len(linreg_ensemble)} models")
    return linreg_ensemble