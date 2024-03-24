from typing import Tuple

import pandas as pd
import os
import glob
from collections import defaultdict
import numpy as np
import json

import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error, precision_score, recall_score, f1_score, confusion_matrix, r2_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from src.estimators import MorpheusFI

from loguru import logger
import copy

pd.options.mode.chained_assignment = None  # default='warn'


def _clean_synth_dataset_name(name: str):
    "Strip anything before the actual data characteristics start"
    return name[name.find("n_R") :]


def read_runtime(dataset_type, compute_type, from_raw=False) -> pd.DataFrame:
    if dataset_type not in ["synthetic", "hamlet", "tpc_ai"]:
        raise ValueError(
            f"Invalid 'dataset_type' specified: {dataset_type}. Must be one of 'synthetic', 'hamlet', 'tpc_ai'"
        )

    if compute_type not in ["gpu", "cpu"]:
        raise ValueError(f"Invalid 'compute_type' specified: {compute_type}. Must be one of 'cpu', 'gpu'")
    res = None
    outfile = f"daic/runtime/{compute_type}/runtime_{dataset_type}.parquet"
    if from_raw or not os.path.exists(outfile):
        dfs = []
        for f in glob.glob(f"daic/runtime/{compute_type}/{dataset_type}/*.log"):
            print(f"Reading {f} for dataset_type={dataset_type} and compute_type={compute_type}")
            temp_df = pd.read_json(f, lines=True)
            temp_df["source_file"] = f

            temp_df["compute_type"] = compute_type
            temp_df["dataset_type"] = dataset_type

            if dataset_type == "synthetic":
                temp_df["dataset"] = temp_df.dataset.apply(_clean_synth_dataset_name)

            if compute_type == "gpu":
                temp_df["compute_unit"] = f.split("/")[-1].split("_")[0].replace(".log", "")
            else:
                temp_df["compute_unit"] = temp_df.num_cores.apply(lambda x: f"CPU {x:02}c")

            dfs.append(temp_df)

        if len(dfs) > 0:
            df = pd.concat(dfs)
            df = df[~df.operator.str.contains("fail")]
            df["times"] = df["times"].apply(lambda x: x[1:])  # first repetition is warm up
            df["times_mean"], df["times_std"] = df["times"].apply(np.mean), df["times"].apply(np.std)
            print(f"writing to {outfile}")
            df.to_parquet(outfile)
            res = df
    else:
        res = pd.read_parquet(outfile)
    return res


def read_features(type_="synthetic") -> pd.DataFrame:
    features = pd.read_json(f"daic/features/features_{type_}.jsonl", lines=True)

    if type_ == "synthetic":
        features["dataset"] = features["dataset"].apply(_clean_synth_dataset_name)

    features.drop_duplicates(["dataset", "join", "operator"], inplace=True)
    return features.reset_index(drop=True)


def read_results(from_parquet=True, add_gpu_chars=True, overwrite=True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reads all results and preprocesses.
    """
    outfile = "daic/preprocessed.parquet"
    if from_parquet and os.path.exists(outfile):
        df = pd.read_parquet(outfile)
    else:
        runtimes, features, data_chars = [], [], []
        for dataset_type in ["synthetic", "hamlet", "tpc_ai"]:
            for compute_type in ["cpu", "gpu"]:
                runtimes.append(read_runtime(dataset_type, compute_type, from_raw=False))
            features.append(read_features(dataset_type))
            data_chars.append(read_data_chars(dataset_type))

        data_chars = pd.concat(data_chars)
        df = pd.concat([x for x in runtimes if x is not None])
        features = pd.concat(features)

        df = preprocess(df, features, data_chars, add_gpu_chars=add_gpu_chars)
        if overwrite:
            print(f"writing to {outfile}")
            df.to_parquet(outfile)

    return df


def read_train_test_validate_X_y(include_cpu=False):
    df = read_results(from_parquet=True)[:10_000]
    if not include_cpu:
        df = df[df.compute_type == "gpu"]
    df.drop(
        columns=["speedup", "num_cores", "materialized_times_mean", "times_mean", "label", "dataset"], inplace=True
    )
    train, test, validate = train_test_validate_split(df, test_fraction=0.3)
    X_train, y_train = train, train["time_saved"]
    X_test, y_test = test, test["time_saved"]
    X_validate, y_validate = validate, validate["time_saved"]

    # preprocess
    transform = feature_transform_pipe(model=None, X_train=X_train, fillna=True)
    logger.info(f"Made train/test/validate split, now transforming Trainset")
    X_train = transform.fit_transform(X_train, y_train)
    logger.info("Transforming Testset")
    X_test = transform.transform(X_test)
    logger.info("Transforming Validate set")
    X_validate = transform.transform(X_validate)

    return X_train, y_train, X_test, y_test, X_validate, y_validate


def describe(df, name):
    print(f"{name} set:")
    print(f"\tRecords: {len(df)}")

    y = df.time_saved > 0.0 if "time_saved" in df.columns else df.label

    pos, neg = len(df[y]), len(df[~y])
    print(f"\tPositive (speedup > 1 with factorizing)/Negative: {pos}/{neg} = {pos/neg:.2f} s")
    print(f"\tDataset types: {df.dataset_type.unique()}")
    print(f"\Compute Units: {df.compute_unit.unique()}")


def train_test_validate_split(df, test_fraction=0.3):
    """In the thesis what is called the test set is actually the validation set. The test set here the synthetic data."""
    df = df.copy()
    to_drop = {"dataset", "source_file", "features"}.intersection(df.columns)
    df.drop(columns=to_drop, inplace=True)
    gpu_col = "compute_unit"
    assert "p100" in df[gpu_col].unique()
    validate = df[(df["compute_unit"] == "p100") | (df["dataset_type"] != "synthetic")]
    train, test = train_test_split(df[~df.index.isin(validate.index)], test_size=test_fraction, random_state=42)
    to_drop = ["dataset_type", "compute_unit"]

    describe(train, "train")
    describe(test, "test")
    describe(validate, "validate")

    train.drop(columns=to_drop, inplace=True)
    test.drop(columns=to_drop, inplace=True)
    validate.drop(columns=to_drop, inplace=True)
    return train, test, validate


def read_data_chars(type_="synthetic", base_path="daic"):
    print(f"reading {base_path}/features/data_chars_{type_}.jsonl")
    data_chars = (
        pd.read_json(f"{base_path}/features/data_chars_{type_}.jsonl", lines=True)
        .groupby(["dataset", "join"], as_index=False)
        .first()[["data_characteristics", "dataset", "join"]]
    )
    data_chars.reset_index(drop=True, inplace=True)

    if type_ == "synthetic":
        data_chars["dataset"] = data_chars["dataset"].apply(_clean_synth_dataset_name)

    data_chars = data_chars.merge(
        pd.json_normalize(data_chars.data_characteristics), left_index=True, right_index=True
    )
    return data_chars


def _ratio(df, column, output_column_name=None, hardware_var=None, override_merge_keys=None):
    if not output_column_name:
        output_column_name = column + "_ratio"
    if not hardware_var:
        hardware_var = "compute_unit"

    merge_keys = tuple(override_merge_keys) if override_merge_keys else ("dataset", "operator", "join", hardware_var)
    df[column].fillna(0.0)
    f = df[df.model == "materialized"][[column, *merge_keys]]
    m = df

    lsuffix = "_materialized"
    df = f.merge(m, on=merge_keys, how="inner", suffixes=(lsuffix, ""))

    df[output_column_name] = df[column + lsuffix] / df[column]

    return df


feature_names = [
    "mem_mat_read",  # 0: materialization(MA) memory read / memory bandwidth
    "mem_mat_write",  # 1: MA memory write / memory bandiwidth
    "mem_fac_read",  # 2: factorization(FA) memory read / memory bandwidth
    "mem_fac_write",  # 3: FA memory write / memory bandwidth
    "comp_scalar_mat",  # 4: scalar ops complexity in MA / total MA complexity
    "comp_lmm_mat",  # 5: LMM ops complexity in MA / total MA complexity
    "comp_rmm_mat",  # 6: RMM ops complexity in MA / total MA complexity
    "comp_scalar_fac",  # 7: scalar ops complexity in FA / total MA complexity
    "comp_lmm_fac",  # 8: LMM ops complexity in FA / total FA complexity
    "comp_rmm_fac",  # 9: RMM ops complexity in FA / total FA complexity
    "comp_mat_col_major",  # 10: Column-major access ops complexity in MA / total MA complexity
    "comp_fac_col_major",  # 11: Column-major access ops complexity in FA / total FA complexity
    "comp_scalar_dense",  # 12: dense scalar ops complexity / parallelism
    "13",
    "14",
    "comp_matrix_dense",  # 15: dense matrix multiplication complexity / paralellism
    "mem_read_scalar_dense",  # 16: dense scalar ops memory read / memory bandwidth
    "mem_write_scalar_dense",  # 17: dense scalar ops memory write / memory bandwidth
    "mem_read_matrix_dense",  # 18: dense MM memory read / memory bandwidth
    "mem_write_matrix_dense",  # 19: dense MM memory write / memory bandwidth
    "mem_read_rowsum",  # 20: Rowsum ops memory read / memory bandwidth
    "mem_write_rowsum",  # 21: Rowsum ops memory write / memory bandiwdth
    "mem_read_colsum",  # 22: Colsum ops memory read / memory bandwidth
    "mem_write_colsum",  # 23: Colsum ops memory write / memory bandwidth
    "24",
    "25",
    "comp_rowsum",  # 26: Rowsum ops complexity / parallelism
    "comp_colsum",  # 27: Colsum ops complexity / parallelism
    "comp_mat",  # 28: total MA complexity / parallelism
    "comp_fac",  # 29: total FA complexity / parallelism
    "comp_ratio",  # 30: complexity ratio
    "tr",  # 31: Morpheous TR
    "fr",  # 32: Morpheous FR
]

model_operators = ["Linear Regression", "Gaussian", "Logistic Regression", "KMeans"]


def read_gpu_chars():
    with open(
        "/home/pepijn/Documents/uni/y5/thesis/amalur/amalur-experiments/results/full_1/daic/features/gpu-characteristics.json"
    ) as f:
        gpu_chars = json.load(f)
    gpu_chars["1080"] = gpu_chars.pop("1080Ti")
    gpu_chars["p100"] = gpu_chars.pop("P100")
    gpu_chars["v100"] = gpu_chars.pop("V100")
    gpu_chars["2080"] = gpu_chars.pop("2080Ti")
    gpu_chars["a40"] = gpu_chars.pop("A40")
    gpu_chars["a10g"] = gpu_chars.pop("A10G")
    gpu_chars["1660"] = gpu_chars.pop("1660Ti")
    gpu_chars_df = pd.DataFrame(gpu_chars).T.apply(pd.to_numeric, errors="ignore")
    gpu_chars_df.rename(columns={x: f"gpu_{x}" for x in gpu_chars_df.columns}, inplace=True)
    return gpu_chars_df


def add_gpu_chars_to_df(df, gpu_col_name="compute_unit"):
    gpu_chars_df = read_gpu_chars()
    gpu_chars_df.index.name = gpu_col_name
    df = df.merge(gpu_chars_df, how="left", left_on=gpu_col_name, right_on=gpu_col_name)
    return df


def preprocess(runtime: pd.DataFrame, features: pd.DataFrame, data_chars: pd.DataFrame, add_gpu_chars=True):
    res = runtime
    # features = features[features.operator.isin(model_operators)]
    # res = res[res.operator.isin(model_operators)]
    res.reset_index(drop=True, inplace=True)
    if "join" not in res.columns:
        res["join"] = "preset"

    res = _ratio(res, "times_mean", "speedup")
    res = _ratio(res, "complexity", "complexity_ratio")

    res = res[res.model == "factorized"][
        [
            "dataset",
            "speedup",
            "operator",
            "num_cores",
            "selectivity",
            "cardinality_T",
            "cardinality_S",
            "join",
            "compute_unit",
            "complexity_ratio",
            "times_mean",
            "source_file",
            "dataset_type",
            "compute_type",
            # 'r_T', 'c_T', 'r_S', 'c_S','Snonzero', 'Tnonzero'
        ]
    ]

    res["label"] = res.speedup > 1.0

    res = pd.merge(
        res,
        features[["dataset", "operator", "features", "join"]],
        how="inner",
        on=["dataset", "operator", "join"],
    )

    if "parallelism" not in res.columns:
        res["parallelism"] = None

    processed_features = res[["features", "parallelism"]].apply(lambda row: postprocessing_features(*row), axis=1)
    res[feature_names] = pd.DataFrame(processed_features.tolist(), index=res.index)

    overlap_cols = set(res.columns).intersection(data_chars.columns).difference({"dataset", "join"})
    overlap_cols.add("data_characteristics")
    data_chars = data_chars.drop(columns=overlap_cols).drop_duplicates(["dataset", "join"])

    res = pd.merge(res, data_chars, how="left", on=["dataset", "join"])

    res["morpheusfi_p"] = res.sparsity_S.apply(len)

    def get_features_morpheusfi(dataset, sparsity_S: list, r_T, r_S):
        if dataset in ["yelp", "movie", "lastfm", "book"]:
            sparsity_Ri, r_Ri = sparsity_S, r_S
        else:
            sparsity_Ri, r_Ri = sparsity_S[1:], r_S[1:]

        q = len([x for x in sparsity_Ri if x > 0.95])
        return q, sparsity_Ri, r_T, r_Ri

    res[["morpheusfi_q", "morpheusfi_eis", "morpheusfi_ns", "morpheusfi_nis"]] = res[
        ["dataset", "sparsity_S", "r_T", "r_S"]
    ].apply(lambda r: get_features_morpheusfi(*r), axis=1, result_type="expand")
    if add_gpu_chars:
        res = add_gpu_chars_to_df(res)
    res["materialized_times_mean"] = res.times_mean * res.speedup
    res["time_saved"] = res.materialized_times_mean - res.times_mean

    def extract_join(row):
        if row["join"] == "preset":
            try:
                return row["dataset"].split("join=")[1].split("/")[0]
            except Exception:
                return "inner"
        return row["join"]

    res["join"] = res[["join", "dataset"]].apply(extract_join, axis=1)
    return res[~res.operator.isin(["Noop", "Materialization"])]


def postprocessing_features(raw_features, parallelism=8000):
    try:
        if not parallelism:
            parallelism = 8000
        raw_features = np.array(raw_features)
        output = np.zeros(33)
        if parallelism > 1000:
            mem_band = 208 * 10**9  # st4 cluster
        else:
            mem_band = 689 * 10**9
        output[0:4] = raw_features[0:4] / mem_band
        output[4:7] = raw_features[4:7] / raw_features[24]
        output[7:10] = raw_features[7:10] / raw_features[25]
        output[10] = raw_features[10] / raw_features[24]
        output[11] = raw_features[11] / raw_features[25]
        output[12] = raw_features[12] / parallelism
        # output[12]=raw_features[12]/raw_features[24]
        # output[13]=raw_features[12]/raw_features[25]
        # output[14]=raw_features[13]/raw_features[24]
        # output[15]=raw_features[13]/raw_features[25]
        output[15] = raw_features[13] / parallelism
        output[16:20] = raw_features[14:18] / mem_band
        output[20:24] = raw_features[[19, 20, 22, 23]] / mem_band
        # output[24:26]=raw_features[[18,21]]/raw_features[24]
        output[26:28] = raw_features[[18, 21]] / parallelism
        # output[26:28]=raw_features[[18,21]]/raw_features[25]
        output[28] = raw_features[24] / parallelism
        output[29] = raw_features[25] / parallelism
        output[30] = raw_features[24] / raw_features[25]
        output[31:33] = raw_features[26:28]
    except Exception as e:
        output = np.zeros(33)
    return output


def main():
    df = read_results(from_parquet=False)
    print(df)


if __name__ == "__main__":
    main()


class ExplodeColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns, max_len):
        self.columns = columns
        self.max_len = max_len

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for column in tqdm(self.columns, desc="Expanding columns"):
            expanded = X[column].apply(pd.Series)
            if expanded.shape[1] < self.max_len:
                for i in range(self.max_len - expanded.shape[1]):
                    expanded[f"missing_{i}"] = np.nan
            elif expanded.shape[1] > self.max_len:
                expanded = expanded.iloc[:, : self.max_len]
            expanded = expanded.rename(columns=lambda x: f"{column}_{x}")
            X = pd.concat([X[:], expanded[:]], axis=1)
            X = X.drop(column, axis=1)
        return X


all_cat_features = [
    "compute_unit",
    "compute_type",
    "join",
    "_architecture",
    "operator",
]
all_cat_features = [*all_cat_features, *[f"gpu_{x}" for x in all_cat_features]]


def feature_transform_pipe(model=None, X_train=None, fillna=False,):
    if X_train is None:
        raise ValueError("X_train must be provided to feature_transform_pipe")
    categorical_features = list(set(all_cat_features).intersection(X_train.columns))
    logger.info(f"{categorical_features}")
    # Convert categorical features to strings
    list_features = [x for x in X_train.columns if X_train[x].apply(lambda x: isinstance(x, (list, np.ndarray))).any()]
    numeric_features = [x for x in X_train.columns if x not in categorical_features + list_features]
    transformers = [
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("list", ExplodeColumns(list_features, 5), list_features),
    ]

    preprocessor = ColumnTransformer(transformers=transformers)
    if fillna:
        imputer = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=0.0)
        pipe = make_pipeline(preprocessor, imputer, model)
    else:
        pipe = make_pipeline(preprocessor, model)
    return pipe


def train_and_score(model, X_train, X_test, y_train, y_test, full_dataset=None, fillna=False, target_col=None):
    pipe = feature_transform_pipe(model=model, X_train=X_train, fillna=fillna)
    pipe.fit(X_train, y_train)
    if target_col is None:
        try:
            target_col = y_test.name
        except:
            pass
    eval_model(pipe, X_test, y_test, full_dataset=full_dataset, target_col=target_col)
    return pipe


def _get_model_name(model):
    return model.steps[-1][0] if isinstance(model, Pipeline) else model.__class__.__name__


def eval_model(model, X_test, y_test, full_dataset=None, plot=True, target_col=None):
    logger.info(f"Model {model.__class__}, {_get_model_name(model)}\n test cols: {X_test.columns}, target_col: {target_col}")
    y_pred = pd.Series(model.predict(X_test), index=X_test.index)

    logger.info(f"Score: {model.score(X_test, y_test)}")

    result, fig, speedup_dict_ = eval_result(
        y_test, y_pred=y_pred, full_dataset=full_dataset, model_name=_get_model_name(model), plot=plot, target_col=target_col
    )
    # if not fig and plot:
    #     fig = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap="bone", text_kw={"size": 20})
    return result, fig, speedup_dict_


def speedup_metrics(index_selector: pd.Series, df: pd.DataFrame) -> dict:
    """
    We're interested in the cases our predictor predicts as positive (factorize).
    For these cases we want to evaluate the time saved by factorizing.
    """
    df = df[index_selector]
    mat_time = df.materialized_times_mean.sum()  # Time without estimator
    fact_time = df.times_mean.sum()  # Time with estimator
    best_time = df[["times_mean", "materialized_times_mean"]].min(axis=1).sum()  # Best possible time
    time_saved = mat_time - fact_time  # Time saved with estimator
    # Average speedup value of positive cases (does not take into account time saved by factorization)
    speedup_avg = df.speedup.mean()
    speedup_real = mat_time / fact_time  # Realized speedup of positive cases
    return {
        "mat_time": mat_time,
        "fact_time": fact_time,
        "best_time": best_time,
        "time_saved": time_saved,
        "speedup_avg": speedup_avg,
        "speedup_real": speedup_real,
    }


def eval_result(y_true, y_pred, full_dataset=None, model_name="", plot=False, target_col=None):
    y_true = y_true.copy()
    res = {}
    if target_col is None:
        target_col = y_true.name
    
    if (np.issubdtype(y_true.dtype, np.number) and np.issubdtype(y_pred.dtype, np.number)):
        scoring_functions = {
            "r2": r2_score,
            "mean_squared_error": mean_squared_error,
        }
        for name, function in scoring_functions.items():
            res[name] = function(y_true, y_pred)
    
    if target_col is not None:        
        if target_col == 'time_saved':
            y_true = y_true > 0.0 if np.issubdtype(y_true.dtype, np.number) else y_true
            y_pred = y_pred > 0.0 if np.issubdtype(y_pred.dtype, np.number) else y_pred
        elif target_col == 'speedup':
            y_true = y_true > 1.0 if np.issubdtype(y_true.dtype, np.number) else y_true
            y_pred = y_pred > 1.0 if np.issubdtype(y_pred.dtype, np.number) else y_pred

    # if y_true.dtype == "float64":
    #     # logger.debug("Assuming y_true is 'time_saved', converting to bool")
    #     y_true = y_true > 0.0

    # if y_pred.dtype == "float64":
    #     logger.info("Assuming y_pred is 'time_saved', converting to bool")
    #     y_pred = y_pred > 0.0

    scoring_functions = {
        "accuracy": accuracy_score,
        "precision": precision_score,
        "recall": recall_score,
        "f1": f1_score,
    }
    for name, function in scoring_functions.items():
        res[name] = function(y_true, y_pred)

    fig = None
    speedup_dict = {}
    if full_dataset is not None:
        if not isinstance(y_true, pd.Series):
            y_true = pd.Series(y_true)
        if not isinstance(y_pred, pd.Series):
            y_pred = pd.Series(y_pred).astype(bool)
            y_pred.index=y_true.index

        if len(y_pred.unique()) < 2:
            logger.warning("WARNING all predicted labels are the same: ", y_pred.unique())

        true_speedup = full_dataset.speedup
        timing_df = full_dataset[["times_mean", "materialized_times_mean", "speedup"]]

        speedup_dict.update(
            **{f"y_true_{key}": value for key, value in speedup_metrics(y_true, timing_df).items()},
            **{f"y_pred_{key}": value for key, value in speedup_metrics(y_pred, timing_df).items()},
        )

        speedup_dict["TP"] = (
            len(true_speedup[y_pred & y_true]),
            true_speedup[y_pred & y_true].mean(),
        )
        speedup_dict["FP"] = (
            len(true_speedup[y_pred & ~y_true]),
            true_speedup[y_pred & ~y_true].mean(),
        )
        speedup_dict["TN"] = (
            len(true_speedup[~y_pred & ~y_true]),
            true_speedup[~y_pred & ~y_true].mean(),
        )
        speedup_dict["FN"] = (
            len(true_speedup[~y_pred & y_true]),
            true_speedup[~y_pred & y_true].mean(),
        )
        res["speedup"] = copy.deepcopy(speedup_dict)

        cf = confusion_matrix(y_true, y_pred)

        group_counts = ["Counts: {0:0.0f}".format(value) for value in cf.flatten()]
        group_percentages = ["Percentages: {0:.2%}".format(value) for value in cf.flatten() / np.sum(cf)]

        group_spdup = [
            "Avg speedups: {0:.2f}".format(value)
            for value in [
                speedup_dict["TN"][1],
                speedup_dict["FP"][1],
                speedup_dict["FN"][1],
                speedup_dict["TP"][1],
            ]
        ]
        group_names = [
            "True Negative",
            "False Positive",
            "False Negative",
            "True Positive",
        ]

        labels = np.asarray(
            [
                f"{v1}\n{v2}\n{v3}\n{v4}"
                for v1, v2, v3, v4 in zip(group_names, group_counts, group_percentages, group_spdup)
            ]
        ).reshape(2, 2)
        fig = None
        if plot:
            fig, axes = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(5, 4.2))
            fig.suptitle(model_name)
            axes.set_title(f"Real speedup of positive samples: {speedup_dict['y_pred_speedup_real']:.2f}, (max: {speedup_dict['y_true_speedup_real']:.2f})")
            sns.heatmap(cf, annot=labels, cmap="Blues", fmt="", ax=axes, cbar=True)
            axes.set_yticklabels(["Materialize", "Factorize"], rotation=90)
            axes.set_xticklabels(["Materialize", "Factorize"])
            axes.set_xlabel("Predicted label")
            axes.set_ylabel("True label")
    return res, fig, speedup_dict


def full_eval(trained_models, X_test, y_test, test, dataset_name=""):
    speedups, result_compare = {}, {}
    for model in trained_models:
        name = _get_model_name(model)
        if isinstance(model, MorpheusFI):
            X_test_filtered = test[model.feature_list]
        else:
            X_test_filtered = X_test
        result, _, stats_dict = eval_model(model, X_test_filtered, y_test, full_dataset=test)
        result_compare[name] = {**result, "stats": stats_dict}
    best_speedup = 0
    for model in result_compare.keys():
        speedup = result_compare[model]["speedup"]
        best_speedup = speedup["y_true_speedup_real"]
        new_dict = {}
        for key, value in speedup.items():
            new_key = key + "_abs"
            if isinstance(value, tuple):
                new_value = value[0]
                new_dict[new_key] = new_value
            new_dict[key] = value[1] if isinstance(value, tuple) else value
        result_compare[model].update(new_dict)

    test_result_compare = pd.DataFrame(result_compare).T.drop(columns=["speedup"])
    test_result_compare["model"] = test_result_compare.index
    melted_df = pd.melt(test_result_compare, id_vars="model", var_name="metric", value_name="metric_value")

    fig, axs = plt.subplot_mosaic("AAAB", figsize=(14, 6))

    def plot_metrics(ax, metrics: list, legend=True):
        print(melted_df.head())
        print(melted_df[melted_df.metric.apply(lambda x: x in metrics)])
        ax.set_axisbelow(True)
        ax.grid(axis="y")
        ax = sns.barplot(
            data=melted_df[melted_df.metric.apply(lambda x: x in metrics)],
            x="metric",
            y="metric_value",
            hue="model",
            #  palette=sns.color_palette('flare'),
            ax=ax,
        )
        if not legend:
            ax.get_legend().remove()
        # Add metric values as text on top of every bar
        for p in ax.patches:
            ax.annotate(
                f"{p.get_height():.2f}",
                (p.get_x() + p.get_width() / 2.0, p.get_height()),
                ha="center",
                va="center",
                xytext=(0, 5),
                textcoords="offset points",
            )

    plot_metrics(axs["A"], ["accuracy", "precision", "recall", "f1"])
    plot_metrics(axs["B"], ["y_pred_speedup_real"], legend=False)
    print(best_speedup)
    axs["B"].axhline(best_speedup, ls="--", color="red", label="Maximum achievable speedup")

    fig.suptitle(f"Performance metrics {dataset_name}")
    return fig, test_result_compare
