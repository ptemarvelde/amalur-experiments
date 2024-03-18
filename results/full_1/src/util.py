from typing import Tuple

import pandas as pd
import os
import glob
from collections import defaultdict
import numpy as np
import json
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

    features.drop_duplicates(["dataset", "join", "operator"], inplace=True )
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


def _ratio(df, column, output_column_name=None, hardware_var=None):
    if not output_column_name:
        output_column_name = column + "_ratio"
    if not hardware_var:
        hardware_var = "compute_unit"

    merge_keys = ("dataset", "operator", "join", hardware_var)
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
    with open ("/home/pepijn/Documents/uni/y5/thesis/amalur/amalur-experiments/results/full_1/daic/features/gpu-characteristics.json") as f:
        gpu_chars = json.load(f)
    gpu_chars['1080'] = gpu_chars.pop('1080Ti')
    gpu_chars['p100'] = gpu_chars.pop('P100')
    gpu_chars['v100'] = gpu_chars.pop('V100')
    gpu_chars['2080'] = gpu_chars.pop('2080Ti')
    gpu_chars['a40'] = gpu_chars.pop('A40')
    gpu_chars['a10g'] = gpu_chars.pop('A10G')
    gpu_chars['1660'] = gpu_chars.pop('1660Ti')
    gpu_chars_df = pd.DataFrame(gpu_chars).T.apply(pd.to_numeric, errors='ignore')
    gpu_chars_df.rename(columns={x: f"gpu_{x}" for x in gpu_chars_df.columns}, inplace=True)
    return gpu_chars_df

def add_gpu_chars_to_df(df, gpu_col_name='compute_unit'):
    gpu_chars_df = read_gpu_chars()
    gpu_chars_df.index.name = gpu_col_name
    df = df.merge(gpu_chars_df, how='left', left_on=gpu_col_name, right_on=gpu_col_name)
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
    res['materialized_times_mean'] = res.times_mean * res.speedup
    res['time_saved'] = res.materialized_times_mean - res.times_mean
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
