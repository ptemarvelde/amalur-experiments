from typing import Tuple

import pandas as pd
import os
import glob
from collections import defaultdict
import numpy as np

pd.options.mode.chained_assignment = None  # default='warn'


def read_runtime_gpu(type_="synthetic", from_raw=False) -> pd.DataFrame:
    if type_ not in ["synthetic", "hamlet"]:
        raise ValueError(
            f"Invalid 'type' specified: {type_}. Must be one of 'synthetic', 'hamlet'"
        )

    outfile = f"daic/runtime/runtime_{type_}.parquet"
    if from_raw or not os.path.exists(outfile):
        dfs = []
        for f in glob.glob(f"daic/runtime/{type_}/*"):
            temp_df = pd.read_json(f, lines=True)
            temp_df["GPU"] = f.split("/")[-1].split("_")[0]
            dfs.append(temp_df)
        df = pd.concat(dfs)
        df.to_parquet(outfile)
        return df
    else:
        return pd.read_parquet(outfile)


def read_features_gpu(type_="synthetic") -> pd.DataFrame:
    features = pd.read_json(f"daic/features/features_{type_}.jsonl", lines=True)
    return features.reset_index(drop=True)


def read_gpu_results(from_parquet=True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reads all results and preprocesses.
    """
    outfile = "daic/all_gpus_preprocessed.parquet"
    if from_parquet and os.path.exists(outfile):
        df = pd.read_parquet(outfile)
    else:
        synth = read_runtime_gpu(type_="synthetic")
        hamlet = read_runtime_gpu(type_="hamlet")
        df = pd.concat([synth, hamlet])

        features_synth = read_features_gpu(type_="synthetic")
        features_hamlet = read_features_gpu(type_="hamlet")
        features = pd.concat([features_synth, features_hamlet])
        
        data_chars_synth = read_data_chars(type_='synthetic')
        data_chars_hamlet = read_data_chars(type_='hamlet')
        data_chars = pd.concat([data_chars_synth, data_chars_hamlet])
        
        df = preprocess(df, features, data_chars)
        df.to_parquet(outfile)

    return df

def read_data_chars(type_='synthetic'):
    data_chars = pd.read_json(f"daic/features/data_chars_{type_}.jsonl", lines=True)[['data_characteristics', 'dataset','join']]
    data_chars.reset_index(drop=True, inplace=True)
    data_chars['dataset'] = data_chars['dataset'].str.replace("/user/data/generated/", "/mnt/data/synthetic/sigmod_extended")
    data_chars = data_chars.merge(pd.json_normalize(data_chars.data_characteristics), left_index=True, right_index=True)
    return data_chars


def _complexity_ratio(df):
    # materialized complexities
    materialized_complexity_dict = defaultdict(lambda: 0)
    for dataset, operator, complexity in df[df.model == "materialized"][
        ["dataset", "operator", "complexity"]
    ].values:
        materialized_complexity_dict[(dataset, operator)] = complexity

    def calc_complexity_ratio(row):
        if row[1] == "Materialization":
            return None
        val = materialized_complexity_dict[(row[0], row[1])]
        if not val:
            return None
        return val / row[2]

    df["complexity_ratio"] = df[["dataset", "operator", "complexity"]].apply(
        calc_complexity_ratio, axis=1
    )
    return df

def _speedup(df):
    hardware_var = "GPU"  # or 'num_cores'

    baseline_lookup_dict = {
        (dataset, operator, x): mean_time
        for (dataset, operator, x, mean_time) in df[
            ((df.model == "materialized") | (df.model == "baseline"))
        ][["dataset", "operator", hardware_var, "times_mean"]].values
    }

    def calc_speedup(row) -> float:
        if row[1] == "Materialization":
            return 0.0
        try:
            baseline = baseline_lookup_dict[(row[0], row[1], row[2])]
        except KeyError:
            return None
        return baseline / row[3]

    df["speedup"] = (
        df[["dataset", "operator", hardware_var, "times_mean"]]
        .apply(calc_speedup, axis=1)
        .dropna()
    )
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

def preprocess(runtime: pd.DataFrame, features: pd.DataFrame, data_chars: pd.DataFrame):
    res = runtime
    features = features[features.operator.isin(model_operators)]
    res = res[res.operator.isin(model_operators)]
    res.reset_index(drop=True, inplace=True)
    if "join" not in res.columns:
        res["join"] = "preset"

    res = _speedup(res)
    res = _complexity_ratio(res)
    
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
            "GPU",
            # 'r_T', 'c_T', 'r_S', 'c_S','Snonzero', 'Tnonzero'
        ]
    ]
    
    res["dataset_type"] = res["dataset"].apply(
        lambda x: "synthetic" if x.startswith("/mnt/data") else "hamlet"
    )
    
    res["label"] = res.speedup > 1.0
    
    res = pd.merge(
        res,
        features[["dataset", "operator", "features", "join"]],
        how="right",
        on=["dataset", "operator", "join"],
    )

    if "parallelism" not in res.columns:
        res["parallelism"] = None
    
    processed_features = res[["features", "parallelism"]].apply(
        lambda row: postprocessing_features(*row), axis=1
    )
    res[feature_names] = pd.DataFrame(processed_features.tolist(), index=res.index)

    overlap_cols = set(res.columns).intersection(data_chars.columns).difference({"dataset", "join"})
    overlap_cols.add("data_characteristics")
    data_chars = data_chars.drop(columns=overlap_cols).drop_duplicates(["dataset", "join"])
    
    res = pd.merge(res, data_chars, how='left', on=['dataset', "join"])   

    
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

    return res


def postprocessing_features(raw_features, parallelism=8000):
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
    return output


def main():
    df = read_gpu_results()
    print(df)


if __name__ == "__main__":
    main()
