import math
from typing import Generator, List
from typing import Tuple
from abc import abstractmethod
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class Classifier:
    _estimator_type = "classifier"  # needed to make ConfusionMatrixDisplay work.

    def __init__(self, feature_list: List[str]):
        self.feature_list = feature_list

    def score(self, X, y):
        pred = self.predict(X)
        return np.count_nonzero((np.array(pred) != np.array(y))) / len(y)

    def predict(self, X):
        res = []
        for features in self.feature_iterator(X):
            res.append(self.optimize(*features))
        return res

    @abstractmethod
    def optimize(self, *args): ...

    def feature_iterator(self, X) -> Generator[Tuple, None, None]:
        yield from X[self.feature_list].values


class Morpheus(Classifier):
    tuple_ratio = 5
    feature_ratio = 1

    def __init__(self, tuple_ratio=5, feature_ratio=1):
        super().__init__(["tr", "fr"])
        self.tuple_ratio = tuple_ratio
        self.feature_ratio = feature_ratio

    def optimize(self, t, f):
        return t > self.tuple_ratio and f > self.feature_ratio


class Amalur(Classifier):
    def __init__(self, complexity_ratio=1.5):
        super().__init__(["comp_ratio"])
        self.complexity_ratio_boundary = complexity_ratio

    def optimize(self, c):
        return c > self.complexity_ratio_boundary


class MorpheusFI(Classifier):
    def __init__(self):
        super().__init__(
            [
                "morpheusfi_q",  # Number of base tables with sparsity < 5%
                "morpheusfi_p",  # Number of base tables
                "morpheusfi_eis",  # List: Sparsity of Ri
                "morpheusfi_ns",  # Number of samples in S
                "morpheusfi_nis",  # List: Number of rows in Ri
            ]
        )

    def optimize(self, q: int, p: int, eis: List[float], ns: int, nis: List[int]) -> bool:
        """Morpheus FI Heuristic decision rule. Choos factorization if this returns True.

        In natural language:
        Choose factorization if either
            - The number of sparse base tables (q) is less than half of the total number of base tables (p)
            - For all dimension tables i the sparsity of Ri times the number of rows in S divided by the number of rows in Ri is greater than 1.
                - If all dim tables are fairly dense and relatively small (compared to Fact table), then factorization is better.

        S: Fact table
        Ri: Dimension table i feature matrix

        Args:
            q (int): number of base tables with sparsity < 5%
            p (int): number of base tables
            eis (List[float]): sparsity of Ri
            ns (int): number of rows in S
            nis (List[int]): number of rows in Ri

        Returns:
            bool: Choose factorization (True) or materialization (False)
        """
        return q < math.floor(p / 2) or (
            math.floor(q >= p / 2) and all([ei * (ns / ni) > 1 for (ei, ni) in zip(eis, nis)])
        )
        
def _get_model_name(model):
    return model.steps[-1][0] if isinstance(model, Pipeline) else model.__class__.__name__

def eval_model(model, X_test, y_test, full_dataset=None, plot=True):
    print(f"Model {model.__class__}, {_get_model_name(model)}\n test cols: {X_test.columns}")
    y_pred = model.predict(X_test)
       
    result, fig, speedup_dict = eval_result(
        y_test, y_pred=y_pred, full_dataset=full_dataset, model_name=_get_model_name(model), plot=plot
    )
    if not fig and plot:
        fig = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap="bone", text_kw={"size": 20})
    return result, fig, speedup_dict

def speedup_metrics(index_selector: pd.Series, df: pd.DataFrame) -> dict:
    """
    We're interested in the cases our predictor predicts as positive (factorize).
    For these cases we want to evaluate the time saved by factorizing.
    """
    df = df[index_selector]
    mat_time = df.materialized_times_mean.sum() # Time without estimator
    fact_time = df.times_mean.sum() # Time with estimator
    best_time = df[["times_mean", "materialized_times_mean"]].min(axis=1).sum() # Best possible time
    time_saved = mat_time - fact_time # Time saved with estimator
    # Average speedup value of positive cases (does not take into account time saved by factorization)
    speedup_avg = df.speedup.mean() 
    speedup_real = mat_time / fact_time # Realized speedup of positive cases
    return {
        "mat_time": mat_time,
        "fact_time": fact_time,
        "best_time" : best_time,
        "time_saved": time_saved,
        "speedup_avg": speedup_avg,
        "speedup_real": speedup_real,
    }

def eval_result(y_test, y_pred, full_dataset=None, model_name="", plot=False):
    y_true = y_test.copy()

    scoring_functions = {
        "accuracy": accuracy_score,
        "precision": precision_score,
        "recall": recall_score,
        "f1": f1_score,
    }
    res = {}
    for name, function in scoring_functions.items():
        res[name] = function(y_true, y_pred)

    fig = None
    speedup_dict = None
    if full_dataset is not None:
        y_true = pd.Series(y_true)
        y_pred = pd.Series(y_pred).astype(bool)

        if len(y_pred.unique()) < 2:
            print("WARNING all predicted labels are the same: ", y_pred.unique())

        y_pred.index = y_true.index
        true_speedup = full_dataset.speedup
        speedup_dict = {}
        timing_df = full_dataset[["times_mean", "materialized_times_mean", "speedup"]]

        speedup_dict.update(
            **{
                f"y_true_{key}": value for key, value in speedup_metrics(y_true, timing_df).items()
            },
            **{
                f"y_pred_{key}": value for key, value in speedup_metrics(y_pred, timing_df).items()
            }
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
        res["speedup"] = speedup_dict

        cf = confusion_matrix(y_true, y_pred)

        group_counts = ["Counts: {0:0.0f}".format(value) for value in cf.flatten()]
        group_percentages = ["Percentages: {0:.2%}".format(value) for value in cf.flatten() / np.sum(cf)]
        #  TODO total train (fact and mat times) for each group
        #  Gives insight into how much time savings is lost.
        group_fact_times = [
            "F times: {0:.0f}".format(value) for value in [speedup_dict["TN"][0], speedup_dict["FP"][0]]
        ]
        group_mat_times = []
        # END TODO

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
            axes.set_title(f"Realized speedup of positive samples: {speedup_dict['y_pred_speedup_real']:.2f}")
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
    best_speedup=0
    for model in result_compare.keys():
        speedup = result_compare[model]["speedup"]
        best_speedup = speedup['y_true_speedup_real']
        new_dict = {}
        for key, value in speedup.items():
            new_key = key + "_abs"
            if isinstance(value, tuple):
                new_value = value[0]
                new_dict[new_key] = new_value
            new_dict[key] = value[1] if isinstance(value, tuple) else value
        result_compare[model].update(new_dict)
        
    test_result_compare = pd.DataFrame(result_compare).T.drop(columns=['speedup'])
    test_result_compare['model'] = test_result_compare.index
    melted_df = pd.melt(test_result_compare, id_vars='model', var_name='metric', value_name='metric_value')
    
    fig, axs = plt.subplot_mosaic("AAAB",figsize=(14,6))
    
    def plot_metrics(ax, metrics: list, legend=True):
        print(melted_df.head())
        print(melted_df[melted_df.metric.apply(lambda x: x in metrics)])
        ax.set_axisbelow(True)
        ax.grid(axis='y')
        ax = sns.barplot(data=melted_df[melted_df.metric.apply(lambda x: x in metrics)], x='metric', y='metric_value', hue='model', 
                        #  palette=sns.color_palette('flare'), 
                         ax=ax)
        if not legend:
            ax.get_legend().remove()
        # Add metric values as text on top of every bar
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 5), textcoords='offset points')

    plot_metrics(axs['A'], ['accuracy', 'precision' , 'recall' , 'f1'])
    plot_metrics(axs['B'], ['y_pred_speedup_real'], legend=False)
    print(best_speedup)
    axs['B'].axhline(best_speedup, ls='--', color='red', label='Maximum achievable speedup')
    
    fig.suptitle(f'Performance metrics {dataset_name}')
    return fig, test_result_compare