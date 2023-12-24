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
    def optimize(self, *args):
        ...

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
                "morpheusfi_q",     # Number of base tables with sparsity < 5%
                "morpheusfi_p",     # Number of base tables
                "morpheusfi_eis",   # List: Sparsity of Ri
                "morpheusfi_ns",    # Number of samples in S
                "morpheusfi_nis",   # List: Number of rows in Ri
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

def eval_model(model, X_test, y_test, speedup=None, plot=False):
    print(f"Model {model.__class__}, test cols: {X_test.columns}")
    y_pred = model.predict(X_test)
    result, fig = eval_result(y_test, y_pred=y_pred,speedup=speedup, model_name=model.__class__.__name__, plot=plot)
    if not fig and plot:
        fig = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap="bone", text_kw={"size": 20})
    return result, fig

def eval_result(y_test, y_pred, speedup=None, model_name='', plot=False):
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
    if speedup is not None:
        y_true = pd.Series(y_true)
        y_pred = pd.Series(y_pred).astype(bool)
        y_pred.index = y_true.index

        best_speedup = speedup[speedup > 1.0].mean()
        speedup_dict = {}
        res["speedup"] = speedup_dict
        speedup_dict["tot_realized_speedup"] = speedup[y_pred].mean()
        speedup_dict["best_speedup"] = len(speedup), best_speedup
        speedup_dict["TP"] = (
            len(speedup[y_pred & y_true]),
            speedup[y_pred & y_true].mean(),
        )
        speedup_dict["FP"] = (
            len(speedup[y_pred & ~y_true]),
            speedup[y_pred & ~y_true].mean(),
        )
        speedup_dict["TN"] = (
            len(speedup[y_pred & ~y_true]),
            speedup[~y_pred & ~y_true].mean(),
        )
        speedup_dict["FN"] = (
            len(speedup[y_pred & ~y_true]),
            speedup[~y_pred & y_true].mean(),
        )

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
            [f"{v1}\n{v3}\n{v4}" for v1, v2, v3, v4 in zip(group_names, group_counts, group_percentages, group_spdup)]
        ).reshape(2, 2)
        fig = None
        if plot:
            fig, axes = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(5, 4.2))
            fig.suptitle(model_name)
            axes.set_title(f"Realized speedup of positive samples: {speedup_dict['tot_realized_speedup']:.2f}")
            sns.heatmap(cf, annot=labels, cmap="Blues", fmt="", ax=axes, cbar=True)
            axes.set_yticklabels(["Materialize", "Factorize"], rotation=90)
            axes.set_xticklabels(["Materialize", "Factorize"])
            axes.set_xlabel("Predicted label")
            axes.set_ylabel("True label")

    return res, fig
