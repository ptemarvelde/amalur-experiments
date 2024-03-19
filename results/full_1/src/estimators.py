import math
from typing import Generator, List
from typing import Tuple
from abc import abstractmethod
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

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


class LogisticRegressionCustomBoundary(LogisticRegression):
    def __init__(self, boundary=0.5, max_iter=300, class_weight='balanced'):
        self.boundary = boundary
        self.__class__.__name__ = f'MyLogReg (boundary at {boundary})'
        super().__init__(max_iter=max_iter, class_weight=class_weight)

    def predict(self, X):
        return super().predict_proba(X)[:, 1] > self.boundary


def is_compute_unit_cpu(row):
    return 'cpu' in row.lower()


class SeparateLogisticRegression:
    _estimator_type = "classifier"

    def __init__(self, train, test, model_features):
        cpu_features = list(
            set(model_features).difference(["memory_bandwidth", "_cores", "processing_power_double_precision"]))
        print(cpu_features)

        train_gpu = train[~train.compute_unit.apply(is_compute_unit_cpu)]
        X_train_gpu, y_train_gpu = train_gpu[cpu_features], train_gpu["label"]
        test_gpu = test[~test.compute_unit.apply(is_compute_unit_cpu)]
        X_test_gpu, y_test_gpu = test_gpu[cpu_features], test_gpu["label"]
        # create model for GPU
        self.log_reg_gpu = train_and_score(
            LogisticRegression(max_iter=500, class_weight="balanced"),
            X_train_gpu,
            X_test_gpu,
            y_train_gpu,
            y_test_gpu,
            fillna=False,
        )

        train_cpu = train[train.compute_unit.apply(is_compute_unit_cpu)]
        X_train_cpu, y_train_cpu = train_cpu[cpu_features], train_cpu["label"]
        test_cpu = test[test.compute_unit.apply(is_compute_unit_cpu)]
        X_test_cpu, y_test_cpu = test_cpu[cpu_features], test_cpu["label"]
        # create model for CPU
        self.log_reg_cpu = train_and_score(
            LogisticRegression(max_iter=500, class_weight="balanced"),
            X_train_cpu,
            X_test_cpu,
            y_train_cpu,
            y_test_cpu,
            fillna=False,
        )

    def score(self, X, y):
        pred = self.predict(X)
        return np.count_nonzero((np.array(pred) != np.array(y))) / len(y)

    def predict(self, X):
        X_cpu = X[X.isna().any(axis=1)]
        X_gpu = X[~X.isna().any(axis=1)]
        X_cpu['pred'] = self.log_reg_cpu.predict(X_cpu)
        X_gpu['pred'] = self.log_reg_gpu.predict(X_gpu)
        pred = pd.concat([X_cpu, X_gpu]).reindex(index=X.index).pred
        return pred


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


