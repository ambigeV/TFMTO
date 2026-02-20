"""Neural Network Weight Training (NN-Training) benchmark problems.

This module provides real-world single-task optimization problems where the
decision variables are the **neural network weights** themselves. The optimization
algorithm (e.g., DE, PSO, GA) directly searches for optimal weight configurations
to minimize classification error or regression loss -- no gradient-based training.

Data is split into train/test sets (70/30 by default). The optimization objective
is the **test set error**: classification error rate or regression MSE.

Datasets used (from scikit-learn):
- Classification: Digits (10 classes), Covertype (7 classes)
- Regression: Diabetes, California Housing

Network architecture: fixed single-hidden-layer MLP with ReLU activation.
Decision variables: flattened weight vector [W1, b1, W2, b2].
"""

import numpy as np
from sklearn.datasets import (
    load_digits, load_diabetes,
    fetch_california_housing, fetch_covtype
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from ddmtolab.Methods.mtop import MTOP


# ---------------------------------------------------------------------------
# Fixed-architecture neural network (numpy-only, no PyTorch needed)
# ---------------------------------------------------------------------------

class _FixedNN:
    """
    Fixed-architecture feedforward neural network parameterized by a flat
    weight vector. Uses ReLU activation on hidden layers and linear output.

    Parameters
    ----------
    layer_sizes : list of int
        Sizes of each layer, e.g. [4, 10, 3] means input=4, hidden=10, output=3.

    Attributes
    ----------
    n_params : int
        Total number of trainable parameters (weights + biases).
    """

    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.n_params = sum(
            layer_sizes[i] * layer_sizes[i + 1] + layer_sizes[i + 1]
            for i in range(len(layer_sizes) - 1)
        )

    def forward(self, X, params):
        """
        Forward pass with given flattened parameters.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, input_dim)
            Input features.
        params : np.ndarray, shape (n_params,)
            Flattened weight vector.

        Returns
        -------
        np.ndarray, shape (n_samples, output_dim)
            Network output (logits for classification, raw values for regression).
        """
        offset = 0
        h = X
        for i in range(len(self.layer_sizes) - 1):
            in_d = self.layer_sizes[i]
            out_d = self.layer_sizes[i + 1]

            W = params[offset:offset + in_d * out_d].reshape(in_d, out_d)
            offset += in_d * out_d
            b = params[offset:offset + out_d]
            offset += out_d

            h = h @ W + b

            # ReLU on all hidden layers (not output)
            if i < len(self.layer_sizes) - 2:
                h = np.maximum(h, 0)

        return h


# ---------------------------------------------------------------------------
# Data loading utilities
# ---------------------------------------------------------------------------

def _load_cls_data(loader_func, max_samples=None, test_ratio=0.3, seed=42):
    """
    Load, standardize, and split a classification dataset.

    Returns
    -------
    tuple
        (X_train, y_train, X_test, y_test, n_classes).
    """
    data = loader_func()
    X = data.data.astype(np.float64)
    y = data.target.astype(np.int64)

    if max_samples and X.shape[0] > max_samples:
        rng = np.random.RandomState(seed)
        idx = rng.choice(X.shape[0], max_samples, replace=False)
        X, y = X[idx], y[idx]

    # Remap labels to 0..n_classes-1 (covtype labels start from 1)
    unique_labels = np.sort(np.unique(y))
    if unique_labels[0] != 0 or not np.array_equal(unique_labels, np.arange(len(unique_labels))):
        label_map = {old: new for new, old in enumerate(unique_labels)}
        y = np.array([label_map[yi] for yi in y])

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    n_classes = len(np.unique(y))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=seed, stratify=y
    )
    return X_train, y_train, X_test, y_test, n_classes


def _load_reg_data(loader_func, max_samples=None, test_ratio=0.3, seed=42):
    """
    Load, standardize, and split a regression dataset.

    Returns
    -------
    tuple
        (X_train, y_train, X_test, y_test).
    """
    data = loader_func()
    X = data.data.astype(np.float64)
    y = data.target.astype(np.float64)

    if max_samples and X.shape[0] > max_samples:
        rng = np.random.RandomState(seed)
        idx = rng.choice(X.shape[0], max_samples, replace=False)
        X, y = X[idx], y[idx]

    scaler_X = StandardScaler()
    X = scaler_X.fit_transform(X)

    scaler_y = StandardScaler()
    y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=seed
    )
    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# NN_Training benchmark class
# ---------------------------------------------------------------------------

class NN_Training:
    """
    Neural Network Weight Training benchmark suite for single-task optimization.

    The decision variables are the **flattened weights and biases** of a
    fixed-architecture MLP. The optimization algorithm directly searches for
    optimal weight configurations -- evaluation is a single forward pass (no
    gradient-based training), making these problems fast to evaluate.

    Data is split into train / test. The optimization objective is the
    **test set error rate** (classification) or **test MSE** (regression).

    Problems are ordered from easy to hard (by dimension & difficulty):

    +-----+-----------------------+---------------+--------+-----------------+
    | P   | Dataset               | Architecture  | Dim    | Task type       |
    +=====+=======================+===============+========+=================+
    | P1  | California Housing    | [8, 10, 1]    | 101    | Regression      |
    | P2  | Diabetes              | [10, 10, 1]   | 121    | Regression      |
    | P3  | Digits                | [64, 10, 10]  | 760    | Classification  |
    | P4  | Covertype             | [54, 20, 7]   | 1247   | Classification  |
    | P5  | Digits (large net)    | [64, 20, 10]  | 1510   | Classification  |
    | P6  | Covertype (large net) | [54, 30, 7]   | 1867   | Classification  |
    +-----+-----------------------+---------------+--------+-----------------+

    Objectives (minimize):

    - Classification: test error rate (1 - accuracy), range [0, 1]
    - Regression: test MSE on standardized targets

    Bounds: [-3, 3] for all weight parameters.

    Parameters
    ----------
    test_ratio : float, optional
        Fraction of data for testing (default 0.3).
    seed : int, optional
        Random seed for train/test split (default 42).
    """

    problem_information = {
        'n_cases': 6,
        'n_tasks': '1',
        'n_dims': '[101, 1867]',
        'n_objs': '1',
        'n_cons': '0',
        'type': 'real_world',
    }

    _WEIGHT_BOUND = 3.0

    def __init__(self, test_ratio=0.3, seed=42):
        self.test_ratio = test_ratio
        self.seed = seed
        self._cache = {}

    # ----- Internal helpers -----

    def _get_cls(self, name, loader_func, arch, max_samples=None):
        """Load (or retrieve cached) classification dataset and build NN."""
        if name not in self._cache:
            X_tr, y_tr, X_te, y_te, n_cls = _load_cls_data(
                loader_func, max_samples, self.test_ratio, self.seed
            )
            nn = _FixedNN(arch)
            self._cache[name] = ('cls', X_tr, y_tr, X_te, y_te, n_cls, nn)
        return self._cache[name]

    def _get_reg(self, name, loader_func, arch, max_samples=None):
        """Load (or retrieve cached) regression dataset and build NN."""
        if name not in self._cache:
            X_tr, y_tr, X_te, y_te = _load_reg_data(
                loader_func, max_samples, self.test_ratio, self.seed
            )
            nn = _FixedNN(arch)
            self._cache[name] = ('reg', X_tr, y_tr, X_te, y_te, nn)
        return self._cache[name]

    def _cls_problem(self, name, loader_func, arch, max_samples=None):
        """Create a single-task classification MTOP (test error objective)."""
        _, X_tr, y_tr, X_te, y_te, n_cls, nn = self._get_cls(
            name, loader_func, arch, max_samples
        )
        dim = nn.n_params

        def objective(x):
            x = np.atleast_2d(x)
            results = np.zeros((x.shape[0], 1))
            for i in range(x.shape[0]):
                logits = nn.forward(X_te, x[i])
                preds = np.argmax(logits, axis=1)
                results[i, 0] = 1.0 - np.mean(preds == y_te)
            return results

        problem = MTOP()
        problem.add_task(
            objective, dim=dim,
            lower_bound=-self._WEIGHT_BOUND,
            upper_bound=self._WEIGHT_BOUND
        )
        return problem

    def _reg_problem(self, name, loader_func, arch, max_samples=None):
        """Create a single-task regression MTOP (test MSE objective)."""
        _, X_tr, y_tr, X_te, y_te, nn = self._get_reg(
            name, loader_func, arch, max_samples
        )
        dim = nn.n_params

        def objective(x):
            x = np.atleast_2d(x)
            results = np.zeros((x.shape[0], 1))
            for i in range(x.shape[0]):
                preds = nn.forward(X_te, x[i]).flatten()
                results[i, 0] = np.mean((preds - y_te) ** 2)
            return results

        problem = MTOP()
        problem.add_task(
            objective, dim=dim,
            lower_bound=-self._WEIGHT_BOUND,
            upper_bound=self._WEIGHT_BOUND
        )
        return problem

    # ----- Single-task problems (easy → hard) -----

    def P1(self) -> MTOP:
        """
        Problem 1: **California Housing** regression.

        Architecture: [8, 10, 1], 101-D. 5000 samples (subsampled).
        Objective: test MSE, minimize.
        """
        return self._reg_problem(
            'california', fetch_california_housing,
            [8, 10, 1], max_samples=5000
        )

    def P2(self) -> MTOP:
        """
        Problem 2: **Diabetes** regression.

        Architecture: [10, 10, 1], 121-D. 442 samples.
        Objective: test MSE, minimize.
        """
        return self._reg_problem(
            'diabetes', load_diabetes,
            [10, 10, 1]
        )

    def P3(self) -> MTOP:
        """
        Problem 3: **Digits** classification (small net).

        Architecture: [64, 10, 10], 760-D. 1797 samples, 10 classes.
        Objective: test error rate, minimize.
        """
        return self._cls_problem(
            'digits_s', load_digits,
            [64, 10, 10]
        )

    def P4(self) -> MTOP:
        """
        Problem 4: **Covertype** classification (medium net).

        Architecture: [54, 20, 7], 1247-D. 5000 samples (subsampled), 7 classes.
        Objective: test error rate, minimize.
        """
        return self._cls_problem(
            'covtype_m', fetch_covtype,
            [54, 20, 7], max_samples=5000
        )

    def P5(self) -> MTOP:
        """
        Problem 5: **Digits** classification (large net).

        Architecture: [64, 20, 10], 1510-D. 1797 samples, 10 classes.
        Objective: test error rate, minimize.
        """
        return self._cls_problem(
            'digits_l', load_digits,
            [64, 20, 10]
        )

    def P6(self) -> MTOP:
        """
        Problem 6: **Covertype** classification (large net).

        Architecture: [54, 30, 7], 1857-D. 5000 samples (subsampled), 7 classes.
        Objective: test error rate, minimize.
        """
        return self._cls_problem(
            'covtype_l', fetch_covtype,
            [54, 30, 7], max_samples=5000
        )
