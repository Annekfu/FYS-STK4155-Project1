"""Data generation and preprocessing for the Runge function.

Project 1 studies Runge's function

    f(x) = 1 / (1 + 25 x^2),   x in [-1, 1]

fitted with polynomials up to high degree. This module provides the data
generator, the polynomial design matrix, the train/test split and the
standardisation used throughout the project.
"""

import numpy as np
from sklearn.model_selection import train_test_split

from common import SEED


def runge(x):
    """Runge's function f(x) = 1 / (1 + 25 x^2)."""
    return 1.0 / (1.0 + 25.0 * x ** 2)


def make_data(n=100, noise=0.1, seed=SEED, uniform=True):
    """Return x and y from the noisy Runge model on [-1, 1].

    noise is the standard deviation of additive Gaussian noise. With
    uniform=True the x values are drawn uniformly, otherwise they are on a
    fixed equidistant grid.
    """
    rng = np.random.default_rng(seed)
    if uniform:
        x = rng.uniform(-1.0, 1.0, n)
    else:
        x = np.linspace(-1.0, 1.0, n)
    y = runge(x) + noise * rng.standard_normal(n)
    return x, y


def polynomial_features(x, degree, intercept=False):
    """Vandermonde design matrix with columns x, x^2, ..., x^degree.

    With intercept=True a leading column of ones is included. The default
    leaves it out, since the project centres the data instead.
    """
    x = np.ravel(x)
    start = 0 if intercept else 1
    return np.column_stack([x ** k for k in range(start, degree + 1)])


def scale_center(X_train, X_test, y_train, y_test):
    """Standardise features on training statistics and centre the target.

    The mean and standard deviation are computed on the training columns
    only, then applied to both sets. This is the leak-free scaling required
    by the project. Returns the transformed arrays plus the statistics
    needed to convert coefficients back to the raw scale.
    """
    x_mean = X_train.mean(axis=0)
    x_std = X_train.std(axis=0)
    x_std[x_std == 0] = 1.0
    X_train_s = (X_train - x_mean) / x_std
    X_test_s = (X_test - x_mean) / x_std

    y_mean = y_train.mean()
    y_train_c = y_train - y_mean
    y_test_c = y_test - y_mean

    stats = {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean}
    return X_train_s, X_test_s, y_train_c, y_test_c, stats


def split(X, y, test_size=0.2, seed=SEED):
    """Thin wrapper around scikit-learn train_test_split with a fixed seed."""
    return train_test_split(X, y, test_size=test_size, random_state=seed)
