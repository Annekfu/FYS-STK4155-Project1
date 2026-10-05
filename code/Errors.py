# LLM-assisted (code level 4): structured and written by Claude (Opus 4.8,
# claude.ai, Sept-Oct 2026) from the author's weekly-exercise solutions.
# The author adapted, tested against the closed-form and library benchmarks,
# commented and verified it. See Appendix A of the report.
"""Error metrics and the bias-variance decomposition helpers.

mse and r2 are the two scores required in part a. The bias_variance
helper turns a matrix of bootstrap predictions into the three terms of
the decomposition of Eq. (2.52).
"""

import numpy as np


def mse(y, y_pred):
    """Mean squared error."""
    y = np.ravel(y)
    y_pred = np.ravel(y_pred)
    return np.mean((y - y_pred) ** 2)


def r2(y, y_pred):
    """R2 score, the fraction of variance explained."""
    y = np.ravel(y)
    y_pred = np.ravel(y_pred)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1.0 - ss_res / ss_tot


def bias_variance(y_test, y_pred, f_test=None):
    """Bias-variance decomposition from a bootstrap prediction matrix.

    y_test has shape (n_test,). y_pred has shape (n_test, n_bootstraps),
    one column per bootstrap training set. Returns the measured test error,
    the squared bias and the variance, averaged over the test points.

    By default the bias is measured against the noisy targets y_test, so it
    contains the noise variance sigma^2 and error = bias + variance holds
    exactly. If the true function f_test is given (known for Runge), the
    bias is measured against it instead; then error = bias + variance +
    mean(eps^2) + 2 mean(eps (f - mean prediction)), where the last term
    vanishes in expectation but not on a finite test set.
    """
    y_test = np.ravel(y_test).reshape(-1, 1)
    target = y_test if f_test is None else np.ravel(f_test).reshape(-1, 1)
    error = np.mean(np.mean((y_test - y_pred) ** 2, axis=1))
    bias = np.mean((target - np.mean(y_pred, axis=1, keepdims=True)) ** 2)
    variance = np.mean(np.var(y_pred, axis=1))
    return error, bias, variance
