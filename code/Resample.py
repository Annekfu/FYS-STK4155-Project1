# LLM-assisted (code level 4): structured and written by Claude (Opus 4.8,
# claude.ai, Sept-Oct 2026) from the author's weekly-exercise solutions.
# The author adapted, tested against the closed-form and library benchmarks,
# commented and verified it. See Appendix A of the report.
# Source: adapted from the bootstrap and own k-fold of the week-36 exercises,
# FYS-STK4155 lecture material (Hjorth-Jensen), https://github.com/CompPhysics/MachineLearning
"""Resampling methods for parts c and d.

bootstrap_bias_variance implements the bias-variance study of part c.
kfold_cv is the own k-fold cross-validation from week 36. sklearn_cv is a
thin wrapper around scikit-learn KFold and cross_val_score, which part d
asks for explicitly, with the own k-fold kept as a cross-check.
"""

import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.utils import resample

from Errors import mse, bias_variance


def bootstrap_bias_variance(X_train, X_test, y_train, y_test, fit_predict,
                            n_bootstraps=100, seed=2026):
    """Bootstrap estimate of test error, squared bias and variance.

    fit_predict(X_tr, y_tr, X_te) returns predictions on X_te for one
    bootstrap training set. Returns error, bias and variance. The caller
    supplies already-scaled design matrices, or a fit_predict that scales
    inside.
    """
    y_test = np.ravel(y_test)
    y_pred = np.empty((y_test.shape[0], n_bootstraps))
    rng = np.random.default_rng(seed)
    for b in range(n_bootstraps):
        idx = rng.integers(0, X_train.shape[0], X_train.shape[0])
        y_pred[:, b] = np.ravel(fit_predict(X_train[idx], np.ravel(y_train)[idx], X_test))
    return bias_variance(y_test, y_pred)


def kfold_cv(x, y, model, k=5, seed=2026):
    """Own k-fold CV. Returns mean and std of the test MSE over folds.

    model must expose fit and predict, for example a scikit-learn pipeline.
    Does not use cross_val_score, per the week-36 requirement.
    """
    rng = np.random.default_rng(seed)
    n = len(x)
    inds = rng.permutation(n)
    folds = np.array_split(inds, k)
    scores = np.zeros(k)
    for j in range(k):
        test_inds = folds[j]
        train_inds = np.concatenate(folds[:j] + folds[j + 1:])
        model.fit(x[train_inds], np.ravel(y)[train_inds])
        y_pred = model.predict(x[test_inds])
        scores[j] = mse(np.ravel(y)[test_inds], y_pred)
    return scores.mean(), scores.std()


def sklearn_cv(x, y, model, k=5, seed=2026):
    """scikit-learn KFold with cross_val_score, returns mean and std MSE."""
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    scores = cross_val_score(model, x, np.ravel(y), cv=kf,
                             scoring="neg_mean_squared_error")
    return -scores.mean(), scores.std()
