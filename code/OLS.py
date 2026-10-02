# LLM-assisted (code level 4): structured and written by Claude (Opus 4.8,
# claude.ai, Sept-Oct 2026) from the author's weekly-exercise solutions.
# The author adapted, tested against the closed-form and library benchmarks,
# commented and verified it. See Appendix A of the report.
# Source: adapted from the SVD / pseudoinverse OLS of the week-35 exercises,
# FYS-STK4155 lecture material (Hjorth-Jensen), https://github.com/CompPhysics/MachineLearning
"""Ordinary least squares for the Runge function, part a.

Own implementation via the singular value decomposition and via the
pseudoinverse, matching the week-35 exercises. Both avoid forming an
explicit inverse.
"""

import numpy as np

from LinearRegression import Regressor


def ols_svd(X, y):
    """OLS coefficients via the SVD, theta = V S^-1 U^T y."""
    y = np.ravel(y)
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    return Vt.T @ (U.T @ y / s)


def ols_pinv(X, y):
    """OLS coefficients via the Moore-Penrose pseudoinverse."""
    y = np.ravel(y)
    return np.linalg.pinv(X.T @ X) @ X.T @ y


class OLS(Regressor):
    """OLS model with a fit that defaults to the SVD solver."""

    def __init__(self, solver="svd"):
        super().__init__()
        self.solver = solver

    def fit(self, X, y):
        if self.solver == "svd":
            self.theta = ols_svd(X, y)
        else:
            self.theta = ols_pinv(X, y)
        return self
