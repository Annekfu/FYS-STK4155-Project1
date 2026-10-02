# LLM-assisted (code level 4): structured and written by Claude (Opus 4.8,
# claude.ai, Sept-Oct 2026) from the author's weekly-exercise solutions.
# The author adapted, tested against the closed-form and library benchmarks,
# commented and verified it. See Appendix A of the report.
# Source: adapted from the closed-form Ridge of the week-37 and week-38 exercises,
# FYS-STK4155 lecture material (Hjorth-Jensen), https://github.com/CompPhysics/MachineLearning
"""Ridge regression for the Runge function, part b.

Own closed-form implementation with the 1/n cost convention, so that the
penalty enters as n*lambda*I. The corresponding scikit-learn alpha is
n*lambda, see the week-37 exercises.
"""

import numpy as np

from LinearRegression import Regressor


def ridge_closed_form(X, y, lam):
    """Ridge coefficients (X^T X + n lambda I)^-1 X^T y."""
    y = np.ravel(y)
    n, p = X.shape
    return np.linalg.solve(X.T @ X + n * lam * np.eye(p), X.T @ y)


class Ridge(Regressor):
    """Ridge model with a fixed penalty lambda."""

    def __init__(self, lam=0.0):
        super().__init__()
        self.lam = lam

    def fit(self, X, y):
        self.theta = ridge_closed_form(X, y, self.lam)
        return self
