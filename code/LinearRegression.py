# LLM-assisted (code level 4): structured and written by Claude (Opus 4.8,
# claude.ai, Sept-Oct 2026) from the author's weekly-exercise solutions.
# The author adapted, tested against the closed-form and library benchmarks,
# commented and verified it. See Appendix A of the report.
"""Common interface for the regression methods.

OLS, Ridge and Lasso share a fit/predict shape. This light base class
records the fitted parameters and provides predict, so the analysis
notebooks can treat the three methods uniformly. The gradient method is
used by the gradient-descent solvers in optimisers.py.
"""

import numpy as np


class Regressor:
    """Base class holding a parameter vector theta."""

    def __init__(self):
        self.theta = None

    def predict(self, X):
        """Prediction X @ theta. Requires a fitted model."""
        if self.theta is None:
            raise RuntimeError("model is not fitted")
        return X @ self.theta

    def cost(self, X, y):
        """Mean squared error cost on the current theta."""
        r = X @ self.theta - np.ravel(y)
        return np.mean(r ** 2)
