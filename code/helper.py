# LLM-assisted (code level 4): structured and written by Claude (Opus 4.8,
# claude.ai, Sept-Oct 2026) from the author's weekly-exercise solutions.
# The author adapted, tested against the closed-form and library benchmarks,
# commented and verified it. See Appendix A of the report.
# Source: adapted from the gradient and Hessian helpers of the week-37 and week-38 exercises,
# FYS-STK4155 lecture material (Hjorth-Jensen), https://github.com/CompPhysics/MachineLearning
"""Gradients, Hessian information and learning-rate helpers.

These support the gradient-descent parts e to h. The analytical gradient
matches the week-35 derivation; the same expression can be produced by
jax.grad of the cost, which is verified in the gradient-descent notebook.
"""

import numpy as np


def ols_ridge_cost(theta, X, y, lam=0.0):
    """Cost (1/n)||X theta - y||^2 + lam ||theta||^2, Eqs. (4.12) and (4.16)."""
    y = np.ravel(y)
    r = X @ theta - y
    return np.mean(r ** 2) + lam * theta @ theta


def ols_ridge_gradient(theta, X, y, lam=0.0):
    """Analytical gradient (2/n) X^T (X theta - y) + 2 lam theta."""
    y = np.ravel(y)
    n = len(y)
    return (2.0 / n) * X.T @ (X @ theta - y) + 2.0 * lam * theta


def hessian_eigs(X, lam=0.0):
    """Eigenvalues of the Hessian (2/n) X^T X + 2 lam I."""
    n = len(X)
    H = (2.0 / n) * X.T @ X + 2.0 * lam * np.eye(X.shape[1])
    return np.linalg.eigvalsh(H)


def learning_rate_bounds(X, lam=0.0):
    """Return gamma_max = 2/lambda_max, gamma_star and the condition number.

    These are the stability bound and the optimal step of plain gradient
    descent from Section 4.5 of the lecture notes.
    """
    eig = hessian_eigs(X, lam)
    lam_min, lam_max = eig.min(), eig.max()
    gamma_max = 2.0 / lam_max
    gamma_star = 2.0 / (lam_max + lam_min)
    kappa = lam_max / lam_min
    return gamma_max, gamma_star, kappa
