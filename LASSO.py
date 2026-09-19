"""Lasso regression via gradient descent, part g.

Lasso has no closed form, so we solve it with the gradient-descent
machinery of parts e and f. The l1 penalty lam*||theta||_1 is not
differentiable at theta_j = 0. Two standard ways to handle this are
provided:

  1. subgradient descent, which uses sign(theta) as a subgradient of the
     absolute value and 0 at the kink;
  2. proximal gradient descent, also called ISTA, which takes a plain
     gradient step on the smooth data term and then applies the soft
     thresholding operator, the proximal operator of the l1 penalty.

ISTA is the more stable of the two and actually drives coefficients to
exactly zero, so it is the recommended solver. Both are checked against
scikit-learn Lasso in the notebook. This is the new work for the project;
the numerical study of learning rate and lambda is done in notebook 4.
"""

import numpy as np

from LinearRegression import Regressor


def soft_threshold(z, thresh):
    """Soft thresholding operator, the proximal operator of the l1 norm."""
    return np.sign(z) * np.maximum(np.abs(z) - thresh, 0.0)


def lasso_ista(X, y, lam, gamma, num_iters=5000, tol=1e-8, theta0=None):
    """Proximal gradient descent for Lasso.

    Minimises (1/n)||X theta - y||^2 + lam ||theta||_1. Each step takes a
    gradient step on the data term with step gamma, then soft-thresholds by
    gamma*lam. Returns the coefficient vector.
    """
    y = np.ravel(y)
    n, p = X.shape
    theta = np.zeros(p) if theta0 is None else np.array(theta0, dtype=float)
    for _ in range(num_iters):
        grad_data = (2.0 / n) * X.T @ (X @ theta - y)
        theta_new = soft_threshold(theta - gamma * grad_data, gamma * lam)
        if np.linalg.norm(theta_new - theta) < tol:
            theta = theta_new
            break
        theta = theta_new
    return theta


def lasso_subgradient(X, y, lam, gamma, num_iters=5000, tol=1e-8, theta0=None):
    """Subgradient descent for Lasso, using sign(theta) at the kink.

    Kept for comparison. It does not set coefficients exactly to zero, which
    is the point made in the discussion of part g.
    """
    y = np.ravel(y)
    n, p = X.shape
    theta = np.zeros(p) if theta0 is None else np.array(theta0, dtype=float)
    for _ in range(num_iters):
        grad_data = (2.0 / n) * X.T @ (X @ theta - y)
        g = grad_data + lam * np.sign(theta)
        theta_new = theta - gamma * g
        if np.linalg.norm(theta_new - theta) < tol:
            theta = theta_new
            break
        theta = theta_new
    return theta


class Lasso(Regressor):
    """Lasso model solved by proximal gradient descent by default."""

    def __init__(self, lam=0.01, gamma=0.1, num_iters=5000, solver="ista"):
        super().__init__()
        self.lam = lam
        self.gamma = gamma
        self.num_iters = num_iters
        self.solver = solver

    def fit(self, X, y):
        if self.solver == "ista":
            self.theta = lasso_ista(X, y, self.lam, self.gamma, self.num_iters)
        else:
            self.theta = lasso_subgradient(X, y, self.lam, self.gamma, self.num_iters)
        return self
