"""Gradient-descent optimisers for parts e, f and h.

One step function holds all five optimisers: plain gradient descent,
momentum, AdaGrad, RMSProp and Adam. The full-gradient loop and the
minibatch SGD loop both call it, so every optimiser works with
minibatches unchanged. Reused directly from the week-37 and week-38
exercises.
"""

import numpy as np


def optimiser_step(method, theta, g, state, t, gamma, beta=0.9, rho=0.99,
                   beta1=0.9, beta2=0.999, eps=1e-8):
    """One update of theta from the gradient g at step t = 1, 2, ...

    state carries the running quantities between steps.
    """
    if method == "plain":
        return theta - gamma * g, state
    if method == "momentum":
        state["v"] = v = beta * state.get("v", 0.0) + gamma * g
        return theta - v, state
    if method == "adagrad":
        state["r"] = r = state.get("r", 0.0) + g * g
        return theta - gamma * g / (np.sqrt(r) + eps), state
    if method == "rmsprop":
        state["r"] = r = state.get("r", 0.0) * rho + (1 - rho) * g * g
        return theta - gamma * g / (np.sqrt(r) + eps), state
    if method == "adam":
        state["m"] = m = beta1 * state.get("m", 0.0) + (1 - beta1) * g
        state["r"] = r = beta2 * state.get("r", 0.0) + (1 - beta2) * g * g
        m_hat = m / (1 - beta1 ** t)
        r_hat = r / (1 - beta2 ** t)
        return theta - gamma * m_hat / (np.sqrt(r_hat) + eps), state
    raise ValueError(f"unknown method {method}")


def optimise(grad, theta0, method, gamma, num_iters=1000, tol=1e-8, **kw):
    """Full-gradient loop. Returns all iterates and the number of steps.

    grad is a function of theta alone. Pass a closure that fixes X, y and
    lambda, or the automatic gradient from jax.grad.
    """
    theta = np.array(theta0, dtype=float)
    state = {}
    history = [theta.copy()]
    t = 0
    for t in range(1, num_iters + 1):
        g = grad(theta)
        theta, state = optimiser_step(method, theta, g, state, t, gamma, **kw)
        history.append(theta.copy())
        if np.linalg.norm(g) < tol:
            break
    return np.array(history), t


def make_batches(n, batch_size, rng):
    """Shuffle the indices and split them into minibatches, Section 4.7."""
    idx = rng.permutation(n)
    return [idx[i:i + batch_size] for i in range(0, n, batch_size)]


def step_length(t, t0, t1):
    """Learning-rate schedule of Eq. (4.40), gamma_t = t0 / (t + t1)."""
    return t0 / (t + t1)


def sgd(grad_batch, theta0, X, y, method="plain", n_epochs=50, batch_size=5,
        gamma=0.1, schedule=None, seed=2026, **kw):
    """Minibatch stochastic gradient descent, Eq. (4.34).

    grad_batch(theta, X_batch, y_batch) returns the minibatch gradient.
    schedule=(t0, t1) replaces the constant gamma by Eq. (4.40). Returns
    the iterate after every epoch.
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    theta = np.array(theta0, dtype=float)
    state = {}
    t = 0
    history = [theta.copy()]
    for epoch in range(n_epochs):
        for batch in make_batches(n, batch_size, rng):
            t += 1
            g = grad_batch(theta, X[batch], np.ravel(y)[batch])
            gamma_t = gamma if schedule is None else step_length(t, *schedule)
            theta, state = optimiser_step(method, theta, g, state, t, gamma_t, **kw)
        history.append(theta.copy())
    return np.array(history)
