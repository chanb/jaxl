import cvxpy as cp
import numpy as np


def linear_kernel(x):
    return x @ x.T


def polynomial_kernel(x, deg, offset=0):
    return np.power(x @ x.T + offset, deg)


# XXX: Assumes linearly separable for now
def primal_svm(train_x, train_y, bias=True):
    assert len(train_x.shape) == 2
    assert len(train_y.shape) == 1

    N, d = train_x.shape

    padded_x = train_x
    if bias:
        d = d + 1
        padded_x = np.concatenate((train_x, np.ones((N, 1))), axis=-1)
    P = np.eye(d)
    if bias:
        P[d - 1] = 0
    q = np.zeros(d)
    G = padded_x * -train_y[:, None]
    h = -np.ones(N)

    primal_var = cp.Variable(d)
    prob = cp.Problem(
        cp.Minimize((1 / 2) * cp.quad_form(primal_var, P) + q.T @ primal_var),
        [G @ primal_var <= h],
    )
    loss = prob.solve(verbose=True, max_iter=50000)
    params = primal_var.value
    if not bias:
        params = np.concatenate([params, [0]])
    return loss, params


def dual_svm(train_x, train_y, kernel=linear_kernel):
    assert len(train_x.shape) == 2
    assert len(train_y.shape) == 1

    N, _ = train_x.shape
    K = kernel(train_x)
    P = (train_y[:, None] @ train_y[None]) * K
    q = -np.ones(N)
    A = train_y[None]
    b = np.zeros(1)
    G = -np.eye(N)
    h = np.zeros(N)

    dual_var = cp.Variable(N)
    prob = cp.Problem(
        cp.Minimize((1 / 2) * cp.quad_form(dual_var, P) + q.T @ dual_var),
        [G @ dual_var <= h, A @ dual_var == b],
    )
    loss = -prob.solve(verbose=True, max_iter=50000)
    alphas = dual_var.value
    return loss, alphas
