#This code is to verify the solver using cvx
import cvxpy as cp
import numpy as np

def solve_aoi_cvx(c, P_budget, w=None, eps=1e-6, gamma=3):
    N = c.shape[0]
    w = np.ones(N) if w is None else w
    x = cp.Variable((N, N), nonneg=True)
    lam = cp.Variable((N, N), nonneg=True)
    a = cp.Variable(N, nonneg=True)
    constraints = []

    # AoI constraint (DCP)
    for j in range(N):
        s_j = cp.sum(lam[:, j]) + eps          # affine & positive
        constraints += [a[j] >= cp.inv_pos(s_j)]

    # bandwidth and power
    for i in range(N):
        constraints += [cp.sum(x[i, :]) <= 1]
        constraints += [cp.sum(cp.power(lam[i, :], 2)) <= P_budget[i]]

    # rate ≤ bandwidth x capacity
    constraints += [lam <= cp.multiply(x, c)]

    prob = cp.Problem(cp.Minimize(w @ a + gamma * cp.sum(x)), constraints)
    prob.solve(solver=cp.SCS, verbose=True)     
    return a.value, x.value, lam.value, prob.value, prob.status
