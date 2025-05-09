# pgd_solver.py
import numpy as np

def _project_simplex(v):
    """Project vector v onto simplex {x >=0, sum x =1}."""
    v_sorted = np.sort(v)[::-1]
    cssv = np.cumsum(v_sorted)
    rho = np.nonzero(v_sorted * np.arange(1, len(v)+1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1)
    return np.maximum(v - theta, 0)

def solve_pgd(c, P_budget, w=None, *, opt_obj=None, gamma=3, steps=500, alpha=0.05, eps=1e-6):
    """
    If opt_obj is provided (CVX optimum), the function also returns gap_hist,
    the relative gap per iteration.
    """
    N = c.shape[0]
    w = np.ones(N) if w is None else w

    # initial feasible point
    x   = np.full((N,N), 1/np.maximum(1,(c>0).sum(1))[:,None]) * (c>0)
    lam = np.minimum(x*c, 0.5)
    a   = 1/(lam.sum(0)+eps)

    obj_hist, grad_hist, viol_hist, gap_hist = [], [], [], []

    for _ in range(steps):
        # ----- gradients -----
        grad_a   = w
        grad_lam = -w[None,:]*(a**2) + 2*lam        # fixed weight factor
        grad_x   = -lam*c + gamma

        # ----- descent -------
        a   -= alpha*grad_a
        lam -= alpha*grad_lam
        x   -= alpha*grad_x

        # ----- projection ----
        a = np.maximum(a, 1/(lam.sum(0)+eps))
        lam = np.clip(lam, 0, x*c)
        lam = np.maximum(lam, 0)        # keep λ non-negative

        for i in range(N):
            x[i,:] = _project_simplex(np.maximum(x[i,:],0)*(c[i,:]>0))
            scale  = np.sqrt(max(1,(lam[i,:]**2).sum()/P_budget[i]))
            lam[i,:] /= scale

        # ----- diagnostics ----
        obj = w @ a + gamma * x.sum()
        obj_hist.append(obj)
        grad_hist.append(np.linalg.norm(grad_lam))
        viol_hist.append(max(0, np.max(lam - x*c)))
        if opt_obj is not None:
            gap_hist.append((obj - opt_obj)/opt_obj)

    return a, x, lam, obj, obj_hist, grad_hist, viol_hist, gap_hist