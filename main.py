import numpy as np, visualize_results as viz
from mesh_topology import create_mesh_network
from pgd_solver import solve_pgd
from aoi_optimizer import solve_aoi_cvx
import matplotlib.pyplot as plt

def uniform_policy(c):
    N=c.shape[0]; outdeg=(c>0).sum(1); x=np.divide(1,outdeg, where=outdeg[:,None]>0)*(c>0)
    lam = 0.5*np.minimum(x*c,1)             # half capacity just as baseline
    a   = 1/(lam.sum(0)+1e-6); obj=a.mean()
    return a,x,lam,obj

def main(save=False):
   # G,c = create_mesh_network(); N=c.shape[0]; P=np.ones(N)*10
    # NEW – bigger, irregular edge mesh
    G, c = create_mesh_network(rows=2, cols=5,      # 16-node grid
                           seed=3,
                           extra_prob=0.05,     # sprinkle extra links
                           cap_low=2, cap_high=12)  # heterogeneous capacities
    N = c.shape[0]
    w = np.ones(N)
    w[0] = 5

    # heterogeneous per-node power budgets (0.2 – 1.0 units)
    P = np.random.uniform(0.05, 0.15, size=N)
    # --- Solvers
    a_cvx,_,_,obj_cvx,_ = solve_aoi_cvx(c,P,w=w)
    a_pgd, x_pgd, lam_pgd, obj_pgd, obj_hist, grad_hist, viol_hist, gap_hist = solve_pgd(c, P, w=w, opt_obj=obj_cvx, steps=700, alpha=0.04)
    a_uni,x_uni,lam_uni,_ = uniform_policy(c)

    print("PGD obj:",obj_pgd,"  CVX obj:",obj_cvx)

    # --- 10 plots
    viz.plot_convergence(obj_hist,               save,1)
    viz.plot_aoi(a_pgd,                      save,2)
    viz.plot_matrix(x_pgd,"x matrix",        save,3)
    viz.plot_matrix(lam_pgd,"λ matrix",      save,4)
    viz.plot_power(lam_pgd,P,                save,5)
    viz.plot_bw(x_pgd,                       save,6)
    viz.plot_util(lam_pgd,x_pgd,c,           save,7)
    viz.plot_inrate(lam_pgd,                 save,8)
    viz.plot_cdf(a_pgd,a_uni,                save,9)
    viz.plot_scatter_power(lam_pgd,          save,10)
    # optional Sankey into node 5
    viz.plot_sankey_into_node(lam_pgd,5,     save,11)
   

    plt.figure(); plt.semilogy(gap_hist)
    plt.xlabel("Iteration"); plt.ylabel("Rel gap to CVX optimum")
    plt.title("PGD Optimality Gap"); #plt.show()


    # 11: objective (log-y)
    plt.figure(); plt.semilogy(obj_hist)
    plt.xlabel("Iteration"); plt.ylabel("Objective")
    plt.title("PGD Convergence (log scale)")
    if save: plt.savefig("Figure_12.png", dpi=300); plt.close()

    # 12: gradient norm
    plt.figure(); plt.semilogy(grad_hist)
    plt.xlabel("Iteration"); plt.ylabel("‖∇f‖")
    plt.title("Gradient Norm vs Iteration")
    if save: plt.savefig("Figure_13.png", dpi=300); plt.close()

    # 13: λ≤xc violation
    plt.figure(); plt.semilogy(viol_hist)
    plt.xlabel("Iteration"); plt.ylabel("Max constraint violation")
    plt.title("Feasibility Convergence")
    if save: plt.savefig("Figure_14.png", dpi=300); plt.close()


    if not save:
        input("Press <Enter> to close figures…")

if __name__=="__main__":
    main(save=False)   # set to True to write Figure_1.png … Figure_11.png
