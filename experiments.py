import time, numpy as np, matplotlib.pyplot as plt
from mesh_topology import create_mesh_network
from pgd_solver import solve_pgd

#Scalability – runtime vs N
sizes = [6, 12, 20, 30]
runtimes = []

for N in sizes:
    # pick rows,cols so rows*cols=N (simple helper)
    rows = int(np.sqrt(N)); cols = int(np.ceil(N/rows))
    G, c = create_mesh_network(rows=rows, cols=cols, seed=0)
    P = np.ones(c.shape[0]) * 10

    t0 = time.time()
    solve_pgd(c, P, steps=700, alpha=0.05)   # keep settings same
    runtimes.append(time.time() - t0)
    print(f"N={N}  time={runtimes[-1]:.2f}s")

plt.figure()
plt.plot(sizes, runtimes, marker='o')
plt.xlabel("Number of nodes N"); plt.ylabel("Runtime (s)")
plt.title("PGD Scalability")
plt.grid(True); plt.show()

#Stress test – AoI vs shrinking power budget
G, c = create_mesh_network(rows=4, cols=4, seed=1, cap_low=5, cap_high=12)   # 16-node example
#budget_levels = np.linspace(10, 1, 10)   # from 10 → 1# --- shrink budgets further --------------------------------------
budget_levels = np.linspace(2.0, 0.1, 10)      # 2 → 0.1 instead of 10 → 1
avg_aoi = []

for P_scale in budget_levels:
    P = np.ones(c.shape[0]) * P_scale
    a, *_ = solve_pgd(c, P, steps=500, alpha=0.05)
    avg_aoi.append(np.mean(a))
    print(f"P={P_scale:.1f}  Avg AoI={avg_aoi[-1]:.3f}")

plt.figure()
plt.plot(budget_levels, avg_aoi, marker='s')
plt.gca().invert_xaxis()   # high→low power left→right
plt.xlabel("Per-node power budget"); plt.ylabel("Average AoI")
plt.title("AoI degradation under power scarcity")
plt.grid(True); plt.show()

#Weight sweep – single node’s AoI vs wₖ
G, c = create_mesh_network(rows=3, cols=3, seed=2, cap_low=3, cap_high=6)   # 9 nodes
#P = np.ones(c.shape[0]) * 10
P = np.ones(c.shape[0]) * 0.5        # VERY tight power budget
#target = 4                    # node index to prioritise
target=0


# >>>>  NEW: give Node‑4 bigger capacity from its neighbours
for i in range(c.shape[0]):
    if c[i, target] > 0:        # if link exists
        c[i, target] = c[target, i] = 15           # plenty of capacity
# <<<<  END PATCH




#weights = np.arange(1, 11)    # 1 … 10
weights = np.arange(1, 51, 5)        # sweep 1,6,11,…,46   ← larger range initial 51
a_target, a_others = [], []

for w_k in weights:
    w = np.ones(c.shape[0])
    w[target] = w_k
    a, *_ = solve_pgd(c, P, w=w, steps=1200, alpha=0.03) #600,0.05
    a_target.append(a[target])
    a_others.append(np.mean(np.delete(a, target)))

plt.figure()
plt.plot(weights, a_target, 'o-', label=f'Node {target} AoI')
plt.plot(weights, a_others, 's--', label='Avg AoI of others')
plt.xlabel(f"Weight w_{target}")
plt.ylabel("AoI")
plt.title("Effect of weight on AoI allocation")
plt.legend(); plt.grid(True); plt.show()