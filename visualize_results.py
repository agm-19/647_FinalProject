import matplotlib.pyplot as plt, seaborn as sns, numpy as np
from matplotlib.sankey import Sankey

# ------------ helper ---------------------------------------------------------
def _finalise(save, fname):
    plt.tight_layout()
    if save:
        plt.savefig(fname, dpi=300)
        plt.close()
    else:
        plt.show(block=False)

# ------------ core plots -----------------------------------------------------
def plot_convergence(hist, save, idx):
    plt.figure(); plt.plot(hist); plt.xlabel("Iter"); plt.ylabel("Σw·AoI")
    plt.title("PGD Convergence"); _finalise(save, f"Figure_{idx}.png")

def plot_aoi(a, save, idx, label="PGD"):
    plt.figure(); plt.bar(range(len(a)), a); plt.xlabel("Node"); plt.ylabel("AoI")
    plt.title(f"AoI per Node ({label})"); _finalise(save, f"Figure_{idx}.png")

def plot_matrix(mat, title, save, idx):
    plt.figure(); sns.heatmap(mat, annot=True, fmt=".2f", cmap="Blues")
    plt.title(title); plt.xlabel("To"); plt.ylabel("From")
    _finalise(save, f"Figure_{idx}.png")

def plot_power(lam, P, save, idx):
    power = np.sum(lam**2, 1); plt.figure(); plt.bar(range(len(power)), power)
    plt.axhline(P[0], ls='--', c='r'); plt.title("Power / Node")
    plt.xlabel("Node"); plt.ylabel("Σ λ²"); _finalise(save, f"Figure_{idx}.png")

def plot_bw(x, save, idx):
    bw = np.sum(x,1); plt.figure(); plt.bar(range(len(bw)), bw)
    plt.axhline(1, ls='--', c='r'); plt.title("Bandwidth Usage / Node")
    plt.xlabel("Node"); plt.ylabel("Σ x"); _finalise(save, f"Figure_{idx}.png")

def plot_util(lam, x, c, save, idx):
    util = np.divide(lam, x*c, out=np.zeros_like(lam), where=(x*c)>0)
    plt.figure(); sns.heatmap(util, annot=True, fmt=".2f", cmap="PuBu")
    plt.title("Link Capacity Utilisation"); _finalise(save, f"Figure_{idx}.png")

def plot_inrate(lam, save, idx):
    inrate = lam.sum(0); plt.figure(); plt.bar(range(len(inrate)), inrate)
    plt.title("Incoming Update Rate per Node"); plt.xlabel("Node"); plt.ylabel("pps")
    _finalise(save, f"Figure_{idx}.png")

def plot_cdf(a_pgd, a_uni, save, idx):
    N = len(a_pgd); xx = np.linspace(0,1,N,endpoint=False)+1/N
    plt.figure(); plt.step(np.sort(a_pgd), xx, label="PGD"); plt.step(np.sort(a_uni), xx, label="Uniform")
    plt.xlabel("AoI"); plt.ylabel("CDF"); plt.title("AoI CDF (PGD vs Uniform)"); plt.legend()
    _finalise(save, f"Figure_{idx}.png")

def plot_scatter_power(lam, save, idx):
    p = lam**2; plt.figure(); plt.scatter(p.flatten(), lam.flatten())
    plt.xlabel("Link Power (λ²)"); plt.ylabel("Rate λ"); plt.title("Power vs Rate per Link")
    _finalise(save, f"Figure_{idx}.png")

def plot_sankey_into_node(lam, k, save, idx):
    inflow = lam[:,k]; labels=[f"{i}→{k}" for i in range(len(inflow))]
    flows = list(-inflow) + [inflow.sum()]
    sank = Sankey(unit=None); sank.add(flows=flows, labels=labels+["Σ in"], orientations=[0]*len(flows))
    sank.finish(); plt.title(f"Sankey of Updates into Node {k}")
    _finalise(save, f"Figure_{idx}.png")
