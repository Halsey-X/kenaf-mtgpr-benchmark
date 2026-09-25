import numpy as np, openpyxl, warnings
warnings.filterwarnings("ignore")
np.random.seed(0)
import sys
sys.path.insert(0, ".")
from mtgpr import MTGPR, _rbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
f = os.path.join(ROOT, "data", "kenaf_pretreatment_dataset.xlsx")
ws = openpyxl.load_workbook(f, data_only=True)["pre"]
rows = list(ws.iter_rows(values_only=True))[2:]
data = np.array([list(r[:7]) for r in rows], float)
X = data[:, :3]; Y = data[:, 3:6]
P, D = 3, 3

# standardize like analysis.py (per-split using support stats)
rng = np.random.RandomState(0)
perm = rng.permutation(31)
sup, que = perm[:8], perm[8:]
def scal(a, axis=0):
    return (a - a.mean(0)) / (a.std(0) + 1e-9)
Xs, Xq = scal(X[sup]), scal(X[que])
Ys = scal(Y[sup])

# ---- gradient check vs finite differences ----
m = MTGPR(rank=2, ard=False, maxiter=1, restarts=1)
theta = m._init_theta(Xs, Ys)
def nll(th): return m._nll_and_grad(th, Xs, Ys)[0]
def grad(th): return m._nll_and_grad(th, Xs, Ys)[1]
g = grad(theta)
eps = 1e-5
fd = np.zeros_like(theta)
for i in range(len(theta)):
    tp = theta.copy(); tp[i] += eps
    tm = theta.copy(); tm[i] -= eps
    fd[i] = (nll(tp) - nll(tm)) / (2 * eps)
rel = np.abs(g - fd) / (np.abs(fd) + 1e-8)
print("grad-check max rel err:", np.max(rel).round(4), "(should be < 1e-2)")
print("analytic sample:", np.round(g[:6], 4))
print("finite-diff  sample:", np.round(fd[:6], 4))

# eps sweep on a couple of indices to adjudicate finite-difference accuracy
print("\n-- eps sweep on index 0 (ls) and 1 (log_sf) --")
for idx in [0, 1]:
    vals = []
    for e in [1e-4, 1e-5, 1e-6, 1e-7]:
        tp = theta.copy(); tp[idx] += e
        tm = theta.copy(); tm[idx] -= e
        vals.append((nll(tp) - nll(tm)) / (2 * e))
    print(f"idx {idx}: analytic={g[idx]:.4f}  fd(eps)=", [round(v, 3) for v in vals])

# ---- basic fit + predict RMSE on this split ----
m2 = MTGPR(rank=2, ard=False, maxiter=120, restarts=3).fit(Xs, Ys)
Yq = m2.predict(Xq)
rmse = np.sqrt(((Yq - scal(Y[que])) ** 2).mean(0))
print("\nMTGPR support=8 query RMSE (cell,hemi,lign):", np.round(rmse, 3))
print("mean-only baseline RMSE:", np.round(np.sqrt(((scal(Y[que]) - scal(Y[sup]).mean(0)) ** 2).mean(0)), 3))
B, corr = m2.coregionalization()
print("\nLearned coregionalization B:\n", np.round(B, 3))
print("Normalized task correlation:\n", np.round(corr, 3))

# ---- quick comparison across ranks on this split ----
for rnk in [1, 2, 3]:
    mm = MTGPR(rank=rnk, ard=False, maxiter=120, restarts=2).fit(Xs, Ys)
    rq = np.sqrt(((mm.predict(Xq) - scal(Y[que])) ** 2).mean(0))
    print(f"rank={rnk} RMSE:", np.round(rq, 3), "mean:", round(rq.mean(), 3))
