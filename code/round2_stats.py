"""Final, internally consistent statistics:
aggregated RMSE = mean of the three per-component RMSE values (same convention
as Table 2), computed per split -> paired t-tests -> Holm-Bonferroni."""
import os, sys, json, math, warnings
import numpy as np
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
BASE = ROOT
sys.path.insert(0, os.path.join(BASE, "code"))
os.chdir(os.path.join(BASE, "code"))
from analysis import (X_raw, Y_raw, N, NC, Scaler, MODELS, run_splits, MTGPR, DEFAULT_MTGPR)
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from scipy import stats

D = 4
V = np.zeros((D - 1, D))
for i in range(D - 1):
    k = i + 1
    V[i, :k] = 1.0; V[i, k] = -k
    V[i, :] /= math.sqrt(k * (k + 1))


def to_parts(Y3):
    return np.hstack([Y3, np.clip(100.0 - Y3.sum(1, keepdims=True), 1.0, None)])


def ilr(p):
    p = np.clip(p, 1e-6, None); p = p / p.sum(1, keepdims=True) * 100.0
    lp = np.log(p)
    return (lp - lp.mean(1, keepdims=True)) @ V.T


def ilr_inv(Z):
    clr = Z @ V
    e = np.exp(clr - clr.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True) * 100.0


print("running 20 splits (support=8)...")
res8, raw8 = run_splits(8)
per = {m: [] for m in MODELS}
for r in range(20):
    for m in MODELS:
        per[m].append(float(np.mean([raw8[m]["RMSE"][c][r] for c in range(NC)])))

extra = ["RSM", "C-MTGPR", "MeanBaseline"]
for m in extra:
    per[m] = []
percomp = {m: {c: [] for c in range(NC)} for m in extra}
for r in range(20):
    perm = np.random.RandomState(r).permutation(N)
    sup, que = perm[:8], perm[8:]
    sx, sy = Scaler().fit(X_raw[sup]), Scaler().fit(Y_raw[sup])
    Xtr, Xte = sx.tf(X_raw[sup]), sx.tf(X_raw[que])
    Ytr_s = sy.tf(Y_raw[sup])
    pf = PolynomialFeatures(degree=2, include_bias=True)
    preds = {
        "RSM": sy.itf(Ridge(alpha=1.0, random_state=0).fit(pf.fit_transform(Xtr), Ytr_s)
                      .predict(pf.transform(Xte))),
        "MeanBaseline": sy.itf(np.tile(Ytr_s.mean(0), (len(que), 1))),
        "C-MTGPR": ilr_inv(MTGPR(**{**DEFAULT_MTGPR, "rng": np.random.default_rng(r)})
                           .fit(Xtr, ilr(to_parts(sy.itf(Ytr_s)))).predict(Xte))[:, :3],
    }
    for m, p in preds.items():
        per[m].append(float(np.mean([np.sqrt(((p[:, c] - Y_raw[que, c]) ** 2).mean())
                                     for c in range(NC)])))
        for c in range(NC):
            percomp[m][c].append(float(np.sqrt(((p[:, c] - Y_raw[que, c]) ** 2).mean())))

allm = MODELS + ["C-MTGPR", "RSM", "MeanBaseline"]
print()
print("aggregated RMSE (mean of 3 per-component RMSE), 20 splits:")
for m in allm:
    a = np.array(per[m])
    print(f"  {m:13s} {a.mean():.3f} +- {a.std():.3f}")


def holm(pairs):
    ps = [p for _, p in pairs]
    mm = len(ps); out = [0.0] * mm; run = 0.0
    for k, idx in enumerate(np.argsort(ps)):
        run = max(run, min(1.0, (mm - k) * ps[idx])); out[idx] = run
    return out


OUT = {"agg": {m: {"mean": float(np.mean(per[m])), "sd": float(np.std(per[m]))} for m in allm},
       "percomp": {m: {str(c): {"mean": float(np.mean(percomp[m][c])),
                                "sd": float(np.std(percomp[m][c]))} for c in range(NC)}
                   for m in extra}}

for ref in ["MTGPR", "C-MTGPR"]:
    print()
    print(f"--- paired t-test vs {ref} (Holm-Bonferroni over "
          f"{len(allm) - 1} comparisons) ---")
    others = [m for m in allm if m != ref]
    ts, ps = [], []
    for b in others:
        t, p = stats.ttest_rel(np.array(per[ref]), np.array(per[b]))
        ts.append(float(t)); ps.append(float(p))
    ha = holm(list(zip(others, ps)))
    tbl = {}
    for i, b in enumerate(others):
        tbl[b] = {"t": ts[i], "p": ps[i], "p_holm": ha[i],
                  "sig_raw": bool(ps[i] < .05), "sig_holm": bool(ha[i] < .05)}
        print(f"  {ref:8s} vs {b:13s} t={ts[i]:+.3f} p={ps[i]:.4f} "
              f"| raw:{'YES' if ps[i] < .05 else 'no ':3s} | Holm={ha[i]:.4f} "
              f"{'YES' if ha[i] < .05 else 'no'}")
    OUT[f"tests_vs_{ref}"] = tbl

with open(os.path.join(BASE, "process_data", "round2_stats.json"), "w") as f:
    json.dump(OUT, f, indent=2)
print("\nSaved round2_stats.json")
