"""Complete the ablation and LOOCV tables in the same convention as the rest of
the manuscript (aggregated metric = mean of the three per-component values):
per-component RMSE/MAE for RSM, C-MTGPR and the mean-prediction baseline."""
import os, sys, json, math, warnings
import numpy as np
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
BASE = ROOT
sys.path.insert(0, os.path.join(BASE, "code"))
os.chdir(os.path.join(BASE, "code"))
from analysis import (X_raw, Y_raw, N, NC, Scaler, MTGPR, DEFAULT_MTGPR)
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

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


EXTRA = ["RSM", "C-MTGPR", "MeanBaseline"]
OUT = {"ablation": {}, "loocv": {}}

# ---------------- ablation ----------------
for ss in [4, 6, 8, 10, 12]:
    acc = {m: {c: {"RMSE": [], "MAE": []} for c in range(NC)} for m in EXTRA}
    for r in range(20):
        perm = np.random.RandomState(r).permutation(N)
        sup, que = perm[:ss], perm[ss:]
        sx, sy = Scaler().fit(X_raw[sup]), Scaler().fit(Y_raw[sup])
        Xtr, Xte = sx.tf(X_raw[sup]), sx.tf(X_raw[que])
        Ytr_s = sy.tf(Y_raw[sup])
        pf = PolynomialFeatures(degree=2, include_bias=True)
        preds = {
            "RSM": sy.itf(Ridge(alpha=1.0, random_state=0)
                          .fit(pf.fit_transform(Xtr), Ytr_s).predict(pf.transform(Xte))),
            "MeanBaseline": sy.itf(np.tile(Ytr_s.mean(0), (len(que), 1))),
            "C-MTGPR": ilr_inv(MTGPR(**{**DEFAULT_MTGPR, "rng": np.random.default_rng(r)})
                               .fit(Xtr, ilr(to_parts(sy.itf(Ytr_s)))).predict(Xte))[:, :3],
        }
        for m, p in preds.items():
            for c in range(NC):
                acc[m][c]["RMSE"].append(float(np.sqrt(((p[:, c] - Y_raw[que, c]) ** 2).mean())))
                acc[m][c]["MAE"].append(float(np.mean(np.abs(p[:, c] - Y_raw[que, c]))))
    OUT["ablation"][str(ss)] = {
        m: {("RMSE_" + ["Cell", "Hemi", "Lign"][c]): float(np.mean(acc[m][c]["RMSE"]))
            for c in range(NC)} |
        {("MAE_" + ["Cell", "Hemi", "Lign"][c]): float(np.mean(acc[m][c]["MAE"]))
         for c in range(NC)} |
        {"agg_RMSE": float(np.mean([np.mean(acc[m][c]["RMSE"]) for c in range(NC)]))}
        for m in EXTRA}
    print(f"ablation n={ss}: " + "  ".join(
        f"{m}={OUT['ablation'][str(ss)][m]['agg_RMSE']:.2f}" for m in EXTRA))

# ---------------- LOOCV ----------------
acc = {m: {c: {"RMSE": [], "MAE": []} for c in range(NC)} for m in EXTRA}
for i in range(N):
    tr = [j for j in range(N) if j != i]; te = [i]
    sx, sy = Scaler().fit(X_raw[tr]), Scaler().fit(Y_raw[tr])
    Xtr, Xte = sx.tf(X_raw[tr]), sx.tf(X_raw[te])
    Ytr_s = sy.tf(Y_raw[tr])
    pf = PolynomialFeatures(degree=2, include_bias=True)
    preds = {
        "RSM": sy.itf(Ridge(alpha=1.0, random_state=0)
                      .fit(pf.fit_transform(Xtr), Ytr_s).predict(pf.transform(Xte))),
        "MeanBaseline": sy.itf(np.tile(Ytr_s.mean(0), (1, 1))),
        "C-MTGPR": ilr_inv(MTGPR(**DEFAULT_MTGPR)
                           .fit(Xtr, ilr(to_parts(sy.itf(Ytr_s)))).predict(Xte))[:, :3],
    }
    for m, p in preds.items():
        for c in range(NC):
            acc[m][c]["RMSE"].append(float(np.sqrt(((p[0, c] - Y_raw[te[0], c]) ** 2).mean())))
            acc[m][c]["MAE"].append(float(np.mean(np.abs(p[0, c] - Y_raw[te[0], c]))))
for m in EXTRA:
    OUT["loocv"][m] = {("RMSE_" + ["Cell", "Hemi", "Lign"][c]): float(np.mean(acc[m][c]["RMSE"]))
                       for c in range(NC)} | \
                      {("MAE_" + ["Cell", "Hemi", "Lign"][c]): float(np.mean(acc[m][c]["MAE"]))
                       for c in range(NC)} | \
                      {"agg_RMSE": float(np.mean([np.mean(acc[m][c]["RMSE"]) for c in range(NC)])),
                       "agg_MAE": float(np.mean([np.mean(acc[m][c]["MAE"]) for c in range(NC)]))}
    print(f"LOOCV {m}: agg RMSE={OUT['loocv'][m]['agg_RMSE']:.3f} "
          f"agg MAE={OUT['loocv'][m]['agg_MAE']:.3f}")

with open(os.path.join(BASE, "process_data", "round2d.json"), "w") as f:
    json.dump(OUT, f, indent=2)
print("Saved round2d.json")
