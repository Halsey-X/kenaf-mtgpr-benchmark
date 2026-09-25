"""Correct LOOCV metrics.

With one held-out sample per fold, the per-fold RMSE of a single component is
mathematically identical to its MAE, which is why the manuscript's Table 5 shows
RMSE == MAE for every model. The quantity a reader assumes ("RMSE") must be
pooled over folds:  RMSE_c = sqrt( mean_i  (y_ic - p_ic)^2 ).
This script recomputes all models that way and saves the fold-level predictions.
"""
import os, sys, json, math, warnings
import numpy as np
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
BASE = ROOT
sys.path.insert(0, os.path.join(BASE, "code"))
os.chdir(os.path.join(BASE, "code"))
from analysis import (X_raw, Y_raw, N, NC, Scaler, MODELS, make_baselines,
                      MTGPR, STPN, DEFAULT_MTGPR, DEFAULT_STPN)
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


ALL = MODELS + ["C-MTGPR", "RSM", "MeanBaseline"]
pred = {m: np.zeros((N, NC)) for m in ALL}

for i in range(N):
    tr = [j for j in range(N) if j != i]; te = [i]
    sx, sy = Scaler().fit(X_raw[tr]), Scaler().fit(Y_raw[tr])
    Xtr, Xte = sx.tf(X_raw[tr]), sx.tf(X_raw[te])
    Ytr_s = sy.tf(Y_raw[tr])
    bl = make_baselines()
    for m in MODELS:
        if m == "ST-PN":
            pred[m][i] = sy.itf(STPN(**{**DEFAULT_STPN, "seed": 7}).fit(Xtr, Ytr_s).predict(Xte))[0]
        elif m == "MTGPR":
            pred[m][i] = sy.itf(MTGPR(**DEFAULT_MTGPR).fit(Xtr, Ytr_s).predict(Xte))[0]
        else:
            pred[m][i] = sy.itf(bl[m].fit(Xtr, Ytr_s).predict(Xte))[0]
    pf = PolynomialFeatures(degree=2, include_bias=True)
    pred["RSM"][i] = sy.itf(Ridge(alpha=1.0, random_state=0)
                            .fit(pf.fit_transform(Xtr), Ytr_s).predict(pf.transform(Xte)))[0]
    pred["MeanBaseline"][i] = sy.itf(Ytr_s.mean(0))
    pred["C-MTGPR"][i] = ilr_inv(MTGPR(**DEFAULT_MTGPR)
                                 .fit(Xtr, ilr(to_parts(sy.itf(Ytr_s)))).predict(Xte))[0][:3]
    if (i + 1) % 10 == 0:
        print(f"  fold {i + 1}/{N}")

NAMES = ["Cell", "Hemi", "Lign"]


def r2(y, p):
    return 1.0 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


OUT = {"loocv": {}, "loocv_r2": {}, "loocv_pred": {}}
print()
print("LOOCV -- RMSE pooled over folds (correct), MAE = mean |error|")
print(f"{'model':13s} {'RMSEcell':>9s} {'RMSEhemi':>9s} {'RMSElign':>9s} "
      f"{'aggRMSE':>8s} {'aggMAE':>7s} {'R2cell':>7s} {'R2hemi':>7s} {'R2lign':>7s}")
for m in ALL:
    per_c_rmse = [float(np.sqrt(np.mean((Y_raw[:, c] - pred[m][:, c]) ** 2))) for c in range(NC)]
    per_c_mae = [float(np.mean(np.abs(Y_raw[:, c] - pred[m][:, c]))) for c in range(NC)]
    r2s = [float(r2(Y_raw[:, c], pred[m][:, c])) for c in range(NC)]
    OUT["loocv"][m] = {f"RMSE_{NAMES[c]}": per_c_rmse[c] for c in range(NC)}
    OUT["loocv"][m].update({f"MAE_{NAMES[c]}": per_c_mae[c] for c in range(NC)})
    OUT["loocv"][m]["agg_RMSE"] = float(np.mean(per_c_rmse))
    OUT["loocv"][m]["agg_MAE"] = float(np.mean(per_c_mae))
    OUT["loocv_r2"][m] = {NAMES[c]: r2s[c] for c in range(NC)}
    OUT["loocv_pred"][m] = pred[m].tolist()
    print(f"{m:13s} {per_c_rmse[0]:9.3f} {per_c_rmse[1]:9.3f} {per_c_rmse[2]:9.3f} "
          f"{np.mean(per_c_rmse):8.3f} {np.mean(per_c_mae):7.3f} "
          f"{r2s[0]:7.3f} {r2s[1]:7.3f} {r2s[2]:7.3f}")

with open(os.path.join(BASE, "process_data", "round2_loocv.json"), "w") as f:
    json.dump(OUT, f, indent=2)
print("\nSaved round2_loocv.json")
