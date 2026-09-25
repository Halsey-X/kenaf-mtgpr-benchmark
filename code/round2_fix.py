"""Round-2 FIX.
1. Implement ILR-constrained multi-task GP (C-MTGPR): enforce non-negativity and
   the mass-balance bound by modelling the 4-part closed composition
   (cellulose, hemicellulose, lignin, other = 100 - sum) in ILR coordinates.
   This converts Limitation (4) of the manuscript into an actual contribution.
2. Re-benchmark C-MTGPR against every model at support=8, across the
   support-size ablation, and under LOOCV.
3. Rebuild the downstream composition -> yield chain with honest LOOCV R2.
"""
import os, sys, json, math, warnings
import numpy as np
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
BASE = ROOT
sys.path.insert(0, os.path.join(BASE, "code"))
os.chdir(os.path.join(BASE, "code"))

from analysis import (X_raw, Y_raw, N, NC, Scaler, MODELS, run_splits,
                      make_baselines, MTGPR, STPN, DEFAULT_MTGPR, DEFAULT_STPN)
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
import openpyxl

SRC = os.path.join(ROOT, "data", "kenaf_pretreatment_dataset.xlsx")
wb = openpyxl.load_workbook(SRC, data_only=True)
ws = wb["pre"]
rows = list(ws.iter_rows(values_only=True))
full = np.array([list(r[:7]) for r in rows[2:]], dtype=float)
Yield_raw = full[:, 6]

OUT = {}
NAMES = ["Cellulose", "Hemicellulose", "Lignin"]

# ---------------- ILR machinery ----------------
D = 4  # C, H, L, other
V = np.zeros((D - 1, D))
for i in range(D - 1):
    k = i + 1
    V[i, :k] = 1.0
    V[i, k] = -k
    V[i, :] /= math.sqrt(k * (k + 1))


def to_parts(Y3):
    s = Y3.sum(1, keepdims=True)
    other = np.clip(100.0 - s, 1.0, None)
    return np.hstack([Y3, other])


def ilr(parts):
    p = np.clip(parts, 1e-6, None)
    p = p / p.sum(1, keepdims=True) * 100.0
    lp = np.log(p)
    return (lp - lp.mean(1, keepdims=True)) @ V.T


def ilr_inv(Z):
    clr = Z @ V
    e = np.exp(clr - clr.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True) * 100.0


def c_mtgpr_fit_predict(Xtr_s, Ytr_s, Xte_s, **kw):
    """Ytr_s / returns are in RAW % units (standardised by caller)."""
    Ztr = ilr(to_parts(Ytr_s))
    k = dict(DEFAULT_MTGPR); k.update(kw)
    m = MTGPR(**k).fit(Xtr_s, Ztr)
    Zte = m.predict(Xte_s)
    return ilr_inv(Zte)[:, :3]


print("=" * 72)
print("sanity: ILR round-trip and closure")
print("=" * 72)
P0 = to_parts(Y_raw)
print("  sum of 3 components: min=%.2f max=%.2f  (other part: %.2f-%.2f)"
      % (Y_raw.sum(1).min(), Y_raw.sum(1).max(), P0[:, 3].min(), P0[:, 3].max()))
rt = ilr_inv(ilr(P0))
print("  ILR round-trip max abs error: %.2e" % np.abs(rt - P0).max())
print("  inverse-ILR closure (row sums, should all be 100): %.4f - %.4f"
      % (rt.sum(1).min(), rt.sum(1).max()))

# ---------------- support = 8, 20 repeats ----------------
print()
print("=" * 72)
print("support = 8, 20 repeats  (all models + RSM + mean + C-MTGPR)")
print("=" * 72)
res8, raw8 = run_splits(8)
per_split = {m: [] for m in MODELS}
for r in range(20):
    for m in MODELS:
        per_split[m].append(float(np.mean([raw8[m]["RMSE"][c][r] for c in range(NC)])))

RSM8, MEAN8, CMT8 = [], [], []
per_comp_cmt = {c: [] for c in range(NC)}
per_comp_neg_viol = []
for r in range(20):
    rng = np.random.RandomState(r)
    perm = rng.permutation(N)
    sup, que = perm[:8], perm[8:]
    sx, sy = Scaler().fit(X_raw[sup]), Scaler().fit(Y_raw[sup])
    Xtr, Xte = sx.tf(X_raw[sup]), sx.tf(X_raw[que])
    Ytr_s = sy.tf(Y_raw[sup])
    # C-MTGPR operates in RAW units -> train on raw, predict raw
    pred = c_mtgpr_fit_predict(Xtr, sy.itf(Ytr_s), Xte, rng=np.random.default_rng(r))
    CMT8.append(float(np.sqrt(((pred - Y_raw[que]) ** 2).mean())))
    for c in range(NC):
        per_comp_cmt[c].append(float(np.sqrt(((pred[:, c] - Y_raw[que, c]) ** 2).mean())))
    per_comp_neg_viol.append(float((pred < 0).sum()))
    # RSM + mean
    pf = PolynomialFeatures(degree=2, include_bias=True)
    Atr = pf.fit_transform(Xtr); Ate = pf.transform(Xte)
    pr = sy.itf(Ridge(alpha=1.0, random_state=0).fit(Atr, Ytr_s).predict(Ate))
    pm = sy.itf(np.tile(Ytr_s.mean(0), (len(que), 1)))
    RSM8.append(float(np.sqrt(((pr - Y_raw[que]) ** 2).mean())))
    MEAN8.append(float(np.sqrt(((pm - Y_raw[que]) ** 2).mean())))
per_split["RSM"] = RSM8
per_split["MeanBaseline"] = MEAN8
per_split["C-MTGPR"] = CMT8

for m in MODELS + ["C-MTGPR", "RSM", "MeanBaseline"]:
    a = np.array(per_split[m])
    print(f"  {m:13s} agg RMSE = {a.mean():.3f} +- {a.std():.3f}")
OUT["support8"] = {m: {"mean": float(np.mean(v)), "sd": float(np.std(v))}
                   for m, v in per_split.items()}
OUT["support8_percomp_CMTGPR"] = {NAMES[c]: {"mean": float(np.mean(per_comp_cmt[c])),
                                             "sd": float(np.std(per_comp_cmt[c]))}
                                  for c in range(NC)}
print("  negative predictions produced by C-MTGPR across 20x23 outputs:", int(sum(per_comp_neg_viol)))
OUT["cmtgpr_neg_predictions"] = int(sum(per_comp_neg_viol))

# ---------------- paired tests + Holm ----------------
print()
print("--- paired t-tests vs C-MTGPR (20 paired splits), Holm-Bonferroni ---")
order_b = ["LR", "Ridge", "RF", "SVR", "MLP", "GPR", "ST-PN", "MTGPR", "RSM"]
cmt = np.array(per_split["C-MTGPR"])
ps, ts = [], []
for b in order_b:
    t, p = stats.ttest_rel(cmt, np.array(per_split[b]))
    ts.append(float(t)); ps.append(float(p))
mm = len(ps); holm = [0.0] * mm; run = 0.0
for k, idx in enumerate(np.argsort(ps)):
    run = max(run, min(1.0, (mm - k) * ps[idx])); holm[idx] = run
for i, b in enumerate(order_b):
    print(f"  C-MTGPR vs {b:7s} t={ts[i]:+.3f} p={ps[i]:.4f} "
          f"| raw:{'YES' if ps[i] < .05 else 'no ':3s} | Holm p_adj={holm[i]:.4f} "
          f"{'YES' if holm[i] < .05 else 'no'}")
OUT["paired_cmtgpr"] = {b: {"t": ts[i], "p": ps[i], "p_holm": holm[i],
                            "sig_raw": bool(ps[i] < .05), "sig_holm": bool(holm[i] < .05)}
                        for i, b in enumerate(order_b)}

# also MTGPR (unconstrained) vs C-MTGPR head-to-head
t, p = stats.ttest_rel(cmt, np.array(per_split["MTGPR"]))
print(f"  [head-to-head] C-MTGPR {cmt.mean():.3f} vs MTGPR {np.mean(per_split['MTGPR']):.3f} "
      f"-> t={t:+.3f} p={p:.4f}")
OUT["cmtgpr_vs_mtgpr"] = {"t": float(t), "p": float(p),
                          "cmtgpr": float(cmt.mean()), "mtgpr": float(np.mean(per_split["MTGPR"]))}

# ---------------- ablation ----------------
print()
print("=" * 72)
print("support-size ablation (incl. C-MTGPR and RSM)")
print("=" * 72)
abl = {}
for ss in [4, 6, 8, 10, 12]:
    rsub, _ = run_splits(ss)
    abl[ss] = {m: float(np.mean([np.mean(rsub[m][c]["RMSE"]) for c in range(NC)])) for m in MODELS}
    cr, rr, mr = [], [], []
    for r in range(20):
        rng = np.random.RandomState(r)
        perm = rng.permutation(N)
        sup, que = perm[:ss], perm[ss:]
        sx, sy = Scaler().fit(X_raw[sup]), Scaler().fit(Y_raw[sup])
        Xtr, Xte = sx.tf(X_raw[sup]), sx.tf(X_raw[que])
        Ytr_s = sy.tf(Y_raw[sup])
        pc = c_mtgpr_fit_predict(Xtr, sy.itf(Ytr_s), Xte, rng=np.random.default_rng(r))
        cr.append(float(np.sqrt(((pc - Y_raw[que]) ** 2).mean())))
        pf = PolynomialFeatures(degree=2, include_bias=True)
        Atr = pf.fit_transform(Xtr); Ate = pf.transform(Xte)
        pr = sy.itf(Ridge(alpha=1.0, random_state=0).fit(Atr, Ytr_s).predict(Ate))
        rr.append(float(np.sqrt(((pr - Y_raw[que]) ** 2).mean())))
        mr.append(float(np.sqrt(((sy.itf(np.tile(Ytr_s.mean(0), (len(que), 1))) - Y_raw[que]) ** 2).mean())))
    abl[ss]["C-MTGPR"] = float(np.mean(cr))
    abl[ss]["RSM"] = float(np.mean(rr))
    abl[ss]["MeanBaseline"] = float(np.mean(mr))
for ss in [4, 6, 8, 10, 12]:
    line = f"  n={ss:2d} " + "  ".join(f"{m}={abl[ss][m]:.2f}" for m in MODELS + ["C-MTGPR", "RSM"])
    print(line)
OUT["ablation"] = {str(k): v for k, v in abl.items()}

# rank of each model at each n (lower=better), among the 10 candidates
print()
cands = MODELS + ["C-MTGPR", "RSM"]
print("  rank of C-MTGPR / MTGPR / RF / RSM by support size:")
for ss in [4, 6, 8, 10, 12]:
    srt = sorted(cands, key=lambda m: abl[ss][m])
    rk = {m: srt.index(m) + 1 for m in cands}
    print(f"    n={ss:2d}: C-MTGPR #{rk['C-MTGPR']}, MTGPR #{rk['MTGPR']}, "
          f"RF #{rk['RF']}, SVR #{rk['SVR']}, RSM #{rk['RSM']}  (of {len(cands)})")
OUT["ranks"] = {str(ss): {m: sorted(cands, key=lambda x: abl[ss][x]).index(m) + 1
                          for m in cands} for ss in [4, 6, 8, 10, 12]}

# ---------------- LOOCV ----------------
print()
print("=" * 72)
print("LOOCV (true RMSE and MAE, incl. C-MTGPR and RSM)")
print("=" * 72)
loocv = {m: {"rmse": [], "mae": []} for m in MODELS + ["C-MTGPR", "RSM"]}
loocv_pred = {m: np.zeros((N, NC)) for m in MODELS + ["C-MTGPR", "RSM"]}
for i in range(N):
    tr = [j for j in range(N) if j != i]; te = [i]
    sx, sy = Scaler().fit(X_raw[tr]), Scaler().fit(Y_raw[tr])
    Xtr, Xte = sx.tf(X_raw[tr]), sx.tf(X_raw[te])
    Ytr_s = sy.tf(Y_raw[tr])
    bl = make_baselines()
    for m in MODELS:
        if m == "ST-PN":
            p = sy.itf(STPN(**{**DEFAULT_STPN, "seed": 7}).fit(Xtr, Ytr_s).predict(Xte))
        elif m == "MTGPR":
            p = sy.itf(MTGPR(**DEFAULT_MTGPR).fit(Xtr, Ytr_s).predict(Xte))
        else:
            p = sy.itf(bl[m].fit(Xtr, Ytr_s).predict(Xte))
        loocv_pred[m][i] = p[0]
        loocv[m]["rmse"].append(float(np.sqrt(((p[0] - Y_raw[te][0]) ** 2).mean())))
        loocv[m]["mae"].append(float(np.mean(np.abs(p[0] - Y_raw[te][0]))))
    pc = c_mtgpr_fit_predict(Xtr, sy.itf(Ytr_s), Xte)
    loocv_pred["C-MTGPR"][i] = pc[0]
    loocv["C-MTGPR"]["rmse"].append(float(np.sqrt(((pc[0] - Y_raw[te][0]) ** 2).mean())))
    loocv["C-MTGPR"]["mae"].append(float(np.mean(np.abs(pc[0] - Y_raw[te][0]))))
    pf = PolynomialFeatures(degree=2, include_bias=True)
    Atr = pf.fit_transform(Xtr); Ate = pf.transform(Xte)
    pr = sy.itf(Ridge(alpha=1.0, random_state=0).fit(Atr, Ytr_s).predict(Ate))
    loocv_pred["RSM"][i] = pr[0]
    loocv["RSM"]["rmse"].append(float(np.sqrt(((pr[0] - Y_raw[te][0]) ** 2).mean())))
    loocv["RSM"]["mae"].append(float(np.mean(np.abs(pr[0] - Y_raw[te][0]))))
loocv_tab = {}
for m in loocv:
    loocv_tab[m] = {"RMSE": float(np.mean(loocv[m]["rmse"])), "MAE": float(np.mean(loocv[m]["mae"]))}
    print(f"  {m:13s} LOOCV RMSE={loocv_tab[m]['RMSE']:.3f}  MAE={loocv_tab[m]['MAE']:.3f}")
OUT["loocv"] = loocv_tab

# per-component LOOCV R2 for C-MTGPR and MTGPR
def r2(y, p):
    return 1.0 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)

print()
print("  per-component LOOCV R2:")
for m in ["MTGPR", "C-MTGPR", "RF"]:
    r2s = [r2(Y_raw[:, c], loocv_pred[m][:, c]) for c in range(NC)]
    print(f"    {m:8s} " + "  ".join(f"{NAMES[c]}={r2s[c]:.3f}" for c in range(NC)))
    OUT.setdefault("loocv_r2_percomp", {})[m] = {NAMES[c]: float(r2s[c]) for c in range(NC)}

# ---------------- downstream ----------------
print()
print("=" * 72)
print("downstream composition -> glucose yield (honest full-fit vs LOOCV)")
print("=" * 72)


def loocv_predict(X, y):
    p = np.zeros(len(y))
    for i in range(len(y)):
        tr = [j for j in range(len(y)) if j != i]
        p[i] = RandomForestRegressor(n_estimators=400, random_state=0).fit(X[tr], y[tr]).predict(X[[i]])[0]
    return p


paths = {"trueComp": Y_raw, "cond": X_raw,
         "predComp_MTGPR": loocv_pred["MTGPR"], "predComp_CMTGPR": loocv_pred["C-MTGPR"]}
for name, Xp in paths.items():
    Xs = Scaler().fit(Xp).tf(Xp)
    ff = RandomForestRegressor(n_estimators=400, random_state=0).fit(Xs, Yield_raw).predict(Xs)
    loo = loocv_predict(Xs, Yield_raw)
    OUT.setdefault("downstream", {})[name] = {
        "r2_fullfit": float(r2(Yield_raw, ff)), "r2_loocv": float(r2(Yield_raw, loo)),
        "rmse_loocv": float(np.sqrt(np.mean((Yield_raw - loo) ** 2)))}
    print(f"  {name:17s} full-fit R2={r2(Yield_raw, ff):.3f}  LOOCV R2={r2(Yield_raw, loo):.3f}  "
          f"LOOCV RMSE={np.sqrt(np.mean((Yield_raw - loo) ** 2)):.3f}")

with open(os.path.join(BASE, "process_data", "round2b.json"), "w") as f:
    json.dump(OUT, f, indent=2)
print()
print("Saved round2b.json")
