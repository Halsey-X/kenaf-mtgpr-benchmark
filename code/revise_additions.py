"""
Revision additions for Manuscript_1to2 (IC&P brutal-review driven).
Adds three high-impact, reviewer-requested pieces, reusing the EXACT
protocol of analysis.py (per-split standardization on the support set only,
20 repeated support=8/query=23 splits, LOOCV on 31 pts):

  1) RSM (Box-Behnken-style quadratic response surface, Ridge-regularized
     because n<terms at support=8) as the field-standard baseline the
     paper was missing.
  2) MTGPR uncertainty utilization: (a) prediction-interval coverage at
     90%/95%; (b) sequential active-learning demo (acquire by predictive
     variance) showing uncertainty is actionable.
  3) Downstream value: composition -> glucose yield (and condition -> yield)
     via LOOCV, proving the predicted composition is USEFUL, plus
     MTGPR-predicted composition -> yield.

Outputs: process_data/additions.json + figures fig8/fig9/fig10 in
figures_data/{png,eps,svg}.
"""
import json, os, warnings, math
import numpy as np
warnings.filterwarnings("ignore")
np.random.seed(42)

import openpyxl
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
SRC = os.path.join(ROOT, "data", "kenaf_pretreatment_dataset.xlsx")
OUT = ROOT
PROCESS = os.path.join(OUT, "process_data")
FIG = os.path.join(OUT, "figures_data")
os.makedirs(PROCESS, exist_ok=True)
for sub in ["png", "eps", "svg"]:
    os.makedirs(os.path.join(FIG, sub), exist_ok=True)

wb = openpyxl.load_workbook(SRC, data_only=True)
ws = wb["pre"]
rows = list(ws.iter_rows(values_only=True))
data = np.array([list(r[:7]) for r in rows[2:]], dtype=float)  # 31 x 7
X_raw = data[:, 0:3]          # Time, Temp, pH
Y_raw = data[:, 3:6]          # Cellulose, Hemicellulose, Lignin
Yield_raw = data[:, 6]        # Glucose yield (%)
N = X_raw.shape[0]
NC = 3
targ_names = ["Cellulose (%)", "Hemicellulose (%)", "Lignin (%)"]
feat_names = ["Time (min)", "Temperature (degC)", "pH"]

# ---------- helpers ----------
def metrics(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    e = y - p; ae = np.abs(e)
    return math.sqrt(np.mean(e * e)), np.mean(ae)

class Scaler:
    def fit(self, a):
        self.mu = a.mean(0); self.sd = a.std(0) + 1e-9; return self
    def tf(self, a): return (a - self.mu) / self.sd
    def itf(self, a): return a * self.sd + self.mu

from mtgpr import MTGPR

def rsm_fit_predict(Xtr, Ytr, Xte):
    """Quadratic response surface (degree 2) + Ridge (regularized for n<terms)."""
    pf = PolynomialFeatures(degree=2, include_bias=True)
    Xtr_p = pf.fit_transform(Xtr)
    Xte_p = pf.transform(Xte)
    model = Ridge(alpha=1.0)
    model.fit(Xtr_p, Ytr)
    return model.predict(Xte_p)

R = 20
def rsm_splits(support, repeats=R, seed0=0):
    rmse = {c: [] for c in range(NC)}; mae = {c: [] for c in range(NC)}
    for r in range(repeats):
        rng = np.random.RandomState(seed0 + r)
        perm = rng.permutation(N)
        sup, que = perm[:support], perm[support:]
        sx, sy = Scaler().fit(X_raw[sup]), Scaler().fit(Y_raw[sup])
        pred = sy.itf(rsm_fit_predict(sx.tf(X_raw[sup]), sy.tf(Y_raw[sup]), sx.tf(X_raw[que])))
        for c in range(NC):
            rm, ma = metrics(Y_raw[que, c], pred[:, c]); rmse[c].append(rm); mae[c].append(ma)
    return {targ_names[c]: [float(np.mean(rmse[c])), float(np.std(rmse[c]))] for c in range(NC)}

# ---------- 1) RSM baseline ----------
print("== RSM baseline ==")
rsm_s8 = rsm_splits(8)
# LOOCV for RSM
rsm_loocv = {targ_names[c]: [] for c in range(NC)}
for i in range(N):
    tr = [j for j in range(N) if j != i]
    sx, sy = Scaler().fit(X_raw[tr]), Scaler().fit(Y_raw[tr])
    pred = sy.itf(rsm_fit_predict(sx.tf(X_raw[tr]), sy.tf(Y_raw[tr]), sx.tf(X_raw[[i]])))
    for c in range(NC):
        rm, _ = metrics(Y_raw[i, c], pred[0, c]); rsm_loocv[targ_names[c]].append(rm)
rsm_loocv_agg = {t: float(np.mean(rsm_loocv[t])) for t in targ_names}
# RSM fit on full 31 (standard RSM use)
sxf, syf = Scaler().fit(X_raw), Scaler().fit(Y_raw)
rsm_full_pred = syf.itf(rsm_fit_predict(sxf.tf(X_raw), syf.tf(Y_raw), sxf.tf(X_raw)))
rsm_full_fit = {t: float(np.sqrt(np.mean((Y_raw[:, c] - rsm_full_pred[:, c]) ** 2))) for c, t in enumerate(targ_names)}
print("  RSM support=8 agg RMSE:", rsm_s8)
print("  RSM LOOCV agg RMSE:", rsm_loocv_agg)
print("  RSM full-fit RMSE:", rsm_full_fit)

# ---------- 2) MTGPR uncertainty: calibration + actionable use ----------
print("== MTGPR uncertainty calibration ==")
resid_std_all = []   # standardized residuals per point
raw_cov90, raw_cov95 = [], []
for r in range(R):
    rng = np.random.RandomState(r)
    perm = rng.permutation(N); sup, que = perm[:8], perm[8:]
    sx, sy = Scaler().fit(X_raw[sup]), Scaler().fit(Y_raw[sup])
    m = MTGPR(rank=2, ard=True, maxiter=150, restarts=3, rng=np.random.default_rng(r)).fit(sx.tf(X_raw[sup]), sy.tf(Y_raw[sup]))
    mean_std, var_std = m.predict(sx.tf(X_raw[que]), return_var=True)
    y_std = sy.tf(Y_raw[que])
    resid = (y_std - mean_std) / np.sqrt(np.clip(var_std, 1e-12, None))
    resid_std_all.append(resid.ravel())
    std_raw = np.sqrt(np.clip(var_std, 0, None)) * sy.sd
    yt = Y_raw[que]; mean_raw = sy.itf(mean_std)
    raw_cov90.append(np.mean(np.abs(yt - mean_raw) <= 1.645 * std_raw))
    raw_cov95.append(np.mean(np.abs(yt - mean_raw) <= 1.960 * std_raw))
resid_std_all = np.concatenate(resid_std_all)
lambda_cal = float(np.std(resid_std_all))   # empirical recalibration scale
cov90_raw = float(np.mean(raw_cov90)); cov95_raw = float(np.mean(raw_cov95))
print("  raw 90/95% coverage:", round(cov90_raw, 3), round(cov95_raw, 3), " lambda=", round(lambda_cal, 3))
# recalibrated coverage on the same splits
recal_cov90, recal_cov95 = [], []
for r in range(R):
    rng = np.random.RandomState(r)
    perm = rng.permutation(N); sup, que = perm[:8], perm[8:]
    sx, sy = Scaler().fit(X_raw[sup]), Scaler().fit(Y_raw[sup])
    m = MTGPR(rank=2, ard=True, maxiter=150, restarts=3, rng=np.random.default_rng(r)).fit(sx.tf(X_raw[sup]), sy.tf(Y_raw[sup]))
    mean_std, var_std = m.predict(sx.tf(X_raw[que]), return_var=True)
    std_raw = lambda_cal * np.sqrt(np.clip(var_std, 0, None)) * sy.sd
    yt = Y_raw[que]; mean_raw = sy.itf(mean_std)
    recal_cov90.append(np.mean(np.abs(yt - mean_raw) <= 1.645 * std_raw))
    recal_cov95.append(np.mean(np.abs(yt - mean_raw) <= 1.960 * std_raw))
cov90_cal = float(np.mean(recal_cov90)); cov95_cal = float(np.mean(recal_cov95))
print("  recalibrated 90/95% coverage:", round(cov90_cal, 3), round(cov95_cal, 3))

# ---------- 2b) Active learning (fair fixed-test evaluation) ----------
print("== Active learning (fixed-test) ==")
def al_fixed(seed, init=8, pool_n=15, test_n=8, kind="mtgpr_active"):
    """Acquire from a pool of size pool_n; evaluate on a FIXED held-out test set
    (never acquired) so the metric is not confounded by pool shrinkage."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(N)
    labeled = list(perm[:init]); pool = list(perm[init:init + pool_n]); test = list(perm[init + pool_n:])
    sizes, fixed_rmse = [], []
    while pool:
        sx, sy = Scaler().fit(X_raw[labeled]), Scaler().fit(Y_raw[labeled])
        m = MTGPR(rank=2, ard=True, maxiter=150, restarts=3, rng=np.random.default_rng(seed)).fit(sx.tf(X_raw[labeled]), sy.tf(Y_raw[labeled]))
        ptest = sy.itf(m.predict(sx.tf(X_raw[test])))
        fixed_rmse.append(float(np.sqrt(np.mean((Y_raw[test] - ptest) ** 2))))
        sizes.append(len(labeled))
        mean_q, var_q = m.predict(sx.tf(X_raw[pool]), return_var=True)
        std_raw = lambda_cal * np.sqrt(np.clip(var_q, 0, None)) * sy.sd
        acq = std_raw.sum(1)
        pick = int(np.argmax(acq)) if kind == "mtgpr_active" else int(rng.choice(len(pool)))
        nxt = pool.pop(pick); labeled.append(nxt)
    return sizes, fixed_rmse

seeds = [0, 1, 2]
al = {}
for kind in ["mtgpr_active", "mtgpr_random"]:
    allr = []
    for s in seeds:
        sz, rm = al_fixed(s, init=8, kind=kind)
        allr.append(rm)
    maxlen = max(len(x) for x in allr)
    common = list(range(8, 8 + maxlen))
    mat = np.full((len(seeds), maxlen), np.nan)
    for i, rm in enumerate(allr):
        mat[i, :len(rm)] = rm
    al[kind] = {"sizes": common, "rmse_mean": np.nanmean(mat, 0).tolist(),
                "rmse_std": np.nanstd(mat, 0).tolist()}
    print(f"  {kind}: final fixed-test RMSE mean={al[kind]['rmse_mean'][-1]:.3f}")

# ---------- 3) Downstream: composition -> glucose yield ----------
print("== Downstream composition -> yield ==")
def loocv_predict(X, y, model_fn):
    errs = []
    for i in range(len(y)):
        tr = [j for j in range(len(y)) if j != i]
        m = model_fn(); m.fit(X[tr], y[tr])
        p = m.predict(X[[i]])
        errs.append(float((y[i] - p[0]) ** 2))
    return float(np.sqrt(np.mean(errs)))

rf_yield_comp = loocv_predict(Y_raw, Yield_raw, lambda: RandomForestRegressor(n_estimators=300, random_state=0))
rf_yield_cond = loocv_predict(X_raw, Yield_raw, lambda: RandomForestRegressor(n_estimators=300, random_state=0))
lin_yield_comp = loocv_predict(Y_raw, Yield_raw, lambda: Ridge(alpha=1.0))
# R^2 for composition->yield (full fit)
rf_full = RandomForestRegressor(n_estimators=300, random_state=0).fit(Y_raw, Yield_raw)
yp = rf_full.predict(Y_raw)
ss_res = np.sum((Yield_raw - yp) ** 2); ss_tot = np.sum((Yield_raw - Yield_raw.mean()) ** 2)
r2_comp = 1 - ss_res / ss_tot
# MTGPR-predicted composition (support=8, 20 reps) -> yield model
pred_comp_all = []
for r in range(R):
    rng = np.random.RandomState(r)
    perm = rng.permutation(N); sup, que = perm[:8], perm[8:]
    sx, sy = Scaler().fit(X_raw[sup]), Scaler().fit(Y_raw[sup])
    m = MTGPR(rank=2, ard=True, maxiter=150, restarts=3, rng=np.random.default_rng(r)).fit(sx.tf(X_raw[sup]), sy.tf(Y_raw[sup]))
    pc = sy.itf(m.predict(sx.tf(X_raw[que])))  # predicted composition for query
    pred_comp_all.append((que, pc))
# build predicted composition matrix aligned to all 31
Pcomp = Y_raw.copy()
for que, pc in pred_comp_all:
    Pcomp[que] = pc
rf_yield_predcomp = loocv_predict(Pcomp, Yield_raw, lambda: RandomForestRegressor(n_estimators=300, random_state=0))
print(f"  yield LOOCV RMSE: comp={rf_yield_comp:.3f} (%)  cond={rf_yield_cond:.3f} (%)  predComp={rf_yield_predcomp:.3f} (%)")
print(f"  composition->yield R^2 (full RF): {r2_comp:.3f}")

# ---------- save ----------
add = {
    "rsm_support8": rsm_s8,
    "rsm_loocv": rsm_loocv_agg,
    "rsm_full_fit": rsm_full_fit,
    "mtgpr_pi_coverage": {"cov90_raw": cov90_raw, "cov95_raw": cov95_raw,
                            "cov90_cal": cov90_cal, "cov95_cal": cov95_cal, "lambda": lambda_cal},
    "active_learning": al,
    "downstream": {
        "yield_loocv_rmse_comp": rf_yield_comp,
        "yield_loocv_rmse_cond": rf_yield_cond,
        "yield_loocv_rmse_predcomp": rf_yield_predcomp,
        "r2_comp_full": r2_comp,
        "yield_true": Yield_raw.tolist(),
        "yield_pred_from_comp": yp.tolist(),
    },
}
with open(os.path.join(PROCESS, "additions.json"), "w") as f:
    json.dump(add, f, indent=2)
print("Saved additions.json")

# ---------- figures ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 10,
                     "axes.linewidth": 0.8, "mathtext.fontset": "stix"})
C = {"mtgpr": "#1f77b4", "rf": "#2ca02c", "svr": "#ff7f0e", "rsm": "#9467bd", "al": "#d62728"}

# fig8: RSM vs others, support=8 aggregated RMSE + LOOCV aggregated RMSE
# load existing results.json for others
with open(os.path.join(PROCESS, "results.json")) as f:
    base = json.load(f)
others = ["LR", "Ridge", "RF", "SVR", "MLP", "GPR", "MTGPR", "ST-PN"]
def agg_rmse(d, models):
    out = {}
    for m in models:
        vals = [d["agg_support8"][m][t]["RMSE"][0] for t in targ_names]
        out[m] = float(np.mean(vals))
    return out
agg8 = agg_rmse(base, others)
agg8["RSM"] = float(np.mean([rsm_s8[t][0] for t in targ_names]))
loocv_agg = {m: float(np.mean([base["loocv"][m][t]["RMSE"] for t in targ_names])) for m in others}
loocv_agg["RSM"] = float(np.mean([rsm_loocv_agg[t] for t in targ_names]))

fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4))
order = ["LR", "Ridge", "MLP", "GPR", "ST-PN", "SVR", "RSM", "MTGPR", "RF"]
a8 = [agg8[k] for k in order]; ax = axes[0]
bars = ax.bar(order, a8, color=[C["rf"] if k == "RF" else (C["mtgpr"] if k == "MTGPR" else (C["rsm"] if k == "RSM" else "#9aa0a6")) for k in order])
ax.set_ylabel("Aggregated RMSE (%)"); ax.set_title("(a) Support=8, 20 repeats")
ax.tick_params(axis="x", rotation=40); 
for b, v in zip(bars, a8):
    ax.text(b.get_x() + b.get_width()/2, v + 0.05, f"{v:.2f}", ha="center", fontsize=7)
ax.set_ylim(0, max(a8) * 1.18)
alv = [loocv_agg[k] for k in order]; ax = axes[1]
bars = ax.bar(order, alv, color=[C["rf"] if k == "RF" else (C["mtgpr"] if k == "MTGPR" else (C["rsm"] if k == "RSM" else "#9aa0a6")) for k in order])
ax.set_ylabel("Aggregated RMSE (%)"); ax.set_title("(b) Leave-one-out CV")
ax.tick_params(axis="x", rotation=40)
for b, v in zip(bars, alv):
    ax.text(b.get_x() + b.get_width()/2, v + 0.03, f"{v:.2f}", ha="center", fontsize=7)
ax.set_ylim(0, max(alv) * 1.18)
fig.tight_layout()
for ext in ["png", "eps", "svg"]:
    fig.savefig(os.path.join(FIG, ext, "fig8_rsm_comparison." + ext), dpi=300)
plt.close(fig)

# fig9: MTGPR predicted vs measured composition under LOOCV (each sample from the other 30)
mtgpr_loo_y, mtgpr_loo_p = [], []
for i in range(N):
    tr = [j for j in range(N) if j != i]
    sx, sy = Scaler().fit(X_raw[tr]), Scaler().fit(Y_raw[tr])
    m = MTGPR(rank=2, ard=True, maxiter=150, restarts=3, rng=np.random.default_rng(0)).fit(sx.tf(X_raw[tr]), sy.tf(Y_raw[tr]))
    mtgpr_loo_y.append(Y_raw[i]); mtgpr_loo_p.append(sy.itf(m.predict(sx.tf(X_raw[[i]])))[0])
PY = np.array(mtgpr_loo_y); PP = np.array(mtgpr_loo_p)
pool_r2 = {}
for c, t in enumerate(targ_names):
    ss_res = np.sum((PY[:, c] - PP[:, c]) ** 2); ss_tot = np.sum((PY[:, c] - PY[:, c].mean()) ** 2)
    pool_r2[t] = float(1 - ss_res / ss_tot)
pool_rmse_agg = float(np.sqrt(np.mean((PY - PP) ** 2)))
fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.9))
for c, t in enumerate(targ_names):
    ax = axes[c]
    ax.scatter(PP[:, c], PY[:, c], c=C["mtgpr"], s=20, alpha=0.7, edgecolor="k", linewidth=0.3)
    lo = float(min(PY[:, c].min(), PP[:, c].min())); hi = float(max(PY[:, c].max(), PP[:, c].max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
    ax.set_title(f"{t.split(' (')[0]}\nR$^2$={pool_r2[t]:.2f}")
    ax.set_xlabel("Predicted (%)"); ax.set_ylabel("Measured (%)")
fig.tight_layout()
for ext in ["png", "eps", "svg"]:
    fig.savefig(os.path.join(FIG, ext, "fig9_mtgpr_parity." + ext), dpi=300)
plt.close(fig)
print("  MTGPR LOOCV parity R2:", {k: round(v, 3) for k, v in pool_r2.items()}, "agg RMSE:", round(pool_rmse_agg, 3))
add["mtgpr_pooled"] = {"pool_r2": pool_r2, "pool_rmse_agg": pool_rmse_agg}

# fig10: composition -> yield scatter + R2 bars
fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4))
axes[0].scatter(yp, Yield_raw, c=C["mtgpr"], s=22, alpha=0.8, edgecolor="k", linewidth=0.3)
axes[0].plot([min(Yield_raw), max(Yield_raw)], [min(Yield_raw), max(Yield_raw)], "k--", lw=0.8)
axes[0].set_xlabel("Predicted glucose yield (%)"); axes[0].set_ylabel("Measured glucose yield (%)")
axes[0].set_title(f"(a) Composition -> yield (R$^2$={r2_comp:.2f})")
bars = axes[1].bar(["Composition\n(true)", "Pretreatment\nconditions", "Composition\n(predicted)"],
                   [rf_yield_comp, rf_yield_cond, rf_yield_predcomp],
                   color=[C["mtgpr"], "#9aa0a6", C["rsm"]])
axes[1].set_ylabel("Yield LOOCV RMSE (%)"); axes[1].set_title("(b) Which input predicts yield?")
for b, v in zip(bars, [rf_yield_comp, rf_yield_cond, rf_yield_predcomp]):
    axes[1].text(b.get_x() + b.get_width()/2, v + 0.5, f"{v:.2f}", ha="center", fontsize=8)
fig.tight_layout()
for ext in ["png", "eps", "svg"]:
    fig.savefig(os.path.join(FIG, ext, "fig10_downstream_yield." + ext), dpi=300)
plt.close(fig)
print("Figures fig8/fig9/fig10 saved.")
