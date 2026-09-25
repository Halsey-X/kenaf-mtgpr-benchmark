"""Generate Fig 1-7 for the 1->2 composition-prediction study (MTGPR-centered),
from process_data/results.json. Each figure is saved in PNG, EPS and SVG into
figures_data/{png,eps,svg}.

Layout: every figure uses constrained_layout to prevent text overlap.
fig6: short y-axis labels to avoid collisions with the colorbar of panel (a).
fig7/fig1: "Tempreture" typo from the raw sheet header is corrected to
"Temperature (°C)".
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
OUT = ROOT
PROCESS = os.path.join(OUT, "process_data")
FIGDATA = os.path.join(OUT, "figures_data")
PNG = os.path.join(FIGDATA, "png")
EPS = os.path.join(FIGDATA, "eps")
SVG = os.path.join(FIGDATA, "svg")
for d in (PNG, EPS, SVG):
    os.makedirs(d, exist_ok=True)

R = json.load(open(os.path.join(PROCESS, "results.json")))
MODELS = ["LR", "Ridge", "RF", "SVR", "MLP", "GPR", "MTGPR", "ST-PN"]
targ = R["targ_names"]; NC = len(targ)
raw_rmse = R["raw_rmse"]; raw_mae = R["raw_mae"]
routes = np.array(R["routes"]); Y = np.array(R["Y_raw"]); X = np.array(R["X_raw"])
route_order = R["route_order"]

plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                     "figure.dpi": 150, "savefig.bbox": "tight", "axes.grid": True,
                     "grid.alpha": 0.3})
COL = {"LR": "#888888", "Ridge": "#4C72B0", "RF": "#55A868", "SVR": "#C44E52",
       "MLP": "#8172B3", "GPR": "#CCB974", "MTGPR": "#E07B39", "ST-PN": "#000000"}

def fix_feat(name):
    """Correct typos coming from the raw sheet header."""
    return name.replace("Tempreture", "Temperature").replace("℃", "°C")

def savefig(fig, name):
    fig.savefig(os.path.join(PNG, name + ".png"), dpi=150)
    fig.savefig(os.path.join(EPS, name + ".eps"))
    fig.savefig(os.path.join(SVG, name + ".svg"))
    plt.close(fig)

COMP_SHORT = ["Cellulose", "Hemicell.", "Lignin"]

# ---------- Fig 1: condition-composition map ----------
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), layout="constrained")
rmarkers = {"Autohydrolysis": "o", "Dilute-acid": "s", "Alkaline": "^"}
for ci, ax in enumerate(axes):
    cvals = Y[:, ci]
    norm = Normalize(cvals.min(), cvals.max())
    for rt in route_order:
        m = routes == rt
        sc = ax.scatter(X[m, 0], X[m, 1], c=cvals[m], cmap="viridis",
                        norm=norm, marker=rmarkers[rt], s=46, alpha=0.88,
                        edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Residence time (min)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title(targ[ci])
fig.colorbar(sc, ax=axes, fraction=0.02, label="Composition (%)")
fig.suptitle("Route-associated pretreatment-condition space and chemical-composition response of kenaf core")
savefig(fig, "fig1_condition_composition")

# ---------- Fig 2: repeated-split error distributions ----------
def agg_per_repeat(raw):
    return {m: np.array(raw[m]).mean(0) for m in MODELS}  # 20 values aggregated over components
rmse_rep = agg_per_repeat(raw_rmse); mae_rep = agg_per_repeat(raw_mae)
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), layout="constrained")
data = [rmse_rep[m] for m in MODELS]
bp = axes[0].boxplot(data, tick_labels=MODELS, patch_artist=True)
for patch, m in zip(bp["boxes"], MODELS): patch.set_facecolor(COL[m]); patch.set_alpha(0.6)
axes[0].set_title("(a) RMSE distribution"); axes[0].set_ylabel("RMSE (%)")
axes[0].tick_params(axis="x", labelsize=8.5)
data = [mae_rep[m] for m in MODELS]
bp = axes[1].boxplot(data, tick_labels=MODELS, patch_artist=True)
for patch, m in zip(bp["boxes"], MODELS): patch.set_facecolor(COL[m]); patch.set_alpha(0.6)
axes[1].set_title("(b) MAE distribution"); axes[1].set_ylabel("MAE (%)")
axes[1].tick_params(axis="x", labelsize=8.5)
cv = [np.std(rmse_rep[m]) / np.mean(rmse_rep[m]) * 100 for m in MODELS]
bars = axes[2].bar(MODELS, cv, color=[COL[m] for m in MODELS], alpha=0.7)
axes[2].set_title("(c) RMSE coefficient of variation"); axes[2].set_ylabel("CV (%)")
for b, v in zip(bars, cv): axes[2].text(b.get_x()+b.get_width()/2, v+0.1, f"{v:.1f}", ha="center", fontsize=8)
axes[2].tick_params(axis="x", labelsize=8.5)
fig.suptitle("Repeated-split prediction error and stability (20 support-query splits, support=8)")
savefig(fig, "fig2_repeat_stability")

# ---------- Fig 3: support-size ablation ----------
abl = R["ablation"]
ssizes = sorted(int(k) for k in abl.keys())
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharex=True, layout="constrained")
for ci, ax in enumerate(axes):
    for m in MODELS:
        ys = [abl[str(s)][m][targ[ci]] for s in ssizes]
        ax.plot(ssizes, ys, marker="o", ms=4, color=COL[m], label=m)
    ax.set_title(targ[ci]); ax.set_xlabel("Support-set size")
    ax.set_ylabel("Mean RMSE (%)")
axes[1].legend(fontsize=7.5, ncol=2, loc="upper right", framealpha=0.9)
fig.suptitle("Effect of support-set size on RMSE across regression models")
savefig(fig, "fig3_ablation")

# ---------- Fig 4: MTGPR hyperparameter sensitivity (ard x rank) ----------
hp = R["hp_sensitivity"]
ranks = sorted({int(r["rank"]) for r in hp[targ[0]]})
ards = sorted({bool(r["ard"]) for r in hp[targ[0]]})
fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), layout="constrained")
for ci, ax in enumerate(axes):
    grid = np.zeros((len(ranks), len(ards)))
    for i, rk in enumerate(ranks):
        for j, ar in enumerate(ards):
            v = [r[targ[ci]] for r in hp[targ[ci]] if int(r["rank"]) == rk and bool(r["ard"]) == ar][0]
            grid[i, j] = v
    im = ax.imshow(grid, aspect="auto", cmap="YlGnBu_r", origin="lower")
    ax.set_xticks(range(len(ards))); ax.set_xticklabels(["shared ls" if not a else "ARD" for a in ards])
    ax.set_yticks(range(len(ranks))); ax.set_yticklabels([str(rk) for rk in ranks])
    ax.set_xlabel("Lengthscale"); ax.set_ylabel("Coregionalization rank")
    ax.set_title(f"{targ[ci]} — RMSE (%)")
    thr = grid.min() + 0.6 * (grid.max() - grid.min())
    for i in range(len(ranks)):
        for j in range(len(ards)):
            ax.text(j, i, f"{grid[i,j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if grid[i, j] > thr else "black")
fig.colorbar(im, ax=axes, fraction=0.02, label="RMSE (%)")
fig.suptitle("MTGPR hyperparameter sensitivity (support=8, 20 repeats): lengthscale mode × coregionalization rank")
savefig(fig, "fig4_mtgpr_hp")

# ---------- Fig 5: LOOCV error distribution ----------
loocv_err = R["loocv_err"]
fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
data = [np.array(loocv_err[m][targ[0]] + loocv_err[m][targ[1]] + loocv_err[m][targ[2]]) for m in MODELS]
bp = ax.boxplot(data, tick_labels=MODELS, patch_artist=True)
for patch, m in zip(bp["boxes"], MODELS): patch.set_facecolor(COL[m]); patch.set_alpha(0.6)
ax.tick_params(axis="x", labelsize=9)
ax.set_ylabel("Absolute prediction error (%)")
ax.set_title("LOOCV absolute error distribution (all three composition components)")
savefig(fig, "fig5_loocv")

# ---------- Fig 6: MTGPR coregionalization (cross-task correlation) ----------
corr = np.array(R["mtgpr_corr"]); B = np.array(R["mtgpr_coreg"])
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
im0 = axes[0].imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
axes[0].set_xticks(range(NC)); axes[0].set_xticklabels(COMP_SHORT, fontsize=9)
axes[0].set_yticks(range(NC)); axes[0].set_yticklabels(COMP_SHORT, fontsize=9)
axes[0].set_title("(a) Cross-task correlation (B normalized)")
for i in range(NC):
    for j in range(NC):
        axes[0].text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center", fontsize=9,
                     color="white" if abs(corr[i,j]) > 0.5 else "black")
fig.colorbar(im0, ax=axes[0], fraction=0.04)
im1 = axes[1].imshow(B, cmap="viridis")
axes[1].set_xticks(range(NC)); axes[1].set_xticklabels(COMP_SHORT, fontsize=9)
axes[1].set_yticks(range(NC)); axes[1].set_yticklabels(COMP_SHORT, fontsize=9)
axes[1].set_title("(b) Coregionalization matrix B = A A$^T$")
for i in range(NC):
    for j in range(NC):
        # viridis: high values are light (yellow) -> black text; low values dark -> white text
        axes[1].text(j, i, f"{B[i,j]:.2f}", ha="center", va="center", fontsize=9,
                     color="black" if B[i, j] > 0.5 * B.max() else "white")
fig.colorbar(im1, ax=axes[1], fraction=0.04)
fig.suptitle("MTGPR intrinsic coregionalization learned on the full 31-sample dataset")
savefig(fig, "fig6_coregionalization")

# ---------- Fig 7: RF descriptor importance ----------
rfi = R["rf_importance"]
feat_labels = [fix_feat(n) for n in R["feat_names"]]
fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
x = np.arange(NC); w = 0.25
for ci, c in enumerate(targ):
    vals = rfi[c]
    ax.bar(x + (ci-1)*w, vals, w, label=c)
ax.set_xticks(x); ax.set_xticklabels(feat_labels)
ax.set_ylabel("RF impurity importance")
ax.set_title("Descriptor importance estimated using random forest regression (per composition component)")
ax.legend(fontsize=8)
savefig(fig, "fig7_importance")

print("Figures generated in", FIGDATA)
print("PNG:", sorted(os.listdir(PNG)))
print("EPS:", sorted(os.listdir(EPS)))
print("SVG:", sorted(os.listdir(SVG)))
