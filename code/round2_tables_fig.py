"""Fill in per-component support=8 numbers for RSM / C-MTGPR / mean baseline,
and draw Fig. 11 exposing full-fit R2 vs LOOCV R2 for the yield chain."""
import os, sys, json, math, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


NAMES = ["Cell", "Hemi", "Lign"]
acc = {m: {c: {"RMSE": [], "MAE": []} for c in range(NC)} for m in ["RSM", "C-MTGPR", "MeanBaseline"]}
agg = {m: [] for m in acc}

for r in range(20):
    rng = np.random.RandomState(r)
    perm = rng.permutation(N)
    sup, que = perm[:8], perm[8:]
    sx, sy = Scaler().fit(X_raw[sup]), Scaler().fit(Y_raw[sup])
    Xtr, Xte = sx.tf(X_raw[sup]), sx.tf(X_raw[que])
    Ytr_s = sy.tf(Y_raw[sup])
    preds = {}
    pf = PolynomialFeatures(degree=2, include_bias=True)
    preds["RSM"] = sy.itf(Ridge(alpha=1.0, random_state=0).fit(pf.fit_transform(Xtr), Ytr_s)
                          .predict(pf.transform(Xte)))
    preds["MeanBaseline"] = sy.itf(np.tile(Ytr_s.mean(0), (len(que), 1)))
    preds["C-MTGPR"] = ilr_inv(MTGPR(**{**DEFAULT_MTGPR, "rng": np.random.default_rng(r)})
                               .fit(Xtr, ilr(to_parts(sy.itf(Ytr_s)))).predict(Xte))[:, :3]
    for m, p in preds.items():
        agg[m].append(float(np.sqrt(((p - Y_raw[que]) ** 2).mean())))
        for c in range(NC):
            acc[m][c]["RMSE"].append(float(np.sqrt(((p[:, c] - Y_raw[que, c]) ** 2).mean())))
            acc[m][c]["MAE"].append(float(np.mean(np.abs(p[:, c] - Y_raw[que, c]))))

out = {}
print("support = 8, 20 repeats -- per-component RMSE / MAE (mean +- sd)")
for m in ["RSM", "C-MTGPR", "MeanBaseline"]:
    row = {}
    for c in range(NC):
        rm = np.array(acc[m][c]["RMSE"]); ma = np.array(acc[m][c]["MAE"])
        row[NAMES[c]] = {"RMSE": [float(rm.mean()), float(rm.std())],
                         "MAE": [float(ma.mean()), float(ma.std())]}
    row["agg_RMSE"] = [float(np.mean(agg[m])), float(np.std(agg[m]))]
    out[m] = row
    print(f"  {m:13s} " + "  ".join(
        f"{NAMES[c]}: {row[NAMES[c]]['RMSE'][0]:.2f}+-{row[NAMES[c]]['RMSE'][1]:.2f} "
        f"({row[NAMES[c]]['MAE'][0]:.2f}+-{row[NAMES[c]]['MAE'][1]:.2f})" for c in range(NC)))
    print(f"                agg RMSE = {row['agg_RMSE'][0]:.3f} +- {row['agg_RMSE'][1]:.3f}")

with open(os.path.join(BASE, "process_data", "round2c.json"), "w") as f:
    json.dump(out, f, indent=2)

# ---------------- Fig 11 ----------------
d = json.load(open(os.path.join(BASE, "process_data", "round2b.json")))
ds = d["downstream"]
labels = [("Measured composition", "trueComp"),
          ("Pretreatment conditions", "cond"),
          ("MTGPR-predicted composition", "predComp_MTGPR")]
ff = [ds[k]["r2_fullfit"] for _, k in labels]
lo = [ds[k]["r2_loocv"] for _, k in labels]

FIG = os.path.join(BASE, "figures_data", "png")
os.makedirs(FIG, exist_ok=True)
x = np.arange(len(labels)); w = 0.36
fig, ax = plt.subplots(figsize=(7.4, 4.4), dpi=300)
b1 = ax.bar(x - w / 2, ff, w, label="Full-fit (in-sample) $R^2$", color="#B0BEC5",
            edgecolor="#37474F", linewidth=0.8, zorder=3)
b2 = ax.bar(x + w / 2, lo, w, label="Leave-one-out $R^2$", color="#1565C0",
            edgecolor="#0D47A1", linewidth=0.8, zorder=3)
ax.axhline(0, color="#37474F", lw=0.9, zorder=4)
for b in list(b1) + list(b2):
    h = b.get_height()
    ax.annotate(f"{h:.2f}", (b.get_x() + b.get_width() / 2, h),
                ha="center", va="bottom" if h >= 0 else "top",
                fontsize=8.5, color="#263238",
                xytext=(0, 3 if h >= 0 else -3), textcoords="offset points")
ax.set_xticks(x)
ax.set_xticklabels([l for l, _ in labels], fontsize=9)
ax.set_ylabel("$R^2$ of predicted glucose yield", fontsize=10)
ax.set_ylim(-0.35, 1.30)
ax.legend(frameon=False, fontsize=9, loc="upper center", ncol=2,
          bbox_to_anchor=(0.5, 1.02), columnspacing=1.4)
ax.grid(axis="y", ls=":", lw=0.6, color="#90A4AE", zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_title("In-sample $R^2$ overstates the composition\u2013yield link", fontsize=10.5, pad=8)
fig.tight_layout()
for ext in ("png", "pdf", "svg"):
    fig.savefig(os.path.join(FIG, f"fig11_downstream_r2.{ext}"), bbox_inches="tight")
print("\nsaved fig11_downstream_r2.png/pdf/svg")
