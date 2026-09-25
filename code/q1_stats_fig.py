# -*- coding: utf-8 -*-
"""Q1 Paired significance for the newly added baselines + figures 12/13."""
import json, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
PROCESS = os.path.join(ROOT, "process_data")
FIGP = os.path.join(ROOT, "figures_data", "png")
R = json.load(open(os.path.join(PROCESS, "q1_results.json"), encoding="utf-8"))
ps = R["support8_per_split"]
MODELS = list(ps.keys())
rng = np.random.default_rng(0)
B = 10000

REF = "RouteMean"
print("=== paired comparison vs %s (support = 8, 20 repeats) ===" % REF)
rows = []
for m in MODELS:
    if m == REF:
        continue
    d = np.array(ps[m]) - np.array(ps[REF])          # positive = worse than RouteMean
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)])
    try:
        p = float(wilcoxon(d).pvalue)
    except Exception:
        p = float("nan")
    rows.append((m, float(d.mean()), float(np.quantile(bs, 0.025)),
                 float(np.quantile(bs, 0.975)), p))

# Benjamini-Hochberg
order = np.argsort([r[4] for r in rows])
adj = [0.0] * len(rows)
prev = 1.0
for rank, idx in enumerate(reversed(order.tolist())):
    k = len(rows) - rank
    val = min(prev, rows[idx][4] * len(rows) / k)
    adj[idx] = float(val)
    prev = val
print("%-12s %8s %20s %9s %9s" % ("model", "diff", "95% CI", "p", "p_BH"))
out = []
for i, (m, dm, lo, hi, p) in enumerate(rows):
    print("%-12s %+8.3f [%6.3f, %6.3f] %9.4f %9.4f" % (m, dm, lo, hi, p, adj[i]))
    out.append({"model": m, "diff": dm, "lo": lo, "hi": hi, "p": p, "p_bh": adj[i]})
R["paired_vs_routemean"] = out

# head-to-head MTGPR vs RF / SVR
print("\n=== MTGPR vs top rivals (paired, bootstrap CI) ===")
h2h = {}
for rival in ["RF", "SVR", "XGB", "PLS", "RSM", "ST-PN"]:
    d = np.array(ps["MTGPR"]) - np.array(ps[rival])
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)])
    p = float(wilcoxon(d).pvalue)
    h2h[rival] = {"diff": float(d.mean()), "lo": float(np.quantile(bs, 0.025)),
                  "hi": float(np.quantile(bs, 0.975)), "p": p}
    print("  MTGPR - %-6s %+6.3f [%6.3f,%6.3f] p=%.4f"
          % (rival, d.mean(), np.quantile(bs, 0.025), np.quantile(bs, 0.975), p))
R["h2h_support8"] = h2h

# ---------------- figures ----------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

loro = R["loro"]
ROUTES = list(loro["per_route"].keys())
show = ["RouteMean", "MTGPR", "RF", "XGB", "SVR", "RSM", "PLS"]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
w = 0.13
x = np.arange(len(ROUTES))
cols = plt.get_cmap("viridis")(np.linspace(0.05, 0.85, len(show)))
for j, m in enumerate(show):
    vals = [loro["per_route"][rt][m] for rt in ROUTES]
    ax.bar(x + (j - len(show) / 2) * w, vals, w, label=m, color=cols[j], edgecolor="white", lw=0.5)
ax.set_xticks(x)
ax.set_xticklabels([r.replace("-", "-\n") for r in ROUTES], fontsize=9)
ax.set_ylabel("Aggregated RMSE (%)", fontsize=10)
ax.set_title("(a) Leave-one-route-out RMSE", fontsize=10, loc="left")
ax.legend(fontsize=7.5, ncol=2, frameon=False)
ax.grid(axis="y", alpha=0.25, lw=0.6)
ax.set_axisbelow(True)

ax = axes[1]
mm = sorted(loro["mean"], key=lambda z: loro["mean"][z])
mm = [m for m in mm if m != "MeanBaseline"] + ["MeanBaseline"]
vals = [loro["mean"][m] for m in mm]
colsm = ["#C0392B" if m == "MTGPR" else ("#8E8E93" if m in ("RouteMean", "MeanBaseline", "GPR")
         else "#4A6FA5") for m in mm]
b = ax.barh(range(len(mm)), vals, color=colsm, edgecolor="white", lw=0.5)
ax.set_yticks(range(len(mm)))
ax.set_yticklabels(mm, fontsize=9)
ax.invert_yaxis()
for i, v in enumerate(vals):
    ax.text(v + 0.12, i, "%.2f" % v, va="center", fontsize=8)
ax.set_xlabel("Mean aggregated RMSE (%) over the three held-out routes", fontsize=9.5)
ax.set_title("(b) Mean LORO error all models", fontsize=10, loc="left")
ax.set_xlim(0, max(vals) * 1.18)
ax.grid(axis="x", alpha=0.25, lw=0.6)
ax.set_axisbelow(True)
plt.tight_layout()
p12 = os.path.join(FIGP, "fig12_loro.png")
plt.savefig(p12, dpi=300, bbox_inches="tight")
print("saved", p12)

fig, ax = plt.subplots(1, 2, figsize=(10.2, 3.9))
c = R["conformal"]["MTGPR"]
ax[0].bar(["GP native\n(nominal 90%)", "Cross-conformal\n(nominal 90%)"],
          [c["cov_native"], c["cov_conf"]],
          color=["#C0392B", "#1F7A4D"], width=0.55, edgecolor="white")
ax[0].axhline(0.90, ls="--", lw=1.1, color="#333333")
ax[0].text(0.52, 0.915, "target 0.90", fontsize=8, color="#333333")
for i, v in enumerate([c["cov_native"], c["cov_conf"]]):
    ax[0].text(i, v + 0.02, "%.3f" % v, ha="center", fontsize=9.5, weight="bold")
ax[0].set_ylim(0, 1.15)
ax[0].set_ylabel("Empirical coverage on query set", fontsize=9.5)
ax[0].set_title("(a) Coverage at support = 8", fontsize=10, loc="left")
ax[0].grid(axis="y", alpha=0.25, lw=0.6)
ax[0].set_axisbelow(True)

cs = R["conformal_by_support"]
ks = sorted(cs, key=lambda z: int(z))
vals = [cs[k]["half_raw"] for k in ks]
ax[1].plot([int(k) for k in ks], vals, "o-", color="#1F3A5F", lw=1.8, ms=6)
for k, v in zip(ks, vals):
    ax[1].text(int(k), v + 0.22, "%.2f" % v, ha="center", fontsize=8.5)
ax[1].set_xlabel("Support-set size (calibration points)", fontsize=9.5)
ax[1].set_ylabel("Conformal half-width (percentage points)", fontsize=9.5)
ax[1].set_title("(b) Interval width vs data available", fontsize=10, loc="left")
ax[1].grid(alpha=0.25, lw=0.6)
ax[1].set_ylim(0, max(vals) * 1.25)
plt.tight_layout()
p13 = os.path.join(FIGP, "fig13_conformal.png")
plt.savefig(p13, dpi=300, bbox_inches="tight")
print("saved", p13)

json.dump(R, open(os.path.join(PROCESS, "q1_results.json"), "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)
print("updated q1_results.json")
