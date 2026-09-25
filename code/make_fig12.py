# -*- coding: utf-8 -*-
"""Regenerate Fig. 12 from the reproducible (5-seed averaged) LORO results."""
import os, io, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
BASE = ROOT
PNG = os.path.join(BASE, "figures_data", "png", "fig12_loro.png")
EPS = os.path.join(BASE, "figures_data", "eps", "fig12_loro.eps")
L = json.load(open(os.path.join(BASE, "process_data", "loro_paired.json"), encoding="utf-8"))["loro_table"]

ROUTES = ["Autohydrolysis", "Dilute-acid", "Alkaline"]
SHORT = ["Auto-\nhydrolysis", "Dilute-\nacid", "Alkaline"]
show = ["MTGPR", "RF", "SVR", "XGB", "ST-PN", "RSM", "PLS", "LR", "GPR", "RouteMean"]
mean = {m: L[m]["mean"] for m in show}
per = {m: [L[m]["per_route"][r][0] for r in ROUTES] for m in show}
sd = {m: max(L[m]["per_route"][r][1] for r in ROUTES) for m in show}

C_TXT, C_BASE, C_GP = "#1F2933", "#8C8C8C", "#3E6DA1"
fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.5))
fig.patch.set_facecolor("white")

# ---------- (a) per held-out route ----------
ax = axes[0]
x = np.arange(len(ROUTES))
w = 0.085
for i, m in enumerate(show):
    off = (i - (len(show) - 1) / 2) * w
    vals = per[m]
    ax.bar(x + off, vals, w,
           color=C_GP if m == "MTGPR" else (C_BASE if m == "RouteMean" else "#BFC7D0"),
           edgecolor="#5A6470" if m in ("MTGPR", "RouteMean") else "none",
           linewidth=1.1 if m in ("MTGPR", "RouteMean") else 0.0,
           label=m, zorder=3)
ax.set_xticks(x)
ax.set_xticklabels(SHORT, fontsize=9)
ax.set_ylabel("Aggregated RMSE (%)", fontsize=10)
ax.set_title("(a) RMSE with each route held out", fontsize=11, weight="bold", pad=8)
ax.grid(axis="y", ls=":", lw=0.6, color="#CCCCCC", zorder=0)
ax.set_axisbelow(True)
ax.set_ylim(0, 21)

# ---------- (b) mean over the three routes ----------
axb = axes[1]
order = sorted(show, key=lambda m: mean[m])
cols = [C_GP if m == "MTGPR" else (C_BASE if m == "RouteMean" else "#BFC7D0") for m in order]
bars = axb.barh(np.arange(len(order)), [mean[m] for m in order],
                color=cols, edgecolor="#5A6470", linewidth=0.8, zorder=3)
for b, m in zip(bars, order):
    edge = C_GP if m == "MTGPR" else (C_BASE if m == "RouteMean" else "none")
    lw = 1.4 if edge != "none" else 0.8
    b.set_edgecolor(edge if edge != "none" else "#8A8A8A")
    b.set_linewidth(lw)
    s = " %.2f" % mean[m] + ("  (seed SD %.2f)" % sd[m] if sd[m] >= 0.01 else "")
    axb.text(mean[m] + 0.18, b.get_y() + b.get_height() / 2, s.strip(),
             va="center", fontsize=9, color=C_TXT)
axb.set_yticks(np.arange(len(order)))
axb.set_yticklabels(order, fontsize=9)
axb.set_xlabel("Mean aggregated RMSE (%)", fontsize=10)
axb.set_title("(b) Mean over the three held-out routes", fontsize=11, weight="bold", pad=8)
axb.grid(axis="x", ls=":", lw=0.6, color="#CCCCCC", zorder=0)
axb.set_axisbelow(True)
axb.set_xlim(0, 18.5)
axb.invert_yaxis()

handles = [plt.Rectangle((0, 0), 1, 1, facecolor=C_GP, ec="#5A6470", lw=1.2),
           plt.Rectangle((0, 0), 1, 1, facecolor=C_BASE, ec="#5A6470", lw=1.2),
           plt.Rectangle((0, 0), 1, 1, facecolor="#BFC7D0", ec="#8A8A8A", lw=0.8)]
fig.legend(handles, ["MTGPR", "Route-mean baseline (trivial)", "other models"],
           loc="lower center", ncol=3, frameon=False, fontsize=9,
           bbox_to_anchor=(0.5, -0.02))
fig.suptitle("Leave-one-route-out generalisation: the Gaussian-process models coincide "
             "with the trivial baseline", fontsize=11.5, weight="bold", y=0.99)
fig.tight_layout(rect=[0, 0.05, 1, 0.93])
os.makedirs(os.path.dirname(EPS), exist_ok=True)
fig.savefig(PNG, dpi=300, facecolor="white")
fig.savefig(EPS, format="eps", facecolor="white")
print("saved", PNG)
print("saved", EPS)
