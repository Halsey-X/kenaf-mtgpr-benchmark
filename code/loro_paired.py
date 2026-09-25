# -*- coding: utf-8 -*-
"""Reproducible LORO + paired tests against the trivial route-mean baseline.

Why this exists:
  q1_upgrade computed LORO with a hard-coded seed=3 that is never stated in the
  Methods, and both ST-PN and MTGPR are stochastic -> Table 6 is not reproducible.
  Here LORO is averaged over SEEDS repeats, which removes the seed dependence *and*
  lets honest paired inference be run against the route-mean baseline.
"""
import sys, io, os, json, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
warnings.filterwarnings("ignore")
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
CODE = os.path.join(ROOT, "code")
OUT = ROOT
sys.path.insert(0, CODE)
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import PolynomialFeatures
from xgboost import XGBRegressor
import analysis as A
from mtgpr import MTGPR

X_raw, Y_raw = A.X_raw, A.Y_raw
N, NC = A.N, A.NC
SC = A.Scaler
routes = np.array(A.routes)
ROUTES = list(A.route_order)
MT_KW = dict(A.DEFAULT_MTGPR)
ST_KW = dict(A.DEFAULT_STPN)
SEEDS = [0, 1, 2, 3, 4]


class PLSAuto:
    def fit(self, X, Y):
        n, p = X.shape
        kmax = max(1, min(p, n - 2))
        best, bk = np.inf, 1
        for k in range(1, kmax + 1):
            errs = []
            for i in range(n):
                tr = [j for j in range(n) if j != i]
                m = PLSRegression(n_components=k, scale=False).fit(X[tr], Y[tr])
                errs.append(float(((m.predict(X[[i]])[0] - Y[i]) ** 2).mean()))
            e = float(np.mean(errs))
            if e < best:
                best, bk = e, k
        self.k = bk
        self.m = PLSRegression(n_components=bk, scale=False).fit(X, Y)
        return self

    def predict(self, X):
        return self.m.predict(X)


class RSM:
    def fit(self, X, Y):
        self.poly = PolynomialFeatures(degree=2, include_bias=True)
        self.m = Ridge(alpha=1.0, random_state=0).fit(self.poly.fit_transform(X), Y)
        return self

    def predict(self, X):
        return self.m.predict(self.poly.transform(X))


def make_std_models():
    return {
        "LR": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=0),
        "PLS": PLSAuto(),
        "RF": RandomForestRegressor(n_estimators=300, random_state=0),
        "XGB": MultiOutputRegressor(XGBRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.1, subsample=1.0,
            colsample_bytree=1.0, objective="reg:squarederror", reg_lambda=1.0,
            random_state=0, n_jobs=1, verbosity=0)),
        "SVR": MultiOutputRegressor(SVR(kernel="rbf", C=1.0, gamma="scale")),
        "MLP": MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=3000,
                            early_stopping=False, random_state=0),
        "GPR": MultiOutputRegressor(GaussianProcessRegressor(random_state=0)),
        "RSM": RSM(),
    }


MODELS = ["LR", "Ridge", "PLS", "RF", "XGB", "SVR", "MLP", "GPR", "MTGPR",
          "ST-PN", "RSM", "RouteMean"]


def fit_all(Xs_raw, Ys_raw, Xq_raw, seed=0):
    sx, sy = SC().fit(Xs_raw), SC().fit(Ys_raw)
    Xs, Xq = sx.tf(Xs_raw), sx.tf(Xq_raw)
    Ys = sy.tf(Ys_raw)
    out = {}
    models = make_std_models()
    for m in MODELS:
        if m == "ST-PN":
            kw = dict(ST_KW); kw["seed"] = seed
            p = sy.itf(A.STPN(**kw).fit(Xs, Ys).predict(Xq))
        elif m == "MTGPR":
            kw = dict(MT_KW); kw["rng"] = np.random.default_rng(seed)
            p = sy.itf(MTGPR(**kw).fit(Xs, Ys).predict(Xq))
        elif m == "RouteMean":
            # training-fold mean = the paper's trivial baseline under LORO
            p = sy.itf(np.tile(Ys.mean(0), (len(Xq_raw), 1)))
        else:
            models[m].fit(Xs, Ys)
            p = sy.itf(np.asarray(models[m].predict(Xq), float))
        out[m] = p
    return out


def agg_rmse(pred, true):
    """Paper convention: mean over the 3 per-component RMSEs."""
    err = pred - true
    return float(np.mean([np.sqrt(np.mean(err[:, c] ** 2)) for c in range(NC)]))


# ---------------- accumulate per-point abs errors over seeds ----------------
per_point = {m: np.zeros((N, NC)) for m in MODELS}
rmse_seed = {rt: {m: [] for m in MODELS} for rt in ROUTES}

for rt in ROUTES:
    te = np.where(routes == rt)[0]
    tr = np.where(routes != rt)[0]
    for s in SEEDS:
        P = fit_all(X_raw[tr], Y_raw[tr], X_raw[te], seed=s)
        for m in MODELS:
            per_point[m][te] += np.abs(P[m] - Y_raw[te]) / len(SEEDS)
            rmse_seed[rt][m].append(agg_rmse(P[m], Y_raw[te]))
    print("  %-16s done" % rt, flush=True)

print("\n=== reproducible LORO (mean +- SD over %d seeds) ===" % len(SEEDS))
print("%-10s " % "model" + " ".join("%-18s" % rt for rt in ROUTES) + "%-16s" % "mean")
loro_tab = {}
for m in MODELS:
    cells, allseed = [], []
    for rt in ROUTES:
        v = rmse_seed[rt][m]
        cells.append("%-18s" % ("%.2f ± %.2f" % (np.mean(v), np.std(v))))
        allseed.append(np.mean(v))
    loro_tab[m] = dict(per_route={rt: [float(np.mean(rmse_seed[rt][m])),
                                       float(np.std(rmse_seed[rt][m]))]
                                  for rt in ROUTES},
                       mean=float(np.mean(allseed)),
                       sd=float(np.std([np.mean(rmse_seed[rt][m]) for rt in ROUTES])))
    print("%-10s " % m + " ".join(cells) + "%-16s" % "%.2f" % np.mean(allseed))

order = np.arange(N)
base = per_point["RouteMean"][order]
print("\n=== paired test vs RouteMean over the 31 held-out points ===")
print("%-10s %8s %8s %8s %10s %18s %7s" %
      ("model", "MAE", "baseMAE", "delta", "wilcox_p", "boot95_CI", "win%"))
res = {}
rng = np.random.RandomState(0)
for m in MODELS:
    d = per_point[m][order].ravel()
    b = base.ravel()
    diff = d - b
    if np.any(np.abs(diff) > 1e-9):
        _, p = stats.wilcoxon(d, b)
    else:
        p = np.nan
    bs = [diff[rng.randint(0, len(diff), len(diff))].mean() for _ in range(5000)]
    res[m] = dict(mean_abs_err=float(d.mean()),
                  delta_vs_base=float(d.mean() - b.mean()),
                  wilcoxon_p=None if np.isnan(p) else float(p),
                  boot_lo=float(np.quantile(bs, 0.025)),
                  boot_hi=float(np.quantile(bs, 0.975)),
                  win_pct=float(np.mean(diff < 0)) * 100)
    print("%-10s %8.3f %8.3f %8.3f %10s %18s %7.1f" %
          (m, d.mean(), b.mean(), res[m]["delta_vs_base"],
           "ref" if np.isnan(p) else "%.4f" % p,
           "ref" if np.isnan(p) else "[%.2f, %.2f]" % (res[m]["boot_lo"], res[m]["boot_hi"]),
           res[m]["win_pct"]))

cand = [m for m in MODELS if res[m]["wilcoxon_p"] is not None]
bh = sorted(cand, key=lambda z: res[z]["wilcoxon_p"])
k = len(bh)
print("\nBH over %d vs-RouteMean comparisons (alpha = 0.05):" % k)
run = 0.0
bh_sig = {}
for i, m in enumerate(bh, 1):
    crit = 0.05 * i / k
    run = max(run, res[m]["wilcoxon_p"])
    ok = run <= crit
    bh_sig[m] = ok
    print("  %2d  %-10s p=%.4f crit %.4f -> %s" % (i, m, res[m]["wilcoxon_p"], crit,
                                                   "SIG" if ok else "n.s."))

print("\nBest learned model per held-out route (mean over seeds):")
learned = ["MTGPR", "RF", "SVR", "XGB", "ST-PN", "RSM", "PLS", "LR", "Ridge", "MLP", "GPR"]
for rt in ROUTES:
    sc_ = {m: float(np.mean(rmse_seed[rt][m])) for m in learned}
    sd_ = {m: float(np.std(rmse_seed[rt][m])) for m in learned}
    rk = sorted(sc_.items(), key=lambda x: x[1])
    print("  %-16s best=%-7s %.2f±%.2f | 2nd=%-7s %.2f±%.2f | RouteMean %.2f" %
          (rt, rk[0][0], rk[0][1], sd_[rk[0][0]], rk[1][0], rk[1][1], sd_[rk[1][0]],
           float(np.mean(rmse_seed[rt]["RouteMean"]))))

json.dump(dict(loro_table=loro_tab, paired=res, bh=bh_sig, seeds=SEEDS),
          open(os.path.join(OUT, "process_data", "loro_paired.json"), "w",
               encoding="utf-8"), indent=2, ensure_ascii=False)
print("\nsaved process_data/loro_paired.json")
