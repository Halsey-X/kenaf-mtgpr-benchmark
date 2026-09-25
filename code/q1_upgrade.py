# -*- coding: utf-8 -*-
"""Q1-level upgrade analyses for the kenaf composition paper.

Adds what a Q1 reviewer would demand and that IS computable from the existing
31-sample matrix:
  A. leave-one-route-out (LORO) generalisation          -- route extrapolation
  B. field-standard PLSR + modern XGBoost baselines     -- benchmark completeness
  C. route-mean baseline                                -- pH/route confound test
  D. cross-conformal prediction intervals               -- replaces "future work"
  E. Friedman + Nemenyi multi-model comparison          -- benchmark protocol
  F. bootstrap CIs on headline metrics
  G. post-hoc simplex projection (mass balance)         -- cheap closure
"""
import json, os, sys, math, warnings
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
CODE = os.path.join(ROOT, "code")
OUT = ROOT
PROCESS = os.path.join(OUT, "process_data")
FIGP = os.path.join(OUT, "figures_data", "png")
FIGE = os.path.join(OUT, "figures_data", "eps")
os.makedirs(PROCESS, exist_ok=True)
sys.path.insert(0, CODE)

import numpy as np
import analysis as A
from mtgpr import MTGPR
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import PolynomialFeatures
from xgboost import XGBRegressor
from scipy.stats import friedmanchisquare, studentized_range

X_raw, Y_raw = A.X_raw, A.Y_raw
N, NC = A.N, A.NC
SC = A.Scaler
TARG = ["Cellulose", "Hemicellulose", "Lignin"]
NAMES = TARG
routes = np.array(A.routes)
ROUTES = list(A.route_order)
MT_KW = dict(A.DEFAULT_MTGPR)
ST_KW = dict(A.DEFAULT_STPN)


# ---------------- helpers ----------------
def rms(a):
    return float(np.sqrt(np.mean(np.asarray(a) ** 2)))


def pooled_rmse(mat):
    """pooled RMSE over rows(samples) x cols(components)."""
    return float(np.sqrt(np.mean(np.asarray(mat) ** 2)))


class PLSAuto:
    """PLSR with latent components chosen by internal leave-one-out."""

    def fit(self, X, Y):
        n, p = X.shape
        kmax = max(1, min(p, n - 2))
        best, bk = np.inf, 1
        for k in range(1, kmax + 1):
            errs = []
            for i in range(n):
                tr = [j for j in range(n) if j != i]
                m = PLSRegression(n_components=k, scale=False)
                m.fit(X[tr], Y[tr])
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


class MeanBaseline:
    def fit(self, X, Y):
        self.mu = Y.mean(0)
        return self

    def predict(self, X):
        return np.tile(self.mu, (len(X), 1))


class RouteMean:
    """Predict the mean composition of the route implied by training samples.
    Tests whether models do more than memorise route identity (pH defines route)."""

    def fit(self, Xs, Ys, Xs_raw):
        r = np.where(Xs_raw[:, 2] == 7, "Autohydrolysis",
             np.where(Xs_raw[:, 2] < 7, "Dilute-acid", "Alkaline"))
        self.lut = {}
        for rt in ROUTES:
            mk = r == rt
            self.lut[rt] = Ys[mk].mean(0) if mk.sum() > 0 else Ys.mean(0)
        return self

    def predict(self, Xq_raw):
        r = np.where(Xq_raw[:, 2] == 7, "Autohydrolysis",
             np.where(Xq_raw[:, 2] < 7, "Dilute-acid", "Alkaline"))
        return np.array([self.lut[x] for x in r])


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
          "ST-PN", "RSM", "RouteMean", "MeanBaseline"]


def predict_std(name, model, Xs, Ys, Xq):
    """X,Y in standardised space -> prediction in standardised space."""
    if name == "MTGPR":
        return MTGPR(**MT_KW).fit(Xs, Ys).predict(Xq)
    return np.asarray(model.predict(Xq), float)


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
        elif m == "MeanBaseline":
            p = sy.itf(MeanBaseline().fit(Xs, Ys).predict(Xq))
        elif m == "RouteMean":
            p = sy.itf(RouteMean().fit(Xs, Ys, Xs_raw).predict(Xq_raw))
        else:
            models[m].fit(Xs, Ys)
            p = sy.itf(predict_std(m, models[m], Xs, Ys, Xq))
        out[m] = p
    return out


RES = {}

# ============ A. repeated support-query benchmark (support = 8) =============
print("=== support = 8, 20 repeats: full 13-model benchmark ===", flush=True)
R = 20
per_split = {m: [] for m in MODELS}                    # aggregated RMSE per split
per_comp = {m: {c: [] for c in range(NC)} for m in MODELS}
per_comp_mae = {m: {c: [] for c in range(NC)} for m in MODELS}
for r in range(R):
    rng = np.random.RandomState(r)
    perm = rng.permutation(N)
    sup, que = perm[:8], perm[8:]
    P = fit_all(X_raw[sup], Y_raw[sup], X_raw[que], seed=r)
    for m in MODELS:
        err = P[m] - Y_raw[que]
        per_split[m].append(pooled_rmse(err))
        for c in range(NC):
            per_comp[m][c].append(rms(err[:, c]))
            per_comp_mae[m][c].append(float(np.mean(np.abs(err[:, c]))))
    if (r + 1) % 5 == 0:
        print("   repeat %d done" % (r + 1), flush=True)

tab2 = {}
for m in MODELS:
    mean_pc = float(np.mean([np.mean(per_comp[m][c]) for c in range(NC)]))
    tab2[m] = {
        "agg_rmse_mean": mean_pc,                    # manuscript convention
        "agg_rmse_mean_pooled": float(np.mean(per_split[m])),
        "rmse": {NAMES[c]: [float(np.mean(per_comp[m][c])), float(np.std(per_comp[m][c]))]
                 for c in range(NC)},
        "mae": {NAMES[c]: [float(np.mean(per_comp_mae[m][c])), float(np.std(per_comp_mae[m][c]))]
                for c in range(NC)},
    }
print("   agg RMSE support=8 (mean of per-component RMSE = manuscript convention):")
for m in sorted(MODELS, key=lambda z: tab2[z]["agg_rmse_mean"]):
    print("      %-12s %.2f" % (m, tab2[m]["agg_rmse_mean"]), flush=True)
RES["support8"] = tab2
RES["support8_per_split"] = per_split

# ==================== E. Friedman + Nemenyi =================================
print("=== Friedman + Nemenyi over the 20 repeated splits ===", flush=True)
M = np.column_stack([per_split[m] for m in MODELS])     # 20 x k
chi2, pf = friedmanchisquare(*[M[:, j] for j in range(M.shape[1])])
ranks = np.array([(-M[i]).argsort().argsort() + 1.0 for i in range(M.shape[0])])
avg_rank = ranks.mean(0)
k = len(MODELS)
q = studentized_range.ppf(0.95, k, np.inf) / math.sqrt(2.0)
cd = q * math.sqrt(k * (k + 1) / (6.0 * M.shape[0]))
print("   Friedman chi2=%.2f p=%.4f ; CD(Nemenyi,alpha=.05)=%.2f" % (chi2, pf, cd), flush=True)
RES["friedman"] = {"chi2": float(chi2), "p": float(pf), "critical_difference": float(cd),
                   "avg_rank": {m: float(avg_rank[i]) for i, m in enumerate(MODELS)}}

# ==================== B. support-size ablation ==============================
print("=== support-size ablation (4..12) ===", flush=True)
abl = {}
for ss in [4, 6, 8, 10, 12]:
    row = {m: [] for m in MODELS}
    rowpc = {m: [] for m in MODELS}
    for r in range(R):
        rng = np.random.RandomState(r)
        perm = rng.permutation(N)
        sup, que = perm[:ss], perm[ss:]
        P = fit_all(X_raw[sup], Y_raw[sup], X_raw[que], seed=r)
        for m in MODELS:
            err = P[m] - Y_raw[que]
            row[m].append(pooled_rmse(err))
            rowpc[m].append(float(np.mean([rms(err[:, c]) for c in range(NC)])))
    abl[ss] = {m: float(np.mean(rowpc[m])) for m in MODELS}
    abl.setdefault("_pooled", {})[ss] = {m: float(np.mean(row[m])) for m in MODELS}
    print("   n=%2d  " % ss + "  ".join("%s=%.2f" % (m, abl[ss][m]) for m in
                                        ["LR", "PLS", "RF", "XGB", "MTGPR", "RSM", "RouteMean"]),
          flush=True)
RES["ablation"] = abl

# ==================== C. LOOCV ==============================================
print("=== LOOCV (pooled RMSE / MAE) ===", flush=True)
loo_pred = {m: np.zeros((N, NC)) for m in MODELS}
for i in range(N):
    tr = [j for j in range(N) if j != i]
    P = fit_all(X_raw[tr], Y_raw[tr], X_raw[[i]], seed=7)
    for m in MODELS:
        loo_pred[m][i] = P[m][0]
    if (i + 1) % 10 == 0:
        print("   LOO %d/%d" % (i + 1, N), flush=True)
loo = {}
for m in MODELS:
    err = loo_pred[m] - Y_raw
    loo[m] = {
        "agg_rmse": float(np.mean([rms(err[:, c]) for c in range(NC)])),
        "agg_rmse_pooled": pooled_rmse(err),
        "agg_mae": float(np.mean([np.mean(np.abs(err[:, c])) for c in range(NC)])),
        "rmse_pc": [rms(err[:, c]) for c in range(NC)],
        "r2_pc": [float(1 - np.sum(err[:, c] ** 2) /
                        np.sum((Y_raw[:, c] - Y_raw[:, c].mean()) ** 2)) for c in range(NC)],
    }
print("   LOOCV pooled RMSE:")
for m in sorted(MODELS, key=lambda z: loo[z]["agg_rmse"]):
    print("      %-12s %.2f  (R2 %s)" % (m, loo[m]["agg_rmse"],
                                         [round(x, 2) for x in loo[m]["r2_pc"]]), flush=True)
RES["loocv"] = loo

# ==================== A. leave-one-route-out ================================
print("=== leave-one-route-out (LORO) ===", flush=True)
loro = {}
for rt in ROUTES:
    te = np.where(routes == rt)[0]
    tr = np.where(routes != rt)[0]
    P = fit_all(X_raw[tr], Y_raw[tr], X_raw[te], seed=3)
    loro[rt] = {m: float(np.mean([rms((P[m] - Y_raw[te])[:, c]) for c in range(NC)]))
                for m in MODELS}
    print("   held-out %-15s " % rt + " ".join(
        "%s=%.2f" % (m, loro[rt][m]) for m in ["PLS", "RF", "XGB", "MTGPR", "RSM", "RouteMean"]),
        flush=True)
loro_mean = {m: float(np.mean([loro[rt][m] for rt in ROUTES])) for m in MODELS}
print("   LORO mean RMSE:", {m: round(loro_mean[m], 2) for m in
                             sorted(MODELS, key=lambda z: loro_mean[z])}, flush=True)
RES["loro"] = {"per_route": loro, "mean": loro_mean}

# ============ D. cross-conformal prediction intervals (support=8) ===========
print("=== cross-conformal (CV+) intervals ===", flush=True)


def conformal_split(sup, que, seed, model="MTGPR", alpha=0.10):
    """Residuals from LOO inside the support -> +-max|resid| band; eval on query."""
    sx, sy = SC().fit(X_raw[sup]), SC().fit(Y_raw[sup])
    Xs, Xq = sx.tf(X_raw[sup]), sx.tf(X_raw[que])
    Ys = sy.tf(Y_raw[sup])
    res = []
    for i in range(len(sup)):
        tr = [j for j in range(len(sup)) if j != i]
        if model == "MTGPR":
            kw = dict(MT_KW); kw["rng"] = np.random.default_rng(seed + 100)
            p = MTGPR(**kw).fit(Xs[tr], Ys[tr]).predict(Xs[[i]])
        elif model == "RF":
            p = RandomForestRegressor(n_estimators=300, random_state=0).fit(
                Xs[tr], Ys[tr]).predict(Xs[[i]])
        res.append(np.abs(Ys[i] - p[0]))
    res = np.array(res)                       # standardised units
    half = np.quantile(res, min(1.0, math.ceil((len(sup) + 1) * (1 - alpha)) / len(sup)), axis=0)
    if model == "MTGPR":
        kw = dict(MT_KW); kw["rng"] = np.random.default_rng(seed + 100)
        mobj = MTGPR(**kw).fit(Xs, Ys)
        mean, var = mobj.predict(Xq, return_var=True)
        native_half = np.sqrt(var)
    else:
        mobj = RandomForestRegressor(n_estimators=300, random_state=0).fit(Xs, Ys)
        mean = mobj.predict(Xq)
        native_half = None
    hit_c = np.mean(np.abs(sy.tf(Y_raw[que]) - mean) <= sy.sd * half)
    out = {"native_half_std": None if native_half is None else
           float(np.mean(native_half)),
           "conf_half_raw": float(np.mean(half * sy.sd)),
           "cov_conf": float(hit_c)}
    if native_half is not None:
        out["cov_native"] = float(np.mean(np.abs(sy.tf(Y_raw[que]) - mean)
                                          <= 1.645 * native_half))
    return out


alpha = 0.10
conf = {}
for mname in ["MTGPR", "RF"]:
    rows = []
    for r in range(R):
        rng = np.random.RandomState(r)
        perm = rng.permutation(N)
        rows.append(conformal_split(perm[:8], perm[8:], seed=r, model=mname, alpha=alpha))
    conf[mname] = {kk: float(np.mean([x.get(kk) for x in rows if x.get(kk) is not None]))
                   for kk in ["conf_half_raw", "cov_conf", "cov_native", "native_half_std"]}
    cnative = conf[mname].get("cov_native")
    print("   %-6s conformal: coverage %.3f  mean half-width %.2f pct-pts | native coverage: %s" %
          (mname, conf[mname]["cov_conf"], conf[mname]["conf_half_raw"],
           "n/a" if cnative is None else "%.3f" % cnative), flush=True)
RES["conformal"] = conf

# how does conformal width shrink with support size?
conf_n = {}
for ss in [8, 12, 16, 20]:
    rows = []
    for r in range(8):
        rng = np.random.RandomState(r)
        perm = rng.permutation(N)
        rows.append(conformal_split(perm[:ss], perm[ss:], seed=r, model="MTGPR", alpha=alpha))
    conf_n[ss] = {"half_raw": float(np.mean([x["conf_half_raw"] for x in rows])),
                  "cov": float(np.mean([x["cov_conf"] for x in rows]))}
    print("   support=%2d -> conformal half-width %.2f pct-pts, coverage %.3f"
          % (ss, conf_n[ss]["half_raw"], conf_n[ss]["cov"]), flush=True)
RES["conformal_by_support"] = conf_n

# ==================== F. bootstrap CIs ======================================
rng = np.random.default_rng(0)
B = 4000
boots = {}
err_mtgpr = {r: None for r in range(R)}
# per-repeat aggregated errors recomputed from stored per-split values
idx = np.arange(R)
for mname in ["MTGPR", "RF", "XGB", "PLS", "RSM", "Ridge", "RouteMean", "MeanBaseline"]:
    vals = np.array(per_split[mname])
    bs = np.array([vals[rng.integers(0, R, R)].mean() for _ in range(B)])
    boots[mname] = {"point": float(vals.mean()),
                    "lo": float(np.quantile(bs, 0.025)),
                    "hi": float(np.quantile(bs, 0.975))}
for mname in ["MTGPR", "RF", "XGB", "PLS", "RSM", "RouteMean"]:
    d = boots[mname]
    print("   %-12s %.2f  [95%% CI %.2f, %.2f]" % (mname, d["point"], d["lo"], d["hi"]), flush=True)
RES["bootstrap"] = boots

# ==================== G. post-hoc simplex projection ========================
print("=== mass balance: violation rate + post-hoc projection ===", flush=True)
sx, sy = SC().fit(X_raw), SC().fit(Y_raw)
mtburg = MTGPR(**MT_KW).fit(sx.tf(X_raw), sy.tf(Y_raw))
Pfull = sy.itf(mtburg.predict(sx.tf(X_raw)))
viol = {"neg": float(np.mean((Pfull < 0).any(1))),
        "sum_gt_100": float(np.mean(Pfull.sum(1) > 100.0)),
        "max_sum": float(Pfull.sum(1).max())}
Q = np.clip(Pfull, 0.0, None)
s = Q.sum(1, keepdims=True)
Qp = np.where(s > 100.0, Q * (100.0 / np.maximum(s, 1e-9)), Q)
proj = {"rmse_before": pooled_rmse(Pfull - Y_raw), "rmse_after": pooled_rmse(Qp - Y_raw)}
print("   violations: any-negative %.1f%% | sum>100 %.1f%% | max sum %.1f"
      % (viol["neg"] * 100, viol["sum_gt_100"] * 100, viol["max_sum"]), flush=True)
print("   full-fit RMSE before %.2f -> after projection %.2f"
      % (proj["rmse_before"], proj["rmse_after"]), flush=True)
RES["mass_balance"] = {"violation": viol, "projection": proj}

with open(os.path.join(PROCESS, "q1_results.json"), "w") as f:
    json.dump(RES, f, indent=2)
print("saved q1_results.json", flush=True)
