# -*- coding: utf-8 -*-
"""Rebuild Table 3 (paired t-tests, MTGPR vs every rival).

Why:  q1_results.json `support8_per_split` stores POOLED RMSE per split, but the
paper declares (2.6.3) the comparison metric as "aggregated ... as the mean of the
three per-component RMSE values".  Those differ (MTGPR 4.106 vs 3.985), so Table 3
cannot be read off stored aggregates.  This script recomputes the 20 support-query
splits under the declared convention, verifies the ten rows already in Table 3,
adds the three missing comparisons (PLS, XGB, RouteMean) and recomputes Holm over
the full family of comparisons.
"""
import sys, io, os, json, math, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
warnings.filterwarnings("ignore")
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
CODE = os.path.join(ROOT, "code")
OUT = ROOT
sys.path.insert(0, CODE)
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.cross_decomposition import PLSRegression
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
import analysis as A
from mtgpr import MTGPR

X_raw, Y_raw, NC, N = A.X_raw, A.Y_raw, A.NC, A.N
Scaler = A.Scaler


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


class MeanBaseline:
    def fit(self, X, Y):
        self.mu = Y.mean(0)
        return self

    def predict(self, X):
        return np.tile(self.mu, (len(X), 1))


D = 4
V = np.zeros((D - 1, D))
for i in range(D - 1):
    k = i + 1
    V[i, :k] = 1.0
    V[i, k] = -k
    V[i, :] /= math.sqrt(k * (k + 1))


def to_parts(Y3):
    other = np.clip(100.0 - Y3.sum(1, keepdims=True), 1.0, None)
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


RIVALS = ["LR", "Ridge", "PLS", "RF", "XGB", "SVR", "MLP", "GPR", "ST-PN",
          "RSM", "RouteMean", "MeanBaseline", "C-MTGPR"]
per_split = {m: [] for m in RIVALS}
mt_splits = []

for r in range(20):
    rng = np.random.RandomState(r)
    perm = rng.permutation(N)
    sup, que = perm[:8], perm[8:]
    sx, sy = Scaler().fit(X_raw[sup]), Scaler().fit(Y_raw[sup])
    Xtr, Xte = sx.tf(X_raw[sup]), sx.tf(X_raw[que])
    Ytr = sy.tf(Y_raw[sup])
    base = A.make_baselines()
    extra = {"PLS": PLSAuto(), "RSM": RSM(), "MeanBaseline": MeanBaseline(),
             "XGB": MultiOutputRegressor(XGBRegressor(
                 n_estimators=300, max_depth=3, learning_rate=0.1, subsample=1.0,
                 colsample_bytree=1.0, objective="reg:squarederror", reg_lambda=1.0,
                 random_state=0, n_jobs=1, verbosity=0))}

    def agg(pred):
        return float(np.mean([np.sqrt(np.mean((pred[:, c] - Y_raw[que, c]) ** 2))
                              for c in range(NC)]))

    kw = dict(A.DEFAULT_MTGPR); kw["rng"] = np.random.default_rng(r)
    mt_splits.append(agg(sy.itf(MTGPR(**kw).fit(Xtr, Ytr).predict(Xte))))

    for m in RIVALS:
        if m == "ST-PN":
            k2 = dict(A.DEFAULT_STPN); k2["seed"] = r
            p = sy.itf(A.STPN(**k2).fit(Xtr, Ytr).predict(Xte))
        elif m == "RouteMean":
            lut = {}
            r_sup = np.asarray(A.routes)[sup]
            for rt in A.route_order:
                mk = r_sup == rt
                lut[rt] = Ytr[mk].mean(0) if mk.sum() > 0 else Ytr.mean(0)
            rq = np.asarray(A.routes)[que]
            p = sy.itf(np.array([lut[x] for x in rq]))
        elif m == "C-MTGPR":
            Ztr = ilr(to_parts(sy.itf(Ytr)))
            k3 = dict(A.DEFAULT_MTGPR); k3["rng"] = np.random.default_rng(r)
            Zte = MTGPR(**k3).fit(Xtr, Ztr).predict(Xte)
            p = ilr_inv(Zte)[:, :3]
        elif m in extra:
            p = sy.itf(np.asarray(extra[m].fit(Xtr, Ytr).predict(Xte), float))
        else:
            p = sy.itf(np.asarray(base[m].fit(Xtr, Ytr).predict(Xte), float))
        per_split[m].append(agg(p))
    print("   split %d/20 done" % (r + 1), flush=True)

mt = np.array(mt_splits)
print("\nMTGPR aggregated RMSE over splits: %.3f  (expect ~3.99)" % mt.mean())
print("MeanBaseline: %.3f (expect ~5.15) ; RouteMean: %.3f (expect ~3.61)"
      % (np.mean(per_split["MeanBaseline"]), np.mean(per_split["RouteMean"])))
print("C-MTGPR: %.3f" % np.mean(per_split["C-MTGPR"]))

rows = []
for m in RIVALS:
    o = np.array(per_split[m])
    t, p = stats.ttest_rel(mt, o)
    rows.append(dict(model=m, t=float(t), p_raw=float(p), diff=float((mt - o).mean())))

# Holm-Bonferroni over the whole family
order = np.argsort([r["p_raw"] for r in rows])
k = len(rows)
run = 0.0
for rank, i in enumerate(order, 1):
    run = max(run, rows[i]["p_raw"])
    rows[i]["p_holm"] = float(min(1.0, run * (k - rank + 1)))

print("\n%-14s %8s %10s %10s %9s" % ("comparison", "t", "p_raw", "p_Holm", "MTGPR-rival"))
for r in rows:
    print("MTGPR vs %-7s %8.2f %10.4f %10.4f %9.3f" %
          (r["model"], r["t"], r["p_raw"], r["p_holm"], r["diff"]))

print("\nsignificant after Holm (alpha = 0.05):")
for r in rows:
    if r["p_holm"] < 0.05:
        print("   %-10s  direction: MTGPR is %s" %
              (r["model"], "BETTER" if r["diff"] < 0 else "WORSE"))

json.dump({"mtgpr_per_split": [float(x) for x in mt],
           "rivals_per_split": {m: [float(x) for x in v] for m, v in per_split.items()},
           "rows": rows},
          open(os.path.join(OUT, "process_data", "table3_rebuilt.json"), "w",
               encoding="utf-8"), indent=2, ensure_ascii=False)
print("\nsaved process_data/table3_rebuilt.json")
