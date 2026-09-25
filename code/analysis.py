"""
1 -> 2 composition-prediction analysis.
Predicts kenaf-core chemical composition (Cellulose, Hemicellulose, Lignin)
from pretreatment descriptors (Time, Temperature, pH).

Mirrors the small-sample protocol of the companion yield paper:
 - 6 baselines (LR, Ridge, RF, SVR, MLP, GPR) + improved single-task
   prototypical network (ST-PN, multi-output extension).
 - 20 repeated random support-query splits (support=8 default).
 - Support-size ablation (4,6,8,10,12).
 - Leave-one-out cross-validation.
 - ST-PN hyperparameter sensitivity.
 - Random-forest descriptor importance.
Outputs results.json for downstream figure / manuscript generation.
"""
import json, os, warnings, math
import numpy as np
warnings.filterwarnings("ignore")
np.random.seed(42)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
SRC = os.path.join(ROOT, "data", "kenaf_pretreatment_dataset.xlsx")
OUT = ROOT
PROCESS = os.path.join(OUT, "process_data")
FIG = os.path.join(OUT, "figures_data")
os.makedirs(PROCESS, exist_ok=True)

import openpyxl
wb = openpyxl.load_workbook(SRC, data_only=True)
ws = wb["pre"]
rows = list(ws.iter_rows(values_only=True))
hdr = rows[1]
data = np.array([list(r[:7]) for r in rows[2:]], dtype=float)  # 31 x 7
X_raw = data[:, 0:3]          # Time, Temperature, pH
Y_raw = data[:, 3:6]          # Cellulose, Hemicellulose, Lignin
feat_names = list(hdr[:3])
targ_names = ["Cellulose (%)", "Hemicellulose (%)", "Lignin (%)"]

# Route assignment by pH (companion paper: autohydrolysis / dilute-acid / alkaline)
ph = X_raw[:, 2]
routes = np.where(ph == 7, "Autohydrolysis",
          np.where(ph < 7, "Dilute-acid", "Alkaline"))
route_order = ["Autohydrolysis", "Dilute-acid", "Alkaline"]

N = X_raw.shape[0]
NC = Y_raw.shape[1]  # 3 components

# ---------- metrics ----------
def metrics(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    e = y - p
    ae = np.abs(e)
    rmse = math.sqrt(np.mean(e * e))
    mae = np.mean(ae)
    maxe = np.max(ae)
    medae = np.median(ae)
    # MAPE safe (composition can be 0 for hemicellulose)
    mask = np.abs(y) > 1e-6
    mape = np.mean(ae[mask] / np.abs(y[mask])) * 100 if mask.sum() > 0 else float("nan")
    return rmse, mae, maxe, medae, mape

METRIC_NAMES = ["RMSE", "MAE", "MaxE", "MedAE", "MAPE"]

# ---------- scaling ----------
class Scaler:
    def fit(self, a):
        self.mu = a.mean(0); self.sd = a.std(0) + 1e-9
        return self
    def tf(self, a):
        return (a - self.mu) / self.sd
    def itf(self, a):
        return a * self.sd + self.mu

# ---------- ST-PN (multi-output) ----------
def relu(a): return np.maximum(a, 0.0)

class STPN:
    def __init__(self, d=8, lr=1e-2, l2=1e-3, epochs=600, patience=20, seed=0):
        self.d, self.lr, self.l2, self.epochs, self.patience = d, lr, l2, epochs, patience
        self.rng = np.random.RandomState(seed)

    def fit(self, Xs, Ys):
        # Xs, Ys standardized (n x 3). Learn W1,b1 (embed), W2 (decode).
        n = Xs.shape[0]
        W1 = self.rng.randn(3, self.d) * 0.5
        b1 = np.zeros(self.d)
        W2 = np.zeros((self.d, NC))
        # train on the full support set for a fixed number of epochs
        # (early stopping on a tiny hold-out is unreliable for n<=12)
        W1, b1, W2 = self._train(W1, b1, W2, Xs, Ys)
        # store
        self.W1, self.b1, self.W2 = W1, b1, W2
        # prototype + target mean from full support
        z = relu(Xs @ W1 + b1)
        self.p = z.mean(0)
        self.ybar = Ys.mean(0)
        return self

    def _grad(self, Xs, Ys, W1, b1, W2, p):
        z = relu(Xs @ W1 + b1)          # training embeddings
        pred = self.ybar0 + (p - z) @ W2  # p = training prototype (shared)
        E = pred - Ys
        n = Xs.shape[0]
        dW2 = (p - z).T @ E / n + self.l2 * W2
        dLdz = -(E @ W2.T)               # because diff = p - z
        da = dLdz * (Xs @ W1 + b1 > 0)
        dW1 = Xs.T @ da / n + self.l2 * W1
        db1 = da.mean(0) + self.l2 * b1
        return dW1, db1, dW2

    def _train(self, W1, b1, W2, Xs, Ys):
        self.ybar0 = Ys.mean(0)
        # training prototype (fixed during training)
        z = relu(Xs @ W1 + b1)
        p = z.mean(0)
        mW1, mb1, mW2 = np.zeros_like(W1), np.zeros_like(b1), np.zeros_like(W2)
        vW1, vb1, vW2 = np.zeros_like(W1), np.zeros_like(b1), np.zeros_like(W2)
        b1m, b2m, eps = 0.9, 0.999, 1e-8
        for t in range(1, self.epochs + 1):
            gW1, gb1, gW2 = self._grad(Xs, Ys, W1, b1, W2, p)
            mW1 = b1m * mW1 + (1 - b1m) * gW1
            mb1 = b1m * mb1 + (1 - b1m) * gb1
            mW2 = b1m * mW2 + (1 - b2m) * gW2
            vW1 = b2m * vW1 + (1 - b2m) * (gW1 ** 2)
            vb1 = b2m * vb1 + (1 - b2m) * (gb1 ** 2)
            vW2 = b2m * vW2 + (1 - b2m) * (gW2 ** 2)
            mW1h = mW1 / (1 - b1m ** t); mb1h = mb1 / (1 - b1m ** t); mW2h = mW2 / (1 - b2m ** t)
            vW1h = vW1 / (1 - b2m ** t); vb1h = vb1 / (1 - b2m ** t); vW2h = vW2 / (1 - b2m ** t)
            W1 = W1 - self.lr * mW1h / (np.sqrt(vW1h) + eps)
            b1 = b1 - self.lr * mb1h / (np.sqrt(vb1h) + eps)
            W2 = W2 - self.lr * mW2h / (np.sqrt(vW2h) + eps)
        return W1, b1, W2

    def predict(self, Xq):
        z = relu(Xq @ self.W1 + self.b1)
        return self.ybar + (self.p - z) @ self.W2

# ---------- baselines ----------
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.multioutput import MultiOutputRegressor
from mtgpr import MTGPR

def make_baselines():
    return {
        "LR": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=0),
        "RF": RandomForestRegressor(n_estimators=300, random_state=0),
        "SVR": MultiOutputRegressor(SVR(kernel="rbf", C=1.0, gamma="scale")),
        "MLP": MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=3000,
                            early_stopping=False, random_state=0),
        "GPR": MultiOutputRegressor(GaussianProcessRegressor(random_state=0)),
    }

MODELS = ["LR", "Ridge", "RF", "SVR", "MLP", "GPR", "MTGPR", "ST-PN"]
DEFAULT_STPN = {"d": 8, "lr": 1e-2, "l2": 1e-1, "epochs": 500}
DEFAULT_MTGPR = {"rank": 2, "ard": True, "maxiter": 150, "restarts": 3}

def fit_predict_baseline(name, model, Xtr, Ytr, Xte):
    model.fit(Xtr, Ytr)
    return model.predict(Xte)

# ---------- protocol ----------
R = 20  # repeats

def run_splits(support, repeats=R, seed0=0, stpn_kwargs=None, mtgpr_kwargs=None):
    """Returns dict: model -> comp -> metric -> array(repeats) for mean metric,
       plus raw RMSE & MAE lists for distributions."""
    res = {m: {c: {met: [] for met in METRIC_NAMES} for c in range(NC)} for m in MODELS}
    raw = {m: {met: [[] for _ in range(NC)] for met in ["RMSE", "MAE"]} for m in MODELS}
    rep_states = []
    for r in range(repeats):
        rng = np.random.RandomState(seed0 + r)
        perm = rng.permutation(N)
        sup, que = perm[:support], perm[support:]
        Xs_raw, Xq_raw = X_raw[sup], X_raw[que]
        Ys_raw, Yq_raw = Y_raw[sup], Y_raw[que]
        # scale
        sx, sy = Scaler().fit(Xs_raw), Scaler().fit(Ys_raw)
        Xs, Xq = sx.tf(Xs_raw), sx.tf(Xq_raw)
        Ys = sy.tf(Ys_raw)
        baselines = make_baselines()
        for m in MODELS:
            if m == "ST-PN":
                kw = dict(DEFAULT_STPN) if stpn_kwargs is None else dict(stpn_kwargs)
                kw["seed"] = seed0 + r
                stpn = STPN(**kw).fit(Xs, Ys)
                pred_q = sy.itf(stpn.predict(Xq))
            elif m == "MTGPR":
                kw = dict(DEFAULT_MTGPR) if mtgpr_kwargs is None else dict(mtgpr_kwargs)
                kw["rng"] = np.random.default_rng(seed0 + r)
                pred_q = sy.itf(MTGPR(**kw).fit(Xs, Ys).predict(Xq))
            else:
                pred_q = sy.itf(fit_predict_baseline(m, baselines[m], Xs, Ys, Xq))
            for c in range(NC):
                yc, pc = Yq_raw[:, c], pred_q[:, c]
                rmse, mae, maxe, medae, mape = metrics(yc, pc)
                res[m][c]["RMSE"].append(rmse)
                res[m][c]["MAE"].append(mae)
                res[m][c]["MaxE"].append(maxe)
                res[m][c]["MedAE"].append(medae)
                res[m][c]["MAPE"].append(mape)
                raw[m]["RMSE"][c].append(rmse)
                raw[m]["MAE"][c].append(mae)
        rep_states.append((sup, que, sx, sy))
    return res, raw

if __name__ == "__main__":
    print("Running 20 repeated splits (support=8)...")
    res8, raw8 = run_splits(8)

    # aggregate mean+-std
    agg = {}
    for m in MODELS:
        agg[m] = {}
        for c in range(NC):
            agg[m][targ_names[c]] = {}
            for met in METRIC_NAMES:
                arr = np.array(res8[m][c][met])
                agg[m][targ_names[c]][met] = [float(arr.mean()), float(arr.std())]

    # ---------- ablation ----------
    print("Running support-size ablation...")
    ablation = {}
    for ss in [4, 6, 8, 10, 12]:
        rsub, _ = run_splits(ss)
        ablation[ss] = {}
        for m in MODELS:
            ablation[ss][m] = {}
            for c in range(NC):
                rmse = np.array(rsub[m][c]["RMSE"]).mean()
                ablation[ss][m][targ_names[c]] = float(rmse)

    # ---------- LOOCV ----------
    print("Running LOOCV...")
    loocv = {m: {targ_names[c]: [] for c in range(NC)} for m in MODELS}
    loocv_err = {m: {targ_names[c]: [] for c in range(NC)} for m in MODELS}
    for i in range(N):
        tr = [j for j in range(N) if j != i]
        te = [i]
        Xs_raw, Xq_raw = X_raw[tr], X_raw[te]
        Ys_raw, Yq_raw = Y_raw[tr], Y_raw[te]
        sx, sy = Scaler().fit(Xs_raw), Scaler().fit(Ys_raw)
        Xs, Xq = sx.tf(Xs_raw), sx.tf(Xq_raw)
        Ys = sy.tf(Ys_raw)
        baselines = make_baselines()
        for m in MODELS:
            if m == "ST-PN":
                stpn = STPN(**{**DEFAULT_STPN, "seed": 7}).fit(Xs, Ys)
                pred_q = sy.itf(stpn.predict(Xq))
            elif m == "MTGPR":
                pred_q = sy.itf(MTGPR(**DEFAULT_MTGPR).fit(Xs, Ys).predict(Xq))
            else:
                pred_q = sy.itf(fit_predict_baseline(m, baselines[m], Xs, Ys, Xq))
            for c in range(NC):
                rmse, mae, maxe, medae, mape = metrics(Yq_raw[:, c], pred_q[:, c])
                loocv[m][targ_names[c]].append([rmse, mae, maxe, medae, mape])
                loocv_err[m][targ_names[c]].append(abs(Yq_raw[:, c] - pred_q[:, c])[0])

    loocv_agg = {}
    for m in MODELS:
        loocv_agg[m] = {}
        for c in range(NC):
            arr = np.array(loocv[m][targ_names[c]])
            loocv_agg[m][targ_names[c]] = {met: float(arr[:, k].mean()) for k, met in enumerate(METRIC_NAMES)}

    # ---------- MTGPR hyperparameter sensitivity (support=8, 20 repeats) ----------
    print("Running MTGPR hyperparameter sensitivity...")
    mtgpr_grid = [(ard, rank) for ard in [False, True] for rank in [1, 2, 3]]
    hp_results = {targ_names[c]: [] for c in range(NC)}
    for ard, rank in mtgpr_grid:
        rmses = {c: [] for c in range(NC)}
        for r in range(R):
            rng = np.random.RandomState(100 + r)
            perm = rng.permutation(N)
            sup, que = perm[:8], perm[8:]
            sx, sy = Scaler().fit(X_raw[sup]), Scaler().fit(Y_raw[sup])
            m = MTGPR(rank=rank, ard=ard, maxiter=150, restarts=3,
                      rng=np.random.default_rng(100 + r)).fit(sx.tf(X_raw[sup]), sy.tf(Y_raw[sup]))
            p = sy.itf(m.predict(sx.tf(X_raw[que])))
            for c in range(NC):
                rmses[c].append(float(np.sqrt(((p[:, c] - Y_raw[que, c]) ** 2).mean())))
        row = {"ard": bool(ard), "rank": int(rank)}
        for c in range(NC):
            row[targ_names[c]] = float(np.mean(rmses[c]))
        for c in range(NC):
            hp_results[targ_names[c]].append(row)
        print(f"  MTGPR ard={ard} rank={rank} -> RMSE(cell,hemi,lign)=",
              [round(row[targ_names[c]], 3) for c in range(NC)])

    # ---------- RF feature importance (full data) ----------
    rf = RandomForestRegressor(n_estimators=400, random_state=0)
    sx_full = Scaler().fit(X_raw)
    rf.fit(sx_full.tf(X_raw), Y_raw)
    # per-component importance needs separate RF per component
    rf_imp = {}
    for c in range(NC):
        rfc = RandomForestRegressor(n_estimators=400, random_state=0)
        rfc.fit(sx_full.tf(X_raw), Y_raw[:, c])
        rf_imp[targ_names[c]] = [float(x) for x in rfc.feature_importances_]

    # ---------- route summary ----------
    route_summary = {}
    for rt in route_order:
        mask = routes == rt
        route_summary[rt] = {
            "n": int(mask.sum()),
            "conditions": {
                "Time (min)": [float(X_raw[mask, 0].min()), float(X_raw[mask, 0].max())],
                "Temp (C)": [float(X_raw[mask, 1].min()), float(X_raw[mask, 1].max())],
                "pH": [float(X_raw[mask, 2].min()), float(X_raw[mask, 2].max())],
            },
            "composition_mean": {targ_names[c]: float(Y_raw[mask, c].mean()) for c in range(NC)},
            "composition_std": {targ_names[c]: float(Y_raw[mask, c].std()) for c in range(NC)},
        }

    # ---------- MTGPR coregionalization (full data) for task-correlation figure ----------
    sx_full = Scaler().fit(X_raw)
    sy_full = Scaler().fit(Y_raw)
    mtgpr_full = MTGPR(**DEFAULT_MTGPR).fit(sx_full.tf(X_raw), sy_full.tf(Y_raw))
    B_full, corr_full = mtgpr_full.coregionalization()
    # average predictive std (uncertainty) on a representative support=8 split's query
    rep = np.random.RandomState(0).permutation(N)
    sup, que = rep[:8], rep[8:]
    sxr, syr = Scaler().fit(X_raw[sup]), Scaler().fit(Y_raw[sup])
    mrep = MTGPR(**DEFAULT_MTGPR).fit(sxr.tf(X_raw[sup]), syr.tf(Y_raw[sup]))
    _, vrep = mrep.predict(sxr.tf(X_raw[que]), return_var=True)
    pred_std = (np.sqrt(vrep) * syr.sd).mean(0).tolist()  # back to raw % scale

    # ---------- save ----------
    out = {
        "n_samples": N,
        "feat_names": feat_names,
        "targ_names": targ_names,
        "route_order": route_order,
        "agg_support8": agg,
        "ablation": ablation,
        "loocv": loocv_agg,
        "hp_sensitivity": hp_results,
        "rf_importance": rf_imp,
        "route_summary": route_summary,
        "raw_rmse": {m: [raw8[m]["RMSE"][c] for c in range(NC)] for m in MODELS},
        "raw_mae": {m: [raw8[m]["MAE"][c] for c in range(NC)] for m in MODELS},
        "loocv_err": loocv_err,
        "mtgpr_coreg": B_full.tolist(),
        "mtgpr_corr": corr_full.tolist(),
        "mtgpr_pred_std": pred_std,
        "routes": routes.tolist(),
        "Y_raw": Y_raw.tolist(),
        "X_raw": X_raw.tolist(),
    }
    with open(os.path.join(PROCESS, "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("Saved results.json. Done.")
