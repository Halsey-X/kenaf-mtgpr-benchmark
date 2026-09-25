"""
MTGPR: Multi-Task Gaussian Process Regression (Intrinsic Coregionalization Model).

Models P correlated targets jointly via a coregionalization kernel
    k((x, t), (x', t')) = B[t, t'] * k_x(x, x')
where B = A A^T is a PSD coregionalization matrix (learned) and k_x is a
shared RBF base kernel over the inputs.  This explicitly captures cross-task
correlation -- the key multi-task structure -- which single-task models ignore.

Fits by maximizing the log marginal likelihood with analytic gradients (L-BFGS-B).
Small-sample friendly (Bayesian) and provides predictive uncertainty.

Interface (matches analysis.py contract):
    m = MTGPR(rank=2, ard=False).fit(Xs, Ys)   # Xs (n,D), Ys (n,P) standardized
    Yq = m.predict(Xq)                          # Xq (m,D) -> Yq (m,P) standardized
"""
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import cholesky, cho_solve, LinAlgError


def _rbf(X1, X2, ls, sf):
    """RBF kernel matrix between X1 (n1,D) and X2 (n2,D): sf^2 * exp(-0.5 * sqdist)."""
    X1s = X1 / ls
    X2s = X2 / ls
    sq = (np.sum(X1s ** 2, 1)[:, None] + np.sum(X2s ** 2, 1)[None, :]
          - 2.0 * (X1s @ X2s.T))
    return sf ** 2 * np.exp(-0.5 * np.maximum(sq, -1e12))


def _rbf_sqdist_components(X1, X2, ls):
    """Return (K0, Dmat) where K0 = exp(-0.5 sqdist) and Dmat[..., d] = dx_d^2."""
    X1s = X1 / ls
    X2s = X2 / ls
    sq = (np.sum(X1s ** 2, 1)[:, None] + np.sum(X2s ** 2, 1)[None, :]
          - 2.0 * (X1s @ X2s.T))
    sq = np.maximum(sq, -1e12)
    # per-dimension squared scaled distance
    n1, n2 = X1.shape[0], X2.shape[0]
    Dmat = np.zeros((X1.shape[1], n1, n2))
    for d in range(X1.shape[1]):
        diff = X1s[:, d][:, None] - X2s[:, d][None, :]
        Dmat[d] = diff ** 2
    return np.exp(-0.5 * sq), Dmat


class MTGPR:
    def __init__(self, rank=2, ard=False, maxiter=120, restarts=3, jitter=1e-6, rng=None):
        self.rank = rank          # rank of coregionalization B = A A^T
        self.ard = ard            # ARD lengthscales (per dim) vs shared
        self.maxiter = maxiter
        self.restarts = restarts
        self.jitter = jitter
        self.rng = np.random.default_rng(0) if rng is None else rng

    # ---- parameter packing ----
    def _n_params(self, D, P):
        return (D if self.ard else 1) + 1 + (P * self.rank) + P

    def _unpack(self, theta, D, P):
        i = 0
        if self.ard:
            ls = np.exp(theta[i:i + D]); i += D
        else:
            ls = np.exp(theta[i]) * np.ones(D); i += 1
        sf = np.exp(theta[i]); i += 1
        A = theta[i:i + P * self.rank].reshape(P, self.rank); i += P * self.rank
        ns = np.exp(theta[i:i + P]); i += P          # noise std per task
        return ls, sf, A, ns

    def _init_theta(self, X, Y):
        D, P = X.shape[1], Y.shape[1]
        if self.ard:
            ls0 = np.log(np.std(X, 0) + 1e-3)
        else:
            ls0 = np.array([np.log(np.std(X) + 1e-3)])
        sf0 = np.array([np.log(np.std(Y) + 1e-3)])
        A0 = self.rng.standard_normal(P * self.rank) * 0.3
        ns0 = np.log(0.1 * (np.std(Y, 0) + 1e-3))
        return np.concatenate([ls0, sf0, A0, ns0])

    # ---- build full (Pn x Pn) training covariance ----
    def _build_K(self, X, ls, sf, A, ns):
        n, P = X.shape[0], A.shape[0]
        Kx = _rbf(X, X, ls, sf)                       # (n,n)
        B = A @ A.T                                    # (P,P)
        K = np.zeros((P * n, P * n))
        for a in range(P):
            for b in range(P):
                K[a * n:(a + 1) * n, b * n:(b + 1) * n] = B[a, b] * Kx
        for a in range(P):
            K[a * n:(a + 1) * n, a * n:(a + 1) * n] += (ns[a] ** 2) * np.eye(n)
        return K, Kx, B

    # ---- negative log marginal likelihood + analytic gradient ----
    def _nll_and_grad(self, theta, X, Y):
        n, D = X.shape[0], X.shape[1]
        P = Y.shape[1]
        ls, sf, A, ns = self._unpack(theta, D, P)
        K, Kx, B = self._build_K(X, ls, sf, A, ns)
        K = K + self.jitter * np.eye(K.shape[0])
        try:
            L = cholesky(K, lower=True)
        except LinAlgError:
            return 1e12, np.zeros_like(theta)
        yvec = Y.T.reshape(-1)                         # task-major
        alpha = cho_solve((L, True), yvec)
        # log|K| via Cholesky diagonal
        logdet = 2.0 * np.sum(np.log(np.diag(L)))
        nll = 0.5 * yvec @ alpha + 0.5 * logdet + 0.5 * K.shape[0] * np.log(2 * np.pi)
        # G = K^{-1} - alpha alpha^T   (correct sign for d NLL / d theta)
        Kinv = cho_solve((L, True), np.eye(K.shape[0]))
        G = Kinv - np.outer(alpha, alpha)

        grad = np.zeros_like(theta)
        # lengthscales (theta = log ls ; the 1/ls cancels -> dK/dtheta = Kx * Dmat)
        _, Dmat = _rbf_sqdist_components(X, X, ls)
        if self.ard:
            for d in range(D):
                dKx = Kx * Dmat[d]
                dK = np.zeros_like(K)
                for a in range(P):
                    for b in range(P):
                        dK[a * n:(a + 1) * n, b * n:(b + 1) * n] = B[a, b] * dKx
                grad[d] = 0.5 * np.sum(G * dK)
        else:
            dKx = Kx * np.sum(Dmat, 0)
            dK = np.zeros_like(K)
            for a in range(P):
                for b in range(P):
                    dK[a * n:(a + 1) * n, b * n:(b + 1) * n] = B[a, b] * dKx
            grad[0] = 0.5 * np.sum(G * dK)
        # signal variance (theta = log sf ; dK/dtheta_sf = 2 * K_signal)
        gi = (D if self.ard else 1)
        Ksig = K.copy()
        for a in range(P):
            Ksig[a * n:(a + 1) * n, a * n:(a + 1) * n] -= (ns[a] ** 2) * np.eye(n)
        grad[gi] = np.sum(G * Ksig)
        # A matrix
        ai = gi + 1
        for q in range(self.rank):
            for p in range(P):
                dK = np.zeros_like(K)
                for b in range(P):
                    dK[p * n:(p + 1) * n, b * n:(b + 1) * n] += A[b, q] * Kx
                for a in range(P):
                    dK[a * n:(a + 1) * n, p * n:(p + 1) * n] += A[a, q] * Kx
                grad[ai + p * self.rank + q] = 0.5 * np.sum(G * dK)
        # noise std
        ni = ai + P * self.rank
        for a in range(P):
            dK = np.zeros_like(K)
            dK[a * n:(a + 1) * n, a * n:(a + 1) * n] += (2.0 * ns[a] ** 2) * np.eye(n)
            grad[ni + a] = 0.5 * np.sum(G * dK)
        return nll, grad

    def fit(self, X, Y):
        X = np.asarray(X, float); Y = np.asarray(Y, float)
        self.D = X.shape[1]; self.P = Y.shape[1]
        best_theta, best_nll = None, 1e18
        bounds = []
        if self.ard:
            bounds += [(-2.0, 2.0)] * self.D
        else:
            bounds += [(-2.0, 2.0)]
        bounds += [(-3.0, 3.0)]                     # log sf
        bounds += [(-3.0, 3.0)] * (self.P * self.rank)  # A
        bounds += [(-5.0, 1.0)] * self.P            # log noise std
        for r in range(self.restarts):
            theta0 = self._init_theta(X, Y)
            if r > 0:
                theta0 = theta0 + self.rng.standard_normal(theta0.shape) * 0.2
            try:
                res = minimize(self._nll_and_grad, theta0, args=(X, Y), jac=True,
                               method="L-BFGS-B", bounds=bounds,
                               options={"maxiter": self.maxiter})
                if res.fun < best_nll:
                    best_nll, best_theta = res.fun, res.x
            except Exception:
                continue
        if best_theta is None:
            best_theta = self._init_theta(X, Y)
        self.theta = best_theta
        ls, sf, A, ns = self._unpack(best_theta, self.D, self.P)
        self.ls, self.sf, self.A, self.ns = ls, sf, A, ns
        self.B = A @ A.T
        # cache for prediction
        K, _, _ = self._build_K(X, ls, sf, A, ns)
        K = K + self.jitter * np.eye(K.shape[0])
        self.L = cholesky(K, lower=True)
        self.yvec = Y.T.reshape(-1)
        self.Xtr = X
        self.fitted = True
        return self

    def predict(self, Xq, return_var=False):
        Xq = np.asarray(Xq, float)
        m = Xq.shape[0]
        n, P = self.Xtr.shape[0], self.P
        Kx_qt = _rbf(Xq, self.Xtr, self.ls, self.sf)    # (m,n)
        Ks = np.zeros((P * m, P * n))
        for a in range(P):
            for b in range(P):
                Ks[a * m:(a + 1) * m, b * n:(b + 1) * n] = self.B[a, b] * Kx_qt
        alpha = cho_solve((self.L, True), self.yvec)
        mean_vec = Ks @ alpha
        mean = mean_vec.reshape(P, m).T                   # (m,P)
        if not return_var:
            return mean
        # predictive variance (diagonal per task)
        v = np.zeros((m, P))
        for a in range(P):
            Kss = _rbf(Xq, Xq, self.ls, self.sf)         # (m,m), shared
            ks = Ks[a * m:(a + 1) * m, :]                # (m, Pn)
            cov = Kss + self.ns[a] ** 2 * np.eye(m) - ks @ cho_solve((self.L, True), ks.T)
            v[:, a] = np.clip(np.diag(cov), 0, None)
        return mean, v

    def coregionalization(self):
        """Return B and its normalized correlation matrix."""
        B = self.B
        d = np.sqrt(np.diag(B))
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = B / np.outer(d, d)
        corr = np.where(np.isfinite(corr), corr, 0.0)
        return B, corr
