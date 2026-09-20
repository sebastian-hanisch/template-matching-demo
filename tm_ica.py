"""Weißung (PCA) und FastICA von Grund auf (numpy): Fixpunktverfahren von Hyvärinen, symmetrisch oder Deflation, drei Kontrastfunktionen."""

from dataclasses import dataclass

import numpy as np

import tm_constants as C


def contrast_g(name, u):
    """(g(u), g'(u)) - Ableitung der Kontrastfunktion G und deren zweite Ableitung."""
    if name == "logcosh":
        t = np.tanh(u)
        return t, 1.0 - t * t
    if name == "exp":
        e = np.exp(-u * u / 2.0)
        return u * e, (1.0 - u * u) * e
    if name == "cube":
        return u ** 3, 3.0 * u * u
    raise ValueError(name)


def contrast_G(name, u):
    if name == "logcosh":
        return np.logaddexp(u, -u) - np.log(2.0)
    if name == "exp":
        return -np.exp(-u * u / 2.0)
    if name == "cube":
        return u ** 4 / 4.0
    raise ValueError(name)


_GAUSS = np.random.default_rng(12345).standard_normal(400_000)


def gaussian_reference(name):
    """E[G(nu)] für eine standardnormale Zufallsvariable (feste Stichprobe, reproduzierbar)."""
    return float(contrast_G(name, _GAUSS).mean())


def non_gaussianity(name, u):
    """J(u) = (E[G(u)] - E[G(nu)])^2 - je größer, desto weniger Gauß'sch (u mit Mittel 0, Varianz 1)."""
    return (float(contrast_G(name, u).mean()) - gaussian_reference(name)) ** 2


@dataclass(frozen=True)
class Whitening:
    mean: np.ndarray              # (n, 1)
    K: np.ndarray                 # (nc, n) Weißungsmatrix
    Z: np.ndarray                 # (nc, T) weiße Daten, Kovarianz = Einheitsmatrix
    eigenvalues: np.ndarray       # alle n Eigenwerte, absteigend


def whiten(X, n_components):
    mean = X.mean(axis=1, keepdims=True)
    Xc = X - mean
    cov = Xc @ Xc.T / X.shape[1]
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    K = (vectors[:, :n_components] / np.sqrt(values[:n_components])).T
    return Whitening(mean, K, K @ Xc, values)


def _sym_decorrelate(W):
    values, vectors = np.linalg.eigh(W @ W.T)
    return vectors @ np.diag(1.0 / np.sqrt(np.maximum(values, 1e-12))) @ vectors.T @ W


@dataclass(frozen=True)
class ICAModel:
    W: np.ndarray                 # (nc, nc) orthogonale Rotation im weißen Raum
    whitening: Whitening
    contrast: str
    method: str
    n_iter: int
    converged: bool
    history: tuple                # W nach jeder Iteration (symmetrisch) bzw. leer

    @property
    def unmixing(self):
        """(nc, n): Schätzquellen = unmixing @ (X - mean)."""
        return self.W @ self.whitening.K

    @property
    def sources(self):
        return self.W @ self.whitening.Z

    def transform(self, X):
        return self.unmixing @ (X - self.whitening.mean)


def fit_ica(X, n_components, contrast=C.DEFAULT_CONTRAST, method=C.DEFAULT_METHOD, init_start=C.DEFAULT_INIT_START, max_iter=C.MAX_ITER, tol=C.TOL, keep_history=True):
    wh = whiten(X, n_components)
    Z = wh.Z
    T = Z.shape[1]
    rng = np.random.default_rng([init_start, 4242])
    W0 = rng.standard_normal((n_components, n_components))
    history, converged, n_iter = [], False, 0
    if method == "symmetric":
        W = _sym_decorrelate(W0)
        for n_iter in range(1, max_iter + 1):
            g, gp = contrast_g(contrast, W @ Z)
            W_new = _sym_decorrelate(g @ Z.T / T - gp.mean(axis=1, keepdims=True) * W)
            delta = np.max(np.abs(np.abs(np.sum(W_new * W, axis=1)) - 1.0))
            W = W_new
            if keep_history:
                history.append(W.copy())
            if delta < tol:
                converged = True
                break
    else:
        W = np.zeros((n_components, n_components))
        converged = True
        total = 0
        for p in range(n_components):
            w = W0[p].copy()
            for k in range(p):
                w -= (w @ W[k]) * W[k]
            w /= np.linalg.norm(w)
            ok = False
            for it in range(1, max_iter + 1):
                g, gp = contrast_g(contrast, w @ Z)
                w_new = Z @ g / T - gp.mean() * w
                for k in range(p):
                    w_new -= (w_new @ W[k]) * W[k]
                w_new /= np.linalg.norm(w_new)
                delta = abs(abs(w_new @ w) - 1.0)
                w = w_new
                if delta < tol:
                    ok = True
                    break
            total = max(total, it)
            converged = converged and ok
            W[p] = w
        n_iter = total
    return ICAModel(W, wh, contrast, method, n_iter, converged, tuple(history))


def pca_components(X, n_components):
    """PCA-Baseline: weiße Hauptkomponenten ohne ICA-Rotation (unkorreliert, aber nicht rotiert)."""
    return whiten(X, n_components).Z
