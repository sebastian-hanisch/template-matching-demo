"""Zeitliche Struktur statt Nicht-Gaußianität: verzögerte Kovarianzen, AMUSE (ein τ) und SOBI (gemeinsame Diagonalisierung mehrerer τ per Jacobi-Rotationen), numpy von Grund auf.
Die Weißung und FastICA (Vergleichsverfahren) stehen wortgleich in sobi_ica.py (aus ica-demo)."""

from dataclasses import dataclass

import numpy as np

import tm_constants as C
from tm_ica import whiten


def lagged_covariances(Z, lags):
    """Symmetrisierte Kovarianzen der weißen Daten bei den Verzögerungen `lags` (Abtastwerte): R_τ = ½ (E[z(t) z(t+τ)ᵀ] + E[z(t+τ) z(t)ᵀ]). Form (len(lags), nc, nc)."""
    T = Z.shape[1]
    out = []
    for tau in lags:
        R = Z[:, : T - tau] @ Z[:, tau:].T / (T - tau)
        out.append(0.5 * (R + R.T))
    return np.array(out)


def lag_set(n_lags, step):
    """τ = step, 2·step, ..., n_lags·step."""
    return tuple(step * (j + 1) for j in range(n_lags))


def off_diagonality(M):
    """Anteil der Energie außerhalb der Diagonalen, gemittelt über den Stapel: 0 = alle Matrizen diagonal."""
    total = (M ** 2).sum()
    diag = (np.einsum("kii->ki", M) ** 2).sum()
    return float((total - diag) / max(total, 1e-300))


def joint_diagonalize(M, max_sweeps=C.JD_MAX_SWEEPS, tol=C.JD_TOL):
    """Näherungsweise gemeinsame Diagonalisierung symmetrischer Matrizen M (k, n, n) mit einer orthogonalen V: Vᵀ M_k V ≈ diagonal (Jacobi-Rotationen, Cardoso & Souloumiac).
    Rückgabe: (V, Verlauf der Nicht-Diagonalität, größter Rotationswinkel je Sweep, konvergiert, V vor dem ersten und nach jedem Sweep)."""
    M = M.copy()
    n = M.shape[1]
    V = np.eye(n)
    history, angles, rotations = [off_diagonality(M)], [], [V.copy()]
    converged = n == 1
    for _sweep in range(max_sweeps if n > 1 else 0):
        biggest = 0.0
        for p in range(n - 1):
            for q in range(p + 1, n):
                g = np.array([M[:, p, p] - M[:, q, q], M[:, p, q] + M[:, q, p]])
                ton = float(g[0] @ g[0] - g[1] @ g[1])
                toff = float(2.0 * g[0] @ g[1])
                theta = 0.5 * np.arctan2(toff, ton + np.hypot(ton, toff))
                biggest = max(biggest, abs(theta))
                if abs(theta) < 1e-15:
                    continue
                c, s = np.cos(theta), np.sin(theta)
                G = np.array([[c, -s], [s, c]])
                pair = [p, q]
                V[:, pair] = V[:, pair] @ G
                M[:, pair, :] = np.einsum("ij,kjl->kil", G.T, M[:, pair, :])
                M[:, :, pair] = np.einsum("kij,jl->kil", M[:, :, pair], G)
        history.append(off_diagonality(M))
        angles.append(biggest)
        rotations.append(V.copy())
        if biggest < tol:
            converged = True
            break
    return V, history, angles, converged, rotations


@dataclass(frozen=True)
class SOBIModel:
    W: np.ndarray                 # (nc, nc) orthogonale Rotation im weißen Raum
    whitening: object
    lags: tuple
    covariances: np.ndarray       # (len(lags), nc, nc) R_τ der weißen Daten
    history: tuple                # Nicht-Diagonalität vor dem ersten Sweep und nach jedem Sweep
    angles: tuple                 # größter Rotationswinkel je Sweep
    rotations: tuple              # V vor dem ersten und nach jedem Sweep (Wᵀ im weißen Raum)
    converged: bool
    kind: str                     # "sobi" oder "amuse"

    @property
    def unmixing(self):
        return self.W @ self.whitening.K

    @property
    def sources(self):
        return self.W @ self.whitening.Z

    def transform(self, X):
        return self.unmixing @ (X - self.whitening.mean)


def fit_sobi(X, n_components, lags):
    """SOBI: weißen, R_τ für alle τ in `lags`, gemeinsam diagonalisieren. Eine einzige Verzögerung ergibt AMUSE (Eigenzerlegung von R_τ)."""
    wh = whiten(X, n_components)
    R = lagged_covariances(wh.Z, lags)
    V, history, angles, converged, rotations = joint_diagonalize(R)
    return SOBIModel(V.T, wh, tuple(lags), R, tuple(history), tuple(angles), tuple(rotations), converged, "sobi")


def fit_amuse(X, n_components, lag):
    """AMUSE (Tong et al.): Eigenzerlegung der symmetrisierten Kovarianz bei einer einzigen Verzögerung. Eigenwerte = Autokorrelationen der Quellen bei τ; nur eindeutig, wenn sie sich unterscheiden."""
    wh = whiten(X, n_components)
    R = lagged_covariances(wh.Z, (lag,))
    values, vectors = np.linalg.eigh(R[0])
    order = np.argsort(values)[::-1]
    return SOBIModel(vectors[:, order].T, wh, (lag,), R, (off_diagonality(R),), (), (), True, "amuse")


def autocorrelation(S, max_lag):
    """Autokorrelationsfunktion je Zeile von S (Zeile mit Mittel 0 und Varianz 1 vorausgesetzt) für τ = 0..max_lag."""
    T = S.shape[1]
    return np.array([[S[i, : T - tau] @ S[i, tau:] / (T - tau) for tau in range(max_lag + 1)] for i in range(S.shape[0])])


def power_spectrum(S, n_bins=200):
    """Geglättetes Leistungsspektrum je Zeile (Periodogramm-Mittel über Segmente der Länge 2·n_bins), Frequenzen in Hz. Rückgabe: (freqs, (k, n_bins+1))."""
    seg = 2 * n_bins
    n_seg = S.shape[1] // seg
    win = np.hanning(seg)
    spec = np.zeros((S.shape[0], n_bins + 1))
    for j in range(n_seg):
        chunk = S[:, j * seg:(j + 1) * seg] * win
        spec += np.abs(np.fft.rfft(chunk, axis=1)) ** 2
    spec /= max(n_seg, 1)
    return np.fft.rfftfreq(seg, 1.0 / C.SAMPLE_RATE), spec
