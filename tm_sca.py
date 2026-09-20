"""Sparse Component Analysis (numpy von Grund auf): aktive Zeitpunkte, Richtungs-Clustering mit Achsen-k-Means (Vorzeichen egal), Rekonstruktion je Zeitpunkt per L1-Minimierung (IRLS)
oder Einzelquellen-Zuordnung. Die Weißung, FastICA und SOBI (Vergleichsverfahren) stehen wortgleich in sca_ica.py und sca_sobi.py."""

from dataclasses import dataclass

import numpy as np

import tm_constants as C

ACTIVE_NOISE_FACTOR = 4.0          # aktiv: Norm des Elektrodenvektors über dem 4-fachen der Rausch-Norm ...
ACTIVE_PEAK_FRACTION = 0.1         # ... mindestens aber 10 % der typischen Spitzen-Norm (sonst würde ein rauschfreies Signal jeden Rest melden)
CORE_PEAK_FRACTION = 0.3           # Richtungen werden nur an den Kernpunkten geschätzt: Norm über 30 % der typischen Spitzen-Norm (Flanken und Nachschlag liegen näher an Überlappung und Rauschen)


def _norms(X):
    return np.linalg.norm(X - np.median(X, axis=1, keepdims=True), axis=0)


def active_samples(X):
    """Maske der Zeitpunkte, an denen mindestens eine Quelle deutlich aktiv ist: ‖x(t)‖ > max(4 · Rausch-Norm, 0.1 · 99.5-%-Quantil der Norm).
    Rausch-Norm = 20-%-Quantil der Normen (die meisten Zeitpunkte sind still)."""
    r = _norms(X)
    noise = np.quantile(r, 0.2)
    threshold = max(ACTIVE_NOISE_FACTOR * noise, ACTIVE_PEAK_FRACTION * np.quantile(r, 0.995))
    return r > threshold


def core_samples(X):
    """Kernpunkte für das Richtungs-Clustering: aktiv UND Norm über 30 % der typischen Spitzen-Norm. Gemessen (5 Neuronen, 2/3/5 Elektroden, 8 feste Datensätze): mit allen aktiven Zeitpunkten verfehlt das Clustering in 6 von 24 Fällen
    eine Mischspalte um mehr als 30 Grad (benachbarte Neuronen verschmelzen), mit den Kernpunkten in keinem."""
    r = _norms(X)
    return active_samples(X) & (r > CORE_PEAK_FRACTION * np.quantile(r, 0.995))


def directions(X, mask):
    """Einheitsvektoren der Elektrodenvektoren an den Zeitpunkten von `mask`, Form (N, n)."""
    V = (X - np.median(X, axis=1, keepdims=True))[:, mask].T
    return V / np.linalg.norm(V, axis=1, keepdims=True)


def _canonical_sign(c):
    """Vorzeichen der Achse festlegen: Summe der Einträge nicht negativ (die Mischspalten der Neuronen sind positiv)."""
    return c if c.sum() >= 0 else -c


@dataclass(frozen=True)
class AxialKMeans:
    centers: np.ndarray           # (k, n) Einheitsachsen, Vorzeichen kanonisch
    labels: np.ndarray            # (N,) Cluster je Richtung
    objective: float              # mittlere |Kosinus-Ähnlichkeit| zur Achse des Clusters (1 = alle Richtungen liegen auf ihren Achsen)
    restart_objectives: tuple     # Zielwert jedes Neustarts (Streuung = Lokale-Minima-Anfälligkeit)
    n_iter: int
    converged: bool


def axial_kmeans(U, k, n_restarts=C.N_RESTARTS, seed=0, max_iter=C.KMEANS_MAX_ITER):
    """Achsen-k-Means: Zuordnung nach maximalem |u·c|, Achse = Hauptvektor der Streumatrix des Clusters; Start per k-means++ auf Achsen; bester von `n_restarts` Läufen."""
    N, n = U.shape
    rng = np.random.default_rng([seed, 777])
    best, objectives = None, []
    for _ in range(n_restarts):
        centers = np.empty((k, n))
        centers[0] = U[rng.integers(N)]
        for j in range(1, k):
            d = np.clip(1.0 - np.max(np.abs(U @ centers[:j].T), axis=1), 0.0, None)
            total = d.sum()
            centers[j] = U[rng.choice(N, p=d / total)] if total > 0 else U[rng.integers(N)]
        labels = np.zeros(N, dtype=int)
        converged = False
        it = 0
        for it in range(1, max_iter + 1):
            sim = np.abs(U @ centers.T)
            new_labels = sim.argmax(axis=1)
            new_centers = centers.copy()
            for j in range(k):
                members = U[new_labels == j]
                if len(members) == 0:                                    # leeres Cluster: an die am schlechtesten erklärte Richtung setzen
                    new_centers[j] = U[np.argmin(sim.max(axis=1))]
                    continue
                values, vectors = np.linalg.eigh(members.T @ members)
                new_centers[j] = vectors[:, -1]
            if it > 1 and np.array_equal(new_labels, labels):
                converged = True
                break
            labels, centers = new_labels, new_centers
        sim = np.abs(U @ centers.T)
        objective = float(sim.max(axis=1).mean())
        objectives.append(objective)
        if best is None or objective > best[0]:
            best = (objective, centers.copy(), sim.argmax(axis=1), it, converged)
    objective, centers, labels, it, converged = best
    centers = np.array([_canonical_sign(c) for c in centers])
    return AxialKMeans(centers, labels, objective, tuple(objectives), it, converged)


def l1_reconstruct(A_hat, V, n_iter=30, ridge=1e-6, decay=0.5):
    """Je Spalte von V (n, N): min ‖s‖₁ s.t. A_hat s = v per IRLS (FOCUSS): s = W Aᵀ (A W Aᵀ + λ I)⁻¹ v mit W = diag(|s| + ε), ε fällt je Iteration um den Faktor `decay`, vektorisiert über alle Zeitpunkte.
    Rückgabe (k, N). Die IRLS-Iteration konvergiert nur langsam gegen die exakte L1-Lösung (Kreuzprüfung gegen ein lineares Programm im Test: 150 Iterationen bei decay 0.85); für die Kennzahlen der Demo genügen 30
    (60-200 Iterationen ändern die Neuronen-Korrelation um höchstens 0.002)."""
    n, k = A_hat.shape
    N = V.shape[1]
    S = np.linalg.pinv(A_hat) @ V                                    # Start: Lösung kleinster Norm
    eye = np.eye(n)
    for it in range(n_iter):
        eps = np.maximum(1e-6, decay ** it * np.abs(S).max(axis=0, keepdims=True))
        w = np.abs(S) + eps                                          # (k, N)
        M = np.einsum("ik,kn,jk->nij", A_hat, w, A_hat) + ridge * eye
        y = np.linalg.solve(M, V.T[:, :, None])[:, :, 0]             # (N, n)
        S = w * (A_hat.T @ y.T)
    return S


def single_reconstruct(A_hat, V):
    """Baseline: jeder Zeitpunkt wird nur der Spalte mit dem größten |Kosinus| zugeordnet, Koeffizient = Projektion. Rückgabe (k, N)."""
    unit = A_hat / np.linalg.norm(A_hat, axis=0, keepdims=True)
    proj = unit.T @ V
    winner = np.abs(unit.T @ (V / np.maximum(np.linalg.norm(V, axis=0, keepdims=True), 1e-12))).argmax(axis=0)
    S = np.zeros((A_hat.shape[1], V.shape[1]))
    S[winner, np.arange(V.shape[1])] = proj[winner, np.arange(V.shape[1])]
    return S


@dataclass(frozen=True)
class SCAModel:
    A_hat: np.ndarray             # (n, k) geschätzte Mischspalten (Norm 1, Vorzeichen kanonisch)
    clustering: AxialKMeans
    mask: np.ndarray              # (T,) aktive Zeitpunkte
    core: np.ndarray              # (T,) Kernpunkte des Clusterings
    noise_variance: float         # Rausch-Varianz je Elektrode (aus den inaktiven Zeitpunkten)
    sources: np.ndarray           # (k, T) geschätzte Quellen (an inaktiven Zeitpunkten 0)
    reconstruction: str


def reconstruct_all(X, A_hat, mask, noise_variance, reconstruction):
    """Quellen (k, T) aus den aktiven Zeitpunkten; an inaktiven Zeitpunkten 0."""
    V = (X - np.median(X, axis=1, keepdims=True))[:, mask]
    S_active = l1_reconstruct(A_hat, V, ridge=max(noise_variance, 1e-9)) if reconstruction == "l1" else single_reconstruct(A_hat, V)
    S = np.zeros((A_hat.shape[1], X.shape[1]))
    S[:, mask] = S_active
    return S


def fit_sca(X, k, reconstruction=C.DEFAULT_RECONSTRUCTION, seed=0, n_restarts=C.N_RESTARTS):
    """SCA: aktive Zeitpunkte -> Richtungen der Kernpunkte -> Achsen-k-Means -> Mischspalten -> Rekonstruktion je aktivem Zeitpunkt (L1 mit Rausch-Regularisierung oder Einzelquelle).
    Bei einer einzigen Elektrode gibt es keine Richtung (alle Spalten gleich)."""
    mask = active_samples(X)
    core = core_samples(X)
    quiet = X[:, ~mask]
    noise_variance = float(np.mean(np.var(quiet, axis=1))) if quiet.shape[1] > 10 else 0.0
    if X.shape[0] == 1 or core.sum() < k:
        A_hat = np.ones((X.shape[0], k)) / np.sqrt(X.shape[0])
        clustering = AxialKMeans(A_hat.T.copy(), np.zeros(int(core.sum()), dtype=int), 0.0, (), 0, False)
        return SCAModel(A_hat, clustering, mask, core, noise_variance, np.zeros((k, X.shape[1])), reconstruction)
    clustering = axial_kmeans(directions(X, core), k, n_restarts, seed)
    A_hat = clustering.centers.T
    return SCAModel(A_hat, clustering, mask, core, noise_variance, reconstruct_all(X, A_hat, mask, noise_variance, reconstruction), reconstruction)


def with_reconstruction(model, X, reconstruction):
    """Dasselbe Clustering, andere Rekonstruktion."""
    if X.shape[0] == 1 or not model.clustering.converged and model.clustering.n_iter == 0:
        return model
    return SCAModel(model.A_hat, model.clustering, model.mask, model.core, model.noise_variance, reconstruct_all(X, model.A_hat, model.mask, model.noise_variance, reconstruction), reconstruction)
