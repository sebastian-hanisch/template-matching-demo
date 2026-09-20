"""Die Standardpipeline des Spike-Sortings (numpy von Grund auf): Detektion per Schwelle, Ausschnitte, Merkmale (PCA, Rohwerte, Spitzenamplituden), k-means und Silhouette-Wahl der Clusterzahl.
Die Vergleichsverfahren ICA, SOBI und SCA stehen wortgleich in tm_ica.py, tm_sobi.py und tm_sca.py."""

from dataclasses import dataclass

import numpy as np

import tm_constants as C

NOISE_FLOOR_FRACTION = 0.002       # Rauschschätzung mindestens 0.2 % der größten Amplitude (sonst wäre die Schwelle bei rauschfreien Daten 0)
SILHOUETTE_MAX_POINTS = 1200       # Silhouette auf höchstens so vielen (zufällig gezogenen) Punkten berechnen


def noise_sigma(X):
    """Robuste Rausch-Standardabweichung je Elektrode: MAD / 0.6745 (Spitzen sind selten und beeinflussen den Median kaum)."""
    med = np.median(X, axis=1, keepdims=True)
    mad = np.median(np.abs(X - med), axis=1) / 0.6745
    return np.maximum(np.maximum(mad, NOISE_FLOOR_FRACTION * np.abs(X - med).max(axis=1)), 1e-12)


def detection_signal(X):
    """d(t) = min_j (x_j(t) - Median_j) / sigma_j: die negativste Auslenkung über alle Elektroden, in Rausch-Standardabweichungen."""
    med = np.median(X, axis=1, keepdims=True)
    return ((X - med) / noise_sigma(X)[:, None]).min(axis=0)


def detect(X, threshold=C.DEFAULT_THRESHOLD, dead_time=C.DEAD_TIME):
    """Zeitpunkte der Minima von d(t) unter -threshold; tiefste zuerst, Mindestabstand `dead_time`. Rückgabe (sortierte Zeiten, d(t))."""
    d = detection_signal(X)
    candidates = np.flatnonzero(d < -threshold)
    taken = np.zeros(len(d), bool)
    times = []
    for t in candidates[np.argsort(d[candidates])]:
        if taken[max(0, t - dead_time): t + dead_time + 1].any():
            continue
        taken[t] = True
        times.append(t)
    return np.sort(np.array(times, dtype=int)), d


def snippets(X, times, before=C.SNIPPET_BEFORE, after=C.SNIPPET_AFTER):
    """Ausschnitte X[:, t - before : t + after] am Minimum ausgerichtet; Spikes zu nah am Rand entfallen. Rückgabe (Zeiten der behaltenen Spikes, (N, n, before + after))."""
    T = X.shape[1]
    keep = (times - before >= 0) & (times + after <= T)
    times = times[keep]
    return times, np.stack([X[:, t - before: t + after] for t in times]) if len(times) else np.zeros((0, X.shape[0], before + after))


@dataclass(frozen=True)
class Features:
    values: np.ndarray            # (N, d) Merkmale je Spike
    kind: str
    explained: np.ndarray         # Varianzanteil der berechneten Hauptkomponenten (nur bei PCA, sonst leer)
    components: np.ndarray        # (p, n * L) Hauptachsen (nur bei PCA)
    mean: np.ndarray              # Mittelwert der Ausschnitte (nur bei PCA)


def pca_features(snips, n_components):
    """Hauptkomponenten der zentrierten, flachgezogenen Ausschnitte per SVD; Vorzeichen: größte Komponente je Achse positiv."""
    flat = snips.reshape(len(snips), -1)
    mean = flat.mean(axis=0)
    U, s, Vt = np.linalg.svd(flat - mean, full_matrices=False)
    p = min(n_components, len(s))
    signs = np.sign(Vt[np.arange(p), np.abs(Vt[:p]).argmax(axis=1)])
    signs[signs == 0] = 1.0
    Vt = Vt[:p] * signs[:, None]
    explained = s[:p] ** 2 / max((s ** 2).sum(), 1e-300)
    return Features((flat - mean) @ Vt.T, "pca", explained, Vt, mean)


def extract_features(snips, kind, n_components=C.DEFAULT_N_COMPONENTS):
    if kind == "pca":
        return pca_features(snips, n_components)
    empty = np.zeros(0)
    if kind == "raw":
        return Features(snips.reshape(len(snips), -1), "raw", empty, np.zeros((0, 0)), empty)
    return Features(snips.min(axis=2), "amplitude", empty, np.zeros((0, 0)), empty)


def squared_distances(A, B):
    """Quadrierte euklidische Abstände aller Zeilen von A zu allen Zeilen von B (über |a|² - 2ab + |b|², schnell auch bei vielen Merkmalen)."""
    d2 = (A ** 2).sum(axis=1)[:, None] - 2.0 * A @ B.T + (B ** 2).sum(axis=1)[None, :]
    return np.maximum(d2, 0.0)


@dataclass(frozen=True)
class KMeans:
    centers: np.ndarray
    labels: np.ndarray
    inertia: float
    restart_inertias: tuple
    n_iter: int


def kmeans(F, k, n_restarts=C.N_RESTARTS, seed=0, max_iter=C.KMEANS_MAX_ITER):
    """k-means (Lloyd) mit k-means++-Start und mehreren Neustarts; das Ergebnis mit der kleinsten Streuung innerhalb der Cluster zählt."""
    N = len(F)
    k = max(1, min(k, N))
    rng = np.random.default_rng([seed, 991])
    best, inertias = None, []
    for _ in range(n_restarts):
        centers = np.empty((k, F.shape[1]))
        centers[0] = F[rng.integers(N)]
        for j in range(1, k):
            d2 = squared_distances(F, centers[:j]).min(axis=1)
            total = d2.sum()
            centers[j] = F[rng.choice(N, p=d2 / total)] if total > 0 else F[rng.integers(N)]
        labels = None
        it = 0
        for it in range(1, max_iter + 1):
            d2 = squared_distances(F, centers)
            new = d2.argmin(axis=1)
            if labels is not None and np.array_equal(new, labels):
                break
            labels = new
            for j in range(k):
                members = F[labels == j]
                centers[j] = members.mean(axis=0) if len(members) else F[d2.min(axis=1).argmax()]
        inertia = float(((F - centers[labels]) ** 2).sum())
        inertias.append(inertia)
        if best is None or inertia < best[0]:
            best = (inertia, centers.copy(), labels.copy(), it)
    inertia, centers, labels, it = best
    order = np.argsort(centers[:, 0])                                   # Reihenfolge der Cluster festlegen (nach dem ersten Merkmal)
    remap = np.empty(k, dtype=int)
    remap[order] = np.arange(k)
    return KMeans(centers[order], remap[labels], inertia, tuple(inertias), it)


def silhouette(F, labels, seed=0, max_points=SILHOUETTE_MAX_POINTS):
    """Mittlere Silhouette (Rousseeuw): (b - a) / max(a, b) mit a = mittlerer Abstand im eigenen Cluster, b = zum nächsten fremden Cluster; bei mehr als max_points Punkten auf einer festen Stichprobe."""
    N = len(F)
    if len(np.unique(labels)) < 2:
        return 0.0
    if N > max_points:
        idx = np.random.default_rng([seed, 5]).choice(N, max_points, replace=False)
        F, labels = F[idx], labels[idx]
    D = np.sqrt(squared_distances(F, F))
    a = np.zeros(len(F))
    b = np.full(len(F), np.inf)
    singleton = np.zeros(len(F), bool)
    for k in np.unique(labels):
        mask = labels == k
        n_k = int(mask.sum())
        to_k = D[:, mask].sum(axis=1)                                       # Summe der Abstände zu den Mitgliedern von Cluster k
        if n_k > 1:
            a[mask] = to_k[mask] / (n_k - 1)                                # eigener Cluster: ohne den Punkt selbst
        else:
            singleton[mask] = True
        b[~mask] = np.minimum(b[~mask], to_k[~mask] / n_k)
    s = (b - a) / np.maximum(np.maximum(a, b), 1e-300)
    s[singleton] = 0.0                                                      # Konvention: Cluster aus einem Punkt zählen 0
    return float(s.mean())


def choose_k(F, k_max=C.K_MAX, seed=0, n_restarts=C.N_RESTARTS):
    """Clusterzahl mit der größten mittleren Silhouette unter k = 2 ... k_max. Rückgabe (k, Silhouette je k)."""
    curve = {}
    for k in range(2, min(k_max, len(F) - 1) + 1):
        curve[k] = silhouette(F, kmeans(F, k, n_restarts, seed).labels, seed)
    if not curve:
        return 1, {}
    best = max(curve, key=curve.get)
    return best, curve


@dataclass(frozen=True)
class Sorting:
    times: np.ndarray             # erkannte Spikezeiten (nach Randfilter)
    all_times: np.ndarray         # erkannte Spikezeiten vor dem Randfilter
    detection: np.ndarray         # d(t)
    snippets: np.ndarray          # (N, n, L)
    features: Features
    clustering: KMeans
    k: int
    silhouette_curve: dict        # k -> Silhouette (nur bei unbekannter Clusterzahl)
    threshold: float


def sort_spikes(X, k, threshold=C.DEFAULT_THRESHOLD, feature=C.DEFAULT_FEATURE, n_components=C.DEFAULT_N_COMPONENTS, cluster_mode=C.DEFAULT_CLUSTER_MODE, seed=0):
    """Die Pipeline: Detektion -> Ausschnitte -> Merkmale -> k-means. `k` ist die (bekannte) Clusterzahl; bei cluster_mode = "silhouette" wird sie gewählt."""
    all_times, d = detect(X, threshold)
    times, snips = snippets(X, all_times)
    if len(times) < 3:
        empty = Features(np.zeros((0, 1)), feature, np.zeros(0), np.zeros((0, 0)), np.zeros(0))
        return Sorting(times, all_times, d, snips, empty, KMeans(np.zeros((0, 1)), np.zeros(len(times), dtype=int), 0.0, (), 0), 0, {}, threshold)
    feats = extract_features(snips, feature, n_components)
    curve = {}
    if cluster_mode == "silhouette":
        k, curve = choose_k(feats.values, seed=seed)
    clustering = kmeans(feats.values, k, seed=seed)
    return Sorting(times, all_times, d, snips, feats, clustering, len(clustering.centers), curve, threshold)
