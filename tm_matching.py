"""Vorlagenabgleich (Matching Pursuit, Kilosort-Stil), numpy von Grund auf.

Das Signal wird als Summe verschobener, skalierter Vorlagen erklärt: X(t) ≈ Σ_e a_e · w_{q_e}(t − t_e) + Rauschen. Gierig wird immer die Vorlage abgezogen, die am besten zu dem passt, was vom Signal übrig ist
(Residuum). Dadurch werden auch überlappende Spikes aufgelöst - dort, wo die Standardpipeline (Ausschnitt -> Merkmale -> Cluster) scheitert.
Die Start-Vorlagen kommen aus der Standardpipeline (tm_algorithm.py); die Verfeinerung schätzt sie aus den Residuen neu."""

from dataclasses import dataclass

import numpy as np

import tm_algorithm as alg
import tm_constants as C

L = C.SNIPPET_BEFORE + C.SNIPPET_AFTER          # Länge einer Vorlage in Abtastwerten
MAX_AMPLITUDE = 1.5                             # größte erlaubte Skalierung einer Vorlage (Kilosort begrenzt die Amplitude ebenfalls); die kleinste ist ein Regler
MAX_EVENTS_FACTOR = 0.05                        # höchstens so viele Ereignisse je Abtastwert (Sicherung gegen Endlosschleifen)


def whiten(X):
    """Je Elektrode Median abziehen und durch die robuste Rausch-Standardabweichung teilen (weißes Rauschen -> Einheitsvarianz). Rückgabe (Xw, Median, sigma)."""
    med = np.median(X, axis=1, keepdims=True)
    sigma = alg.noise_sigma(X)
    return (X - med) / sigma[:, None], med, sigma


def cross_correlations(W):
    """XC[q, p, d + L - 1] = Σ_j Σ_l W[q, j, l] · W[p, j, l + d] für alle Verschiebungen d in (-L, L): wie sich die Korrelation der Vorlage q ändert, wenn Vorlage p abgezogen wird."""
    Q = W.shape[0]
    XC = np.zeros((Q, Q, 2 * L - 1))
    for d in range(-(L - 1), L):
        lo, hi = max(0, -d), min(L, L - d)
        XC[:, :, d + L - 1] = np.einsum("qjl,pjl->qp", W[:, :, lo:hi], W[:, :, lo + d: hi + d])
    return XC


def correlation_traces(Xw, W):
    """Z[q, s] = ⟨Xw[:, s : s + L], W[q]⟩ / |W[q]|: wie gut Vorlage q ab Abtastwert s zum Signal passt. Unter reinem, weißem Rauschen ist Z ungefähr standardnormalverteilt."""
    n, T = Xw.shape
    S = max(T - L + 1, 0)
    Z = np.zeros((W.shape[0], S))
    for q in range(W.shape[0]):
        norm = np.sqrt((W[q] ** 2).sum())
        if norm < 1e-12 or S == 0:
            continue
        c = sum(np.correlate(Xw[j], W[q, j], mode="valid") for j in range(n))
        Z[q] = c / norm
    return Z


@dataclass(frozen=True)
class Pursuit:
    times: np.ndarray             # Zeit der Spitze (Start + SNIPPET_BEFORE) je Ereignis, nach Zeit sortiert
    templates: np.ndarray         # Nummer der Vorlage je Ereignis
    amplitudes: np.ndarray        # Skalierung der Vorlage je Ereignis
    scores: np.ndarray            # Prüfgröße z beim Abziehen
    residual: np.ndarray          # (n, T) Signal nach Abziehen aller Ereignisse
    order: np.ndarray             # Reihenfolge, in der die Ereignisse abgezogen wurden (Indizes in die obigen Felder)


def matching_pursuit(Xw, W, threshold, min_amplitude=C.DEFAULT_MIN_AMPLITUDE, max_amplitude=MAX_AMPLITUDE):
    """Gierig: die größte Prüfgröße z über alle Vorlagen und Zeiten wählen, solange z >= threshold; Vorlage mit Amplitude a = c/|w|² abziehen (a < min_amplitude: kein Spike, sondern Rest eines unvollkommen abgezogenen; a > max_amplitude: auf max_amplitude begrenzt);
    die Prüfgrößen der Umgebung exakt aktualisieren (Kreuzkorrelation der Vorlagen)."""
    n, T = Xw.shape
    Q = W.shape[0]
    R = Xw.copy()
    empty = Pursuit(np.zeros(0, dtype=int), np.zeros(0, dtype=int), np.zeros(0), np.zeros(0), R, np.zeros(0, dtype=int))
    if Q == 0 or T < L:
        return empty
    Z = correlation_traces(Xw, W)
    norm = np.sqrt((W ** 2).sum(axis=(1, 2)))
    norm = np.maximum(norm, 1e-12)
    XC = cross_correlations(W) / norm[:, None, None]          # auf z-Einheiten umgerechnet (Zeile q durch |w_q| geteilt)
    S = Z.shape[1]
    lo_a, hi_a = min_amplitude, max_amplitude
    events = []
    max_events = int(MAX_EVENTS_FACTOR * T) + 10
    while len(events) < max_events:
        flat = int(np.argmax(Z))
        q0, s = divmod(flat, S)
        z = Z[q0, s]
        if z < threshold:
            break
        a_raw = float(z / norm[q0])                              # c / |w|² = z / |w|
        if a_raw < lo_a:                                         # zu kleiner Anteil einer Vorlage: Rest eines unvollkommen abgezogenen Spikes, kein eigener Spike
            Z[q0, s] = -np.inf
            continue
        a = min(a_raw, hi_a)
        events.append((s, q0, a, z))
        lo, hi = max(0, s - L + 1), min(S, s + L)
        d = np.arange(lo, hi) - s                                # Verschiebung der betroffenen Startpunkte
        Z[:, lo:hi] -= a * XC[:, q0, d + L - 1]
        Z[q0, s] = -np.inf                                       # dieselbe Vorlage am selben Ort nicht noch einmal (relevant, wenn die Amplitude begrenzt wurde)
    if not events:
        return empty
    ev = np.array(events)
    for s, q0, a, _ in events:
        R[:, int(s): int(s) + L] -= a * W[int(q0)]
    starts = ev[:, 0].astype(int)
    order_time = np.argsort(starts, kind="stable")
    inverse = np.empty_like(order_time)
    inverse[order_time] = np.arange(len(order_time))
    return Pursuit(starts[order_time] + C.SNIPPET_BEFORE, ev[order_time, 1].astype(int), ev[order_time, 2], ev[order_time, 3], R, inverse)


def cleaned_snippets(W, pursuit):
    """Je Ereignis der Ausschnitt ohne alle anderen abgezogenen Spikes (Residuum plus eigener Beitrag), geteilt durch die Amplitude: so sähe der Spike allein aus. Form (Ereignisse, n, L)."""
    if len(pursuit.times) == 0:
        return np.zeros((0, W.shape[1], L))
    starts = pursuit.times - C.SNIPPET_BEFORE
    return np.stack([(pursuit.residual[:, s: s + L] + a * W[q]) / a for s, q, a in zip(starts, pursuit.templates, pursuit.amplitudes)])


def average_templates(W, pursuit):
    """Verfeinerung "neu mitteln": je Vorlage der Mittelwert der bereinigten Ausschnitte ihrer Ereignisse. Vorlagen ohne Ereignisse bleiben unverändert."""
    cleaned = cleaned_snippets(W, pursuit)
    new = W.copy()
    for q in range(W.shape[0]):
        members = pursuit.templates == q
        if members.any():
            new[q] = cleaned[members].mean(axis=0)
    return new


def recluster_templates(W, pursuit, k, seed=0):
    """Verfeinerung "neu clustern": die bereinigten Ausschnitte (ohne Kollisions-Verunreinigung) noch einmal wie in der Standardpipeline clustern (PCA, k-means) - so kann sich auch die Zuordnung ändern,
    nicht nur die Form der Vorlagen. Vorlage = Mittel der bereinigten Ausschnitte je Cluster. Bei zu wenigen Ereignissen bleiben die Vorlagen unverändert."""
    cleaned = cleaned_snippets(W, pursuit)
    if len(cleaned) < max(k, 2) + 2:
        return W
    feats = alg.extract_features(cleaned, "pca", C.DEFAULT_N_COMPONENTS)
    labels = alg.kmeans(feats.values, k, seed=seed).labels
    groups = [cleaned[labels == c].mean(axis=0) for c in range(k) if (labels == c).any()]
    return np.array(groups)


def initial_templates(Xw, sorting):
    """Start-Vorlagen aus den Clustern der Standardpipeline (in Rausch-Einheiten): Mittel der Ausschnitte je Cluster. Rückgabe (Vorlagen (Q, n, L), Clusternummern)."""
    labels = sorting.clustering.labels
    keep = np.array([i for i in range(len(sorting.times))], dtype=int)
    templates, ids = [], []
    for c in range(sorting.k):
        members = keep[labels == c]
        if len(members) == 0:
            continue
        snips = np.stack([Xw[:, t - C.SNIPPET_BEFORE: t - C.SNIPPET_BEFORE + L] for t in sorting.times[members]])
        templates.append(snips.mean(axis=0))
        ids.append(c)
    n = Xw.shape[0]
    return (np.array(templates) if templates else np.zeros((0, n, L))), ids


@dataclass(frozen=True)
class Matching:
    sorting: alg.Sorting          # die Standardpipeline (liefert die Start-Vorlagen und dient als Vergleich)
    Xw: np.ndarray                # (n, T) weiß gemachtes Signal
    sigma: np.ndarray
    templates_start: np.ndarray   # (Q, n, L) Start-Vorlagen
    templates: np.ndarray         # (Q, n, L) Vorlagen nach der Verfeinerung
    history: tuple                # Vorlagen nach jeder Runde (Eintrag 0 = Start)
    pursuit_start: Pursuit        # Ergebnis des Abgleichs mit den Start-Vorlagen
    pursuit: Pursuit              # Ergebnis des letzten Abgleichs
    threshold: float
    min_amplitude: float
    refine: str
    rounds: int


def run_matching(X, k, threshold=C.DEFAULT_MATCH_THRESHOLD, min_amplitude=C.DEFAULT_MIN_AMPLITUDE, refine=C.DEFAULT_REFINE, rounds=C.DEFAULT_ROUNDS, cluster_mode=C.DEFAULT_CLUSTER_MODE, seed=0, sorting=None):
    """Standardpipeline für die Start-Vorlagen, dann Abgleich. Verfeinerung (`refine`, `rounds` Runden): "none" = nur der erste Abgleich, "average" = Vorlagen aus den bereinigten Ausschnitten neu mitteln,
    "recluster" = die bereinigten Ausschnitte neu clustern; nach jeder Runde erneut abgleichen. `sorting` kann eine schon berechnete Pipeline-Sortierung sein."""
    if sorting is None:
        sorting = alg.sort_spikes(X, k, cluster_mode=cluster_mode, seed=seed)
    Xw, _, sigma = whiten(X)
    if len(sorting.times) < 3 or sorting.k == 0:
        W = np.zeros((0, X.shape[0], L))
    else:
        W, _ = initial_templates(Xw, sorting)
    first = matching_pursuit(Xw, W, threshold, min_amplitude)
    pursuit, history, cur = first, [W], W
    for _ in range(rounds if refine != "none" else 0):
        cur = average_templates(cur, pursuit) if refine == "average" else recluster_templates(cur, pursuit, sorting.k, seed)
        pursuit = matching_pursuit(Xw, cur, threshold, min_amplitude)
        history.append(cur)
    return Matching(sorting, Xw, sigma, W, cur, tuple(history), first, pursuit, threshold, min_amplitude, refine, rounds)
