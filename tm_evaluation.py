"""Auswertung der Vorlagenabgleich-Demo: Zuordnung, Spitzen-Treffer (aus ica/sobi/sca-demo übernommen), Sortiergüte von Pipeline und Vorlagenabgleich mit derselben Auswertung (Detektion, Sortiergenauigkeit,
Kollisionen, Geisterereignisse), Orakel-Vorlagen als Obergrenze, Sweeps, Vergleichstabellen und das Urteil."""

from dataclasses import dataclass

import numpy as np

import tm_algorithm as alg
import tm_constants as C
import tm_ica as ica
import tm_matching as tmm
import tm_sca as sca
import tm_scenario as sc
import tm_sobi as sobi


@dataclass(frozen=True)
class Settings:
    threshold: float = C.DEFAULT_MATCH_THRESHOLD
    min_amplitude: float = C.DEFAULT_MIN_AMPLITUDE
    refine: str = C.DEFAULT_REFINE
    rounds: int = C.DEFAULT_ROUNDS
    cluster_mode: str = C.DEFAULT_CLUSTER_MODE
    contrast: str = C.DEFAULT_CONTRAST
    init_start: int = C.DEFAULT_INIT_START


def make_dataset(m=C.DEFAULT_N_NEURONS, n=C.DEFAULT_N_ELECTRODES, rate_scale=C.DEFAULT_RATE_SCALE, similarity=C.DEFAULT_SIMILARITY, jitter=C.DEFAULT_JITTER, noise=C.DEFAULT_NOISE,
                 n_samples=C.DEFAULT_N_SAMPLES, seed=C.DEFAULT_SEED):
    return sc.make_dataset(m, n, rate_scale, similarity, jitter, noise, n_samples, seed)


def n_components(ds):
    """Die Zahl der Quellen wird als bekannt angenommen (Standardannahme von FastICA und SOBI); bei weniger Elektroden als Quellen bleibt nur n."""
    return min(ds.n_electrodes, ds.S.shape[0])


def correlation_matrix(S, estimates):
    """|Korrelation| (k, nc) zwischen wahren Quellen und Schätzungen."""
    a = (S - S.mean(axis=1, keepdims=True)) / S.std(axis=1, keepdims=True)
    b = estimates - estimates.mean(axis=1, keepdims=True)
    sd = b.std(axis=1, keepdims=True)
    b = b / np.where(sd > 0, sd, 1.0)
    return np.abs(a @ b.T) / S.shape[1]


def assign(C_abs):
    """Optimale Zuordnung Quelle -> Schätzung (jede Schätzung höchstens einer Quelle), Summe der |Korrelationen| maximal. Exakt per Bitmasken-DP.
    Rückgabe: Liste je Quelle mit dem Index der Schätzung oder -1 (nicht zugeordnet, nur wenn es weniger Schätzungen als Quellen gibt)."""
    k, nc = C_abs.shape
    best = {}

    def solve(row, used):
        if row == k:
            return 0.0, ()
        key = (row, used)
        if key in best:
            return best[key]
        result = (solve(row + 1, used)[0], (-1,) + solve(row + 1, used)[1])
        for j in range(nc):
            if not used >> j & 1:
                value, rest = solve(row + 1, used | 1 << j)
                if value + C_abs[row, j] > result[0] + 1e-12:
                    result = (value + C_abs[row, j], (j,) + rest)
        best[key] = result
        return result

    return list(solve(0, 0)[1])


def matched(S, estimates):
    """Zuordnung + Vorzeichen-/Skalenkorrektur per Regression: (Zuordnung, |Korrelation| je Quelle, Schätzquellen in Skala und Vorzeichen der wahren Quellen)."""
    cm = correlation_matrix(S, estimates)
    idx = assign(cm)
    corr = np.array([cm[i, j] if j >= 0 else 0.0 for i, j in enumerate(idx)])
    aligned = np.zeros_like(S)
    for i, j in enumerate(idx):
        if j >= 0:
            e = estimates[j] - estimates[j].mean()
            aligned[i] = (e @ (S[i] - S[i].mean()) / (e @ e)) * e + S[i].mean()
    return idx, corr, aligned


def sir_db(corr):
    """Signal-zu-Interferenz in dB aus der Korrelation: rho^2 / (1 - rho^2)."""
    r2 = np.clip(np.asarray(corr) ** 2, 1e-9, 1.0 - 1e-9)
    return 10.0 * np.log10(r2 / (1.0 - r2))


def amari_index(P):
    """Amari-Index einer (nc, k)-Matrix P = Entmischung x Mischung; 0 = perfekt (nur Permutation und Skalierung), 1 = schlechtestmöglich; verallgemeinert auf nicht quadratische P."""
    P = np.abs(P)
    k = P.shape[1]
    rows = (P.sum(axis=1) / P.max(axis=1) - 1.0).sum()
    cols = (P.sum(axis=0) / P.max(axis=0) - 1.0).sum()
    d = max(k, P.shape[0])
    return float((rows + cols) / (2.0 * d * (d - 1))) if d > 1 else 0.0


# --- Spike-Erkennung auf den Schätzquellen ------------------------------------------------------------------------------------
DETECT_MIN_SEPARATION = 15         # Abtastwerte zwischen zwei erkannten Spitzen
DETECT_TOLERANCE = 4               # erlaubte Abweichung zur wahren Spitze
DETECT_SIGMA_FACTOR = 4.0          # Schwelle: 4 robuste Standardabweichungen (MAD) ...
DETECT_DEPTH_FRACTION = 0.3        # ... mindestens aber 30 % der typischen Spitzentiefe (sonst würde ein rauschfreies Signal jeden Rest melden)


def detect_spikes(x):
    """Negative Spitzen von x (Vorzeichen bereits wie beim Neuron): tiefste zuerst, Mindestabstand; Schwelle max(4 sigma_MAD, 0.3 x typische Tiefe); typische Tiefe = Median der (höchstens) 10 tiefsten Spitzen."""
    sigma = np.median(np.abs(x - np.median(x))) / 0.6745
    threshold = -DETECT_SIGMA_FACTOR * sigma
    taken = np.zeros(len(x), bool)
    peaks = []
    for t in np.argsort(x):
        if x[t] > threshold:
            break
        if taken[max(0, t - DETECT_MIN_SEPARATION): t + DETECT_MIN_SEPARATION + 1].any():
            continue
        taken[t] = True
        peaks.append(t)
        if len(peaks) == 10:                                             # typische Tiefe = Median der 10 tiefsten Spitzen (ab hier gilt die Schwelle laufend)
            threshold = min(threshold, DETECT_DEPTH_FRACTION * float(np.median(x[peaks])))
    if 0 < len(peaks) < 10:                                              # weniger als 10 Spitzen: typische Tiefe = Median der gefundenen
        threshold = min(threshold, DETECT_DEPTH_FRACTION * float(np.median(x[peaks])))
        peaks = [t for t in peaks if x[t] <= threshold]
    return np.sort(np.array(peaks, dtype=int))


def spike_f1(detected, true):
    """F1 der erkannten gegen die wahren Spitzenzeiten (Zuordnung je wahrer Spitze zur nächsten, jede erkannte höchstens einmal)."""
    if len(detected) == 0 or len(true) == 0:
        return 0.0
    used = np.zeros(len(detected), bool)
    tp = 0
    for t in true:
        d = np.abs(detected - t)
        j = int(np.argmin(d))
        if d[j] <= DETECT_TOLERANCE and not used[j]:
            used[j] = True
            tp += 1
    if tp == 0:
        return 0.0
    precision, recall = tp / len(detected), tp / len(true)
    return 2 * precision * recall / (precision + recall)



def excess_kurtosis(x):
    x = np.asarray(x, dtype=float)
    x = (x - x.mean(axis=-1, keepdims=True)) / x.std(axis=-1, keepdims=True)
    return (x ** 4).mean(axis=-1) - 3.0








# --- Zuordnung erkannter und wahrer Spikes ---------------------------------------------------------------------------------------------


def truth_spikes(ds):
    """Alle wahren Spikes nach Zeit sortiert: (Zeiten der negativen Spitze, Neuron, Kollision). Kollision = ein anderer Spike beginnt höchstens COLLISION_WINDOW Abtastwerte vor oder nach diesem."""
    times = np.concatenate(ds.spike_times)
    neuron = np.concatenate([np.full(len(t), i) for i, t in enumerate(ds.spike_times)])
    starts = np.concatenate(ds.spike_starts)
    order = np.argsort(times, kind="stable")
    times, neuron, starts = times[order], neuron[order], starts[order]
    all_starts = np.sort(starts)
    left = np.searchsorted(all_starts, starts - C.COLLISION_WINDOW, side="left")
    right = np.searchsorted(all_starts, starts + C.COLLISION_WINDOW, side="right")
    return times, neuron, (right - left) > 1


def match_detections(detected, truth_times, tolerance=C.MATCH_TOLERANCE):
    """Je erkannter Spitze der Index der nächsten noch freien wahren Spitze innerhalb `tolerance`, sonst -1 (jede wahre Spitze höchstens einmal)."""
    out = np.full(len(detected), -1, dtype=int)
    used = np.zeros(len(truth_times), bool)
    for i, t in enumerate(detected):
        lo, hi = int(np.searchsorted(truth_times, t - tolerance, side="left")), int(np.searchsorted(truth_times, t + tolerance, side="right"))
        best, best_d = -1, tolerance + 1
        for c in range(lo, hi):
            if not used[c] and abs(int(truth_times[c]) - int(t)) < best_d:
                best, best_d = c, abs(int(truth_times[c]) - int(t))
        if best >= 0 and best_d <= tolerance:
            out[i] = best
            used[best] = True
    return out


@dataclass(frozen=True)
class EventResult:
    """Sortiergüte einer Liste von Ereignissen (Zeit, Cluster/Vorlage); für die Pipeline und den Vorlagenabgleich dieselbe Auswertung."""
    recall: float                 # Anteil der wahren Spikes, die erkannt wurden
    precision: float              # Anteil der erkannten Ereignisse, die zu einem wahren Spike gehören
    n_detected: int
    n_true: int
    n_ghosts: int                 # erkannte Ereignisse ohne wahren Spike ("Geister")
    missed_collision_share: float  # Anteil der verpassten Spikes, die Kollisionen sind
    accuracy: float               # richtig sortierte Anteil der erkannten wahren Spikes
    accuracy_single: float
    accuracy_collision: float
    confusion: np.ndarray         # (m, k) Zahl erkannter wahrer Spikes je Neuron und Cluster
    cluster_of_neuron: list       # je Neuron das zugeordnete Cluster (-1 = keins)
    neuron_of_event: np.ndarray   # je Ereignis das Neuron seines Clusters (-1 = Cluster ohne Neuron)
    truth_of_event: np.ndarray    # je Ereignis das wahre Neuron (-1 = falsch erkannt)
    collision_of_event: np.ndarray
    f1: float                     # mittlerer Spitzen-F1 der Neuronen (Ereignisse seines Clusters gegen die wahren Spitzenzeiten)
    f1_per_neuron: np.ndarray
    collision_share: float        # Anteil der wahren Spikes, die Kollisionen sind
    majority_baseline: float      # Genauigkeit, wenn jeder Spike dem häufigsten Neuron zugeordnet würde


def evaluate_events(ds, times, labels, k):
    m = ds.n_neurons
    times = np.asarray(times, dtype=int)
    labels = np.asarray(labels, dtype=int)
    k = max(int(k), 1)
    tt, tn, tc = truth_spikes(ds)
    mi = match_detections(times, tt)
    ok = mi >= 0
    matched_truth = np.full(len(times), -1)
    matched_truth[ok] = tn[mi[ok]]
    collision = np.zeros(len(times), bool)
    collision[ok] = tc[mi[ok]]
    confusion = np.zeros((m, k))
    for lab, y in zip(labels[ok], tn[mi[ok]]):
        confusion[y, lab] += 1
    idx = assign(confusion) if ok.any() else [-1] * m
    cmap = {c: i for i, c in enumerate(idx) if c >= 0}
    neuron_of_event = np.array([cmap.get(int(l), -1) for l in labels], dtype=int) if len(labels) else np.zeros(0, dtype=int)
    correct = (neuron_of_event == matched_truth) & ok
    n_ok = max(int(ok.sum()), 1)
    single, coll = ok & ~collision, ok & collision
    missed = np.ones(len(tt), bool)
    missed[mi[ok]] = False
    f1s = []
    for i in range(m):
        c = idx[i]
        detected_i = times[labels == c] if c >= 0 else np.array([], dtype=int)
        f1s.append(spike_f1(detected_i, ds.spike_times[i]))
    counts = np.bincount(tn, minlength=m)
    return EventResult(
        recall=float(ok.sum() / max(len(tt), 1)), precision=float(ok.sum() / max(len(times), 1)), n_detected=int(len(times)), n_true=int(len(tt)), n_ghosts=int((~ok).sum()),
        missed_collision_share=float(tc[missed].mean()) if missed.any() else 0.0,
        accuracy=float(correct.sum() / n_ok), accuracy_single=float(correct[single].sum() / max(single.sum(), 1)) if single.any() else float("nan"),
        accuracy_collision=float(correct[coll].sum() / max(coll.sum(), 1)) if coll.any() else float("nan"),
        confusion=confusion, cluster_of_neuron=list(idx), neuron_of_event=neuron_of_event, truth_of_event=matched_truth, collision_of_event=collision,
        f1=float(np.mean(f1s)), f1_per_neuron=np.array(f1s), collision_share=float(tc.mean()), majority_baseline=float(counts.max() / max(counts.sum(), 1)))


def evaluate_pursuit(ds, pursuit, n_templates):
    return evaluate_events(ds, pursuit.times, pursuit.templates, max(n_templates, 1))


def evaluate_pipeline(ds, sorting):
    return evaluate_events(ds, sorting.times, sorting.clustering.labels, sorting.k)


def oracle_templates(ds):
    """Die wahren Vorlagen in Rausch-Einheiten (nur als Messreferenz - in der Praxis unbekannt): Mittel des rauschfreien Signals über die Spikes eines Neurons, die nicht mit einem anderen überlappen."""
    med = np.median(ds.X, axis=1, keepdims=True)
    Xc = (ds.X_clean - med) / alg.noise_sigma(ds.X)[:, None]
    tt, tn, tc = truth_spikes(ds)
    L = tmm.L
    T = Xc.shape[1]
    out = []
    for i in range(ds.n_neurons):
        sel = (tn == i) & ~tc
        if sel.sum() < 2:
            sel = tn == i
        snips = [Xc[:, t - C.SNIPPET_BEFORE: t - C.SNIPPET_BEFORE + L] for t in tt[sel] if t - C.SNIPPET_BEFORE >= 0 and t - C.SNIPPET_BEFORE + L <= T]
        out.append(np.mean(snips, axis=0) if snips else np.zeros((ds.n_electrodes, L)))
    return np.array(out)


# --- Vergleichsverfahren: Spitzen-F1 mit derselben Definition -----------------------------------------------------------------------


def comparator_f1(ds, settings):
    """Mittlerer Spitzen-F1 der Neuronen für ICA, SOBI und SCA auf denselben Daten (Schätzung -> Zuordnung per Korrelation -> Schwelle -> Treffer), dazu die Neuronen-Korrelation."""
    m = ds.n_neurons
    nc = n_components(ds)
    out = {}
    estimates = {
        "ica": ica.fit_ica(ds.X, nc, settings.contrast, C.DEFAULT_METHOD, settings.init_start).sources,
        "sobi": sobi.fit_sobi(ds.X, nc, C.SOBI_LAGS).sources,
        "sca": sca.fit_sca(ds.X, m, "l1", seed=settings.init_start).sources,
    }
    for name, est in estimates.items():
        idx, corr, aligned = matched(ds.S, est)
        out[name] = {"f1": float(np.mean([spike_f1(detect_spikes(aligned[i]), ds.spike_times[i]) for i in range(m)])), "corr": float(corr[:m].mean())}
    return out


# --- Gesamtanalyse ------------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Analysis:
    ds: sc.Dataset
    settings: Settings
    sorting: alg.Sorting          # die Standardpipeline (Start-Vorlagen und Vergleich)
    pipe: EventResult
    matching: tmm.Matching
    tm: EventResult               # Ergebnis des Vorlagenabgleichs
    oracle: EventResult           # Abgleich mit den wahren Vorlagen (Obergrenze, nur als Messreferenz)
    comparators: dict             # "ica", "sobi", "sca" -> {"f1", "corr"}


def analyse(ds, settings, with_comparators=True):
    sorting = alg.sort_spikes(ds.X, ds.n_neurons, cluster_mode=settings.cluster_mode, seed=settings.init_start)
    pipe = evaluate_pipeline(ds, sorting)
    matching = tmm.run_matching(ds.X, ds.n_neurons, settings.threshold, settings.min_amplitude, settings.refine, settings.rounds, settings.cluster_mode, settings.init_start, sorting)
    tm = evaluate_pursuit(ds, matching.pursuit, matching.templates.shape[0])
    oracle_pursuit = tmm.matching_pursuit(matching.Xw, oracle_templates(ds), settings.threshold, settings.min_amplitude)
    oracle = evaluate_pursuit(ds, oracle_pursuit, ds.n_neurons)
    comparators = comparator_f1(ds, settings) if with_comparators else {}
    return Analysis(ds, settings, sorting, pipe, matching, tm, oracle, comparators)


def start_result(a):
    """Auswertung des ersten Abgleichs mit den Start-Vorlagen (vor jeder Verfeinerung)."""
    return evaluate_pursuit(a.ds, a.matching.pursuit_start, a.matching.templates_start.shape[0])


def round_f1(a):
    """Spitzen-F1 des Abgleichs nach jeder Verfeinerungsrunde (Eintrag 0 = Start-Vorlagen)."""
    s = a.settings
    out = []
    for W in a.matching.history:
        pursuit = tmm.matching_pursuit(a.matching.Xw, W, s.threshold, s.min_amplitude)
        out.append(evaluate_pursuit(a.ds, pursuit, W.shape[0]).f1)
    return out


def analyse_for(params, settings, with_comparators=True):
    """`params` = (m, n, rate_scale, similarity, jitter, noise, n_samples, seed)."""
    m, n, rate_scale, similarity, jitter, noise, n_samples, seed = params
    return analyse(make_dataset(m, n, rate_scale, similarity, jitter, noise, n_samples, seed), settings, with_comparators)


# --- Sweeps und Tabellen ---------------------------------------------------------------------------------------------------------------------

SWEEP_VALUES = {
    "n_electrodes": (1, 2, 3, 4, 6, 8),
    "rate_scale": (0.25, 0.5, 1.0, 2.0, 3.0, 4.0),
    "noise": (0.0, 0.2, 0.4, 0.7, 1.0),
    "similarity": (0.0, 0.25, 0.5, 0.75, 1.0),
    "jitter": (0.0, 0.05, 0.1, 0.2, 0.3),
    "n_neurons": (2, 3, 4, 5),
    "threshold": (2.0, 3.0, 5.0, 8.0, 12.0),
    "min_amplitude": (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9),
    "rounds": (0, 1, 2, 3, 5),
}
SWEEP_LABELS = {"n_electrodes": "Anzahl Elektroden", "rate_scale": "Feuerrate (Faktor)", "noise": "Rauschen (relativ zum Neuronen-Signal)", "similarity": "Wellenform-Ähnlichkeit (0 = alle gleich, 1 = wie gewohnt)",
                "jitter": "Amplitudenschwankung", "n_neurons": "Anzahl Neuronen", "threshold": "Schwelle z des Abgleichs", "min_amplitude": "Kleinste erlaubte Amplitude (Anteil der Vorlage)",
                "rounds": "Verfeinerungsrunden (0 = keine)"}
DATA_PARAMETERS = {"n_electrodes": "n", "rate_scale": "rate_scale", "noise": "noise", "similarity": "similarity", "jitter": "jitter", "n_neurons": "m"}     # Regler der Daten (Rest: Regler des Abgleichs)
COMPARATORS = ("ica", "sobi", "sca")
COMPARATOR_ONLY_DATA = True         # ICA/SOBI/SCA hängen nur von den Daten ab, nicht von den Reglern des Abgleichs


def _summarise(x, per_seed):
    row = {"x": x}
    for key in per_seed[0]:
        arr = np.array([r[key] for r in per_seed], dtype=float)
        row[key] = float(np.nanmean(arr)) if not np.isnan(arr).all() else float("nan")
        row[key + "_std"] = float(np.nanstd(arr)) if not np.isnan(arr).all() else float("nan")
    return row


def _record(a):
    out = {"f1": a.tm.f1, "accuracy": a.tm.accuracy, "recall": a.tm.recall, "precision": a.tm.precision, "collision_accuracy": a.tm.accuracy_collision, "collision_share": a.tm.collision_share,
           "pipe_f1": a.pipe.f1, "pipe_accuracy": a.pipe.accuracy, "pipe_collision_accuracy": a.pipe.accuracy_collision, "oracle_f1": a.oracle.f1, "ghosts": float(a.tm.n_ghosts)}
    for name in COMPARATORS:
        out[name] = a.comparators[name]["f1"] if a.comparators else float("nan")
    return out


def sweep(parameter, values=None, settings=Settings(), **base):
    """Mittel und Streuung (über die festen Sweep-Datensätze) der Kennzahlen des Abgleichs, der Pipeline, des Orakels und - bei Reglern der Daten - von ICA, SOBI und SCA."""
    values = SWEEP_VALUES[parameter] if values is None else values
    rows = []
    for x in values:
        per_seed = []
        for seed in C.SWEEP_SEEDS:
            if parameter in DATA_PARAMETERS:
                kw = dict(base)
                kw[DATA_PARAMETERS[parameter]] = x
                a = analyse(make_dataset(seed=seed, **kw), settings)
            else:
                if parameter == "rounds":
                    s = Settings(**{**settings.__dict__, "refine": "none" if x == 0 else settings.refine, "rounds": max(int(x), 1)})
                else:
                    s = Settings(**{**settings.__dict__, parameter: x})
                a = analyse(make_dataset(seed=seed, **base), s, with_comparators=False)
            per_seed.append(_record(a))
        rows.append(_summarise(x, per_seed))
    return rows


REFINE_SCENES = (
    ("Vier Neuronen, vier Elektroden", dict(m=4, n=4)),
    ("Hohe Feuerrate (Faktor 2)", dict(m=4, n=4, rate_scale=2.0)),
    ("Fünf Neuronen, vier Elektroden", dict(m=5, n=4)),
    ("Eine Elektrode (4 Neuronen)", dict(m=4, n=1)),
)
REFINE_METHODS = ("pipeline", "none", "average", "recluster", "oracle")


def refine_table(settings=Settings(), **base):
    """Vorlagenqualität: Spitzen-F1 der Pipeline, des Abgleichs ohne Verfeinerung, mit "neu mitteln", mit "neu clustern" (jeweils `settings.rounds` Runden) und mit den wahren Vorlagen (Orakel), Mittel und Minimum über die Sweep-Datensätze."""
    rows = []
    for label, scene in REFINE_SCENES:
        kw = dict(base)
        kw.update(scene)
        acc = {name: [] for name in REFINE_METHODS}
        for seed in C.SWEEP_SEEDS:
            ds = make_dataset(seed=seed, **kw)
            sorting = alg.sort_spikes(ds.X, ds.n_neurons, cluster_mode="known", seed=settings.init_start)
            acc["pipeline"].append(evaluate_pipeline(ds, sorting).f1)
            for refine in ("none", "average", "recluster"):
                mt = tmm.run_matching(ds.X, ds.n_neurons, settings.threshold, settings.min_amplitude, refine, settings.rounds, "known", settings.init_start, sorting)
                acc[refine].append(evaluate_pursuit(ds, mt.pursuit, mt.templates.shape[0]).f1)
            po = tmm.matching_pursuit(mt.Xw, oracle_templates(ds), settings.threshold, settings.min_amplitude)
            acc["oracle"].append(evaluate_pursuit(ds, po, ds.n_neurons).f1)
        row = {"scene": label}
        for name, vals in acc.items():
            row[name] = float(np.mean(vals))
            row[name + "_min"] = float(np.min(vals))
        rows.append(row)
    return rows


SCENES = (
    ("Genug Elektroden (4 Neuronen, 6 Elektroden)", dict(m=4, n=6)),
    ("Zwei Elektroden (4 Neuronen)", dict(m=4, n=2)),
    ("Eine Elektrode (4 Neuronen)", dict(m=4, n=1)),
    ("Fünf Neuronen, vier Elektroden", dict(m=5, n=4)),
    ("Hohe Feuerrate (Faktor 2, 4 Neuronen, 4 Elektroden)", dict(m=4, n=4, rate_scale=2.0)),
    ("Ähnliche Wellenformen (Ähnlichkeit 0, 4 Neuronen, 4 Elektroden)", dict(m=4, n=4, similarity=0.0)),
    ("Amplitudenschwankung (0.3, 4 Neuronen, 4 Elektroden)", dict(m=4, n=4, jitter=0.3)),
    ("Starkes Rauschen (Rauschen 2.0, 4 Neuronen, 4 Elektroden)", dict(m=4, n=4, noise=2.0)),
)
SCENE_METHODS = ("pipeline", "matching", "ica", "sobi", "sca")


def scene_table(settings=Settings(), **base):
    """Pipeline, Vorlagenabgleich, ICA, SOBI und SCA in acht Szenen (Spitzen-F1 der Neuronen), Mittel und Spanne über die Sweep-Datensätze."""
    rows = []
    for label, scene in SCENES:
        kw = dict(base)
        kw.update(scene)
        acc = {name: [] for name in SCENE_METHODS}
        acc["accuracy"], acc["pipe_accuracy"] = [], []
        for seed in C.SWEEP_SEEDS:
            a = analyse(make_dataset(seed=seed, **kw), settings)
            acc["pipeline"].append(a.pipe.f1)
            acc["matching"].append(a.tm.f1)
            acc["accuracy"].append(a.tm.accuracy)
            acc["pipe_accuracy"].append(a.pipe.accuracy)
            for name in COMPARATORS:
                acc[name].append(a.comparators[name]["f1"])
        row = {"scene": label}
        for name, vals in acc.items():
            row[name] = float(np.mean(vals))
            row[name + "_min"], row[name + "_max"] = float(np.min(vals)), float(np.max(vals))
        rows.append(row)
    return rows


# --- Urteil ------------------------------------------------------------------------------------------------------------------------------

VERDICT_PRECISION = 0.9           # Genauigkeit der Detektion des Abgleichs darunter: Geisterereignisse
VERDICT_RECALL = 0.7              # Trefferquote des Abgleichs darunter: zu viele Spikes verpasst (Schwelle zu hoch oder Rauschen)
VERDICT_ORACLE_GAP = 0.10         # F1-Abstand zum Abgleich mit den wahren Vorlagen, ab dem die Vorlagen als Ursache gelten
VERDICT_GAIN = 0.05               # F1-Gewinn des Abgleichs gegenüber der Pipeline, ab dem "Abgleich besser" gilt
VERDICT_GOOD = 0.9                # Spitzen-F1, ab dem der Abgleich als gut gilt


def verdict(a):
    """(Art, Code, Kennzahlen). Nur mit großer Marge - kein Urteil nahe an einer Schwelle. Reihenfolge: Geister, verpasste Spikes, falsche Neuronenzahl, falsche Vorlagen, Abgleich besser, sonst gut/schlecht."""
    ds, t, p, o = a.ds, a.tm, a.pipe, a.oracle
    m = ds.n_neurons
    data = {"f1": t.f1, "pipe_f1": p.f1, "oracle_f1": o.f1, "accuracy": t.accuracy, "pipe_accuracy": p.accuracy, "accuracy_collision": t.accuracy_collision, "pipe_accuracy_collision": p.accuracy_collision,
            "recall": t.recall, "pipe_recall": p.recall, "precision": t.precision, "ghosts": t.n_ghosts, "n_events": t.n_detected, "n_true": t.n_true, "collision_share": t.collision_share,
            "k": a.sorting.k, "m": m, "n": ds.n_electrodes, "refine": a.settings.refine, "min_amplitude": a.settings.min_amplitude, "threshold": a.settings.threshold,
            **{f"{k}_f1": v["f1"] for k, v in a.comparators.items()}}
    if t.precision < VERDICT_PRECISION:
        return "warning", "ghosts", data
    if t.recall < VERDICT_RECALL:
        return "warning", "missed", data
    if a.settings.cluster_mode == "silhouette" and a.sorting.k != m:
        return "warning", "wrong_k", data
    if o.f1 - t.f1 > VERDICT_ORACLE_GAP:
        return "warning", "templates_wrong", data
    if t.f1 - p.f1 >= VERDICT_GAIN:
        return "success", "matching_better", data
    if t.f1 >= VERDICT_GOOD:
        return "success", "matching_ok", data
    return "warning", "matching_poor", data
