"""Mehrelektroden-Szenario der Spike-Sorting-Demo: dasselbe Array wie in ica-demo, sobi-demo und sca-demo (Neuronen, Wellenformen, Feuern, Mischung, Rauschen - dort wortgleich übernommen),
ohne Laufzeitverzögerung und ohne Hintergrund. Neu: die Ähnlichkeit der Wellenformen und die Amplitudenschwankung je Spike. Bei Ähnlichkeit 1 und Schwankung 0 sind alle Daten
identisch zu denen der Vorgänger.

Nur numpy. Jede Quelle hat einen eigenen Zufallsstrom (Seed, Nummer): Neuron 0 feuert für einen Seed immer gleich, egal wie viele Neuronen oder Elektroden eingestellt sind."""

from dataclasses import dataclass

import numpy as np

import tm_constants as C


@dataclass(frozen=True)
class Dataset:
    S: np.ndarray                 # (m, T) wahre Quellen (Neuronen), jede Zeile hat Varianz 1
    X: np.ndarray                 # (n, T) Elektrodensignale (mit Rauschen)
    X_clean: np.ndarray           # (n, T) ohne Rauschen
    A: np.ndarray                 # (n, m) Mischmatrix
    spike_times: tuple            # je Neuron: Zeitpunkte der negativen Spitze (Abtastwerte)
    spike_starts: tuple           # je Neuron: Startzeitpunkte der Spikes (Abtastwerte)
    active: np.ndarray            # (m, T) bool: Neuron ist an diesem Zeitpunkt aktiv (Wellenform über 10 % der Spitzenhöhe)
    noise_sigma: float
    n_neurons: int
    n_electrodes: int
    seed: int

    @property
    def kinds(self):
        return ("neuron",) * self.n_neurons


def neuron_sigma(index, similarity=1.0):
    """Breite der Spitze; Ähnlichkeit 0 = alle Neuronen haben SIGMA_CENTER, 1 = die eigenen Breiten."""
    return C.SIGMA_CENTER + similarity * (C.NEURON_SIGMAS[index] - C.SIGMA_CENTER)


def spike_waveform(index, similarity=1.0):
    """Biphasische Wellenform: tiefe negative Spitze bei PEAK_INDEX, flacher positiver Nachschlag."""
    sigma = neuron_sigma(index, similarity)
    t = np.arange(C.WAVEFORM_LENGTH, dtype=float)
    return -np.exp(-((t - C.PEAK_INDEX) / sigma) ** 2) + C.OVERSHOOT * np.exp(-((t - (C.PEAK_INDEX + 2.5 * sigma)) / (1.6 * sigma)) ** 2)


def firing_times(index, n_samples, seed, rate_scale=1.0):
    """Poisson-artiges Feuern mit Refraktärzeit: Abstände = Refraktärzeit + Exponentialverteilung. Spike-Startzeitpunkte (Abtastwerte)."""
    rng = np.random.default_rng([seed, index])
    mean_isi = C.SAMPLE_RATE / (C.NEURON_RATES[index] * rate_scale)
    n_draw = int(n_samples / (mean_isi - C.REFRACTORY) * 1.5) + 20
    isi = C.REFRACTORY + rng.exponential(mean_isi - C.REFRACTORY, n_draw)
    times = np.cumsum(isi) - isi[0] + rng.uniform(0, mean_isi)
    return times[times < n_samples - C.WAVEFORM_LENGTH].astype(int)


def neuron_source(index, n_samples, seed, rate_scale=1.0, similarity=1.0, jitter=0.0):
    starts = firing_times(index, n_samples, seed, rate_scale)
    impulses = np.zeros(n_samples)
    if jitter > 0:
        amplitudes = 1.0 + jitter * np.random.default_rng([seed, 3000 + index]).standard_normal(len(starts))
    else:
        amplitudes = np.ones(len(starts))
    impulses[starts] = amplitudes
    s = np.convolve(impulses, spike_waveform(index, similarity))[:n_samples]
    return s, starts


def electrode_positions(n):
    return np.array([[0.5, 0.0]]) if n == 1 else np.column_stack([np.linspace(0.0, 1.0, n), np.zeros(n)])


def mixing_matrix(m, n):
    """(n, m): Neuronen mit 1/(d^2+eps)-Abfall (Spitzenamplitude je Neuron)."""
    pos = electrode_positions(n)
    cols = []
    for i in range(m):
        d = np.linalg.norm(pos - np.array(C.NEURON_POSITIONS[i]), axis=1)
        a = 1.0 / (d ** 2 + C.DISTANCE_EPS)
        cols.append(C.NEURON_AMPLITUDES[i] * a / a.max())
    return np.column_stack(cols)


def make_dataset(n_neurons, n_electrodes, rate_scale, similarity, jitter, noise, n_samples, seed):
    m, n = n_neurons, n_electrodes
    sources, raw, starts_list = [], [], []
    for i in range(m):
        s, starts = neuron_source(i, n_samples, seed, rate_scale, similarity, jitter)
        raw.append(s)
        sources.append(s)
        starts_list.append(starts)
    S = np.array(sources)
    S = (S - S.mean(axis=1, keepdims=True)) / S.std(axis=1, keepdims=True)
    # Zeit der negativen Spitze: Start + Lage des Minimums der Wellenform (bei Ähnlichkeit 1 und Standardformen = PEAK_INDEX)
    peak_offsets = [int(np.argmin(spike_waveform(i, similarity))) for i in range(m)]
    spike_times = tuple(starts + off for starts, off in zip(starts_list, peak_offsets))
    active = np.abs(np.array(raw)) > C.ACTIVE_THRESHOLD
    A = mixing_matrix(m, n)
    X_clean = A @ S
    sigma = noise * X_clean.std()
    X = X_clean + sigma * np.random.default_rng([seed, 1000]).standard_normal(X_clean.shape)
    return Dataset(S, X, X_clean, A, spike_times, tuple(starts_list), active, float(sigma), m, n, seed)
