import numpy as np
import pytest
from sklearn.cluster import KMeans as SkKMeans
from sklearn.decomposition import PCA as SkPCA
from sklearn.metrics import silhouette_score

import tm_algorithm as alg
import tm_constants as C
import tm_evaluation as ev
import tm_ica as ica
import tm_sca as sca
import tm_scenario as sc
import tm_sobi as sobi


def _blobs(k=3, per=80, d=2, spread=0.15, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-3, 3, (k, d)) * 2
    F = np.vstack([c + spread * rng.standard_normal((per, d)) for c in centers])
    return F, np.repeat(np.arange(k), per), centers


# --- Rauschschätzung und Detektion ------------------------------------------------------------------------------------------------------


def test_noise_sigma_recovers_the_standard_deviation_and_ignores_rare_spikes():
    rng = np.random.default_rng(0)
    X = 0.7 * rng.standard_normal((3, 30000))
    X[:, 1000:1010] -= 30.0
    assert np.allclose(alg.noise_sigma(X), 0.7, rtol=0.05)


def test_noise_sigma_has_a_floor_for_noise_free_data():
    X = np.zeros((2, 1000))
    X[0, 100] = -5.0
    assert (alg.noise_sigma(X) > 0).all()


def test_detect_finds_planted_spikes_respects_dead_time_and_reports_the_minimum():
    rng = np.random.default_rng(1)
    X = 0.1 * rng.standard_normal((2, 5000))
    plants = [300, 1200, 2500, 4000]
    for t in plants:
        X[0, t - 2: t + 3] -= np.array([1.0, 2.0, 4.0, 2.0, 1.0])
    X[0, 1208] -= 3.0                                                                    # zweiter Ausschlag innerhalb der Totzeit
    times, d = alg.detect(X, 5.0)
    assert list(times) == plants and d.shape == (5000,) and d[300] < -5.0
    times2, _ = alg.detect(X, 5.0, dead_time=3)
    assert 1208 in times2 and len(times2) == 5


def test_detect_uses_the_most_negative_electrode_and_ignores_positive_deflections():
    rng = np.random.default_rng(2)
    X = 0.1 * rng.standard_normal((3, 3000))
    X[2, 500] -= 4.0
    X[0, 900] += 4.0
    times, _ = alg.detect(X, 5.0)
    assert list(times) == [500]


def test_snippets_are_aligned_at_the_minimum_and_edge_spikes_are_dropped():
    X = np.zeros((2, 200))
    for t in (50, 120):
        X[:, t] = -3.0
    times, snips = alg.snippets(X, np.array([3, 50, 120, 195]))
    assert list(times) == [50, 120] and snips.shape == (2, 2, 30) and (snips[:, :, C.SNIPPET_BEFORE] == -3.0).all()


# --- PCA ---------------------------------------------------------------------------------------------------------------------------------


def test_pca_features_match_scikit_learn_up_to_sign_and_report_the_explained_variance():
    rng = np.random.default_rng(3)
    snips = rng.standard_normal((200, 3, 30)) @ np.diag(np.linspace(2.0, 0.2, 30))
    ours = alg.pca_features(snips, 4)
    ref = SkPCA(n_components=4).fit(snips.reshape(200, -1))
    assert np.allclose(np.abs(ours.values), np.abs(ref.transform(snips.reshape(200, -1))), atol=1e-9)
    assert np.allclose(ours.explained, ref.explained_variance_ratio_, atol=1e-9) and ours.components.shape == (4, 90)
    assert np.allclose(ours.values.mean(axis=0), 0.0, atol=1e-9)


def test_feature_kinds_have_the_documented_shapes():
    snips = np.random.default_rng(4).standard_normal((50, 3, 30))
    assert alg.extract_features(snips, "raw").values.shape == (50, 90) and alg.extract_features(snips, "amplitude").values.shape == (50, 3)
    assert np.allclose(alg.extract_features(snips, "amplitude").values, snips.min(axis=2)) and alg.extract_features(snips, "pca", 2).values.shape == (50, 2)


# --- k-means und Silhouette -----------------------------------------------------------------------------------------------------------


def test_kmeans_recovers_well_separated_blobs_like_scikit_learn():
    F, truth, _ = _blobs()
    ours = alg.kmeans(F, 3, seed=1)
    ref = SkKMeans(3, n_init=10, random_state=0).fit(F)
    assert abs(ours.inertia - ref.inertia_) < 1e-6 * ref.inertia_
    conf = np.zeros((3, 3))
    for a, b in zip(truth, ours.labels):
        conf[a, b] += 1
    assert (conf.max(axis=1) == 80).all() and len(ours.restart_inertias) == C.N_RESTARTS


def test_kmeans_is_deterministic_orders_clusters_by_the_first_feature_and_handles_k_larger_than_n():
    F, _, _ = _blobs()
    a, b = alg.kmeans(F, 3, seed=2), alg.kmeans(F, 3, seed=2)
    assert np.array_equal(a.labels, b.labels) and (np.diff(a.centers[:, 0]) >= 0).all()
    assert alg.kmeans(F[:2], 5).centers.shape[0] == 2


def test_silhouette_matches_scikit_learn():
    F, truth, _ = _blobs(spread=0.9)
    assert abs(alg.silhouette(F, truth) - silhouette_score(F, truth)) < 1e-9
    km = alg.kmeans(F, 4, seed=1).labels
    assert abs(alg.silhouette(F, km) - silhouette_score(F, km)) < 1e-9 and alg.silhouette(F, np.zeros(len(F), dtype=int)) == 0.0


def test_silhouette_uses_a_fixed_subsample_for_large_inputs():
    F, truth, _ = _blobs(per=800)
    assert alg.silhouette(F, truth) == alg.silhouette(F, truth) and 0.0 < alg.silhouette(F, truth) <= 1.0


def test_choose_k_picks_the_true_number_for_clean_blobs_and_returns_the_curve():
    F, _, _ = _blobs(k=4, spread=0.1)
    k, curve = alg.choose_k(F, 7, seed=1)
    assert k == 4 and sorted(curve) == [2, 3, 4, 5, 6, 7] and curve[4] == max(curve.values())


# --- Gesamtpipeline -------------------------------------------------------------------------------------------------------------------


def test_sort_spikes_on_easy_data_finds_all_neurons():
    ds = ev.make_dataset(m=2, n=4, noise=0.05)
    srt = alg.sort_spikes(ds.X, 2, 4.5, "pca", 3, "known", seed=1)
    r = ev.evaluate_pipeline(ds, srt)
    assert srt.k == 2 and r.recall > 0.9 and r.precision > 0.99 and r.accuracy > 0.99 and len(srt.times) == len(srt.snippets) == len(srt.clustering.labels)


def test_sort_spikes_with_almost_no_spikes_does_not_crash():
    X = 0.1 * np.random.default_rng(5).standard_normal((2, 2000))
    srt = alg.sort_spikes(X, 3, 20.0)
    assert srt.k == 0 and len(srt.times) == 0


def test_sort_spikes_is_deterministic():
    ds = ev.make_dataset()
    a, b = alg.sort_spikes(ds.X, 4, seed=3), alg.sort_spikes(ds.X, 4, seed=3)
    assert np.array_equal(a.clustering.labels, b.clustering.labels) and np.array_equal(a.times, b.times)


# --- Vergleichsverfahren aus den Vorgänger-Demos (eingefroren) ---------------------------------------------------------------------


def test_fastica_copy_reproduces_the_frozen_reference():
    """Eingefrorener Wert aus ica-demo (4 Neuronen, 6 Elektroden, Rauschen 0.05, Seed 7): Neuronen-Korrelation 0.981."""
    ds = ev.make_dataset(n=6)
    m = ica.fit_ica(ds.X, 4, "logcosh", "symmetric", 1)
    assert m.converged and abs(ev.matched(ds.S, m.sources)[1].mean() - 0.981) < 2e-3


def test_sobi_copy_reproduces_the_frozen_reference():
    """Eingefrorener Wert aus sobi-demo (nur Neuronen, mittlere Verzögerungen): SOBI 0.978."""
    ds = ev.make_dataset(n=6)
    m = sobi.fit_sobi(ds.X, 4, C.SOBI_LAGS)
    assert m.converged and abs(ev.matched(ds.S, m.sources)[1].mean() - 0.978) < 2e-3


def test_sca_copy_reproduces_the_frozen_reference():
    """Eingefrorener Wert aus sca-demo (4 Neuronen, 2 Elektroden, Seed 7, L1): Neuronen-Korrelation 0.952."""
    ds = ev.make_dataset(n=2)
    m = sca.fit_sca(ds.X, 4, "l1", seed=1)
    assert abs(ev.matched(ds.S, m.sources)[1].mean() - 0.952) < 5e-3


# --- Szenario ----------------------------------------------------------------------------------------------------------------------------


def test_scenario_is_the_one_of_the_predecessors_for_default_similarity_and_jitter():
    """Eingefroren aus ica/sobi/sca-demo (Seed 7, Rauschen 0.05, 6 Elektroden, keine Hintergrundquelle)."""
    ds = sc.make_dataset(4, 6, 1.0, 1.0, 0.0, 0.05, 20000, 7)
    assert abs(np.abs(ds.S[0][:3000]).sum() - 436.61272090571026) < 1e-6 and abs(np.abs(ds.X[:, :2000]).sum() - 4981.157973724114) < 1e-6


def test_spike_times_point_at_the_negative_peak_for_all_similarities():
    for similarity in (0.0, 0.5, 1.0):
        ds = ev.make_dataset(noise=0.0, similarity=similarity)
        for i in range(ds.n_neurons):
            for t in ds.spike_times[i][:15]:
                assert ds.S[i][t - 3: t + 4].argmin() == 3


def test_similarity_zero_gives_identical_waveforms_and_one_gives_the_original_widths():
    assert np.allclose(sc.spike_waveform(0, 0.0), sc.spike_waveform(3, 0.0)) and not np.allclose(sc.spike_waveform(0, 1.0), sc.spike_waveform(3, 1.0))
    assert sc.neuron_sigma(0, 1.0) == C.NEURON_SIGMAS[0] and sc.neuron_sigma(0, 0.0) == C.SIGMA_CENTER


def test_jitter_scales_single_spikes_but_keeps_the_spike_times():
    plain = ev.make_dataset(noise=0.0, jitter=0.0)
    jit = ev.make_dataset(noise=0.0, jitter=0.3)
    assert all(np.array_equal(a, b) for a, b in zip(plain.spike_times, jit.spike_times))
    heights = np.array([-jit.X_clean[0, t] / -plain.X_clean[0, t] for t in plain.spike_times[0][:60] if abs(plain.X_clean[0, t]) > 0.1])
    assert 0.15 < heights.std() < 0.5


def test_streams_are_independent_of_other_settings():
    a = ev.make_dataset(m=2, n=3, seed=11)
    b = ev.make_dataset(m=5, n=8, seed=11)
    assert np.array_equal(a.spike_times[0], b.spike_times[0]) and np.allclose(a.S[0], b.S[0])


def test_constants_are_consistent():
    assert len(C.NEURON_SIGMAS) == len(C.NEURON_RATES) == len(C.NEURON_AMPLITUDES) == len(C.NEURON_POSITIONS) == C.N_NEURONS_MAX
    assert C.SNIPPET_BEFORE + C.SNIPPET_AFTER == C.WAVEFORM_LENGTH and set(C.CLUSTER_MODES) == set(C.CLUSTER_MODE_LABELS) and set(C.REFINE_MODES) == set(C.REFINE_LABELS)
    assert C.MATCH_THRESHOLD_MIN <= C.DEFAULT_MATCH_THRESHOLD <= C.MATCH_THRESHOLD_MAX and C.MIN_AMPLITUDE_MIN <= C.DEFAULT_MIN_AMPLITUDE <= C.MIN_AMPLITUDE_MAX and C.ROUNDS_MIN <= C.DEFAULT_ROUNDS <= C.ROUNDS_MAX and C.DEFAULT_REFINE in C.REFINE_MODES
