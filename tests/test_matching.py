"""Vorlagenabgleich: Handinstanzen (Passung, Kreuzkorrelation, Abziehen, Amplitudengrenzen), Verfeinerung und die Ausrichtung der Start-Vorlagen."""

import numpy as np
import pytest

import tm_algorithm as alg
import tm_constants as C
import tm_evaluation as ev
import tm_matching as tmm

L = tmm.L


def _templates(q=2, n=3, seed=0, scale=3.0):
    return np.random.default_rng(seed).standard_normal((q, n, L)) * scale


def _signal(W, events, T=500):
    X = np.zeros((W.shape[1], T))
    for s, q, a in events:
        X[:, s: s + L] += a * W[q]
    return X


def test_whiten_gives_median_zero_and_unit_noise_and_ignores_spikes():
    rng = np.random.default_rng(1)
    X = 5.0 + np.array([[2.0], [0.5]]) * rng.standard_normal((2, 20000))
    X[0, 100:130] -= 60.0                                                            # ein Spike beeinflusst die robuste Schätzung kaum
    Xw, med, sigma = tmm.whiten(X)
    assert np.allclose(np.median(Xw, axis=1), 0.0, atol=1e-12) and np.allclose(sigma, [2.0, 0.5], rtol=0.03)
    assert np.allclose(np.median(np.abs(Xw), axis=1) / 0.6745, 1.0, rtol=0.03)


def test_correlation_traces_match_a_brute_force_computation():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((3, 200))
    W = _templates()
    Z = tmm.correlation_traces(X, W)
    assert Z.shape == (2, 200 - L + 1)
    for q in (0, 1):
        for s in (0, 17, 100, 170):
            assert Z[q, s] == pytest.approx((X[:, s: s + L] * W[q]).sum() / np.sqrt((W[q] ** 2).sum()))


def test_correlation_of_pure_noise_is_standard_normal():
    """Beleg für 'unter reinem Rauschen ungefähr standardnormalverteilt' (Erwartungswert 0, Standardabweichung 1)."""
    Z = tmm.correlation_traces(np.random.default_rng(3).standard_normal((3, 30000)), _templates())
    assert abs(Z.mean()) < 0.05 and abs(Z.std() - 1.0) < 0.05


def test_cross_correlations_match_a_brute_force_computation():
    W = _templates(q=3)
    XC = tmm.cross_correlations(W)
    for d in (-29, -5, 0, 7, 29):
        for q, p in ((0, 1), (2, 2), (1, 0)):
            expected = sum((W[q, :, l] * W[p, :, l + d]).sum() for l in range(L) if 0 <= l + d < L)
            assert XC[q, p, d + L - 1] == pytest.approx(expected)
    assert XC[1, 1, L - 1] == pytest.approx((W[1] ** 2).sum())                         # Verschiebung 0 = Energie


def test_isolated_events_are_recovered_exactly_and_the_residual_vanishes():
    W = _templates()
    X = _signal(W, [(100, 0, 1.0), (300, 1, 1.2)])
    p = tmm.matching_pursuit(X, W, threshold=5.0)
    assert list(p.times - C.SNIPPET_BEFORE) == [100, 300] and list(p.templates) == [0, 1] and np.allclose(p.amplitudes, [1.0, 1.2])
    assert np.abs(p.residual).max() < 1e-9


def test_overlapping_events_are_resolved_with_the_right_times_and_templates():
    W = _templates(seed=4)
    X = _signal(W, [(100, 0, 1.0), (112, 1, 0.9)])
    p = tmm.matching_pursuit(X, W, threshold=5.0)
    assert list(p.times - C.SNIPPET_BEFORE) == [100, 112] and list(p.templates) == [0, 1]
    assert np.allclose(p.amplitudes, [1.0, 0.9], atol=0.15) and np.abs(p.residual).max() < 0.3 * np.abs(X).max()      # gieriges Abziehen ist bei Überlappung nur näherungsweise exakt


def test_the_pursuit_updates_its_scores_exactly_like_a_full_recomputation():
    """Der Kern der Beschleunigung: die Aktualisierung über Kreuzkorrelationen entspricht der neu berechneten Passung des Residuums (kein Rest oberhalb der Schwelle)."""
    W = _templates(seed=5)
    rng = np.random.default_rng(6)
    X = _signal(W, [(60, 0, 1.0), (75, 1, 1.1), (200, 1, 0.8), (210, 0, 1.3), (350, 0, 1.0)]) + 0.02 * rng.standard_normal((3, 500))
    p = tmm.matching_pursuit(X, W, threshold=5.0)
    assert len(p.times) == 5
    assert tmm.correlation_traces(p.residual, W).max() < 5.0


def test_amplitudes_below_the_minimum_are_not_events_and_above_the_maximum_are_clipped():
    W = _templates()
    X = _signal(W, [(100, 0, 0.3), (300, 1, 3.0)])
    assert list(tmm.matching_pursuit(X, W, 5.0, min_amplitude=0.5).templates) == [1]
    low = tmm.matching_pursuit(X, W, 5.0, min_amplitude=0.2)
    assert 100 in (low.times - C.SNIPPET_BEFORE) and 300 in (low.times - C.SNIPPET_BEFORE)
    assert low.amplitudes[list(low.times - C.SNIPPET_BEFORE).index(300)] == pytest.approx(tmm.MAX_AMPLITUDE)          # begrenzt; der Rest darüber wird als weitere Ereignisse abgezogen, die Schleife endet


def _waveform(sigma):
    t = np.arange(L)
    return -np.exp(-(((t - 8) / sigma) ** 2)) + 0.4 * np.exp(-(((t - (8 + 2.5 * sigma)) / (1.6 * sigma)) ** 2))


def test_without_a_minimum_amplitude_leftovers_become_ghost_events():
    """Beleg für 'Ohne Amplitudengrenze': eine um 20 % zu breite Vorlage lässt einen Rest übrig, der ohne Grenze bei kleinem Rauschen als weiterer Spike gemeldet wird (Beleg an den Daten: test_scenario_and_evaluation)."""
    W = np.stack([np.tile(_waveform(3.0) * 30, (3, 1)), np.tile(_waveform(2.0) * 25, (3, 1))])
    X = np.zeros((3, 500))
    X[:, 100: 100 + L] += W[0]
    Wm = W.copy()
    Wm[0] = np.tile(_waveform(3.6) * 30, (3, 1))
    clean = tmm.matching_pursuit(X, Wm, 5.0, min_amplitude=0.5)
    ghosts = tmm.matching_pursuit(X, Wm, 5.0, min_amplitude=0.0)
    assert len(clean.times) == 1 and clean.times[0] - C.SNIPPET_BEFORE == 100 and len(ghosts.times) >= 3


def test_no_templates_or_a_short_signal_give_no_events():
    X = np.random.default_rng(8).standard_normal((3, 400))
    assert len(tmm.matching_pursuit(X, np.zeros((0, 3, L)), 5.0).times) == 0
    assert len(tmm.matching_pursuit(X[:, :10], _templates(), 5.0).times) == 0
    assert len(tmm.matching_pursuit(X, np.zeros((1, 3, L)), 5.0).times) == 0            # Vorlage aus Nullen: nichts passt


def test_pursuit_is_deterministic_and_sorted_by_time():
    ds = ev.make_dataset(rate_scale=2.0)
    Xw, _, _ = tmm.whiten(ds.X)
    a = alg.sort_spikes(ds.X, 4, seed=1)
    W, _ = tmm.initial_templates(Xw, a)
    p1, p2 = tmm.matching_pursuit(Xw, W, 5.0), tmm.matching_pursuit(Xw, W, 5.0)
    assert (p1.times == p2.times).all() and (p1.templates == p2.templates).all() and (np.diff(p1.times) >= 0).all()
    assert sorted(p1.order) == list(range(len(p1.times)))


def test_start_templates_are_aligned_at_the_spike_minimum():
    """Die Vorlagen sind am Minimum ausgerichtet (Index SNIPPET_BEFORE): der Abgleich meldet Ereignisse an derselben Stelle wie die Detektion."""
    ds = ev.make_dataset()
    a = alg.sort_spikes(ds.X, 4, seed=1)
    Xw, _, _ = tmm.whiten(ds.X)
    W, ids = tmm.initial_templates(Xw, a)
    assert W.shape == (4, 4, L) and ids == [0, 1, 2, 3]
    for w in W:
        strongest = w[np.argmax(np.abs(w).max(axis=1))]
        assert abs(int(np.argmin(strongest)) - C.SNIPPET_BEFORE) <= 1


def test_cleaned_snippets_of_isolated_events_equal_the_template():
    W = _templates()
    X = _signal(W, [(100, 0, 1.0), (300, 1, 1.2)])
    p = tmm.matching_pursuit(X, W, 5.0)
    cleaned = tmm.cleaned_snippets(W, p)
    assert np.allclose(cleaned[0], W[0]) and np.allclose(cleaned[1], W[1])


def test_cleaned_snippets_remove_the_overlapping_neighbour():
    W = _templates(seed=9)
    X = _signal(W, [(100, 0, 1.0), (112, 1, 1.0)])
    p = tmm.matching_pursuit(X, W, 5.0)
    cleaned = tmm.cleaned_snippets(W, p)
    raw = X[:, 100: 100 + L]
    assert np.linalg.norm(cleaned[0] - W[0]) < 0.3 * np.linalg.norm(raw - W[0])       # der Ausschnitt mit Nachbar ist weit von der Vorlage entfernt, der bereinigte fast gleich


def test_average_refinement_moves_a_wrong_template_towards_the_truth():
    """Neu mitteln: ein zu breit geschätzte Vorlage wird aus den bereinigten Ausschnitten ihrer Ereignisse wieder die wahre (Handinstanz; an den Daten hilft es viel weniger, siehe test_scenario_and_evaluation)."""
    W = np.stack([np.tile(_waveform(3.0) * 30, (3, 1)), np.tile(_waveform(2.0) * 25, (3, 1))])
    X = np.zeros((3, 900))
    for s in range(50, 800, 100):
        X[:, s: s + L] += W[0]
    Wm = W.copy()
    Wm[0] = np.tile(_waveform(3.6) * 30, (3, 1))
    p = tmm.matching_pursuit(X, Wm, 5.0)
    new = tmm.average_templates(Wm, p)
    unit = lambda w: w / np.linalg.norm(w)
    assert np.linalg.norm(unit(new[0]) - unit(W[0])) < 1e-9 < np.linalg.norm(unit(Wm[0]) - unit(W[0]))                # die Form stimmt wieder (nur die Skalierung der Startvorlage bleibt)
    assert np.linalg.norm(new[0] - W[0]) < 0.6 * np.linalg.norm(Wm[0] - W[0]) and (new[1] == Wm[1]).all()           # Vorlage ohne Ereignisse bleibt unverändert


def test_recluster_needs_enough_events_and_otherwise_keeps_the_templates():
    W = _templates()
    p = tmm.matching_pursuit(_signal(W, [(100, 0, 1.0), (300, 1, 1.0)]), W, 5.0)
    assert tmm.recluster_templates(W, p, 4) is W
    ds = ev.make_dataset(rate_scale=2.0)
    Xw, _, _ = tmm.whiten(ds.X)
    Wn, _ = tmm.initial_templates(Xw, alg.sort_spikes(ds.X, 4, seed=1))
    pn = tmm.matching_pursuit(Xw, Wn, 5.0)
    assert tmm.recluster_templates(Wn, pn, 4).shape == (4, 4, L)


def test_run_matching_history_rounds_and_a_given_sorting_give_the_same_result():
    ds = ev.make_dataset(rate_scale=2.0)
    srt = alg.sort_spikes(ds.X, 4, seed=1)
    a = tmm.run_matching(ds.X, 4, seed=1, sorting=srt)
    b = tmm.run_matching(ds.X, 4, seed=1)
    assert (a.pursuit.times == b.pursuit.times).all() and (a.pursuit.templates == b.pursuit.templates).all()
    assert len(a.history) == 1 + C.DEFAULT_ROUNDS and a.refine == C.DEFAULT_REFINE and (a.history[0] == a.templates_start).all() and (a.history[-1] == a.templates).all()
    none = tmm.run_matching(ds.X, 4, refine="none", rounds=5, seed=1, sorting=srt)
    assert len(none.history) == 1 and none.pursuit is none.pursuit_start or (none.pursuit.times == none.pursuit_start.times).all()


def test_run_matching_with_almost_no_spikes_does_not_crash():
    X = 0.1 * np.random.default_rng(5).standard_normal((2, 2000))
    mt = tmm.run_matching(X, 3, seed=1)
    assert mt.templates.shape[0] == 0 and len(mt.pursuit.times) == 0
