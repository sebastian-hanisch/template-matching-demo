"""Auswertung und die Aussagen der App als Tests: jede Zahl in den Hilfetexten, Tabellen und Presets ist hier über die festen Sweep-Datensätze belegt (Toleranzen bewusst weit).
k-means (Start-Vorlagen) ist datensatzabhängig - deshalb Mittel und Spannen, nicht einzelne Datensätze."""

from functools import lru_cache

import numpy as np
import pytest

import tm_algorithm as alg
import tm_constants as C
import tm_evaluation as ev
import tm_matching as tmm
import tm_scenario as sc


@lru_cache(maxsize=None)
def _cached(items, settings, comparators):
    return tuple(ev.analyse(ev.make_dataset(seed=s, **dict(items)), settings, with_comparators=comparators) for s in C.SWEEP_SEEDS)


def _analyses(settings=ev.Settings(), comparators=False, **kw):
    return _cached(tuple(sorted(kw.items())), settings, comparators)


def _mean(analyses, group, attr="f1"):
    return float(np.mean([getattr(getattr(a, group), attr) for a in analyses]))


def _cmp(analyses, name):
    return float(np.mean([a.comparators[name]["f1"] for a in analyses]))


# --- Szenario: bit-identisch zum Vorgänger --------------------------------------------------------------------------------------------


def test_default_dataset_is_bit_identical_to_the_spike_sorting_demo():
    ds = sc.make_dataset(4, 4, 1.0, 1.0, 0.0, 0.05, 20000, 7)
    # Float-Werte mit enger Toleranz: die Summe hängt von der Summationsreihenfolge der numpy-Version/CPU ab (CI-Runner weicht in der 13. Stelle ab)
    assert float(ds.X.sum()) == pytest.approx(20.409393146494438, rel=1e-9) and float(ds.X[0, 100]) == pytest.approx(0.17004588550776134, rel=1e-9) and float(ds.X[3, 5000]) == pytest.approx(0.15260309252966886, rel=1e-9)
    assert sum(len(t) for t in ds.spike_times) == 227


# --- Zuordnung und Kennzahlen: Handinstanzen -----------------------------------------------------------------------------------------


def test_match_detections_hand_instance():
    truth = np.array([100, 200, 205, 400])
    out = ev.match_detections(np.array([98, 203, 210, 300, 401]), truth)
    assert list(out) == [0, 2, -1, -1, 3]                                                 # 203 liegt näher an 205 als an 200; 210 findet nichts Freies in Reichweite, 300 hat keinen Partner
    assert list(ev.match_detections(np.array([200, 201]), np.array([200]))) == [0, -1]     # jede wahre Spitze nur einmal


def test_truth_spikes_marks_collisions_by_start_distance():
    ds = ev.make_dataset(noise=0.0)
    times, neuron, coll = ev.truth_spikes(ds)
    assert (np.diff(times) >= 0).all() and len(times) == sum(len(t) for t in ds.spike_times) and set(neuron) == set(range(4))
    starts = np.sort(np.concatenate(ds.spike_starts))
    for s in ds.spike_starts[0][:30]:
        near = ((np.abs(starts - s) <= C.COLLISION_WINDOW).sum() > 1)
        idx = np.flatnonzero((neuron == 0))[list(ds.spike_starts[0]).index(s)]
        assert coll[idx] == near


def test_evaluate_events_on_a_perfect_case_and_with_ghosts_and_without_events():
    ds = ev.make_dataset(m=2, n=4, noise=0.0, rate_scale=0.25)
    tt, tn, tc = ev.truth_spikes(ds)
    r = ev.evaluate_events(ds, tt, tn, 2)                                                  # die wahren Spikes als Ereignisse: alles richtig
    assert r.recall == r.precision == r.accuracy == 1.0 and r.f1 == pytest.approx(1.0) and r.n_ghosts == 0 and r.confusion.shape == (2, 2)
    swapped = ev.evaluate_events(ds, tt, 1 - tn, 2)                                         # Cluster vertauscht: die optimale Zuordnung gleicht das aus
    assert swapped.accuracy == 1.0
    ghost_times = np.concatenate([tt, [tt[-1] + 500]])
    g = ev.evaluate_events(ds, ghost_times, np.concatenate([tn, [0]]), 2)
    assert g.n_ghosts == 1 and g.precision == pytest.approx(len(tt) / (len(tt) + 1)) and g.recall == 1.0
    e = ev.evaluate_events(ds, np.zeros(0, dtype=int), np.zeros(0, dtype=int), 1)
    assert e.recall == 0.0 and e.n_detected == 0 and e.f1 == 0.0 and e.cluster_of_neuron == [-1, -1]


def test_evaluate_events_confusion_matrix_and_split_accuracies_are_consistent():
    a = ev.analyse(ev.make_dataset(rate_scale=2.0), ev.Settings(), with_comparators=False)
    for r in (a.pipe, a.tm):
        assert r.confusion.sum() == round(r.recall * r.n_true) and 0 <= r.accuracy <= 1 and r.accuracy_single >= r.accuracy_collision
        assert abs(r.accuracy * r.confusion.sum() - sum(r.confusion[i, c] for i, c in enumerate(r.cluster_of_neuron) if c >= 0)) < 1e-9
        assert len(r.neuron_of_event) == len(r.truth_of_event) == len(r.collision_of_event) == r.n_detected
        assert r.n_ghosts == int((r.truth_of_event < 0).sum())


def test_oracle_templates_have_the_right_shape_and_alignment_and_the_oracle_matching_is_good():
    ds = ev.make_dataset()
    W = ev.oracle_templates(ds)
    assert W.shape == (4, 4, tmm.L)
    for w in W:
        strongest = w[np.argmax(np.abs(w).max(axis=1))]
        assert abs(int(np.argmin(strongest)) - C.SNIPPET_BEFORE) <= 1
    assert ev.analyse(ds, ev.Settings(), with_comparators=False).oracle.f1 > 0.95


def test_analysis_has_the_expected_structure_and_comparators_only_when_asked():
    a = ev.analyse(ev.make_dataset(), ev.Settings())
    assert set(a.comparators) == {"ica", "sobi", "sca"} and a.tm.n_true == a.pipe.n_true == a.oracle.n_true
    assert ev.analyse(ev.make_dataset(), ev.Settings(), with_comparators=False).comparators == {}
    b = ev.analyse_for((4, 4, 2.0, 0.5, 0.2, 0.7, 20000, 7), ev.Settings(), with_comparators=False)
    assert b.ds.n_neurons == 4 and len(ev.round_f1(b)) == 1 + C.DEFAULT_ROUNDS and ev.start_result(b).n_true == b.tm.n_true


def test_round_f1_starts_with_the_start_templates_and_recluster_improves_it_on_high_rate():
    a = ev.analyse(ev.make_dataset(rate_scale=2.0, seed=100000), ev.Settings(), with_comparators=False)
    f = ev.round_f1(a)
    assert f[0] == pytest.approx(ev.start_result(a).f1) and f[-1] == pytest.approx(a.tm.f1) and f[-1] > f[0] + 0.1


# --- Verdict-Codes ---------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("params,kwargs,codes", [
    ((4, 4, 1.0, 1.0, 0.0, 0.05), {}, {"matching_ok", "matching_better"}),
    ((4, 4, 2.0, 1.0, 0.0, 0.05), {}, {"matching_better"}),
    ((4, 1, 1.0, 1.0, 0.0, 0.05), {}, {"templates_wrong"}),
    ((5, 4, 1.0, 1.0, 0.0, 0.05), {}, {"matching_better", "templates_wrong"}),
    ((4, 4, 1.0, 1.0, 0.0, 0.05), {"min_amplitude": 0.0}, {"ghosts"}),
    ((4, 4, 2.0, 1.0, 0.0, 0.05), {"refine": "none"}, {"templates_wrong", "matching_better"}),
    ((4, 4, 1.0, 1.0, 0.0, 0.05), {"cluster_mode": "silhouette"}, {"wrong_k"}),
    ((4, 4, 1.0, 1.0, 0.3, 0.05), {}, {"matching_better"}),
])
def test_verdict_codes_hold_on_several_datasets(params, kwargs, codes):
    for seed in (7,) + C.SWEEP_SEEDS:
        a = ev.analyse_for(params + (20000, seed), ev.Settings(**kwargs), with_comparators=False)
        assert ev.verdict(a)[1] in codes, (seed, ev.verdict(a)[1])


def test_verdict_missed_and_poor_and_levels():
    a = ev.analyse_for((4, 4, 1.0, 1.0, 0.0, 3.0, 20000, 7), ev.Settings(), with_comparators=False)                # weit über der Reglergrenze: nur die Verdict-Zweige prüfen
    assert ev.verdict(a)[:2] == ("warning", "missed")
    good = ev.analyse_for((4, 4, 2.0, 1.0, 0.0, 0.05, 20000, 7), ev.Settings(), with_comparators=False)
    assert ev.verdict(good)[0] == "success" and ev.verdict(good)[1] == "matching_better"
    ghosts = ev.analyse_for((4, 4, 1.0, 1.0, 0.0, 0.05, 20000, 7), ev.Settings(min_amplitude=0.0), with_comparators=False)
    assert ev.verdict(ghosts)[0] == "warning"
    data = ev.verdict(good)[2]
    assert {"f1", "pipe_f1", "oracle_f1", "accuracy_collision", "pipe_accuracy_collision", "precision", "ghosts", "collision_share", "k", "m"} <= set(data)


def test_sweep_seeds_are_separate_from_demo_seeds():
    assert min(C.SWEEP_SEEDS) >= 100_000 > C.DEFAULT_SEED and len(C.SWEEP_SEEDS) >= 5


def test_sweeps_and_tables_have_the_expected_shape_and_are_deterministic():
    rows = ev.sweep("n_electrodes", values=(2, 4))
    assert rows == ev.sweep("n_electrodes", values=(2, 4)) and [r["x"] for r in rows] == [2, 4]
    assert all(set(r) >= {"f1", "pipe_f1", "oracle_f1", "accuracy", "recall", "precision", "collision_accuracy", "pipe_collision_accuracy", "collision_share", "ica", "sobi", "sca", "f1_std", "ghosts"} for r in rows)
    thr = ev.sweep("threshold", values=(3.0, 5.0))
    assert np.isnan(thr[0]["ica"]) and thr[0]["recall"] > 0.9
    rounds = ev.sweep("rounds", values=(0, 2))
    assert [r["x"] for r in rounds] == [0, 2]
    table = ev.refine_table()
    assert [r["scene"] for r in table] == [s for s, _ in ev.REFINE_SCENES] and all(set(r) >= set(ev.REFINE_METHODS) for r in table)
    scenes = ev.scene_table()
    assert [r["scene"] for r in scenes] == [s for s, _ in ev.SCENES] and all(r["matching_min"] <= r["matching"] <= r["matching_max"] for r in scenes)


def test_rounds_sweep_zero_means_no_refinement():
    a = ev.sweep("rounds", values=(0,))[0]["f1"]
    b = np.mean([ev.analyse(ev.make_dataset(seed=s), ev.Settings(refine="none"), with_comparators=False).tm.f1 for s in C.SWEEP_SEEDS])
    assert a == pytest.approx(b)


# --- Aussagen der App: Hilfetexte der Seitenleiste ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("m,tm,pipe", [(2, 1.00, 0.99), (3, 1.00, 0.98), (4, 0.99, 0.94), (5, 0.91, 0.74)])
def test_neuron_help_text(m, tm, pipe):
    a = _analyses(m=m)
    assert abs(_mean(a, "tm") - tm) < 0.03 and abs(_mean(a, "pipe") - pipe) < 0.03


@pytest.mark.parametrize("n,tm,pipe", [(1, 0.55, 0.51), (2, 0.98, 0.95), (4, 0.99, 0.94)])
def test_electrode_help_text(n, tm, pipe):
    a = _analyses(n=n)
    assert abs(_mean(a, "tm") - tm) < 0.03 and abs(_mean(a, "pipe") - pipe) < 0.03


def test_signal_separators_fail_with_one_electrode_where_the_matching_still_runs():
    a = _analyses(n=1, comparators=True)
    assert abs(_cmp(a, "ica") - 0.16) < 0.03 and abs(_cmp(a, "sobi") - 0.16) < 0.03 and abs(_cmp(a, "sca") - 0.05) < 0.04


@pytest.mark.parametrize("rate,collision,tm,pipe", [(0.25, 0.02, 1.00, 0.99), (1.0, 0.16, 0.99, 0.94), (2.0, 0.34, 0.97, 0.77), (4.0, 0.57, 0.93, 0.59)])
def test_rate_help_text(rate, collision, tm, pipe):
    a = _analyses(rate_scale=rate)
    assert abs(_mean(a, "tm", "collision_share") - collision) < 0.03 and abs(_mean(a, "tm") - tm) < 0.03 and abs(_mean(a, "pipe") - pipe) < 0.04


@pytest.mark.parametrize("similarity,tm,pipe", [(0.0, 0.98, 0.82), (1.0, 0.99, 0.94)])
def test_similarity_help_text(similarity, tm, pipe):
    a = _analyses(similarity=similarity)
    assert abs(_mean(a, "tm") - tm) < 0.03 and abs(_mean(a, "pipe") - pipe) < 0.04


@pytest.mark.parametrize("jitter,tm,pipe", [(0.2, 0.98, 0.91), (0.3, 0.95, 0.84)])
def test_jitter_help_text(jitter, tm, pipe):
    a = _analyses(jitter=jitter)
    assert abs(_mean(a, "tm") - tm) < 0.03 and abs(_mean(a, "pipe") - pipe) < 0.04


def test_noise_help_text():
    a = _analyses(noise=1.0, comparators=True)
    assert abs(_mean(a, "tm") - 0.99) < 0.02 and abs(_mean(a, "pipe") - 0.90) < 0.03
    assert abs(_cmp(a, "ica") - 0.43) < 0.05 and abs(_cmp(a, "sobi") - 0.38) < 0.05 and abs(_cmp(a, "sca") - 0.86) < 0.05


def test_threshold_help_text():
    """Beleg: bei geringem Rauschen ist alles von 2 bis 12 gleich gut (0.99); bei Rauschen 1.0 sind es 0.99 (Schwelle 5) gegen 0.93 (12)."""
    for thr in (2.0, 5.0, 12.0):
        assert abs(_mean(_analyses(ev.Settings(threshold=thr)), "tm") - 0.99) < 0.02
    assert abs(_mean(_analyses(ev.Settings(threshold=5.0), noise=1.0), "tm") - 0.99) < 0.02 and abs(_mean(_analyses(ev.Settings(threshold=12.0), noise=1.0), "tm") - 0.93) < 0.03


def test_min_amplitude_help_text():
    """Beleg: ohne Grenze F1 0.35; ab 0.2 gut (0.98); 0.3-0.7 gleich gut (0.98-0.99); bei 0.9 nur 68 % der überlappenden Spikes richtig; etwa dreimal so viele Ereignisse wie Spikes ohne Grenze."""
    zero = _analyses(ev.Settings(min_amplitude=0.0))
    assert abs(_mean(zero, "tm") - 0.35) < 0.06 and 2.4 < np.mean([a.tm.n_detected / a.tm.n_true for a in zero]) < 3.4
    assert abs(_mean(_analyses(ev.Settings(min_amplitude=0.2)), "tm") - 0.98) < 0.02
    for amin in (0.3, 0.5, 0.7):
        assert 0.97 < _mean(_analyses(ev.Settings(min_amplitude=amin)), "tm") < 1.0
    assert abs(_mean(_analyses(ev.Settings(min_amplitude=0.9)), "tm", "accuracy_collision") - 0.68) < 0.05


def test_refine_help_text():
    """Beleg: bei doppelter Feuerrate keine / neu mitteln 0.84 und neu clustern 0.97; bei fünf Neuronen 0.77 / 0.79 / 0.91; wahre Vorlagen 0.97 / 0.96."""
    high = {r: _mean(_analyses(ev.Settings(refine=r), rate_scale=2.0), "tm") for r in C.REFINE_MODES}
    assert abs(high["none"] - 0.84) < 0.03 and abs(high["average"] - 0.84) < 0.03 and abs(high["recluster"] - 0.97) < 0.02
    five = {r: _mean(_analyses(ev.Settings(refine=r), m=5), "tm") for r in C.REFINE_MODES}
    assert abs(five["none"] - 0.77) < 0.04 and abs(five["average"] - 0.79) < 0.04 and abs(five["recluster"] - 0.91) < 0.03
    assert abs(_mean(_analyses(rate_scale=2.0), "oracle") - 0.97) < 0.02 and abs(_mean(_analyses(m=5), "oracle") - 0.96) < 0.02


def test_rounds_help_text():
    """Beleg: eine Runde genügt: bei doppelter Feuerrate 0.97 nach einer, zwei und drei Runden, bei fünf Neuronen 0.91."""
    for rounds in (1, 2, 3):
        assert abs(_mean(_analyses(ev.Settings(rounds=rounds), rate_scale=2.0), "tm") - 0.97) < 0.02
    assert abs(_mean(_analyses(ev.Settings(rounds=1), m=5), "tm") - 0.91) < 0.03


# --- Aussagen der App: Presets und Grenzen-Tabelle ----------------------------------------------------------------------------------------


def test_default_preset_numbers():
    """Beleg für 'Vier Neuronen, vier Elektroden': F1 0.99 gegen 0.94, überlappende Spikes 93 % gegen 85 % richtig, mit wahren Vorlagen 0.99."""
    a = _analyses()
    assert abs(_mean(a, "tm") - 0.99) < 0.02 and abs(_mean(a, "pipe") - 0.94) < 0.02 and abs(_mean(a, "oracle") - 0.99) < 0.02
    assert abs(_mean(a, "tm", "accuracy_collision") - 0.93) < 0.04 and abs(_mean(a, "pipe", "accuracy_collision") - 0.85) < 0.04


def test_one_electrode_preset_numbers():
    a = _analyses(n=1)
    assert abs(_mean(a, "oracle") - 0.94) < 0.03 and _mean(a, "tm") < 0.7 and _mean(a, "oracle") - _mean(a, "tm") > 0.3 and min(x.oracle.f1 - x.tm.f1 for x in a) > 0.05


def test_five_neurons_preset_numbers_at_the_preset_seed():
    a = ev.analyse(ev.make_dataset(m=5, seed=C.DEFAULT_SEED), ev.Settings(), with_comparators=False)
    assert 0.65 < a.tm.f1 < 0.8 and abs(a.oracle.f1 - 0.96) < 0.03


def test_high_rate_loses_eight_percent_of_the_spikes_at_factor_four():
    assert abs((1 - _mean(_analyses(rate_scale=4.0), "tm", "recall")) - 0.08) < 0.03


def test_start_pipeline_is_the_predecessor_pipeline():
    """Die Pipeline-Werte dieser Demo sind die des Vorgänger-Stücks (Standardfall: F1 0.945, Sortiergenauigkeit 0.98)."""
    a = _analyses()
    assert abs(_mean(a, "pipe") - 0.945) < 0.02 and abs(_mean(a, "pipe", "accuracy") - 0.98) < 0.02
    assert all(x.sorting.k == 4 for x in a) and isinstance(a[0].sorting, alg.Sorting)
