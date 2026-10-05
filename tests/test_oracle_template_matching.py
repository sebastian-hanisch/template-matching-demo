"""Orakel-Tests des Vorlagenabgleichs (unabhängige Rechenwege): gieriges Matching Pursuit gegen eine naive Fassung, die die Passung nach jedem Abziehen vollständig neu berechnet,
Kreuzkorrelationen und Passung gegen scipy.signal.correlate, bereinigte Ausschnitte per Schleife, Zuordnung erkannter Spitzen per Vollsuche, Ereignis-Auswertung gegen scipy linear_sum_assignment."""

import numpy as np
import pytest

import tm_algorithm as alg
import tm_constants as C
import tm_evaluation as ev
import tm_matching as tmm

scipy_opt = pytest.importorskip("scipy.optimize")
scipy_signal = pytest.importorskip("scipy.signal")

L = tmm.L


def _naive_pursuit(X, W, threshold, lo_a, hi_a):
    """Gierig wie in der Demo, aber ohne Aktualisierungstrick: nach jedem Abziehen werden alle Passungen aus dem Residuum neu berechnet (bereits gewählte Orte bleiben gesperrt)."""
    n, T = X.shape
    Q, S = W.shape[0], T - L + 1
    R = X.copy()
    norm = np.sqrt((W ** 2).sum(axis=(1, 2)))
    blocked = np.zeros((Q, S), bool)
    events = []
    for _ in range(int(tmm.MAX_EVENTS_FACTOR * T) + 10):
        Z = np.full((Q, S), -np.inf)
        for q in range(Q):
            for s in range(S):
                if not blocked[q, s]:
                    Z[q, s] = (R[:, s:s + L] * W[q]).sum() / norm[q]
        q0, s = divmod(int(np.argmax(Z)), S)
        z = Z[q0, s]
        if z < threshold:
            break
        a_raw = z / norm[q0]
        blocked[q0, s] = True
        if a_raw < lo_a:
            continue
        a = min(a_raw, hi_a)
        events.append((s, q0, a))
        R[:, s:s + L] -= a * W[q0]
    return events, R


def test_matching_pursuit_equals_the_naive_greedy_with_full_recomputation():
    rng = np.random.default_rng(0)
    n_events = 0
    for _ in range(25):
        Q, n, T = int(rng.integers(1, 4)), int(rng.integers(1, 4)), int(rng.integers(120, 200))
        W = rng.standard_normal((Q, n, L)) * rng.uniform(1.5, 4)
        X = rng.standard_normal((n, T)) * rng.uniform(0.1, 1.0)
        for _ in range(int(rng.integers(2, 7))):
            s = int(rng.integers(0, T - L))
            X[:, s:s + L] += rng.uniform(0.4, 1.8) * W[int(rng.integers(0, Q))]
        thr, lo, hi = float(rng.uniform(3, 8)), float(rng.choice([0.0, 0.2, 0.5])), float(rng.choice([1.0, 1.5]))
        p = tmm.matching_pursuit(X, W, thr, lo, hi)
        ref, R_ref = _naive_pursuit(X, W, thr, lo, hi)
        got = sorted((int(t) - C.SNIPPET_BEFORE, int(q), float(a)) for t, q, a in zip(p.times, p.templates, p.amplitudes))
        want = sorted((int(s), int(q), float(a)) for s, q, a in ref)
        assert len(got) == len(want) and all(g[:2] == w[:2] and abs(g[2] - w[2]) < 1e-6 for g, w in zip(got, want))
        assert np.allclose(p.residual, R_ref, atol=1e-6)
        assert [(int(p.times[j]) - C.SNIPPET_BEFORE, int(p.templates[j])) for j in p.order] == [(s, q) for s, q, _ in ref]        # `order` = Reihenfolge des Abziehens
        n_events += len(want)
    assert n_events > 40


def test_correlation_traces_and_cross_correlations_match_scipy():
    rng = np.random.default_rng(1)
    n, T, Q = 2, 120, 3
    X, W = rng.standard_normal((n, T)), rng.standard_normal((Q, n, L))
    Z = tmm.correlation_traces(X, W)
    XC = tmm.cross_correlations(W)
    for q in range(Q):
        assert np.allclose(Z[q], sum(scipy_signal.correlate(X[j], W[q, j], mode="valid") for j in range(n)) / np.sqrt((W[q] ** 2).sum()))
        for p in range(Q):
            full = sum(scipy_signal.correlate(W[q, j], W[p, j], mode="full") for j in range(n))
            assert np.allclose(XC[q, p], [full[(L - 1) - d] for d in range(-(L - 1), L)])


def test_cleaned_snippets_and_averaged_templates_match_a_direct_construction():
    rng = np.random.default_rng(2)
    n, T = 2, 300
    W = rng.standard_normal((2, n, L)) * 3
    X = rng.standard_normal((n, T)) * 0.3
    for s in (20, 32, 100, 105, 200):
        X[:, s:s + L] += rng.uniform(0.7, 1.3) * W[int(rng.integers(0, 2))]
    p = tmm.matching_pursuit(X, W, 5.0, 0.3)
    starts = p.times - C.SNIPPET_BEFORE
    cleaned = tmm.cleaned_snippets(W, p)
    for e in range(len(starts)):
        snip = X[:, starts[e]:starts[e] + L].copy()
        for e2 in range(len(starts)):
            if e2 != e:
                for l in range(L):
                    l2 = l + int(starts[e] - starts[e2])
                    if 0 <= l2 < L:
                        snip[:, l] -= p.amplitudes[e2] * W[p.templates[e2], :, l2]
        assert np.allclose(cleaned[e], snip / p.amplitudes[e], atol=1e-8)
    avg = tmm.average_templates(W, p)
    for q in range(2):
        members = p.templates == q
        assert np.allclose(avg[q], cleaned[members].mean(axis=0) if members.any() else W[q])


def test_match_detections_equals_a_full_search_also_for_crowded_truth():
    """Regression: früher wurden nur vier Nachbarn der Einfügestelle geprüft; bei mehreren gleichzeitigen wahren Spitzen blieben freie unentdeckt (Ereignisse des Abgleichs können dichter liegen als 15 Abtastwerte)."""
    assert list(ev.match_detections(np.array([60] * 5), np.array([60] * 6))) == [0, 1, 2, 3, 4]
    rng = np.random.default_rng(3)
    for _ in range(40):
        truth = np.sort(rng.integers(0, 300, int(rng.integers(5, 80))))
        det = np.sort(rng.integers(-5, 305, int(rng.integers(1, 80))))
        used, want = [False] * len(truth), []
        for t in det:
            best, best_d = -1, C.MATCH_TOLERANCE + 1
            for c, tt in enumerate(truth):
                if not used[c] and abs(int(tt) - int(t)) < best_d:
                    best, best_d = c, abs(int(tt) - int(t))
            if best >= 0:
                used[best] = True
            want.append(best)
        assert list(ev.match_detections(det, truth)) == want


def test_evaluate_events_of_a_pursuit_matches_an_independent_reference():
    rng = np.random.default_rng(4)
    for i in range(6):
        m = int(rng.integers(3, 6))
        ds = ev.make_dataset(m=m, n=int(rng.integers(2, 5)), rate_scale=float(rng.choice([1, 2, 4])), n_samples=8000, seed=int(rng.integers(0, 10**6)))
        mt = tmm.run_matching(ds.X, m, refine="none")
        times, labels, k = mt.pursuit.times, mt.pursuit.templates, max(mt.templates.shape[0], 1)
        r = ev.evaluate_events(ds, times, labels, k)
        truth = sorted(((int(t), j, int(s)) for j in range(m) for t, s in zip(ds.spike_times[j], ds.spike_starts[j])), key=lambda x: x[0])
        coll = [any((o[1], o[2]) != (j, s) and abs(o[2] - s) <= C.COLLISION_WINDOW for o in truth) for (t, j, s) in truth]
        used, mt_idx = [False] * len(truth), []
        for t in times:
            best, best_d = -1, C.MATCH_TOLERANCE + 1
            for c, tr in enumerate(truth):
                if not used[c] and abs(tr[0] - int(t)) < best_d:
                    best, best_d = c, abs(tr[0] - int(t))
            if best >= 0:
                used[best] = True
            mt_idx.append(best)
        ok = [c >= 0 for c in mt_idx]
        assert abs(r.recall - sum(ok) / len(truth)) < 1e-12 and r.n_ghosts == len(times) - sum(ok) and abs(r.precision - sum(ok) / max(len(times), 1)) < 1e-12
        conf = np.zeros((m, k))
        for q in range(len(times)):
            if ok[q]:
                conf[truth[mt_idx[q]][1], labels[q]] += 1
        assert np.array_equal(conf, r.confusion)
        rr, cc = scipy_opt.linear_sum_assignment(conf, maximize=True)
        assert abs(r.accuracy - conf[rr, cc].sum() / max(sum(ok), 1)) < 1e-12
        cmap = {c: j for j, c in enumerate(r.cluster_of_neuron) if c >= 0}
        correct = [ok[q] and cmap.get(int(labels[q]), -1) == truth[mt_idx[q]][1] for q in range(len(times))]
        single = [q for q in range(len(times)) if ok[q] and not coll[mt_idx[q]]]
        if single:
            assert abs(r.accuracy_single - sum(correct[q] for q in single) / len(single)) < 1e-12
        assert abs(r.collision_share - np.mean(coll)) < 1e-12


def test_oracle_templates_are_the_noise_free_mean_of_the_non_colliding_spikes():
    ds = ev.make_dataset(seed=5)
    W = ev.oracle_templates(ds)
    med = np.median(ds.X, axis=1, keepdims=True)
    Xc = (ds.X_clean - med) / alg.noise_sigma(ds.X)[:, None]
    truth = sorted(((int(t), j, int(s)) for j in range(ds.n_neurons) for t, s in zip(ds.spike_times[j], ds.spike_starts[j])), key=lambda x: x[0])
    for j in range(ds.n_neurons):
        singles = [t for (t, jj, s) in truth if jj == j and not any((o[1], o[2]) != (jj, s) and abs(o[2] - s) <= C.COLLISION_WINDOW for o in truth)
                   and t - C.SNIPPET_BEFORE >= 0 and t - C.SNIPPET_BEFORE + L <= Xc.shape[1]]
        assert np.allclose(W[j], np.mean([Xc[:, t - C.SNIPPET_BEFORE:t - C.SNIPPET_BEFORE + L] for t in singles], axis=0))
