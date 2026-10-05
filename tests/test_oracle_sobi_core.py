"""Orakel-Tests (unabhängige Rechenwege): verzögerte Kovarianzen per Schleife, gemeinsame Diagonalisierung gegen exakt diagonalisierbare Stapel und Stationarität,
Weißung gegen sklearn-PCA, Zuordnung gegen scipy linear_sum_assignment, Korrelationen gegen numpy corrcoef, Amari-Index gegen die Formel von Hand, Leistungsspektrum gegen scipy.signal.welch,
Spike-Erkennung per Schleife, Spitzen-F1 gegen maximale Zuordnung, Zufallsniveau der Trennschärfe per Quadratur."""

import numpy as np
import pytest

import tm_sobi as alg
import tm_ica as ica
import tm_evaluation as ev

scipy_signal = pytest.importorskip("scipy.signal")
scipy_opt = pytest.importorskip("scipy.optimize")
sk_decomposition = pytest.importorskip("sklearn.decomposition")


def _sym(rng, n):
    B = rng.standard_normal((n, n))
    return B + B.T


def test_lagged_covariances_match_a_loop_over_time():
    rng = np.random.default_rng(0)
    for _ in range(25):
        n, T = int(rng.integers(1, 5)), int(rng.integers(30, 80))
        Z = rng.standard_normal((n, T))
        lags = tuple(int(t) for t in rng.choice(np.arange(1, 20), 3, replace=False))
        R = alg.lagged_covariances(Z, lags)
        for k, tau in enumerate(lags):
            M = sum(np.outer(Z[:, t], Z[:, t + tau]) for t in range(T - tau)) / (T - tau)
            assert np.allclose(R[k], 0.5 * (M + M.T))


def test_off_diagonality_matches_a_loop():
    rng = np.random.default_rng(1)
    M = rng.standard_normal((3, 4, 4))
    off = sum(M[k, i, j] ** 2 for k in range(3) for i in range(4) for j in range(4) if i != j)
    assert abs(alg.off_diagonality(M) - off / (M ** 2).sum()) < 1e-12


def test_joint_diagonalization_recovers_random_exactly_diagonalizable_stacks():
    rng = np.random.default_rng(2)
    for _ in range(40):
        n, K = int(rng.integers(2, 6)), int(rng.integers(2, 7))
        Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
        M = np.array([Q @ np.diag(rng.standard_normal(n)) @ Q.T for _ in range(K)])
        V, history, angles, converged, _ = alg.joint_diagonalize(M)
        assert converged and np.allclose(V.T @ V, np.eye(n), atol=1e-10)
        assert max(np.abs(V.T @ A @ V - np.diag(np.diag(V.T @ A @ V))).max() for A in M) < 1e-10


def test_joint_diagonalization_of_a_noisy_stack_is_a_stationary_point_of_the_criterion():
    """Näherungsweise diagonalisierbar: keine kleine Rotation in einer Koordinatenebene senkt das Kriterium noch (Kriterium von Hand als Summe der Außerdiagonalquadrate)."""
    rng = np.random.default_rng(3)

    def crit(M, V):
        return sum(((V.T @ A @ V) ** 2).sum() - (np.diag(V.T @ A @ V) ** 2).sum() for A in M)

    for _ in range(15):
        Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
        M = np.array([Q @ np.diag(rng.standard_normal(3)) @ Q.T + 0.15 * _sym(rng, 3) for _ in range(4)])
        V = alg.joint_diagonalize(M)[0]
        base = crit(M, V)
        for p, q in ((0, 1), (0, 2), (1, 2)):
            for t in (1e-3, -1e-3):
                G = np.eye(3)
                G[p, p] = G[q, q] = np.cos(t)
                G[p, q], G[q, p] = -np.sin(t), np.sin(t)
                assert crit(M, V @ G) >= base - 1e-10


def test_amuse_eigenvalues_are_the_lagged_autocorrelations():
    """Weiße Mischung von AR(1)-Quellen: die Eigenwerte von R_tau sind phi^tau (Theorie)."""
    rng = np.random.default_rng(4)
    phis = np.array([0.9, 0.6, 0.3])
    S = np.array([(lambda s: (s - s.mean()) / s.std())(scipy_signal.lfilter([1.0], [1.0, -p], rng.standard_normal(60000))) for p in phis])
    X = rng.standard_normal((4, 3)) @ S
    m = alg.fit_amuse(X, 3, 3)
    assert np.allclose(np.sort(np.linalg.eigvalsh(m.covariances[0]))[::-1], phis ** 3, atol=0.03)


def test_sobi_unmixes_a_noise_free_ar_mixture_by_the_amari_formula_of_hand():
    rng = np.random.default_rng(5)
    S = np.array([(lambda s: (s - s.mean()) / s.std())(scipy_signal.lfilter([1.0], [1.0, -p], rng.standard_normal(30000))) for p in (0.95, 0.5, -0.5)])
    A = rng.standard_normal((5, 3))
    P = np.abs(alg.fit_sobi(A @ S, 3, tuple(range(1, 11))).unmixing @ A)
    ref = (sum(P[i].sum() / P[i].max() - 1 for i in range(3)) + sum(P[:, j].sum() / P[:, j].max() - 1 for j in range(3))) / (2 * 3 * 2)
    assert abs(ev.amari_index(P) - ref) < 1e-12 and ref < 0.1


def test_whitening_matches_sklearn_pca_up_to_sign():
    rng = np.random.default_rng(6)
    for _ in range(10):
        n = int(rng.integers(2, 6))
        X = rng.standard_normal((n, n)) @ (rng.standard_normal((n, 300)) * rng.uniform(0.5, 3, (n, 1)))
        nc = int(rng.integers(1, n + 1))
        Zp = sk_decomposition.PCA(n_components=nc, whiten=True).fit(X.T).transform(X.T).T * np.sqrt(300 / 299)
        assert np.allclose(np.abs(ica.whiten(X, nc).Z), np.abs(Zp), atol=1e-8)


def test_assignment_matches_scipy_including_ties_and_rectangular_shapes():
    rng = np.random.default_rng(7)
    for i in range(80):
        k, nc = int(rng.integers(1, 6)), int(rng.integers(1, 6))
        Cm = rng.uniform(0, 1, (k, nc))
        if i % 4 == 0:
            Cm = np.round(Cm, 1)
        idx = ev.assign(Cm)
        used = [c for c in idx if c >= 0]
        assert len(used) == len(set(used)) == min(k, nc)
        r, c = scipy_opt.linear_sum_assignment(Cm, maximize=True)
        assert abs(sum(Cm[a, b] for a, b in enumerate(idx) if b >= 0) - Cm[r, c].sum()) < 1e-9


def test_correlation_matrix_matches_numpy_corrcoef_and_matching_aligns_scale_and_sign():
    rng = np.random.default_rng(8)
    S = rng.standard_normal((3, 400))
    S = (S - S.mean(1, keepdims=True)) / S.std(1, keepdims=True)
    E = rng.standard_normal((4, 400)) + 0.5 * S[:1]
    assert np.allclose(ev.correlation_matrix(S, E), np.abs(np.corrcoef(np.vstack([S, E]))[:3, 3:]), atol=1e-10)
    scale, perm = rng.choice([-1, 1], 3) * rng.uniform(1, 4, 3), rng.permutation(3)
    E = (S * scale[:, None])[perm] + 0.05 * rng.standard_normal((3, 400))
    idx, corr, aligned = ev.matched(S, E)
    assert idx == [int(np.where(perm == i)[0][0]) for i in range(3)]
    for i, j in enumerate(idx):
        e, s = E[j] - E[j].mean(), S[i] - S[i].mean()
        assert np.allclose(aligned[i], (e @ s) / (e @ e) * e + S[i].mean(), atol=1e-10)


def test_autocorrelation_and_power_spectrum_match_numpy_and_scipy():
    rng = np.random.default_rng(9)
    S = rng.standard_normal((2, 400))
    S = (S - S.mean(1, keepdims=True)) / S.std(1, keepdims=True)
    acf = alg.autocorrelation(S, 12)
    for r in range(2):
        full = np.correlate(S[r], S[r], "full")[399:]
        assert np.allclose(acf[r], full[:13] / (400 - np.arange(13)))
    X = rng.standard_normal((2, 1000))
    freqs, spec = alg.power_spectrum(X, 100)
    f2, p2 = scipy_signal.welch(X, fs=1e4, window=np.hanning(200), nperseg=200, noverlap=0, detrend=False, scaling="spectrum")
    ratio = spec[:, 1:-1] / p2[:, 1:-1]
    assert np.allclose(freqs, f2) and np.allclose(ratio, ratio[0, 0], rtol=1e-6)


def test_spike_detection_matches_a_loop_and_f1_matches_the_maximal_matching():
    rng = np.random.default_rng(10)
    for _ in range(25):
        x = rng.standard_normal(3000) * 0.3
        pk = np.sort(rng.choice(np.arange(20, 2980, 40), int(rng.integers(0, 40)), replace=False))
        x[pk] -= rng.uniform(3, 10, len(pk))
        thr = -4 * np.median(np.abs(x - np.median(x))) / 0.6745
        ref = []
        for t in np.argsort(x):
            if x[t] > thr:
                break
            if any(abs(int(t) - u) <= 15 for u in ref):
                continue
            ref.append(int(t))
            if len(ref) == 10:
                thr = min(thr, 0.3 * float(np.median(x[ref])))
        if 0 < len(ref) < 10:                                                  # weniger als 10 Spitzen: Tiefenschwelle über den Median der gefundenen
            thr = min(thr, 0.3 * float(np.median(x[ref])))
            ref = [u for u in ref if x[u] <= thr]
        assert sorted(ref) == list(ev.detect_spikes(x))
    for _ in range(60):
        true = np.sort(rng.choice(np.arange(0, 400, 7), int(rng.integers(1, 40)), replace=False))
        det = np.unique(np.sort(rng.choice(np.arange(0, 400, 16), int(rng.integers(1, 22)), replace=False)) + int(rng.integers(-3, 4)))
        ok = (np.abs(true[:, None] - det[None, :]) <= 4).astype(float)
        r, c = scipy_opt.linear_sum_assignment(ok, maximize=True)
        tp = ok[r, c].sum()
        ref = 0.0 if tp == 0 else 2 * tp / (len(det) + len(true))
        assert abs(ev.spike_f1(det, true) - ref) < 1e-12
