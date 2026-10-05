import numpy as np

import tm_evaluation as ev


def _noise(n=3000):
    """Standardnormalrauschen, bei 3.5 sigma gekappt: robuste Schwelle 4 sigma liegt bei etwa -4."""
    return np.clip(np.random.default_rng(1).standard_normal(n), -3.5, 3.5)


def test_depth_threshold_applies_with_fewer_than_ten_spikes():
    """Regression: die Mindesttiefe (30 % der typischen Spitzentiefe) galt erst ab 10 gefundenen Spitzen. Bei weniger
    (kurze Aufnahmen) blieb ein Rauschausreißer über 4 sigma als Falschtreffer stehen, obwohl dieselbe Spur mit mehr
    Spitzen ihn verwirft. Mini-Instanz von Hand: vier Spitzen bei -20, Ausreißer bei -4.6; Median der Tiefen -20, 30 % = -6."""
    x = _noise()
    deep = [500, 1000, 1500, 2000]
    x[deep] = -20.0
    x[2500] = -4.6
    assert list(ev.detect_spikes(x)) == deep


def test_depth_threshold_is_the_same_with_few_and_many_spikes():
    x = _noise()
    deep = list(range(100, 2900, 200))                                    # 14 tiefe Spitzen
    x[deep] = -20.0
    x[2950] = -4.6
    assert list(ev.detect_spikes(x)) == deep
    few = deep[:4]
    y = _noise()
    y[few] = -20.0
    y[2950] = -4.6
    assert list(ev.detect_spikes(y)) == few


def test_few_spikes_of_similar_depth_all_survive_and_edge_cases():
    x = _noise()
    x[[300, 900, 1500]] = [-10.0, -8.0, -12.0]                            # 30 % von -10 = -3: alle über 4 sigma bleiben
    assert list(ev.detect_spikes(x)) == [300, 900, 1500]
    one = _noise()
    one[1200] = -9.0
    assert list(ev.detect_spikes(one)) == [1200]
    assert len(ev.detect_spikes(_noise())) == 0                           # keine Spitze: leeres Ergebnis
    quiet = 0.001 * np.clip(np.random.default_rng(2).standard_normal(1000), -3.5, 3.5)
    quiet[[200, 600]] = -1.0
    assert list(ev.detect_spikes(quiet)) == [200, 600]                    # sehr leises Rauschen: die Spitzen bleiben
