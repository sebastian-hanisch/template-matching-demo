"""Jedes Preset zeigt, was sein Name und seine Hilfe behaupten (Bänder mit dem ausgelieferten Code kalibriert, bewusst weit)."""

import pytest

import tm_constants as C
from tm_evaluation import Settings, analyse_for, verdict


def _measure(p):
    params = (p["m"], p["n"], p["rate_scale"], p["similarity"], p["jitter"], p["noise"], p["n_samples"], p["seed"])
    a = analyse_for(params, Settings(p["threshold"], p["min_amplitude"], p["refine"], p["rounds"], p["cluster_mode"]), with_comparators=False)
    return {"verdict": verdict(a)[1], "recall": a.tm.recall, "precision": a.tm.precision, "f1": a.tm.f1, "pipe_f1": a.pipe.f1, "oracle_f1": a.oracle.f1}


def test_every_preset_has_help_and_bands():
    assert set(C.PRESETS) == set(C.PRESET_HELP) == set(C.PRESET_EXPECTED_BANDS)
    assert len(C.PRESETS) == 6


def test_preset_settings_are_within_slider_bounds():
    for p in C.PRESETS.values():
        assert C.N_NEURONS_MIN <= p["m"] <= C.N_NEURONS_MAX and C.N_ELECTRODES_MIN <= p["n"] <= C.N_ELECTRODES_MAX
        assert C.RATE_SCALE_MIN <= p["rate_scale"] <= C.RATE_SCALE_MAX and abs(p["rate_scale"] * 4 - round(p["rate_scale"] * 4)) < 1e-9
        assert C.SIMILARITY_MIN <= p["similarity"] <= C.SIMILARITY_MAX and C.JITTER_MIN <= p["jitter"] <= C.JITTER_MAX and C.NOISE_MIN <= p["noise"] <= C.NOISE_MAX
        assert C.N_SAMPLES_MIN <= p["n_samples"] <= C.N_SAMPLES_MAX and p["n_samples"] % 1000 == 0
        assert C.MATCH_THRESHOLD_MIN <= p["threshold"] <= C.MATCH_THRESHOLD_MAX and abs(p["threshold"] * 2 - round(p["threshold"] * 2)) < 1e-9
        assert C.MIN_AMPLITUDE_MIN <= p["min_amplitude"] <= C.MIN_AMPLITUDE_MAX and abs(p["min_amplitude"] * 20 - round(p["min_amplitude"] * 20)) < 1e-9
        assert p["refine"] in C.REFINE_MODES and C.ROUNDS_MIN <= p["rounds"] <= C.ROUNDS_MAX and p["cluster_mode"] in C.CLUSTER_MODES


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_preset_stays_inside_its_bands(name):
    measured = _measure(C.PRESETS[name])
    for key, expected in C.PRESET_EXPECTED_BANDS[name].items():
        value = measured[key]
        if key == "verdict":
            assert value in expected, f"{key}: {value}"
        else:
            lo, hi = expected
            assert lo <= value <= hi, f"{key}: {value} nicht in [{lo}, {hi}]"
