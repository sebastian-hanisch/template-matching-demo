"""Rauchtests der Streamlit-Oberfläche per AppTest: Standard, jedes Preset, Randgrößen, Schritt-Zustand, ausgeblendete Regler, Zusatz-Experimente, Achsensperre."""

import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import tm_constants as C
from tm_presets import PRESET_KEYS

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app.py"
SUCCESS_PRESETS = ("Vier Neuronen, vier Elektroden", "Hohe Feuerrate")


def _run(setup=None, timeout=300):
    at = AppTest.from_file(str(APP), default_timeout=timeout)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    if setup is not None:
        setup(at)
        at.run()
        assert not at.exception, [e.value for e in at.exception]
    return at


def _apply(at, p):
    for key, state_key in PRESET_KEYS.items():
        at.session_state[state_key] = p[key]


def test_default_renders_without_exception():
    at = _run()
    assert any("Abgleich in Aktion" in m.value for m in at.markdown)
    assert not at.error and not at.warning and len(at.success) == 1


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_renders(name):
    at = _run(lambda a: _apply(a, C.PRESETS[name]))
    assert len(at.success) == (name in SUCCESS_PRESETS) and bool(at.warning) == (name not in SUCCESS_PRESETS)


def test_extreme_settings_render():
    def one_electrode(at):
        at.session_state["n_electrodes_slider"] = 1
    _run(one_electrode)

    def small(at):
        at.session_state["n_neurons_slider"] = C.N_NEURONS_MIN
        at.session_state["n_samples_slider"] = C.N_SAMPLES_MIN
        at.session_state["noise_slider"] = C.NOISE_MAX
        at.session_state["rate_slider"] = C.RATE_SCALE_MIN
        at.session_state["threshold_slider"] = C.MATCH_THRESHOLD_MAX
        at.session_state["min_amplitude_slider"] = C.MIN_AMPLITUDE_MAX
        at.session_state["refine_select"] = "none"
    _run(small)

    def large(at):
        at.session_state["n_neurons_slider"] = C.N_NEURONS_MAX
        at.session_state["n_electrodes_slider"] = C.N_ELECTRODES_MAX
        at.session_state["rate_slider"] = C.RATE_SCALE_MAX
        at.session_state["similarity_slider"] = C.SIMILARITY_MIN
        at.session_state["jitter_slider"] = C.JITTER_MAX
        at.session_state["n_samples_slider"] = C.N_SAMPLES_MAX
        at.session_state["cluster_mode_select"] = "silhouette"
        at.session_state["threshold_slider"] = C.MATCH_THRESHOLD_MIN
        at.session_state["min_amplitude_slider"] = C.MIN_AMPLITUDE_MIN
        at.session_state["refine_select"] = "average"
        at.session_state["rounds_slider"] = C.ROUNDS_MAX
    _run(large)


def test_no_spikes_detected_renders():
    def setup(at):
        at.session_state["noise_slider"] = C.NOISE_MAX
        at.session_state["rate_slider"] = C.RATE_SCALE_MIN
        at.session_state["n_neurons_slider"] = 2
        at.session_state["threshold_slider"] = C.MATCH_THRESHOLD_MAX
    for step in (1, 2, 3, 4, 5, 6):
        at = _run(setup)
        at.session_state["tm_step"] = step
        at.run()
        assert not at.exception


def test_rounds_slider_is_hidden_without_refinement_and_its_value_is_kept():
    at = _run(lambda a: (a.session_state.__setitem__("refine_select", "recluster"), a.session_state.__setitem__("rounds_slider", 4)))
    assert any(s.key == "rounds_slider" for s in at.slider)
    at.session_state["refine_select"] = "none"
    at.run()
    assert not at.exception and not any(s.key == "rounds_slider" for s in at.slider)
    at.session_state["refine_select"] = "average"
    at.run()
    assert not at.exception and [s for s in at.slider if s.key == "rounds_slider"][0].value == 4


def test_rounds_sweep_option_disappears_without_refinement():
    at = _run()
    sel = [s for s in at.selectbox if s.key == "sweep_select"][0]
    assert "rounds" in sel.options or "Verfeinerungsrunden" in sel.options
    sel.select("rounds")
    at.run()
    at.session_state["refine_select"] = "none"
    at.run()
    assert not at.exception and [s for s in at.selectbox if s.key == "sweep_select"][0].value != "rounds"


def test_step_state_resets_when_the_data_or_settings_change_and_survives_reruns():
    at = _run()
    at.session_state["tm_step"] = 4
    at.run()
    assert not at.exception and at.session_state["tm_step"] == 4
    at.session_state["noise_slider"] = 0.2
    at.run()
    assert not at.exception and at.session_state["tm_step"] == 1


@pytest.mark.parametrize("step", [1, 2, 3, 4, 5, 6])
def test_every_step_renders(step):
    def setup(at):
        at.session_state["tm_step"] = step
    _run(setup)


@pytest.mark.parametrize("step", [5, 6])
def test_late_steps_render_with_one_electrode_and_without_refinement(step):
    def setup(at):
        at.session_state["n_electrodes_slider"] = 1
        at.session_state["refine_select"] = "none"
        at.session_state["tm_step"] = step
    _run(setup)


def test_window_start_is_clamped_when_the_recording_gets_shorter():
    at = _run(lambda a: a.session_state.__setitem__("window_start", 1500))
    at.session_state["n_samples_slider"] = C.N_SAMPLES_MIN
    at.run()
    assert not at.exception and at.session_state["window_start"] == 0


def test_every_sweep_parameter_and_the_experiments_run_on_demand():
    at = _run()
    for parameter in ("rate_scale", "noise", "similarity", "jitter", "threshold", "min_amplitude", "rounds", "n_neurons", "n_electrodes"):
        [s for s in at.selectbox if s.key == "sweep_select"][0].select(parameter)
        at.run()
        assert not at.exception, [e.value for e in at.exception]
    for key in ("refine_start", "scenes_start"):
        [b for b in at.button if b.key == key][0].click()
        at.run()
        assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["refine_on"] and at.session_state["scenes_on"]


def test_every_figure_of_the_visualisation_module_is_axis_locked():
    source = (ROOT / "tm_visualization.py").read_text(encoding="utf-8")
    assert len(re.findall(r"return lock_axes\(fig\)", source)) == len(re.findall(r"^def build_", source, flags=re.M))
    assert len(re.findall(r"^\s+return fig$", source, flags=re.M)) == 1


def test_every_plotly_chart_has_an_explicit_key():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    calls = re.findall(r"plotly_chart\(", source)
    keyed = re.findall(r"plotly_chart\(.*?key=\"[a-z_]+\"\)", source)
    assert len(calls) == len(keyed) >= 14
