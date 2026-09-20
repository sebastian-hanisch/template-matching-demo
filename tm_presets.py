"""SETTING_SPECS-Permalink-Muster, Presets und Zufalls-Seed-Button (Standardmuster aus dem OR-Demo-Portfolio, siehe ss_presets.py in spike-sorting-demo)."""

import math
import random
from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

import tm_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: Optional[float] = None
    hi: Optional[float] = None


def _choice(options):
    def cast(value):
        value = str(value)
        if value not in options:
            raise ValueError(value)
        return value
    return cast


SETTING_SPECS = {
    "n_neurons_slider": SettingSpec("m", int, C.DEFAULT_N_NEURONS, C.N_NEURONS_MIN, C.N_NEURONS_MAX),
    "n_electrodes_slider": SettingSpec("n", int, C.DEFAULT_N_ELECTRODES, C.N_ELECTRODES_MIN, C.N_ELECTRODES_MAX),
    "rate_slider": SettingSpec("rate", float, C.DEFAULT_RATE_SCALE, C.RATE_SCALE_MIN, C.RATE_SCALE_MAX),
    "similarity_slider": SettingSpec("sim", float, C.DEFAULT_SIMILARITY, C.SIMILARITY_MIN, C.SIMILARITY_MAX),
    "jitter_slider": SettingSpec("jitter", float, C.DEFAULT_JITTER, C.JITTER_MIN, C.JITTER_MAX),
    "noise_slider": SettingSpec("noise", float, C.DEFAULT_NOISE, C.NOISE_MIN, C.NOISE_MAX),
    "n_samples_slider": SettingSpec("T", int, C.DEFAULT_N_SAMPLES, C.N_SAMPLES_MIN, C.N_SAMPLES_MAX),
    "threshold_slider": SettingSpec("thr", float, C.DEFAULT_MATCH_THRESHOLD, C.MATCH_THRESHOLD_MIN, C.MATCH_THRESHOLD_MAX),
    "min_amplitude_slider": SettingSpec("amin", float, C.DEFAULT_MIN_AMPLITUDE, C.MIN_AMPLITUDE_MIN, C.MIN_AMPLITUDE_MAX),
    "refine_select": SettingSpec("refine", _choice(C.REFINE_MODES), C.DEFAULT_REFINE),
    "rounds_slider": SettingSpec("rounds", int, C.DEFAULT_ROUNDS, C.ROUNDS_MIN, C.ROUNDS_MAX),
    "cluster_mode_select": SettingSpec("kmode", _choice(C.CLUSTER_MODES), C.DEFAULT_CLUSTER_MODE),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, 2_000_000_000),
}
PRESET_KEYS = {"m": "n_neurons_slider", "n": "n_electrodes_slider", "rate_scale": "rate_slider", "similarity": "similarity_slider", "jitter": "jitter_slider", "noise": "noise_slider",
               "n_samples": "n_samples_slider", "threshold": "threshold_slider", "min_amplitude": "min_amplitude_slider", "refine": "refine_select", "rounds": "rounds_slider",
               "cluster_mode": "cluster_mode_select", "seed": "seed_input"}
KEPT = {"rounds_slider": "_rounds_kept"}        # bei ausgeblendetem Regler (Verfeinerung "keine") bleibt sein Wert hier erhalten


def init_session_state_defaults():
    """Fehlende Zustände auffüllen; die Zahl der Runden (nur bei Verfeinerung sichtbar, sonst vom Widget-Zustand gelöscht) kehrt zum zuletzt gewählten Wert zurück."""
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = st.session_state.get(KEPT[state_key], spec.default) if state_key in KEPT else spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
                if isinstance(value, float) and not math.isfinite(value):
                    continue
                if spec.lo is not None:
                    value = max(spec.lo, value)
                if spec.hi is not None:
                    value = min(spec.hi, value)
                st.session_state[state_key] = value
            except (ValueError, TypeError):
                pass
    st.session_state["rate_slider"] = round(st.session_state.get("rate_slider", C.DEFAULT_RATE_SCALE) * 4) / 4
    st.session_state["similarity_slider"] = round(st.session_state.get("similarity_slider", C.DEFAULT_SIMILARITY) * 20) / 20
    st.session_state["jitter_slider"] = round(st.session_state.get("jitter_slider", C.DEFAULT_JITTER) * 20) / 20
    st.session_state["threshold_slider"] = round(st.session_state.get("threshold_slider", C.DEFAULT_MATCH_THRESHOLD) * 2) / 2
    st.session_state["min_amplitude_slider"] = round(st.session_state.get("min_amplitude_slider", C.DEFAULT_MIN_AMPLITUDE) * 20) / 20
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        st.session_state[state_key] = C.PRESETS[name][key]
    st.session_state["_rounds_kept"] = C.PRESETS[name]["rounds"]


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, 2_000_000_000)
