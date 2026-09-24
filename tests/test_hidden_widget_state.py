"""Ausblendbare Regler (KEPT): Permalink und Preset legen ihren Wert nur in KEPT ab, nie im Zustand des Reglers - sonst zeigt der Regler später den Mindestwert, während die App mit dem gesetzten Wert rechnet
(nur im echten Browser sichtbar, AppTest sieht es nicht; der Zustand ist hier die prüfbare Invariante)."""

import importlib
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

MODULE = next(Path(__file__).resolve().parent.parent.glob("*_presets.py")).stem
P = importlib.import_module(MODULE)


def _load(module):
    import importlib

    import streamlit as st
    P = importlib.import_module(module)
    P.load_permalink_settings()
    st.session_state["stale"] = sorted(k for k in P.KEPT if k in st.session_state)
    st.session_state["kept"] = sorted(v for v in P.KEPT.values() if v in st.session_state)


def _preset(module, name):
    import importlib

    import streamlit as st
    P = importlib.import_module(module)
    P.apply_preset(name)
    st.session_state["stale"] = sorted(k for k in P.KEPT if k in st.session_state)
    st.session_state["kept"] = sorted(v for v in P.KEPT.values() if v in st.session_state)
    for key in P.KEPT:
        P.seed_widget(key)
    st.session_state["seeded"] = [st.session_state[k] == st.session_state[v] for k, v in P.KEPT.items()]


def test_permalink_writes_only_kept_for_hidden_widgets():
    at = AppTest.from_function(_load, args=(MODULE,))
    for state_key, spec in P.SETTING_SPECS.items():
        if state_key in P.KEPT:
            at.query_params[spec.url_param] = str(spec.default)
    at.run()
    assert not at.exception
    assert at.session_state["stale"] == []
    assert at.session_state["kept"] == sorted(P.KEPT.values())


@pytest.mark.parametrize("name", list(P.C.PRESETS))
def test_preset_writes_only_kept_for_hidden_widgets_and_seed_widget_restores_them(name):
    at = AppTest.from_function(_preset, args=(MODULE, name))
    at.run()
    assert not at.exception
    assert at.session_state["stale"] == []
    assert at.session_state["kept"] == sorted(P.KEPT.values())
    assert all(at.session_state["seeded"])
