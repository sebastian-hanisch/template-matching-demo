"""Vorlagenabgleich (Matching Pursuit, Kilosort-Stil) an Mehrelektroden-Signalen - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo EIN Verfahren - den Vorlagenabgleich des Spike-Sortings - und lässt stattdessen das Beispiel wachsen.
Fünftes Stück der Quellentrennung-Linie der "Konzepte"-Reihe und Nachfolger der Spike-Sorting-Standardpipeline: der Abgleich löst die überlappenden Spikes auf, an denen die Pipeline scheitert.
Wo er dafür seine Vorlagen herbekommt und was dann schiefgehen kann, wird hier gemessen. Siehe README für die Einordnung.

Lauffähig mit: streamlit run app.py
"""

import time

import numpy as np
import streamlit as st

import tm_constants as C
import tm_matching as tmm
from tm_evaluation import (
    SWEEP_LABELS, Settings, analyse_for, make_dataset, oracle_templates, refine_table, round_f1, scene_table, start_result, sweep, truth_spikes, verdict,
)
from tm_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from tm_visualization import (
    CLUSTER_COLORS,
    build_confusion,
    build_correlation,
    build_features,
    build_layout,
    build_method_bars,
    build_raster,
    build_refine_bars,
    build_rounds,
    build_scenes,
    build_subtraction,
    build_sweep,
    build_templates,
    build_traces,
    source_color,
    source_labels,
    template_neurons,
)

st.set_page_config(page_title="Vorlagenabgleich – Sebastian Hanisch", layout="wide")

STEP_LABELS = {1: "1 · Signal", 2: "2 · Start-Vorlagen", 3: "3 · Abgleich", 4: "4 · Abziehen", 5: "5 · Verfeinerung", 6: "6 · Ergebnis"}
WINDOW_WIDTH_MS = 60
SWEEP_OPTIONS = {"n_electrodes": "Anzahl Elektroden", "rate_scale": "Feuerrate", "noise": "Rauschen", "similarity": "Wellenform-Ähnlichkeit", "jitter": "Amplitudenschwankung", "n_neurons": "Anzahl Neuronen",
                 "threshold": "Schwelle z", "min_amplitude": "Kleinste Amplitude", "rounds": "Verfeinerungsrunden"}


def _pct(x):
    return "–" if np.isnan(x) else f"{x:.0%}"


@st.cache_data(show_spinner=False)
def _dataset(m, n, rate_scale, similarity, jitter, noise, n_samples, seed):
    return make_dataset(m, n, rate_scale, similarity, jitter, noise, n_samples, seed)


@st.cache_data(show_spinner=False)
def _analysis(data_params, settings):
    return analyse_for(data_params, settings)


@st.cache_data(show_spinner=False)
def _round_f1(data_params, settings):
    return round_f1(analyse_for(data_params, settings, with_comparators=False))


@st.cache_data(show_spinner=False)
def _oracle(data_params):
    return oracle_templates(make_dataset(*data_params))


@st.cache_data(show_spinner=False)
def _sweep(parameter, base, settings):
    m, n, rate_scale, similarity, jitter, noise, n_samples = base
    settings = Settings(**{**settings.__dict__, "cluster_mode": "known"})                # Sweeps rechnen mit bekannter Neuronenzahl
    return sweep(parameter, settings=settings, m=m, n=n, rate_scale=rate_scale, similarity=similarity, jitter=jitter, noise=noise, n_samples=n_samples)


@st.cache_data(show_spinner=False)
def _refine_rows(base, settings):
    m, n, rate_scale, similarity, jitter, noise, n_samples = base
    return refine_table(settings, similarity=similarity, jitter=jitter, noise=noise, n_samples=n_samples)


@st.cache_data(show_spinner=False)
def _scenes(base, settings):
    m, n, rate_scale, similarity, jitter, noise, n_samples = base
    settings = Settings(**{**settings.__dict__, "cluster_mode": "known"})
    return scene_table(settings, noise=noise, n_samples=n_samples)


st.title("🧩 Vorlagenabgleich – Spikes auflösen, die sich überlappen")
st.markdown(
    """
Die Standardpipeline des Spike-Sortings (Vorgänger-Stück) sortiert **Ereignisse**: Schwelle, Ausschnitt, Merkmale, Clustering. Ihre Schwäche: wo zwei Neuronen fast gleichzeitig feuern, ergibt der Ausschnitt eine **Mischform**,
die zu keinem Cluster passt - der zweite Spike geht verloren oder verfälscht das Clustering. Der **Vorlagenabgleich** (Kilosort-Stil) dreht die Sicht um: jedes Neuron hat eine **Vorlage** (seine typische Wellenform über alle Elektroden),
und die Aufnahme wird als **Summe verschobener, skalierter Vorlagen** erklärt. Gierig wird immer die Vorlage abgezogen, die am besten zu dem passt, was vom Signal übrig ist - das **Residuum**. Überlappende Spikes sind dann
kein Problem mehr: sie werden nacheinander abgezogen. Die Demo zeigt das - und woher die Vorlagen kommen, denn **die** sind die eigentliche Schwierigkeit.
Wie das Verfahren funktioniert, erklärt der aufgeklappte Abschnitt direkt darunter.
"""
)
st.caption(
    "Anders als die Fall-Demos im Portfolio, die an einem Anwendungsfall mehrere Verfahren vergleichen, zeigt diese Demo - fünftes Stück der Quellentrennung-Linie der \"Konzepte\"-Reihe, Nachfolger der Spike-Sorting-Standardpipeline - "
    "**ein** Verfahren an einem wachsenden Beispiel. Das Array ist dasselbe wie in der Pipeline-, ICA-, SOBI- und SCA-Demo."
)

with st.expander("So funktioniert der Vorlagenabgleich", expanded=True):
    st.markdown(
        """
1. **Start-Vorlagen.** Die Standardpipeline aus dem Vorgänger-Stück (Schwelle, Ausschnitte, PCA, k-means) liefert erste Cluster; die Vorlage eines Neurons ist der **Mittelwert** der Ausschnitte seines Clusters, über alle Elektroden.
   Alle Signale werden vorher je Elektrode durch ihre Rausch-Standardabweichung geteilt (das Rauschen hat dann überall Einheitsvarianz).
2. **Abgleich.** Für jede Vorlage $q$ und jeden Zeitpunkt $s$ wird gemessen, wie gut das Signal dort zur Vorlage passt: $z_q(s) = \\langle x_{s:s+30}, w_q\\rangle / \\lVert w_q \\rVert$. Unter reinem Rauschen ist $z$ ungefähr standardnormalverteilt -
   ein $z$ über der **Schwelle** ist kein Zufall.
3. **Abziehen.** Die beste Passung (größtes $z$ über alle Vorlagen und Zeiten) wird als Spike gezählt und mit ihrer besten **Amplitude** $a = \\langle x, w_q\\rangle / \\lVert w_q \\rVert^2$ vom Signal abgezogen; danach werden die Passungen in der Umgebung
   aktualisiert und der nächste Spike gesucht. Eine Amplitude unter der **kleinsten erlaubten Amplitude** zählt nicht als Spike, sondern als Rest eines unvollkommen abgezogenen (sonst meldet der Abgleich Geister); über 1.5 wird begrenzt.
4. **Verfeinerung.** Die Start-Vorlagen sind aus Clustern gemittelt, in denen überlappende Spikes stecken. Nach dem Abgleich lässt sich jeder Spike **ohne die anderen abgezogenen Spikes** ansehen (Residuum plus eigener Beitrag): die Vorlagen werden daraus neu gemittelt
   ("neu mitteln": verbessert die Form, ändert aber nicht, welcher Spike zu welchem Neuron gehört) oder die bereinigten Spikes werden noch einmal geclustert ("neu clustern": kann auch falsche Cluster ändern). Dann wird erneut abgeglichen.
5. **Bewertung** (nur hier möglich, weil die Wahrheit bekannt ist): Ereignisse werden den wahren Spikes zugeordnet (Zeitfenster ±0.6 ms), Vorlagen und Neuronen optimal einander zugeordnet. Für Pipeline und Abgleich gilt dieselbe Auswertung.

Was der Abgleich **verlangt**: Vorlagen, die stimmen, und ein Signal, das die Summe der Vorlagen plus Rauschen ist. Was er **nicht** verlangt: mindestens so viele Elektroden wie Neuronen oder Spikes, die sich nicht überlappen.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_cols = st.columns(len(C.PRESETS))
for i, name in enumerate(C.PRESETS.keys()):
    with preset_cols[i]:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name])

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    n_neurons = st.slider(
        "Neuronen", *bounds("n_neurons_slider"), key="n_neurons_slider",
        help="Spitzenartige Quellen wie in den anderen Demos. Bei 4 Elektroden Spitzen-F1 des Abgleichs (Pipeline): 1.00 (0.99) bei 2 Neuronen, 1.00 (0.98) bei 3, 0.99 (0.94) bei 4, 0.91 (0.74) bei 5 - "
             "bei 5 Neuronen liegen die Wellenformen dichter und die Start-Cluster sind öfter falsch.",
    )
    n_electrodes = st.slider(
        "Elektroden", *bounds("n_electrodes_slider"), key="n_electrodes_slider",
        help="Aufnahmestellen auf einer Zeile. Der Abgleich arbeitet wie die Pipeline auch mit einer, hilft dort aber nicht: Spitzen-F1 0.55 (Pipeline 0.51); mit zwei 0.98 (0.95), mit vier 0.99 (0.94). "
             "ICA, SOBI und SCA brauchen mehr: bei einer Elektrode 0.16 / 0.16 / 0.05.",
    )
    rate_scale = st.slider(
        "Feuerrate (Faktor)", *bounds("rate_slider"), key="rate_slider", step=0.25,
        help="Faktor auf die Feuerraten (20-35 Hz). Mehr Feuern heißt mehr überlappende Spikes: Anteil bei Faktor 0.25 / 1 / 2 / 4: 2 % / 16 % / 34 % / 57 %. Spitzen-F1 des Abgleichs 1.00 / 0.99 / 0.97 / 0.93, der Pipeline 0.99 / 0.94 / 0.77 / 0.59 - "
             "dort spielt der Abgleich seinen Vorteil aus.",
    )
    similarity = st.slider(
        "Wellenform-Ähnlichkeit", *bounds("similarity_slider"), key="similarity_slider", step=0.05,
        help="1 = die Neuronen haben verschieden breite Spitzen, 0 = alle dieselbe Form; dann unterscheiden sie sich nur über ihre Amplitudenverhältnisse an den Elektroden. Spitzen-F1 des Abgleichs 0.98 bei Ähnlichkeit 0 (Pipeline 0.82), 0.99 bei 1 (0.94).",
    )
    jitter = st.slider(
        "Amplitudenschwankung", *bounds("jitter_slider"), key="jitter_slider", step=0.05,
        help="Streuung der Spitzenhöhe von Spike zu Spike (relativ). Der Abgleich skaliert jede Vorlage frei (0.5-1.5) und kommt damit gut zurecht: Spitzen-F1 0.95 bei 0.3 (Pipeline 0.84), 0.98 bei 0.2 (0.91).",
    )
    noise = st.slider(
        "Rauschen", *bounds("noise_slider"), key="noise_slider", step=0.05,
        help="Sensorrauschen relativ zum Neuronen-Signal. Der Abgleich mittelt über alle 30 Abtastwerte und alle Elektroden: Spitzen-F1 0.99 bei 1.0 (Pipeline 0.90); ICA und SOBI leiden mehr (0.43 / 0.38), SCA weniger (0.86).",
    )
    n_samples = st.slider(
        "Länge der Aufnahme", *bounds("n_samples_slider"), key="n_samples_slider", step=1000,
        help="Abtastwerte bei 10 kHz. Mehr Länge = mehr Spikes zum Mitteln der Vorlagen.",
    )
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)

    st.markdown("**Vorlagenabgleich**")
    threshold = st.slider(
        "Schwelle z", *bounds("threshold_slider"), key="threshold_slider", step=0.5,
        help="Wie gut eine Vorlage passen muss, gemessen in Standardabweichungen des Rauschens. Bei geringem Rauschen ist alles von 2 bis 12 gleich gut (Spitzen-F1 0.99); erst bei starkem Rauschen kostet eine hohe Schwelle Spikes: "
             "bei Rauschen 1.0 sind es 0.99 (Schwelle 5), aber 0.93 bei 12. Niedrige Schwellen schaden nicht, weil die kleinste erlaubte Amplitude Geister abfängt.",
    )
    min_amplitude = st.slider(
        "Kleinste erlaubte Amplitude", *bounds("min_amplitude_slider"), key="min_amplitude_slider", step=0.05,
        help="Kleinster Anteil einer Vorlage, der noch als eigener Spike zählt. Ohne Grenze (0) bleibt nach jedem Abziehen ein kleiner Rest, der bei wenig Rauschen weit über der Schwelle liegt und als weiterer Spike gemeldet wird: "
             "Spitzen-F1 0.35. Ab 0.2 ist es wieder gut (0.98), 0.3-0.7 sind gleich gut (0.98-0.99); bei 0.9 gehen überlappende Spikes verloren (nur 68 % der Kollisionen richtig).",
    )
    refine = st.selectbox(
        "Verfeinerung der Vorlagen", C.REFINE_MODES, key="refine_select", format_func=lambda r: C.REFINE_LABELS[r],
        help="Keine: die Start-Vorlagen aus den Pipeline-Clustern bleiben. Neu mitteln: jede Vorlage aus den bereinigten Ausschnitten ihrer Spikes. Neu clustern: die bereinigten Ausschnitte werden noch einmal geclustert - "
             "das ist das Einzige, das auch falsche Cluster verändert: bei doppelter Feuerrate 0.97 (keine und neu mitteln 0.84), bei fünf Neuronen 0.91 (0.77 / 0.79).",
    )
    if refine != "none":
        rounds = st.slider(
            "Runden", *bounds("rounds_slider"), key="rounds_slider",
            help="Wie oft die Vorlagen neu geschätzt und erneut abgeglichen werden. Eine Runde genügt: bei doppelter Feuerrate 0.97 nach einer, zwei und drei Runden, bei fünf Neuronen 0.91.",
        )
        st.session_state["_rounds_kept"] = rounds
    else:
        rounds = int(st.session_state.get("_rounds_kept", C.DEFAULT_ROUNDS))
    cluster_mode = st.selectbox(
        "Neuronenzahl (für die Start-Vorlagen)", C.CLUSTER_MODES, key="cluster_mode_select", format_func=lambda c: C.CLUSTER_MODE_LABELS[c],
        help="Bekannt: die Pipeline bekommt die wahre Neuronenzahl. Unbekannt: die Silhouette wählt k unter 2-8 - sie wählt erfahrungsgemäß zu viele (siehe Pipeline-Demo), und jedes Cluster wird zu einer Vorlage.",
    )

    st.button("🎲 Neue Aufnahme generieren", width="stretch", on_click=randomize_seed, help="Würfelt einen neuen Zufalls-Seed für Spikezeiten und Rauschen.")

sync_query_params({
    "n_neurons_slider": int(n_neurons), "n_electrodes_slider": int(n_electrodes), "rate_slider": float(rate_scale), "similarity_slider": float(similarity), "jitter_slider": float(jitter),
    "noise_slider": noise, "n_samples_slider": int(n_samples), "threshold_slider": float(threshold), "min_amplitude_slider": float(min_amplitude), "refine_select": refine, "rounds_slider": int(rounds),
    "cluster_mode_select": cluster_mode, "seed_input": int(seed),
})

data_params = (int(n_neurons), int(n_electrodes), float(rate_scale), float(round(similarity, 2)), float(round(jitter, 2)), float(noise), int(n_samples), int(seed))
settings = Settings(threshold=float(threshold), min_amplitude=float(round(min_amplitude, 2)), refine=refine, rounds=int(rounds), cluster_mode=cluster_mode)
with st.spinner("Sortiere die Spikes..."):
    ds = _dataset(*data_params)
    analysis = _analysis(data_params, settings)
pipe, tm_res, orc, match = analysis.pipe, analysis.tm, analysis.oracle, analysis.matching
m_n, n_el = ds.n_neurons, ds.n_electrodes
labels = source_labels(ds)
colors = [source_color(i) for i in range(m_n)]
level, code, vd = verdict(analysis)
data_key = data_params + (settings,)
truth_times, truth_neuron, truth_collision = truth_spikes(ds)
W_start, W_final = match.templates_start, match.templates
start_res = start_result(analysis)
neurons_start = template_neurons(start_res.cluster_of_neuron, W_start.shape[0])
neurons_final = template_neurons(tm_res.cluster_of_neuron, W_final.shape[0])
has_templates = W_start.shape[0] > 0
best_electrode = int((W_start ** 2).sum(axis=(0, 2)).argmax()) if has_templates else 0

# --- Abgleich in Aktion ---------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Abgleich in Aktion")
if "tm_step" not in st.session_state or st.session_state.get("tm_step_owner") != data_key:
    st.session_state["tm_step"] = 1
    st.session_state["tm_step_owner"] = data_key
duration_ms = ds.S.shape[1] * 1000.0 / C.SAMPLE_RATE
max_start = int(duration_ms - WINDOW_WIDTH_MS)
if st.session_state.get("window_start", 0) > max_start:
    st.session_state["window_start"] = 0
step_col, play_col, win_col = st.columns([4, 2, 3])
with step_col:
    step = st.select_slider("Schritt", options=list(STEP_LABELS), key="tm_step", format_func=lambda s: STEP_LABELS[s])
with play_col:
    auto_play = st.button("▶️ Abspielen", width="stretch")
with win_col:
    window = st.slider(f"Zeitfenster ({WINDOW_WIDTH_MS} ms) ab [ms]", 0, max_start, key="window_start", step=10, help="Welchen Ausschnitt der Aufnahme die Zeitreihen zeigen.")
view_slot = st.empty()


def _render(current_step):
    with view_slot.container():
        if current_step == 1:
            c1, c2 = st.columns([3, 2])
            c1.markdown("**Die Elektrodensignale** (▼ = wahre Spitzen der Neuronen)")
            c1.plotly_chart(build_traces([f"E{j + 1}" for j in range(n_el)], list(ds.X), window, WINDOW_WIDTH_MS, normalise=False), width="stretch", key="step_electrodes")
            c2.markdown("**Ort von Neuronen und Elektroden**")
            c2.plotly_chart(build_layout(ds), width="stretch", key="step_layout")
            st.markdown("**Die wahren Neuronen** (unbekannt in der Praxis)")
            st.plotly_chart(build_traces(labels, list(ds.S), window, WINDOW_WIDTH_MS, colors, [ds.spike_times[i] for i in range(m_n)]), width="stretch", key="step_sources")
        elif current_step == 2:
            if has_templates:
                sorting = analysis.sorting
                F2 = sorting.features.values[:, :2]
                c1, c2 = st.columns([2, 3])
                c1.markdown("**Cluster der Standardpipeline** (Kreuze: überlappende Spikes)")
                centers = np.array([F2[sorting.clustering.labels == j].mean(axis=0) for j in range(sorting.k) if (sorting.clustering.labels == j).any()])
                c1.plotly_chart(build_features(F2, sorting.clustering.labels, pipe.collision_of_event, "PC 1", "PC 2", centers, CLUSTER_COLORS), width="stretch", key="step_clusters")
                c2.markdown("**Start-Vorlagen** (Mittel je Cluster; farbig = zugeordnetes Neuron, grau gestrichelt = die wahren Vorlagen zur Kontrolle)")
                c2.plotly_chart(build_templates(W_start, neurons_start, W_oracle=_oracle(data_params)), width="stretch", key="step_templates")
            else:
                st.info("Die Pipeline hat (fast) keine Spikes gefunden - es gibt keine Vorlagen.")
        elif current_step == 3:
            if has_templates:
                st.markdown("**Passung jeder Vorlage zum Signal** (z, vor dem Abziehen; Rauten = gefundene Ereignisse)")
                st.plotly_chart(build_correlation(tmm.correlation_traces(match.Xw, W_start), settings.threshold, match.pursuit_start, neurons_start, window, WINDOW_WIDTH_MS), width="stretch", key="step_correlation")
            else:
                st.info("Ohne Vorlagen gibt es nichts abzugleichen.")
        elif current_step == 4:
            if has_templates:
                st.markdown(f"**Abziehen** (Elektrode {best_electrode + 1}, die mit den größten Vorlagen)")
                st.plotly_chart(build_subtraction(match.Xw, W_start, match.pursuit_start, neurons_start, best_electrode, window, WINDOW_WIDTH_MS), width="stretch", key="step_subtraction")
            else:
                st.info("Ohne Vorlagen gibt es nichts abzuziehen.")
        elif current_step == 5:
            if has_templates:
                c1, c2 = st.columns([3, 2])
                c1.markdown("**Vorlagen vor (gepunktet) und nach (durchgezogen) der Verfeinerung**; grau gestrichelt: die wahren Vorlagen")
                c1.plotly_chart(build_templates(W_final, neurons_final, W_start=W_start, W_oracle=_oracle(data_params), neurons_start=neurons_start), width="stretch", key="step_refine_templates")
                c2.markdown("**Spitzen-F1 nach jeder Runde**")
                if refine != "none":
                    c2.plotly_chart(build_rounds(_round_f1(data_params, settings), pipe.f1, orc.f1), width="stretch", key="step_rounds")
                else:
                    c2.info("Verfeinerung ist ausgeschaltet: es bleiben die Start-Vorlagen.")
            else:
                st.info("Ohne Vorlagen gibt es nichts zu verfeinern.")
        else:
            c1, c2 = st.columns([3, 2])
            c1.markdown("**Raster: wahre Spikes (Striche) und ihre Sortierung durch den Abgleich (Punkte)**")
            c1.plotly_chart(build_raster(truth_times, truth_neuron, match.pursuit.times, tm_res.neuron_of_event, window, WINDOW_WIDTH_MS, m_n), width="stretch", key="step_raster")
            c2.markdown("**Verwechslungsmatrix** (Zeilen: wahre Neuronen, Spalten: Vorlagen)")
            c2.plotly_chart(build_confusion(tm_res.confusion, tm_res.cluster_of_neuron), width="stretch", key="step_confusion")


if auto_play:
    for s in STEP_LABELS:
        _render(s)
        time.sleep(1.2)
    step = 6
else:
    _render(step)

if step == 1:
    st.caption(f"{m_n} Neuronen feuern (insgesamt {tm_res.n_true} Spikes); {n_el} Elektrode(n) messen Mischungen. Der Abgleich sieht nur die Elektrodenspuren - die wahren Neuronen und ihre Spitzen stehen nur zur Bewertung da. "
               f"Anteil der Spikes, die mit einem anderen überlappen (Beginn höchstens 1.2 ms auseinander): {tm_res.collision_share:.0%}.")
elif step == 2:
    st.caption(f"Die Pipeline hat {analysis.sorting.k} Cluster gefunden{' (wie vorgegeben)' if cluster_mode == 'known' else ' (per Silhouette gewählt, wahr: ' + str(m_n) + ')'}; ihre Mittelwerte sind die Start-Vorlagen. "
               "Sie stecken voller überlappender Spikes (Kreuze im Merkmalsraum): das Mittel weicht deshalb von der wahren Vorlage ab (grau gestrichelt) - je häufiger die Überlappung, desto mehr. "
               f"Die Pipeline selbst erreicht Spitzen-F1 {pipe.f1:.2f}, der Abgleich mit diesen Vorlagen ohne Verfeinerung {start_res.f1:.2f}.")
elif step == 3:
    st.caption("Jede Kurve zeigt, wie gut eine Vorlage ab diesem Zeitpunkt zum Signal passt; ihr Maximum liegt genau am Spike des zugehörigen Neurons. Bei überlappenden Spikes zeigen zwei Vorlagen gleichzeitig hohe Werte. "
               f"Der Abgleich zieht immer den höchsten Wert zuerst ab und aktualisiert die Umgebung. Gefunden: {start_res.n_detected} Ereignisse (Trefferquote {start_res.recall:.0%}, Genauigkeit {start_res.precision:.0%}). "
               "Werte über dem Achsenende sind abgeschnitten.")
elif step == 4:
    st.caption("Oben das Signal (grau) und die abgezogenen Vorlagen (farbig, mit ihrer jeweiligen Amplitude); unten das Residuum. Bei einem einzelnen Spike bleibt nur Rauschen zurück - bei überlappenden Spikes wird erst der eine, "
               "dann der andere abgezogen und das Residuum ist am Ende auch dort flach. Ein Rest, der größer als das Rauschen bleibt, ist ein Spike, den der Abgleich nicht erklären konnte (falsche oder fehlende Vorlage).")
elif step == 5:
    st.caption({
        "none": "Keine Verfeinerung: es bleiben die Start-Vorlagen.",
        "average": f"Neu mitteln ({settings.rounds} Runde(n)): jede Vorlage aus den bereinigten Ausschnitten ihrer Spikes - die Kollisions-Verunreinigung fällt heraus, die Zuordnung der Spikes zu Vorlagen bleibt aber, wie sie ist. Das bringt meist wenig: die Vorlagen waren schon Mittel über viele Spikes, das eigentliche Problem sind falsche Cluster.",
        "recluster": f"Neu clustern ({settings.rounds} Runde(n)): die bereinigten Ausschnitte werden noch einmal per PCA und k-means gruppiert - ohne die Kollisions-Ausreißer sieht k-means die Neuronen richtig, auch wenn die Start-Cluster falsch waren.",
    }[refine] + f" Spitzen-F1 jetzt {tm_res.f1:.2f}; mit den wahren Vorlagen (Orakel, nur zur Kontrolle) wären es {orc.f1:.2f}.")
else:
    st.caption(f"Der Abgleich hat {tm_res.n_detected} Ereignisse gefunden ({tm_res.n_true} wahre Spikes). Sortiergenauigkeit {tm_res.accuracy:.0%} (nicht überlappende Spikes {_pct(tm_res.accuracy_single)}, "
               f"überlappende {_pct(tm_res.accuracy_collision)}; Pipeline: {_pct(pipe.accuracy_single)} / {_pct(pipe.accuracy_collision)}); ein Klassifikator, der immer das häufigste Neuron nennt, läge bei {tm_res.majority_baseline:.0%}.")

st.markdown("---")

# --- Ergebnis --------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Was der Abgleich gefunden hat - im Vergleich mit Pipeline, ICA, SOBI und SCA")
st.caption(
    "Spitzen-F1: je Neuron die Ereignisse seiner Vorlage (bzw. seines Clusters, bzw. die Spitzen seiner geschätzten Spur bei ICA, SOBI, SCA) gegen die wahren Spitzenzeiten (Toleranz ±4 Abtastwerte), gemittelt über die Neuronen - "
    "für alle Verfahren dieselbe Definition. Pipeline und Abgleich müssen zusätzlich jeden Spike der richtigen Neuronen-Nummer zuordnen."
)
best_cmp = max(analysis.comparators.values(), key=lambda v: v["f1"])["f1"] if analysis.comparators else float("nan")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Trefferquote (Abgleich)", f"{tm_res.recall:.0%}", delta=f"Pipeline {pipe.recall:.0%}", delta_color="off", help="Anteil der wahren Spikes, die als Ereignis gefunden wurden; darunter zum Vergleich die Trefferquote der Pipeline.")
m2.metric("Sortiergenauigkeit", f"{tm_res.accuracy:.0%}", delta=f"Pipeline {pipe.accuracy:.0%}", delta_color="off", help="Anteil der gefundenen wahren Spikes, die dem richtigen Neuron zugeordnet sind; darunter die Genauigkeit der Pipeline.")
m3.metric("Spitzen-F1 (Abgleich)", f"{tm_res.f1:.2f}", delta=f"{tm_res.f1 - pipe.f1:+.2f} ggü. Pipeline", delta_color="normal", help="Mittlerer Spitzen-F1 der Neuronen; im Delta der Abstand zur Standardpipeline.")
m4.metric("Geisterereignisse", f"{tm_res.n_ghosts}", delta=f"von {tm_res.n_detected} Ereignissen", delta_color="off", help="Ereignisse ohne wahren Spike: Reste unvollkommen abgezogener Spikes oder Rauschen.")

_t = vd
_cmp = f"ICA, SOBI und SCA erreichen hier {_t.get('ica_f1', float('nan')):.2f} / {_t.get('sobi_f1', float('nan')):.2f} / {_t.get('sca_f1', float('nan')):.2f}"
if code == "ghosts":
    st.warning(f"⚠️ Nur {_t['precision']:.0%} der {_t['n_events']} Ereignisse gehören zu einem wahren Spike, {_t['ghosts']} sind **Geister**: nach jedem Abziehen bleibt ein kleiner Rest, und bei so geringem Rauschen liegt schon ein Rest von wenigen Prozent "
               f"weit über der Schwelle. Die kleinste erlaubte Amplitude ({_t['min_amplitude']:g}) fängt solche Reste ab - höher stellen (bei geringem Rauschen genügt 0.2), bei starkem Rauschen auch die Schwelle. Spitzen-F1 {_t['f1']:.2f}.")
elif code == "missed":
    st.warning(f"⚠️ Der Abgleich findet nur {_t['recall']:.0%} der Spikes: bei diesem Rauschen liegen viele Passungen unter der Schwelle z = {_t['threshold']:g}. Schwelle senken - zu niedrige schaden kaum, weil die Amplitudengrenze Geister abfängt. Spitzen-F1 {_t['f1']:.2f}.")
elif code == "wrong_k":
    st.warning(f"⚠️ Die Silhouette hat {_t['k']} Cluster gewählt, wahr sind {_t['m']}: jedes Cluster wird zu einer Vorlage, die überzähligen Vorlagen gehören keinem Neuron und ziehen Spikes an sich. Spitzen-F1 {_t['f1']:.2f} (Pipeline {_t['pipe_f1']:.2f}, "
               f"mit den wahren Vorlagen {_t['oracle_f1']:.2f}). Der Abgleich braucht die richtige Neuronenzahl.")
elif code == "templates_wrong":
    if refine != "recluster":
        st.warning(f"⚠️ Die Vorlagen sind verfälscht: mit den wahren Vorlagen erreichte der Abgleich {_t['oracle_f1']:.2f}, mit diesen nur {_t['f1']:.2f}. Sie sind aus Clustern gemittelt, die durch überlappende Spikes falsch geschnitten sein können - "
                   "die Verfeinerung \"bereinigte Spikes neu clustern\" ist einen Versuch wert.")
    else:
        st.warning(f"⚠️ Auch das Neu-Clustern findet die Neuronen nicht: Spitzen-F1 {_t['f1']:.2f} gegen {_t['oracle_f1']:.2f} mit den wahren Vorlagen. Die Vorlagen sind falsch (Neuronen verschmolzen oder geteilt), der Abgleich selbst würde funktionieren. "
                   f"Die Pipeline erreicht {_t['pipe_f1']:.2f} - die Schwäche liegt im Clustering (k-means), das die Vorlagen liefert, nicht im Abgleich.")
elif code == "matching_better":
    st.success(f"✅ Der Abgleich löst die Überlappungen auf: Spitzen-F1 {_t['f1']:.2f} gegen {_t['pipe_f1']:.2f} der Pipeline; von den überlappenden Spikes sortiert er {_pct(_t['accuracy_collision'])} richtig, die Pipeline {_pct(_t['pipe_accuracy_collision'])} "
               f"({_t['collision_share']:.0%} aller Spikes überlappen). {_cmp} - sie lösen Überlappung auch auf, brauchen aber genug Elektroden und das Mischungsmodell.")
elif code == "matching_ok":
    st.success(f"✅ Der Abgleich sortiert gut: Spitzen-F1 {_t['f1']:.2f}, die Pipeline kommt hier auf {_t['pipe_f1']:.2f}. Bei nur {_t['collision_share']:.0%} überlappender Spikes ist der Vorsprung klein - mehr Feuerrate zeigt ihn. {_cmp}.")
else:
    st.warning(f"⚠️ Spitzen-F1 nur {_t['f1']:.2f} (Pipeline {_t['pipe_f1']:.2f}, mit den wahren Vorlagen {_t['oracle_f1']:.2f}).")

t1, t2 = st.columns(2)
with t1:
    st.markdown("**Raster: wahre Spikes (Striche) und ihre Sortierung durch den Abgleich (Punkte)**")
    st.plotly_chart(build_raster(truth_times, truth_neuron, match.pursuit.times, tm_res.neuron_of_event, window, WINDOW_WIDTH_MS, m_n), width="stretch", key="res_raster")
with t2:
    st.markdown("**Spitzen-F1: Abgleich, Pipeline, ICA, SOBI und SCA**")
    st.plotly_chart(build_method_bars(tm_res, pipe, analysis.comparators), width="stretch", key="res_bars")
st.caption("Punkte in der Zeile eines Neurons, ohne dass darüber ein Strich steht, sind falsch zugeordnet; Striche ohne Punkt darunter sind verpasste Spikes. ICA, SOBI und SCA stehen hier nur als Vergleich: sie trennen Signale und sortieren keine Ereignisse; "
           "ihr F1 kommt aus der Spitzenerkennung auf der geschätzten Neuronen-Spur.")

st.markdown("---")

# --- Sweeps ----------------------------------------------------------------------------------------------------------------------------

st.subheader("📐 Wie stark hängt das Ergebnis von Elektroden, Feuerrate, Rauschen, Wellenformen und den Reglern des Abgleichs ab?")
sweep_options = [p for p in SWEEP_OPTIONS if p != "rounds" or refine != "none"]
if st.session_state.get("sweep_select") not in sweep_options:
    st.session_state["sweep_select"] = sweep_options[0]
sweep_param = st.selectbox("Welcher Regler soll durchgefahren werden?", sweep_options, format_func=lambda p: SWEEP_OPTIONS[p], key="sweep_select")
current = {"n_electrodes": int(n_electrodes), "rate_scale": float(rate_scale), "noise": float(noise), "similarity": float(similarity), "jitter": float(jitter), "n_neurons": int(n_neurons),
           "threshold": float(threshold), "min_amplitude": float(min_amplitude), "rounds": int(rounds)}[sweep_param]
with st.spinner("Rechne den Sweep über 5 feste Datensätze..."):
    rows = _sweep(sweep_param, data_params[:7], settings)
st.plotly_chart(build_sweep(rows, SWEEP_LABELS[sweep_param], current=current, with_comparators=sweep_param in ("n_electrodes", "rate_scale", "noise", "similarity", "jitter", "n_neurons")), width="stretch", key="sweep_chart")
st.caption("Mittel und Streuung (Band) über 5 feste Sweep-Datensätze (getrennt vom Seed oben); alle anderen Regler wie in der Seitenleiste, aber mit bekannter Neuronenzahl. k-means ist nicht bei jedem Datensatz gleich gut - einzelne Datensätze schlagen stark aus. "
           "Die Regler des Abgleichs betreffen nur ihn (keine Vergleichslinien für ICA, SOBI, SCA); die gepunktete Linie zeigt, was mit den wahren Vorlagen möglich wäre. Verfeinerungsrunden 0 = keine Verfeinerung.")

st.markdown("---")

# --- Vorlagenqualität ------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Woher die Vorlagen kommen: Verfeinerung im Vergleich")
if st.button("Verfeinerungen vergleichen (dauert einige Sekunden)", key="refine_start"):
    st.session_state["refine_on"] = True
if st.session_state.get("refine_on"):
    with st.spinner("Vergleiche 4 Szenen × 5 Datensätze × 5 Vorlagenquellen..."):
        refine_rows = _refine_rows(data_params[:7], settings)
    st.plotly_chart(build_refine_bars(refine_rows), width="stretch", key="refine_chart")
    st.table({
        "Szene": [r["scene"] for r in refine_rows],
        "Pipeline": [f"{r['pipeline']:.2f}" for r in refine_rows],
        "keine": [f"{r['none']:.2f}" for r in refine_rows],
        "neu mitteln": [f"{r['average']:.2f}" for r in refine_rows],
        "neu clustern": [f"{r['recluster']:.2f} (min {r['recluster_min']:.2f})" for r in refine_rows],
        "wahre Vorlagen": [f"{r['oracle']:.2f}" for r in refine_rows],
    })
    st.caption("Spitzen-F1, Mittel über 5 feste Datensätze; Schwelle, kleinste Amplitude und Runden wie in der Seitenleiste, mit bekannter Neuronenzahl. Die letzte Spalte ist die Obergrenze: der Abgleich mit den wahren Vorlagen (in der Praxis unbekannt). "
               "Wo \"neu clustern\" sie erreicht, war die Überlappung das Problem; wo nicht (eine Elektrode), sind es die Cluster selbst.")

st.markdown("---")

# --- Szenen ----------------------------------------------------------------------------------------------------------------------------

st.subheader("🧩 Wer sortiert was: acht Szenen im Vergleich")
if st.button("Acht Szenen vergleichen (dauert einige Sekunden)", key="scenes_start"):
    st.session_state["scenes_on"] = True
if st.session_state.get("scenes_on"):
    with st.spinner("Vergleiche 8 Szenen × 5 Datensätze × 5 Verfahren..."):
        scene_rows = _scenes(data_params[:7], settings)
    st.plotly_chart(build_scenes(scene_rows), width="stretch", key="scenes_chart")
    st.table({
        "Szene": [r["scene"] for r in scene_rows],
        "Pipeline": [f"{r['pipeline']:.2f}" for r in scene_rows],
        "Abgleich": [f"{r['matching']:.2f} ({r['matching_min']:.2f}-{r['matching_max']:.2f})" for r in scene_rows],
        "ICA": [f"{r['ica']:.2f}" for r in scene_rows],
        "SOBI": [f"{r['sobi']:.2f}" for r in scene_rows],
        "SCA": [f"{r['sca']:.2f}" for r in scene_rows],
    })
    st.caption("Jede Szene legt Neuronen- und Elektrodenzahl und ihre Besonderheit fest; Rauschen (sofern die Szene es nicht setzt), Länge und Regler des Abgleichs wie in der Seitenleiste, mit bekannter Neuronenzahl. Mittel und Spanne über 5 feste Datensätze.")

st.markdown("---")

# --- Grenzen -----------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden - und wer danach kommt")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Vorlagen stimmen** | Kommen sie aus falschen Clustern, stimmt auch der Abgleich nicht: mit einer Elektrode 0.55 statt 0.94 mit den wahren Vorlagen (Preset "Eine Elektrode"). | Bessere Clusterverfahren für die Start-Vorlagen (siehe Clustering-Linie); Vorlagen aus Wissen über die Neuronen |
| **Das Signal ist die Summe der Vorlagen plus weißes Rauschen** | Korreliertes Rauschen lässt Passungen zufällig ausschlagen (hier wäre eine Weißung nötig - dieses Rauschen ist weiß). Nichtlineare Überlagerung wird nicht erklärt. | Weißung (Kilosort), Filter |
| **Die Vorlage ändert sich nicht (bis auf die Amplitude)** | Drift und Verformung (hier nicht modelliert) lassen Reste stehen, die als Geister auftauchen; die Amplitudengrenze fängt sie nur zum Teil ab. | Vorlagen, die nachgeführt werden |
| **Gieriges Abziehen findet die richtige Zerlegung** | Bei dichter Überlappung ist die beste Einzelpassung nicht immer die richtige; bei Preset "Hohe Feuerrate" mit Faktor 4 gehen 8 % der Spikes verloren. | Gemeinsame Optimierung mehrerer Ereignisse |
| **Neuronenzahl bekannt** | Jedes Start-Cluster wird eine Vorlage; die Silhouette wählt zu viele. | Übersplitten und Verschmelzen nach Regeln oder von Hand |
| **Wellenform trägt die Information** | Ein anderer Ansatz nutzt statt der Form die **Zeitverzögerungen** desselben Spikes über mehrere Elektroden (Verzögerungsgraph, Nachbarschaftsmengen, Clique-Überdeckungen) - in der Dissertation des Autors (Universität Rostock, 2017) untersucht. | **Verzögerungsgraph** (späteres Stück des Zweigs; dieses Stück misst nur den Vorlagenabgleich) |
"""
)
st.caption("Die genannten Verfahren sind die nächsten Stücke der Quellentrennung-Linie; hier steht nur, welche Annahme sie jeweils lockern. Ein Leistungsvergleich mit dem Verzögerungsgraph-Ansatz wird in dieser Demo nicht behauptet.")

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Modell.** Nach der Weißung ($x_j \leftarrow (x_j - \tilde x_j)/\hat\sigma_j$ je Elektrode $j$, robuste Schätzung $\hat\sigma_j$ aus dem Median der Abweichungen): $x(t) = \sum_{e} a_e\, w_{q_e}(t - t_e) + \varepsilon(t)$ - Ereignis $e$ hat eine Vorlage $q_e$, eine Zeit $t_e$ und eine Amplitude $a_e$;
die Vorlagen $w_q \in \mathbb{R}^{n \times 30}$ sind das Muster eines Neurons über alle Elektroden.

**Passung.** $c_q(s) = \sum_j \sum_{l=0}^{29} x_j(s + l)\, w_{q,j}(l)$, $z_q(s) = c_q(s)/\lVert w_q \rVert$. Unter reinem, weißem Rauschen ist $z_q(s) \sim \mathcal{N}(0, 1)$. Die beste Amplitude für Vorlage $q$ am Ort $s$ ist $a = c_q(s)/\lVert w_q \rVert^2 = z_q(s)/\lVert w_q \rVert$.

**Gieriges Abziehen (Matching Pursuit).** Wiederhole: wähle $(q, s) = \arg\max z_q(s)$; ist $z < \theta$ (Schwelle), Ende. Ist $a < a_{\min}$ (kleinste erlaubte Amplitude), verwirf $(q, s)$; sonst setze $a \leftarrow \min(a, 1.5)$, ziehe $a\,w_q$ ab
und aktualisiere die Passungen in der Umgebung exakt über die Kreuzkorrelation der Vorlagen: $z_p(s') \leftarrow z_p(s') - a \cdot \langle w_p(\cdot), w_q(\cdot + s' - s)\rangle / \lVert w_p \rVert$.

**Verfeinerung.** Bereinigter Ausschnitt eines Ereignisses: $\tilde s_e = (r(t_e - 8 : t_e + 22) + a_e w_{q_e})/a_e$ mit dem Residuum $r$ nach allen Abzügen. *Neu mitteln:* $w_q \leftarrow \operatorname{mean}\{\tilde s_e : q_e = q\}$.
*Neu clustern:* PCA und k-means auf allen $\tilde s_e$, $w_q \leftarrow$ Mittel des Clusters $q$. Danach erneut abgleichen.

**Bewertung.** Ereignisse und wahre Spitzen werden zugeordnet, wenn ihre Zeiten höchstens 6 Abtastwerte auseinanderliegen (jede wahre höchstens einmal). Vorlage $\leftrightarrow$ Neuron: optimale Zuordnung auf der Verwechslungsmatrix (Bitmasken-DP).
**Sortiergenauigkeit** = richtig zugeordnete / gefundene wahre Spikes. **Überlappung:** ein anderer Spike beginnt höchstens 12 Abtastwerte vor oder nach diesem. **Spitzen-F1** je Neuron aus den Zeiten seiner Ereignisse (Toleranz ±4), gemittelt.
**Orakel:** der Abgleich mit den wahren Vorlagen (Mittel des rauschfreien Signals über die nicht überlappenden Spikes eines Neurons) - eine Obergrenze, in der Praxis unbekannt.

**Grenzen.** (1) Die Vorlagen kommen aus einem Clustering, dessen Fehler sie erben. (2) Gierig ist nicht optimal. (3) Vorlagen sind konstant angenommen. (4) Die Neuronenzahl ist bekannt (oder schlecht geschätzt).

Implementiert in `tm_matching.py` (Weißung, Passung, Abziehen, Verfeinerung), `tm_algorithm.py` (Detektion, Ausschnitte, PCA, k-means, Silhouette - die Pipeline des Vorgänger-Stücks, wortgleich), `tm_ica.py`, `tm_sobi.py` und `tm_sca.py`
(Vergleichsverfahren, wortgleich aus den Vorgänger-Demos), `tm_scenario.py` (Mehrelektroden-Generator), `tm_evaluation.py` (Zuordnung, Kennzahlen, Sweeps, Tabellen, Urteil).
        """
    )

st.markdown("---")

st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
