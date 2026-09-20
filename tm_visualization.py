"""Plotly-Visualisierungen der Vorlagenabgleich-Demo: Elektrodenlayout, Signalspuren, Vorlagen, Korrelationsspuren des Abgleichs, Subtraktion (Signal und Residuum), Verfeinerung, Verwechslungsmatrix,
Raster, Kennzahlen-Balken, Sweeps, Vorlagenqualität und Szenen-Vergleich. Alle Figuren laufen durch `lock_axes` (Touch-Scrolling-Konvention des Portfolios)."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import tm_constants as C
from tm_scenario import electrode_positions

BLUE, ORANGE, GREEN, RED, GRAY, PURPLE, TEAL = "#1f77b4", "#d68a2e", "#2ca02c", "#d62728", "#8a8f98", "#8e5fbf", "#00838f"
NEURON_COLORS = ("#1f77b4", "#d68a2e", "#2ca02c", "#8e5fbf", "#c2185b")
CLUSTER_COLORS = NEURON_COLORS + ("#00838f", "#6d4c41", "#455a64", "#9e9d24")
METHOD_COLORS = {"pipeline": BLUE, "matching": TEAL, "oracle": GRAY, "ica": ORANGE, "sobi": PURPLE, "sca": GREEN}
METHOD_NAMES = {"pipeline": "Pipeline", "matching": "Vorlagenabgleich", "oracle": "Abgleich mit wahren Vorlagen", "ica": "ICA", "sobi": "SOBI", "sca": "SCA"}
REFINE_LABELS = {"pipeline": "Pipeline", "none": "Abgleich, keine Verfeinerung", "average": "Abgleich, neu mitteln", "recluster": "Abgleich, neu clustern", "oracle": "Abgleich mit wahren Vorlagen"}
REFINE_COLORS = {"pipeline": BLUE, "none": "#9fc5e8", "average": "#4fb3bf", "recluster": TEAL, "oracle": GRAY}


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def source_color(i):
    return NEURON_COLORS[i % len(NEURON_COLORS)]


def source_labels(ds):
    return [f"Neuron {i + 1}" for i in range(ds.n_neurons)]


def template_neurons(cluster_of_neuron, n_templates):
    """Je Vorlage das Neuron, dem sie (optimal) zugeordnet ist; -1 = keinem Neuron zugeordnet."""
    out = [-1] * n_templates
    for i, c in enumerate(cluster_of_neuron):
        if 0 <= c < n_templates:
            out[c] = i
    return out


def template_color(neuron, q):
    return source_color(neuron) if neuron >= 0 else GRAY


def _window(t0, width):
    fs = C.SAMPLE_RATE
    return int(t0 * fs / 1000), int((t0 + width) * fs / 1000)


def build_layout(ds):
    """Elektroden (Quadrate auf y = 0) und Neuronen (Kreise, Größe = Spitzenamplitude)."""
    pos = electrode_positions(ds.n_electrodes)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pos[:, 0], y=pos[:, 1], mode="markers+text", text=[f"E{j + 1}" for j in range(ds.n_electrodes)], textposition="bottom center", name="Elektroden",
                             marker=dict(symbol="square", size=14, color=GRAY), hoverinfo="skip"))
    for i in range(ds.n_neurons):
        x, y = C.NEURON_POSITIONS[i]
        fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers+text", text=[f"N{i + 1}"], textposition="top center", name=f"Neuron {i + 1}", hoverinfo="skip",
                                 marker=dict(size=10 + 14 * C.NEURON_AMPLITUDES[i], color=source_color(i), opacity=0.85)))
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(range=[-0.1, 1.1], title="Ort (willkürliche Einheit)", zeroline=False),
                      yaxis=dict(range=[-0.15, 0.7], title="Abstand", zeroline=False), showlegend=False)
    return lock_axes(fig)


def build_traces(labels, arrays, t0, width, colors=None, spike_times=None, height=None, normalise=True):
    """Gestapelte Spuren eines Zeitfensters [t0, t0 + width) in ms (Abtastrate 10 kHz); optional Markierungen der Spitzen je Zeile."""
    fs = C.SAMPLE_RATE
    lo, hi = _window(t0, width)
    fig = go.Figure()
    n = len(arrays)
    scale_all = max(float(np.abs(a).max()) for a in arrays) if not normalise else None
    for r, (label, y) in enumerate(zip(labels, arrays)):
        seg = y[lo:hi]
        scale = float(np.abs(y).max()) if normalise else scale_all
        offset = (n - 1 - r) * 1.3
        color = colors[r] if colors else BLUE
        fig.add_trace(go.Scatter(x=np.arange(lo, hi) * 1000.0 / fs, y=offset + seg / max(scale, 1e-12), mode="lines", line=dict(color=color, width=1.2), name=label, hoverinfo="skip"))
        if spike_times is not None and r < len(spike_times):
            marks = [t for t in spike_times[r] if lo <= t < hi]
            if marks:
                fig.add_trace(go.Scatter(x=np.array(marks) * 1000.0 / fs, y=[offset + 0.75] * len(marks), mode="markers", marker=dict(symbol="triangle-down", size=7, color=color), hoverinfo="skip", showlegend=False))
    fig.update_layout(height=height or max(180, 42 * n + 60), margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
                      xaxis=dict(title="Zeit [ms]"), yaxis=dict(tickmode="array", tickvals=[(n - 1 - r) * 1.3 for r in range(n)], ticktext=list(labels), zeroline=False))
    return lock_axes(fig)


def _strongest(W, k=4):
    """Die (höchstens k) Elektroden mit der größten Vorlagenenergie, nach Elektrodennummer sortiert."""
    if W.shape[0] == 0:
        return list(range(min(k, W.shape[1])))
    energy = (W ** 2).sum(axis=(0, 2))
    return sorted(int(j) for j in np.argsort(energy)[::-1][:k])


def build_templates(W, neurons, W_start=None, W_oracle=None, neurons_start=None):
    """Vorlagen je Elektrode (Spalten = die stärksten Elektroden): durchgezogen die aktuellen Vorlagen (Farbe = zugeordnetes Neuron), gepunktet die Start-Vorlagen, grau gestrichelt die wahren Vorlagen (nur zur Kontrolle)."""
    neurons_start = neurons if neurons_start is None else neurons_start
    electrodes = _strongest(W if W.shape[0] else (W_oracle if W_oracle is not None else W))
    fig = make_subplots(rows=1, cols=max(len(electrodes), 1), subplot_titles=[f"Elektrode {j + 1}" for j in electrodes], shared_yaxes=True, horizontal_spacing=0.03)
    t = (np.arange(C.SNIPPET_BEFORE + C.SNIPPET_AFTER) - C.SNIPPET_BEFORE) * 1000.0 / C.SAMPLE_RATE
    for c, j in enumerate(electrodes, start=1):
        if W_oracle is not None:
            for i in range(len(W_oracle)):
                fig.add_trace(go.Scatter(x=t, y=W_oracle[i, j], mode="lines", line=dict(color=GRAY, width=1.5, dash="dash"), hoverinfo="skip", showlegend=False), row=1, col=c)
        if W_start is not None:
            for q in range(len(W_start)):
                fig.add_trace(go.Scatter(x=t, y=W_start[q, j], mode="lines", line=dict(color=template_color(neurons_start[q] if q < len(neurons_start) else -1, q), width=1.5, dash="dot"), hoverinfo="skip", showlegend=False), row=1, col=c)
        for q in range(len(W)):
            fig.add_trace(go.Scatter(x=t, y=W[q, j], mode="lines", line=dict(color=template_color(neurons[q] if q < len(neurons) else -1, q), width=3), hoverinfo="skip", showlegend=False), row=1, col=c)
    fig.update_xaxes(title="ms")
    fig.update_yaxes(title_text="Rausch-σ", col=1)
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10))
    return lock_axes(fig)


def build_correlation(Z, threshold, pursuit, neurons, t0, width):
    """Wie gut jede Vorlage zum Signal passt (Prüfgröße z, vor dem Abziehen), Schwelle gestrichelt; Rauten = gefundene Ereignisse (Farbe = Vorlage). Werte über dem Achsenende werden abgeschnitten."""
    fs = C.SAMPLE_RATE
    lo, hi = _window(t0, width)
    top = max(8.0 * threshold, 40.0)
    fig = go.Figure()
    for q in range(Z.shape[0]):
        seg = np.minimum(Z[q, max(lo - C.SNIPPET_BEFORE, 0): hi - C.SNIPPET_BEFORE], top)
        x = (np.arange(len(seg)) + max(lo - C.SNIPPET_BEFORE, 0) + C.SNIPPET_BEFORE) * 1000.0 / fs
        fig.add_trace(go.Scatter(x=x, y=seg, mode="lines", line=dict(color=template_color(neurons[q] if q < len(neurons) else -1, q), width=1.3), hoverinfo="skip", showlegend=False))
    fig.add_hline(y=threshold, line=dict(color=RED, dash="dash", width=1.5), annotation_text=f"Schwelle z = {threshold:g}", annotation_position="top left")
    inside = (pursuit.times >= lo) & (pursuit.times < hi)
    for e in np.flatnonzero(inside):
        q, t = int(pursuit.templates[e]), int(pursuit.times[e])
        z = min(float(Z[q, t - C.SNIPPET_BEFORE]), top)
        fig.add_trace(go.Scatter(x=[t * 1000.0 / fs], y=[z], mode="markers", marker=dict(symbol="diamond", size=9, color=template_color(neurons[q] if q < len(neurons) else -1, q), line=dict(color="black", width=1)),
                                 hoverinfo="skip", showlegend=False))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="Zeit [ms]"), yaxis=dict(title="z (Passung der Vorlage)", range=[-3, top * 1.05]))
    return lock_axes(fig)


def build_subtraction(Xw, W, pursuit, neurons, electrode, t0, width):
    """Oben: das Signal einer Elektrode (grau) und die abgezogenen Vorlagen (farbig, Skalierung mit ihrer Amplitude); unten: was übrig bleibt (Residuum) - mit einem Rest von Rauschen, wenn alles erklärt ist."""
    fs = C.SAMPLE_RATE
    lo, hi = _window(t0, width)
    x = np.arange(lo, hi) * 1000.0 / fs
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, subplot_titles=(f"Signal (Elektrode {electrode + 1}) und abgezogene Vorlagen", "Residuum nach dem Abziehen aller Vorlagen"))
    fig.add_trace(go.Scatter(x=x, y=Xw[electrode, lo:hi], mode="lines", line=dict(color=GRAY, width=2), hoverinfo="skip", showlegend=False), row=1, col=1)
    for e in range(len(pursuit.times)):
        s = int(pursuit.times[e]) - C.SNIPPET_BEFORE
        if s + W.shape[2] < lo or s >= hi:
            continue
        q = int(pursuit.templates[e])
        xs = np.arange(s, s + W.shape[2])
        keep = (xs >= lo) & (xs < hi)
        fig.add_trace(go.Scatter(x=xs[keep] * 1000.0 / fs, y=(pursuit.amplitudes[e] * W[q, electrode])[keep], mode="lines", line=dict(color=template_color(neurons[q] if q < len(neurons) else -1, q), width=2),
                                 hoverinfo="skip", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=pursuit.residual[electrode, lo:hi], mode="lines", line=dict(color=RED, width=1.3), hoverinfo="skip", showlegend=False), row=2, col=1)
    fig.update_xaxes(title="Zeit [ms]", row=2, col=1)
    fig.update_yaxes(title_text="Rausch-σ")
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=30, b=10))
    return lock_axes(fig)


def build_rounds(f1_by_round, pipe_f1, oracle_f1):
    """Spitzen-F1 des Abgleichs nach jeder Verfeinerungsrunde (Runde 0 = Start-Vorlagen aus der Pipeline); zum Vergleich die Pipeline und der Abgleich mit den wahren Vorlagen."""
    xs = list(range(len(f1_by_round)))
    fig = go.Figure(go.Scatter(x=xs, y=f1_by_round, mode="lines+markers", line=dict(color=TEAL, width=3), marker=dict(size=9), name="Vorlagenabgleich", hoverinfo="skip"))
    fig.add_hline(y=pipe_f1, line=dict(color=BLUE, dash="dash"), annotation_text="Pipeline", annotation_position="bottom right")
    fig.add_hline(y=oracle_f1, line=dict(color=GRAY, dash="dot"), annotation_text="mit wahren Vorlagen", annotation_position="top right")
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="Verfeinerungsrunde", dtick=1), yaxis=dict(title="Spitzen-F1", range=[0, 1.1]), showlegend=False)
    return lock_axes(fig)


def build_features(F, color_index, collision, title_x="Merkmal 1", title_y="Merkmal 2", centers=None, palette=None):
    """Merkmalsraum (erste zwei Merkmale): Punkt-Farbe = `color_index` (Cluster oder wahres Neuron; -1 = falsch erkannt, rot); Kollisions-Spikes als Kreuze; optional Cluster-Zentren."""
    palette = palette or NEURON_COLORS
    one_d = F.shape[1] < 2
    x = F[:, 0]
    y = np.random.default_rng(0).uniform(-1, 1, len(F)) if one_d else F[:, 1]
    fig = go.Figure()
    for j in sorted(set(int(v) for v in color_index)):
        for is_coll in (False, True):
            sel = (color_index == j) & (collision == is_coll)
            if sel.any():
                color = RED if j < 0 else palette[j % len(palette)]
                fig.add_trace(go.Scattergl(x=x[sel], y=y[sel], mode="markers", marker=dict(size=6 if is_coll else 5, color=color, symbol="x" if is_coll else "circle", opacity=0.7), hoverinfo="skip", showlegend=False))
    if centers is not None and len(centers):
        cy = np.zeros(len(centers)) if one_d else centers[:, 1]
        fig.add_trace(go.Scatter(x=centers[:, 0], y=cy, mode="markers", marker=dict(size=13, color="black", symbol="diamond-open", line=dict(width=2)), hoverinfo="skip", showlegend=False))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title=title_x), yaxis=dict(title="(zufällig gestreut)" if one_d else title_y, showticklabels=not one_d), showlegend=False)
    return lock_axes(fig)


def build_confusion(confusion, cluster_of_neuron):
    """Verwechslungsmatrix: Zeilen = wahre Neuronen, Spalten = Vorlagen bzw. Cluster (nach der optimalen Zuordnung sortiert, nicht zugeordnete hinten); die Diagonale ist die richtige Sortierung."""
    m, k = confusion.shape
    order = [c for c in cluster_of_neuron if c >= 0] + [c for c in range(k) if c not in cluster_of_neuron]
    Z = confusion[:, order]
    labels = [f"Vorlage {c + 1}" for c in order]
    fig = go.Figure(go.Heatmap(z=Z, x=labels, y=[f"Neuron {i + 1}" for i in range(m)], colorscale="Blues", text=Z.astype(int), texttemplate="%{text}", showscale=False, hoverinfo="skip"))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=60 + 42 * m, margin=dict(l=10, r=10, t=10, b=10))
    return lock_axes(fig)


def build_raster(truth_times, truth_neuron, detected, neuron_of_event, t0, width, n_neurons):
    """Spikes im Zeitfenster: je Neuron eine Zeile mit den wahren Spikes (oben, Strich) und den ihm zugeordneten Ereignissen (unten, Punkt); falsch zugeordnete liegen in der Zeile des falschen Neurons."""
    fs = C.SAMPLE_RATE
    lo, hi = _window(t0, width)
    fig = go.Figure()
    for i in range(n_neurons):
        base = (n_neurons - 1 - i) * 1.0
        tsel = truth_times[(truth_neuron == i) & (truth_times >= lo) & (truth_times < hi)]
        dsel = detected[(neuron_of_event == i) & (detected >= lo) & (detected < hi)]
        if len(tsel):
            fig.add_trace(go.Scatter(x=tsel * 1000.0 / fs, y=np.full(len(tsel), base + 0.25), mode="markers", marker=dict(symbol="line-ns-open", size=14, color=source_color(i), line=dict(width=2)), hoverinfo="skip", showlegend=False))
        if len(dsel):
            fig.add_trace(go.Scatter(x=dsel * 1000.0 / fs, y=np.full(len(dsel), base - 0.05), mode="markers", marker=dict(symbol="circle", size=8, color=source_color(i)), hoverinfo="skip", showlegend=False))
    fig.update_layout(height=60 + 60 * n_neurons, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="Zeit [ms]", range=[lo * 1000.0 / fs, hi * 1000.0 / fs]),
                      yaxis=dict(tickmode="array", tickvals=[(n_neurons - 1 - i) * 1.0 + 0.1 for i in range(n_neurons)], ticktext=[f"Neuron {i + 1}" for i in range(n_neurons)], zeroline=False))
    return lock_axes(fig)


def build_method_bars(tm, pipe, comparators):
    """Spitzen-F1 von Vorlagenabgleich, Pipeline und (falls berechnet) ICA, SOBI und SCA."""
    names = ["pipeline", "matching"] + [n for n in ("ica", "sobi", "sca") if n in comparators]
    f1 = [pipe.f1, tm.f1] + [comparators[n]["f1"] for n in names[2:]]
    fig = go.Figure(go.Bar(x=[METHOD_NAMES[n] for n in names], y=f1, marker_color=[METHOD_COLORS[n] for n in names], text=[f"{v:.2f}" for v in f1], textposition="outside", hoverinfo="skip"))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(title="Spitzen-F1 der Neuronen", range=[0, 1.15]), showlegend=False)
    return lock_axes(fig)


def build_sweep(rows, xlabel, current=None, log=False, with_comparators=True):
    """Links: Spitzen-F1 des Vorlagenabgleichs (mit Streuung), der Pipeline (gestrichelt), der wahren Vorlagen (gepunktet) und der Vergleichsverfahren; rechts: Trefferquote, Genauigkeit der Detektion und Kollisionsanteil."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Spitzen-F1", "Detektion und Kollisionen"), horizontal_spacing=0.12)
    xs = [r["x"] for r in rows]
    y, sd = np.array([r["f1"] for r in rows]), np.array([r["f1_std"] for r in rows])
    fig.add_trace(go.Scatter(x=xs + xs[::-1], y=list(y + sd) + list(y - sd)[::-1], fill="toself", fillcolor=TEAL, opacity=0.15, line=dict(width=0), hoverinfo="skip", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=xs, y=y, mode="lines+markers", name="Vorlagenabgleich", line=dict(color=TEAL, width=2.5), hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=xs, y=[r["pipe_f1"] for r in rows], mode="lines+markers", name="Pipeline", line=dict(color=BLUE, width=1.8, dash="dash"), hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=xs, y=[r["oracle_f1"] for r in rows], mode="lines+markers", name="mit wahren Vorlagen", line=dict(color=GRAY, width=1.5, dash="dot"), hoverinfo="skip"), row=1, col=1)
    if with_comparators:
        for name in ("ica", "sobi", "sca"):
            fig.add_trace(go.Scatter(x=xs, y=[r[name] for r in rows], mode="lines+markers", name=METHOD_NAMES[name], line=dict(color=METHOD_COLORS[name], width=1.6), hoverinfo="skip"), row=1, col=1)
    for key, name, color, dash in (("recall", "Trefferquote", GREEN, "solid"), ("precision", "Genauigkeit der Detektion", RED, "solid"), ("collision_accuracy", "richtig sortierte Kollisionen (Abgleich)", TEAL, "solid"),
                                   ("pipe_collision_accuracy", "richtig sortierte Kollisionen (Pipeline)", BLUE, "dash"), ("collision_share", "Kollisionsanteil", GRAY, "dot")):
        fig.add_trace(go.Scatter(x=xs, y=[r[key] for r in rows], mode="lines+markers", name=name, line=dict(color=color, width=2, dash=dash), hoverinfo="skip"), row=1, col=2)
    fig.update_xaxes(title=xlabel, type="log" if log else "linear")
    fig.update_yaxes(range=[0, 1.05])
    if current is not None:
        for col in (1, 2):
            fig.add_vline(x=current, line=dict(color=RED, dash="dash"), row=1, col=col)
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.35))
    return lock_axes(fig)


def build_refine_bars(rows):
    """Spitzen-F1 nach Vorlagenquelle je Szene (Mittel über die Sweep-Datensätze)."""
    labels = [r["scene"].replace(" (", "<br>(").replace(", ", ",<br>") for r in rows]
    fig = go.Figure()
    for name in ("pipeline", "none", "average", "recluster", "oracle"):
        y = [r[name] for r in rows]
        fig.add_trace(go.Bar(x=labels, y=y, name=REFINE_LABELS[name], marker_color=REFINE_COLORS[name], text=[f"{v:.2f}" for v in y], textposition="outside", hoverinfo="skip"))
    fig.update_layout(height=420, barmode="group", margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(title="Spitzen-F1 der Neuronen", range=[0, 1.2]), legend=dict(orientation="h", y=-0.35))
    return lock_axes(fig)


def build_scenes(rows):
    """Pipeline, Vorlagenabgleich, ICA, SOBI und SCA: Spitzen-F1 je Szene; Balken = Mittel, Fehlerbalken = Spanne über die Sweep-Datensätze."""
    labels = [r["scene"].replace(" (", "<br>(") for r in rows]
    fig = go.Figure()
    for name in ("pipeline", "matching", "ica", "sobi", "sca"):
        y = [r[name] for r in rows]
        fig.add_trace(go.Bar(x=labels, y=y, name=METHOD_NAMES[name], marker_color=METHOD_COLORS[name], text=[f"{v:.2f}" for v in y], textposition="outside", hoverinfo="skip",
                             error_y=dict(type="data", symmetric=False, array=[r[name + "_max"] - r[name] for r in rows], arrayminus=[r[name] - r[name + "_min"] for r in rows])))
    fig.update_layout(height=460, barmode="group", margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(title="Spitzen-F1 der Neuronen", range=[0, 1.2]), legend=dict(orientation="h", y=-0.5))
    return lock_axes(fig)
