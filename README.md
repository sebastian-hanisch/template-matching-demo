# Vorlagenabgleich – Spikes auflösen, die sich überlappen – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-template-matching-demo.streamlit.app/)**

Fünftes Stück der **Quellentrennung-Linie** der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning" und **Nachfolger der Spike-Sorting-Standardpipeline**:
anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo **ein** Verfahren – den **Vorlagenabgleich** (Matching Pursuit im Kilosort-Stil) – an einem wachsenden Beispiel,
mit der **Standardpipeline**, **ICA**, **SOBI** und **SCA** als Vergleich. Vehikel: dasselbe **Mehrelektroden-Array** wie in [spike-sorting-demo](../spike-sorting-demo), [ica-demo](../ica-demo), [sobi-demo](../sobi-demo) und [sca-demo](../sca-demo)
(dort übernommen, per Test gegen eingefrorene Werte geprüft); neu sind die Regler des Abgleichs (Schwelle, kleinste erlaubte Amplitude, Verfeinerung der Vorlagen).

**Einordnung in die Reihe (die Kanten des Graphen):** die Standardpipeline sortiert **Ereignisse** (Schwelle → Ausschnitt → Merkmale → Clustering) und **verliert überlappende Spikes**: zwei fast gleichzeitige Spikes ergeben eine Mischform, die zu keinem Cluster passt.
Der Vorlagenabgleich dreht die Sicht um: jedes Neuron hat eine **Vorlage** (seine Wellenform über alle Elektroden), die Aufnahme wird als **Summe verschobener, skalierter Vorlagen** erklärt, und gierig wird immer die am besten passende abgezogen – überlappende Spikes werden nacheinander aufgelöst.
Wie die Pipeline arbeitet der Abgleich auch mit **einer** Elektrode (ICA, SOBI und SCA brauchen mehr). Seine Vorlagen bekommt er aus den Clustern der Pipeline – und **die** sind die eigentliche Schwierigkeit.
```
ica-demo → sobi-demo → sca-demo
        ↘ nmf-demo (Nicht-Negativität statt Unabhängigkeit; einkanalfähig)
pca-demo + Clustering-Linie → spike-sorting-demo (Standardpipeline)
                              → template-matching-demo (Vorlagenabgleich: löst Überlappung auf)
                              → delay-graph-demo (Verzögerungsgraph: Zeitverzögerungen statt Wellenform)
```

| Frage | Ergebnis (4 Neuronen, 4 Elektroden, Rauschen 0.05, 20000 Abtastwerte; Mittel über 5 feste Datensätze, Seeds 100000–100004; Verfeinerung "neu clustern", 2 Runden) |
|---|---|
| Grundfall | ✅ Spitzen-F1 **0.99** (Pipeline 0.94); von den überlappenden Spikes (16 % aller) sortiert der Abgleich **93 %** richtig, die Pipeline 85 %. Mit den wahren Vorlagen (Orakel) wäre es ebenfalls 0.99 |
| Feuerrate | ✅ Kollisionsanteil 2 / 16 / 34 / 57 % bei Faktor 0.25 / 1 / 2 / 4; Spitzen-F1 des Abgleichs **1.00 / 0.99 / 0.97 / 0.93**, der Pipeline 0.99 / 0.94 / 0.77 / 0.59. Bei Faktor 4 gehen 8 % der Spikes verloren (Greedy ist nicht optimal) |
| Eine Elektrode | ❌ der Abgleich **hilft nicht**: 0.55 gegen 0.51 der Pipeline – die Vorlagen kommen aus Clustern, die schon falsch sind. **Mit den wahren Vorlagen 0.94**: der Abgleich selbst kann es, die Vorlagen fehlen. ICA / SOBI / SCA: 0.16 / 0.16 / 0.05 |
| Fünf Neuronen | ⚠️ 0.91 gegen 0.74 der Pipeline (Orakel 0.96), aber stark datensatzabhängig (schlechtester Datensatz 0.69) |
| Verfeinerung der Vorlagen | Bei doppelter Feuerrate: keine 0.84, **neu mitteln 0.84**, **neu clustern 0.97** (wahre Vorlagen 0.97); bei fünf Neuronen 0.77 / 0.79 / 0.91 (Orakel 0.96). Nur das Neu-Clustern ändert falsche Cluster – und eine Runde genügt |
| Amplitudengrenze | ⚠️ ohne Grenze (0): **etwa dreimal so viele Ereignisse wie Spikes** (Geister), Spitzen-F1 0.35; ab 0.2 gut (0.98), 0.3–0.7 gleich gut (0.98–0.99); bei 0.9 nur 68 % der überlappenden Spikes richtig |
| Schwelle z | bei geringem Rauschen ist alles von 2 bis 12 gleich gut (0.99); bei Rauschen 1.0: 0.99 (Schwelle 5), **0.93 bei 12**. Niedrige Schwellen schaden nicht – die Amplitudengrenze fängt Geister ab |
| Rauschen | ✅ robust: 0.99 bei Rauschen 1.0 (Pipeline 0.90, ICA 0.43, SOBI 0.38, SCA 0.86) |
| Wellenform-Ähnlichkeit / Amplitudenschwankung | ✅ 0.98 bei Ähnlichkeit 0 (Pipeline 0.82); 0.95 bei Schwankung 0.3 (Pipeline 0.84) |
| Rechenzeit | ✅ 0.15 s für die Analyse im Grundfall; 2.9 s im Extremfall (T = 40000, 8 Elektroden, 5 Neuronen, vierfache Feuerrate, 5 Runden) |

## Was die Demo zeigt

1. **Abgleich in Aktion** (Schritt-Slider + Abspielen, Zeitfenster-Regler): **Signal** → **Start-Vorlagen** (Cluster der Pipeline im Merkmalsraum; Vorlagen als Mittel je Cluster über die stärksten Elektroden, daneben die wahren Vorlagen zur Kontrolle) →
   **Abgleich** (Passung z jeder Vorlage über der Zeit, Schwelle, gefundene Ereignisse) → **Abziehen** (Signal mit abgezogenen Vorlagen oben, Residuum unten) → **Verfeinerung** (Vorlagen vor und nach; Spitzen-F1 nach jeder Runde gegen Pipeline und wahre Vorlagen) → **Ergebnis** (Raster, Verwechslungsmatrix).
2. **Was der Abgleich gefunden hat – im Vergleich mit Pipeline, ICA, SOBI und SCA:** Trefferquote, Sortiergenauigkeit, Spitzen-F1 (für alle Verfahren dieselbe Definition), Geisterereignisse; Raster; Urteil
   (Codes: Geister → verpasste Spikes → falsche Neuronenzahl → **falsche Vorlagen** (Abstand zum Abgleich mit den wahren Vorlagen) → Abgleich besser → gut / schlecht).
3. **📐 Sweeps** über Elektroden, Feuerrate, Rauschen, Wellenform-Ähnlichkeit, Amplitudenschwankung, Neuronenzahl, Schwelle, kleinste Amplitude und Verfeinerungsrunden (feste Datensätze ab 100000, Streuung; die Pipeline gestrichelt und die wahren Vorlagen gepunktet in jedem Diagramm; Sweeps rechnen mit bekannter Neuronenzahl).
4. **🔬 Woher die Vorlagen kommen:** vier Szenen × Pipeline / keine Verfeinerung / neu mitteln / neu clustern / wahre Vorlagen (Experiment auf Abruf).
5. **🧩 Wer sortiert was:** acht Szenen für Pipeline, Abgleich, ICA, SOBI und SCA mit Spanne über die Datensätze (Experiment auf Abruf).
6. **🚧 Grenzen:** Tabelle "Annahme – was passiert – wer setzt an" (Vorlagen müssen stimmen, weißes Rauschen, konstante Vorlagen, Greedy, Neuronenzahl, Verzögerungsgraph).

Regler: Neuronen (2–5), Elektroden (1–8), Feuerrate (×0.25–×4), Wellenform-Ähnlichkeit (0–1), Amplitudenschwankung (0–0.3), Rauschen, Länge, Schwelle z (2–12), kleinste erlaubte Amplitude (0–0.9), Verfeinerung (keine / neu mitteln / neu clustern), Runden (1–5; nur bei Verfeinerung sichtbar, Wert bleibt erhalten), Neuronenzahl für die Start-Vorlagen (bekannt / Silhouette).

## Messwerte der Presets (Seed 7; sie prüfen sich mit weiten Bändern selbst)

| Preset | Trefferquote | Genauigkeit der Detektion | Spitzen-F1 Abgleich | Pipeline | wahre Vorlagen | Urteil |
|---|---|---|---|---|---|---|
| Vier Neuronen, vier Elektroden | 0.99 | 1.00 | 0.986 | 0.938 | 0.981 | Abgleich sortiert gut |
| Hohe Feuerrate (×2) | 0.98 | 0.99 | 0.952 | 0.653 | 0.953 | Abgleich besser |
| Eine Elektrode | 0.92 | 1.00 | 0.396 | 0.383 | 0.918 | falsche Vorlagen |
| Fünf Neuronen | 0.96 | 0.99 | 0.718 | 0.662 | 0.957 | falsche Vorlagen |
| Ohne Amplitudengrenze | 1.00 | 0.31 | 0.338 | 0.938 | 0.556 | Geister |
| Hohe Feuerrate, keine Verfeinerung | 0.92 | 0.98 | 0.666 | 0.653 | 0.953 | falsche Vorlagen |

## Modell und Verfahren

- **Szenario** (`tm_scenario.py`): wie in der spike-sorting-demo (10 kHz; Neuronen mit biphasischer Wellenform und Poisson-artigem Feuern mit 2 ms Refraktärzeit, Mischung ∝ 1/(d² + ε), Rauschen relativ zum Neuronen-Signal, Wellenform-Ähnlichkeit, Amplitudenschwankung); bei den Standardwerten bit-identisch (per Test).
- **Vorlagenabgleich** (`tm_matching.py`, numpy von Grund auf): **Weißung** je Elektrode (Median abziehen, durch die robuste Rausch-Standardabweichung teilen); **Start-Vorlagen** = Mittel der Ausschnitte je Cluster der Pipeline; **Passung** z_q(s) = ⟨x_{s:s+30}, w_q⟩ / ‖w_q‖ (unter weißem Rauschen ≈ N(0, 1), per Test);
  **gieriges Abziehen**: größtes z über alle Vorlagen und Zeiten, Amplitude a = z/‖w_q‖ (a < kleinste Amplitude: verwerfen, a > 1.5: begrenzen), die Passungen in der Umgebung werden **exakt über die Kreuzkorrelation der Vorlagen aktualisiert** statt neu berechnet (gegen die vollständige Neuberechnung geprüft);
  **Verfeinerung**: bereinigter Ausschnitt = (Residuum + eigener Beitrag) / Amplitude; "neu mitteln" mittelt sie je Vorlage, "neu clustern" gruppiert sie noch einmal per PCA und k-means.
- **Pipeline** (`tm_algorithm.py`): die des Vorgänger-Stücks, wortgleich (Detektion, Ausschnitte, PCA, k-means, Silhouette; gegen scikit-learn geprüft) – liefert die Start-Vorlagen und den Vergleich.
- **Vergleich:** ICA, SOBI und SCA wortgleich aus den Vorgänger-Demos (`tm_ica.py`, `tm_sobi.py`, `tm_sca.py`): Spitzen-F1 auf der geschätzten Neuronen-Spur – dieselbe Definition wie für Pipeline und Abgleich.
- **Auswertung** (`tm_evaluation.py`): Zuordnung erkannter ↔ wahrer Spitzen (±6 Abtastwerte, jede wahre höchstens einmal), Kollisionen (ein anderer Spike beginnt höchstens 12 Abtastwerte vor oder nach diesem), Vorlage ↔ Neuron per optimaler Zuordnung (Bitmasken-DP), Sortiergenauigkeit gesamt / einzeln / Kollision, Geisterereignisse,
  **Orakel-Vorlagen** (Mittel des rauschfreien Signals über die nicht überlappenden Spikes eines Neurons – nur als Messreferenz), Sweeps, Vergleichs- und Vorlagen-Tabellen, Urteil.

## Was nicht funktioniert hat / Grenzen

- **Die erste Fassung meldete vier von fünf Ereignissen als Geister** (Genauigkeit der Detektion 0.21, Spitzen-F1 0.46): Matching Pursuit ist bei Überlappung nur näherungsweise exakt, und schon ein Rest von wenigen Prozent liegt bei so geringem Rauschen weit über einer Schwelle in Rausch-Standardabweichungen (z ≈ 20–30 für eine 10 %-Abweichung bei den hier vorkommenden Vorlagenlängen).
  Erst die **kleinste erlaubte Amplitude** – eine Passung, die nur einen Bruchteil einer Vorlage erklärt, ist kein Spike – löste das (0.98 ab 0.2). Die Schwelle allein tut es nicht: sie ist von 2 bis 12 wirkungslos, solange das Rauschen klein ist.
- **Das Problem sind nicht die Überlappungen im Mittel, sondern falsche Cluster.** Vor dem Bau war die Vermutung, dass Kollisions-Ausreißer die Mittelwerte verfälschen und "neu mitteln" das behebt. Gemessen: es ändert nichts (0.84 gegen 0.84 bei doppelter Feuerrate). Was hilft, ist das **Neu-Clustern** der bereinigten Spikes (0.97):
  ohne die Kollisions-Ausreißer sieht k-means die Neuronen richtig. Den Median statt des Mittelwerts für die Vorlagen zu nehmen brachte nichts (0.85 gegen 0.84 bei doppelter Feuerrate, bei fünf Neuronen schlechter) und wurde verworfen; weitere Runden ändern ab der ersten nichts mehr.
- **Bei einer Elektrode hilft auch das nicht** (0.55 gegen 0.94 mit den wahren Vorlagen): die Cluster sind dort nicht wegen der Überlappung falsch, sondern weil k-means die falsche Aufteilung mit der kleineren Streuung bevorzugt (siehe spike-sorting-demo). Der Abgleich erbt den Fehler seiner Vorlagen.
- **Die Vorlagen erben ihre Skalierung:** die Verfeinerung teilt durch die Amplitude, deshalb bleibt die Skalierung der Start-Vorlage erhalten (die Form wird richtig, die Größe nicht unbedingt) – im Abgleich fängt die freie Amplitude das ab.
- **Gieriges Abziehen ist nicht optimal:** bei dichter Überlappung (Faktor 4, 57 % Kollisionen) gehen 8 % der Spikes verloren; eine gemeinsame Optimierung mehrerer Ereignisse wäre der nächste Schritt und ist nicht enthalten.
- **Weißung nur je Elektrode:** hier ist das Rauschen weiß und unkorreliert; bei korreliertem Rauschen (Kilosort weißt über alle Elektroden) wäre eine volle Weißung nötig. **Keine Drift**: die Vorlagen sind konstant angenommen.
- **Die Neuronenzahl ist bekannt** (außer im Silhouette-Modus, wo jedes zu viel gewählte Cluster zu einer Vorlage wird); Sweeps und Szenen rechnen immer mit bekannter Neuronenzahl.
- **Der Verzögerungsgraph-Ansatz** (die Dissertation des Autors, Universität Rostock 2017: statt der Wellenform die Zeitverzögerungen desselben Spikes über mehrere Elektroden) wird nur genannt; **ein Leistungsvergleich mit ihm ist nicht Teil dieser Demo.**
- **Synthetische Daten:** feste Spitzenform je Neuron, exakt lineare Mischung, weißes Gauß'sches Rauschen, Elektroden auf einer Zeile. Literatur nur mit Namen: das Verfahren folgt dem Kilosort-Ansatz (Pachitariu et al.), vereinfacht.

## Verifikation

- Matching Pursuit mit Handinstanzen (isolierte Ereignisse exakt zurückgewonnen, Residuum 0; überlappende Ereignisse mit richtigen Zeiten und Vorlagen; Amplitudengrenzen; leere Eingaben); **Passung gegen eine Brute-Force-Rechnung**; unter reinem Rauschen Mittel 0 und Standardabweichung 1; **Kreuzkorrelationen gegen Brute-Force**;
  die exakte Aktualisierung der Passungen gegen die vollständige Neuberechnung des Residuums; Ausrichtung der Start-Vorlagen am Minimum; bereinigte Ausschnitte (isoliert = Vorlage, mit Nachbar = Nachbar entfernt); "neu mitteln" macht die Form einer falschen Vorlage wieder richtig (Handinstanz); Determinismus.
- Auswertung mit Handinstanzen (Zuordnung, Kollisionen, perfekter Fall, vertauschte Cluster, ein Geist, keine Ereignisse); Verwechslungsmatrix und Genauigkeitsaufteilung konsistent; Orakel-Vorlagen (Form, Ausrichtung); Szenario bit-genau gegen die Vorgänger-Demo; Pipeline, ICA, SOBI, SCA gegen scikit-learn bzw. eingefrorene Werte.
- **Alle Zahlen der App-Texte sind als Tests hinterlegt** (Neuronen, Elektroden, Feuerrate und Kollisionsanteil, Ähnlichkeit, Schwankung, Rauschen, Schwelle, kleinste Amplitude, Verfeinerung, Runden, Preset-Hilfen, Grenzen-Tabelle; jeweils Mittel über die festen Sweep-Datensätze mit weiten Toleranzen);
  Verdict-Codes über sechs Datensätze; alle 6 Presets in Bändern; AppTest-Rauchtests (Default, jedes Preset, jeder Schritt auch ohne erkannte Spikes und mit einer Elektrode, Randgrößen, ausgeblendete Rundenzahl behält ihren Wert, Sweep-Optionen, Experimente auf Abruf), Achsensperre und explizite Schlüssel aller Figuren.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App: Schritte, Ergebnis, 📐 Sweeps, 🔬 Vorlagenqualität, 🧩 Szenen, 🚧 Grenzen, Mathe |
| `tm_matching.py` | Weißung, Passung, Kreuzkorrelation, gieriges Abziehen, Verfeinerung, `run_matching` |
| `tm_algorithm.py` | die Standardpipeline des Vorgänger-Stücks (Detektion, Ausschnitte, PCA, k-means, Silhouette) |
| `tm_ica.py`, `tm_sobi.py`, `tm_sca.py` | Vergleichsverfahren (wortgleich aus ica/sobi/sca-demo) |
| `tm_scenario.py`, `tm_constants.py` | Mehrelektroden-Generator; Konstanten, Presets |
| `tm_evaluation.py` | Zuordnung, Kollisionen, Sortiergüte, Orakel-Vorlagen, Vergleich, Sweeps, Tabellen, Urteil |
| `tm_presets.py`, `tm_visualization.py` | Permalink/Presets, Plotly-Figuren (achsengesperrt) |
| `tests/` | Abgleich (Handinstanzen, Kreuzprüfungen), Pipeline-Kopie, Szenario, Aussagen der App, Presets, AppTest |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

Teil des [Operations-Research-Demo-Portfolios](https://sebastianhanisch.net/demos.html) von
[Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
Interesse an einer maßgeschneiderten Lösung? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html).
