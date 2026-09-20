"""Defaults, Slider-Grenzen und feste Szenario-Größen der Vorlagenabgleich-Demo. Das Szenario ist das der spike-sorting-demo (Neuronen, Wellenformen, Feuerraten, Mischung wortgleich, dazu Wellenform-Ähnlichkeit
und Amplitudenschwankung); neu sind die Regler des Vorlagenabgleichs."""

# --- Szenario (fest, wortgleich aus ica/sobi/sca-demo) ------------------------------------------------------------------------
SAMPLE_RATE = 10_000                       # Hz
WAVEFORM_LENGTH = 30                       # Abtastwerte je Spike
PEAK_INDEX = 8                             # Lage der negativen Spitze in der Wellenform
OVERSHOOT = 0.4                            # Höhe des positiven Nachschlags
REFRACTORY = 20                            # Abtastwerte (2 ms)
NEURON_SIGMAS = (2.0, 3.0, 2.5, 4.0, 3.5)          # Breite der Spitze je Neuron
SIGMA_CENTER = 3.0                         # Ähnlichkeit 0: alle Neuronen bekommen diese Breite
NEURON_RATES = (20.0, 28.0, 35.0, 24.0, 31.0)      # Feuerrate in Hz (bei Feuerraten-Faktor 1)
NEURON_AMPLITUDES = (1.0, 0.8, 1.2, 0.7, 0.9)      # Spitzenamplitude an der nächsten Elektrode
NEURON_POSITIONS = ((0.10, 0.30), (0.35, 0.20), (0.60, 0.35), (0.85, 0.25), (0.50, 0.55))   # Elektroden liegen bei y = 0, x in [0, 1]
DISTANCE_EPS = 0.05
ACTIVE_THRESHOLD = 0.1                     # ein Neuron gilt an einem Zeitpunkt als aktiv, wenn seine Wellenform dort mehr als 10 % der Spitzenhöhe erreicht

# --- Regler ---------------------------------------------------------------------------------------------------------------------
DEFAULT_N_NEURONS = 4
N_NEURONS_MIN, N_NEURONS_MAX = 2, 5
DEFAULT_N_ELECTRODES = 4
N_ELECTRODES_MIN, N_ELECTRODES_MAX = 1, 8
DEFAULT_RATE_SCALE = 1.0
RATE_SCALE_MIN, RATE_SCALE_MAX = 0.25, 4.0
DEFAULT_SIMILARITY = 1.0
SIMILARITY_MIN, SIMILARITY_MAX = 0.0, 1.0                                # 1 = Wellenformen wie gewohnt, 0 = alle Neuronen haben dieselbe Form
DEFAULT_JITTER = 0.0
JITTER_MIN, JITTER_MAX = 0.0, 0.3                                        # Streuung der Spitzenhöhe je Spike (relativ)
DEFAULT_NOISE = 0.05
NOISE_MIN, NOISE_MAX = 0.0, 1.0
DEFAULT_N_SAMPLES = 20_000
N_SAMPLES_MIN, N_SAMPLES_MAX = 5_000, 40_000
DEFAULT_THRESHOLD = 4.5                                                  # Schwelle der Start-Pipeline (fest): Rausch-Standardabweichungen
FEATURES = ("pca", "raw", "amplitude")
DEFAULT_FEATURE = "pca"                                                  # Merkmale der Start-Pipeline (fest)
DEFAULT_N_COMPONENTS = 3
CLUSTER_MODES = ("known", "silhouette")
CLUSTER_MODE_LABELS = {"known": "bekannt (wie eingestellt)", "silhouette": "unbekannt (per Silhouette wählen)"}
DEFAULT_CLUSTER_MODE = "known"
K_MAX = 8                                                                # größte Clusterzahl bei der Silhouette-Wahl
CONTRASTS = ("logcosh", "exp", "cube")
CONTRAST_LABELS = {"logcosh": "log cosh (robust)", "exp": "Gauß-Ableitung (sehr robust)", "cube": "Kurtosis (u³)"}
DEFAULT_CONTRAST = "logcosh"
METHODS = ("symmetric", "deflation")
DEFAULT_METHOD = "symmetric"
INIT_STARTS = (1, 2, 3, 4, 5)
DEFAULT_INIT_START = 1
DEFAULT_SEED = 7

# --- Vorlagenabgleich ---------------------------------------------------------------------------------------------------------
DEFAULT_MATCH_THRESHOLD = 5.0
MATCH_THRESHOLD_MIN, MATCH_THRESHOLD_MAX = 2.0, 12.0                     # Schwelle z der Passung (unter Rauschen ungefähr standardnormalverteilt)
DEFAULT_MIN_AMPLITUDE = 0.5
MIN_AMPLITUDE_MIN, MIN_AMPLITUDE_MAX = 0.0, 0.9                          # kleinster erlaubter Anteil der Vorlage (Skalierung)
REFINE_MODES = ("none", "average", "recluster")
REFINE_LABELS = {"none": "keine (nur Start-Vorlagen)", "average": "Vorlagen neu mitteln", "recluster": "bereinigte Spikes neu clustern"}
DEFAULT_REFINE = "recluster"
DEFAULT_ROUNDS = 2
ROUNDS_MIN, ROUNDS_MAX = 1, 5

# --- Pipeline (fest) --------------------------------------------------------------------------------------------------------------
SNIPPET_BEFORE = 8                         # Abtastwerte vor dem Minimum
SNIPPET_AFTER = 22                         # ... und ab dem Minimum (Länge 30 = Wellenformlänge)
DEAD_TIME = 15                             # Abtastwerte Mindestabstand zweier erkannter Spitzen
MATCH_TOLERANCE = 6                        # erkannte und wahre Spitze gelten als dieselbe, wenn ihre Zeiten höchstens so weit auseinanderliegen
COLLISION_WINDOW = 12                      # ein wahrer Spike ist "Kollision", wenn ein anderer Spike höchstens so viele Abtastwerte entfernt beginnt (die Spitzen sind 4-8 Abtastwerte breit)
N_RESTARTS = 10                            # Neustarts des k-means
KMEANS_MAX_ITER = 100

# --- Vergleichsverfahren (aus ica/sobi/sca-demo) ---------------------------------------------------------------------------------
RECONSTRUCTIONS = ("l1", "single")
DEFAULT_RECONSTRUCTION = "l1"
MAX_ITER = 200                             # FastICA
TOL = 1e-6
JD_MAX_SWEEPS = 100                        # SOBI: Jacobi-Sweeps
JD_TOL = 1e-8
SOBI_LAGS = tuple(range(2, 21, 2))         # Verzögerungen des SOBI-Vergleichs

# --- Auswertung -----------------------------------------------------------------------------------------------------------------
SWEEP_SEEDS = (100000, 100001, 100002, 100003, 100004)            # feste Datensätze der Sweeps, getrennt vom Demo-Seed


# --- Presets ---------------------------------------------------------------------------------------------------------------------


def _preset(**kw):
    base = dict(m=DEFAULT_N_NEURONS, n=DEFAULT_N_ELECTRODES, rate_scale=DEFAULT_RATE_SCALE, similarity=DEFAULT_SIMILARITY, jitter=DEFAULT_JITTER, noise=DEFAULT_NOISE, n_samples=DEFAULT_N_SAMPLES,
                threshold=DEFAULT_MATCH_THRESHOLD, min_amplitude=DEFAULT_MIN_AMPLITUDE, refine=DEFAULT_REFINE, rounds=DEFAULT_ROUNDS, cluster_mode=DEFAULT_CLUSTER_MODE, seed=DEFAULT_SEED)
    base.update(kw)
    return base


PRESETS = {
    "Vier Neuronen, vier Elektroden": _preset(),
    "Hohe Feuerrate": _preset(rate_scale=2.0),
    "Eine Elektrode": _preset(n=1),
    "Fünf Neuronen": _preset(m=5),
    "Ohne Amplitudengrenze": _preset(min_amplitude=0.0),
    "Hohe Feuerrate, keine Verfeinerung": _preset(rate_scale=2.0, refine="none"),
}
PRESET_HELP = {
    "Vier Neuronen, vier Elektroden": "Der Grundfall: der Abgleich erreicht Spitzen-F1 0.99 (Pipeline 0.94) und sortiert von den überlappenden Spikes 93 % richtig (Pipeline 85 %); mit den wahren Vorlagen wäre es kaum besser (0.99).",
    "Hohe Feuerrate": "Doppelte Feuerrate: jeder dritte Spike überlappt mit einem anderen, die Pipeline fällt auf Spitzen-F1 0.77, der Abgleich hält 0.97 - so gut wie mit den wahren Vorlagen (0.97). Genau dafür ist er gedacht.",
    "Eine Elektrode": "Nur eine Elektrode: der Abgleich hilft nicht (0.55 gegen 0.51 der Pipeline) - seine Vorlagen kommen aus den Clustern der Pipeline, und die sind hier schon falsch. Mit den wahren Vorlagen wären es 0.94: "
                      "der Abgleich selbst kann es, die Vorlagen fehlen.",
    "Fünf Neuronen": "Fünf Neuronen: die Wellenformen liegen dichter, k-means bildet in manchen Aufnahmen falsche Cluster - dann sind die Vorlagen falsch. Im Mittel über fünf Aufnahmen 0.91 gegen 0.74 der Pipeline, in dieser Aufnahme nur 0.72 "
                      "(Orakel: 0.96): die Streuung von Aufnahme zu Aufnahme ist groß.",
    "Ohne Amplitudengrenze": "Die kleinste erlaubte Amplitude ist 0: nach jedem abgezogenen Spike bleibt ein kleiner Rest, der bei so geringem Rauschen weit über der Schwelle liegt - der Abgleich meldet ihn als weiteren Spike. "
                             "Es entstehen etwa dreimal so viele Ereignisse wie Spikes (Spitzen-F1 0.35); ab einer Grenze von 0.2 ist es wieder gut (0.98).",
    "Hohe Feuerrate, keine Verfeinerung": "Dieselbe Aufnahme wie 'Hohe Feuerrate', aber die Start-Vorlagen bleiben unverändert: die Cluster der Pipeline sind durch die überlappenden Spikes teilweise falsch geschnitten, und der Abgleich erbt das: "
                                         "Spitzen-F1 0.84 statt 0.97. 'Vorlagen neu mitteln' ändert daran nichts (0.84) - erst 'bereinigte Spikes neu clustern' holt den Rest.",
}
# Bänder (Seed des Presets; Werte mit dem ausgelieferten Code kalibriert, bewusst weit): Spitzen-F1 des Abgleichs (f1), der Pipeline (pipe_f1), mit wahren Vorlagen (oracle_f1), Trefferquote (recall),
# Genauigkeit der Detektion (precision), erlaubte Urteile (verdict)
PRESET_EXPECTED_BANDS = {
    "Vier Neuronen, vier Elektroden": {"f1": (0.93, 1.0), "pipe_f1": (0.85, 0.98), "recall": (0.93, 1.0), "verdict": ("matching_ok", "matching_better")},
    "Hohe Feuerrate": {"f1": (0.88, 1.0), "pipe_f1": (0.5, 0.85), "recall": (0.9, 1.0), "verdict": ("matching_better",)},
    "Eine Elektrode": {"f1": (0.2, 0.65), "oracle_f1": (0.8, 1.0), "verdict": ("templates_wrong",)},
    "Fünf Neuronen": {"f1": (0.55, 0.95), "oracle_f1": (0.85, 1.0), "verdict": ("templates_wrong", "matching_better")},
    "Ohne Amplitudengrenze": {"f1": (0.15, 0.6), "precision": (0.1, 0.6), "verdict": ("ghosts",)},
    "Hohe Feuerrate, keine Verfeinerung": {"f1": (0.5, 0.9), "oracle_f1": (0.85, 1.0), "verdict": ("templates_wrong", "matching_better")},
}
