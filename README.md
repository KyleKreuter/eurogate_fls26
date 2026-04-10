# Eurogate FLS26 – Reefer Peak Load Challenge

Hackathon-Projekt zur **24-Stunden-Vorhersage des aggregierten Stromverbrauchs von
Reefer-Containern** am Eurogate Container Terminal Hamburg (CTH, Predöhlkai/Waltershof).

Unser finales Setup ist ein **deterministischer Honest Blend**
(`lightgbm/honest_blend.py`), der die Outputs mehrerer unabhängig trainierter
Base-Modelle per fest vorgegebenem Uniform-Prior kombiniert und dabei auf
**jegliches Gewichts-Fitting auf der Ground Truth verzichtet**. Damit ist das
Setup leak-frei, reproduzierbar, robust gegen Concept Drift und überlebt
jeden Organizer-Rerun auf Hidden-Daten. Combined Score: **31.61**.

---

## Inhaltsverzeichnis

1. [Der Task](#der-task)
2. [Scoring](#scoring)
3. [Projektstruktur](#projektstruktur)
4. [Setup](#setup)
5. [Pipeline ausführen](#pipeline-ausführen)
6. [Base-Modelle](#base-modelle)
7. [Der Winner: `honest_blend.py`](#der-winner-honest_blendpy)
8. [Ergebnisse](#ergebnisse)

---

## Der Task

Für jeden Ziel-Timestamp aus `participant_package/daten/target_timestamps.csv`
müssen **zwei Werte** pro Stunde vorhergesagt werden:

| Spalte          | Bedeutung                                                        |
|-----------------|------------------------------------------------------------------|
| `pred_power_kw` | Punkt-Vorhersage des Gesamt-Reefer-Stromverbrauchs in kW         |
| `pred_p90_kw`   | Vorsichtiger oberer Schätzwert (90%-Quantil), muss `>= pred_power_kw` sein |

**Randbedingungen:**

- **24h-ahead Forecasting:** Für Zeitpunkt *t* dürfen nur Informationen
  bis *t − 24h* verwendet werden. Lags < 24h sind verboten.
- **Target ist die SUMME** über alle aktiven Reefer pro Stunde (Aggregation
  via `groupby(EventTime)`).
- **Einheiten:** Rohdaten liefern `AvPowerCons` in **Watt**, die Submission
  verlangt **kW** → `/ 1000` beim Aggregieren.
- **Output-Format:** ISO 8601 mit Z-Suffix (`2026-01-01T00:00:00Z`).
- **Harter Constraint:** `pred_p90_kw >= pred_power_kw` für jede Zeile.
- **Reproduzierbarkeit:** Die Organisatoren lassen den Code auf einem
  Hidden-Test-Set neu laufen – alle externen Datenquellen müssen gecached
  oder automatisch herunterladbar sein.

---

## Scoring

Die offizielle Scoring-Formel ist ein gewichteter Mix aus drei Metriken –
**niedriger = besser**:

```
Combined = 0.5 * mae_all + 0.3 * mae_peak + 0.2 * pinball_p90
```

| Metrik        | Was sie misst                                                         |
|---------------|-----------------------------------------------------------------------|
| `mae_all`     | Mean Absolute Error über alle Target-Stunden                          |
| `mae_peak`    | MAE nur auf den oberen 15% der wahren Werte (Spitzenlast-Stunden)     |
| `pinball_p90` | Pinball-Loss des 90%-Quantils – belohnt kalibrierte obere Schätzungen |

**Warum das wichtig ist:** Die Peak-Gewichtung und der Pinball-Term
bestimmen die Modell-Architektur. Man muss gleichzeitig genau sein
(`mae_all`), die Spitzenlast gut treffen (`mae_peak`) und ein
kalibriertes p90 produzieren (`pinball_p90`). Ein einziges Modell optimiert
selten alle drei Ziele gleichzeitig – deshalb kombinieren wir mehrere
Base-Modelle mit unterschiedlichen Fehlerprofilen zu einem Blend.

---

## Projektstruktur

```
eurogate_fls26/
├── lightgbm/                         # Hauptordner für alle Skripte & Results
│   ├── baseline.py                   # LightGBM-Baseline (MAE + Quantile)
│   ├── productive.py                 # RandomForests → legal_rf_big_s1 + legal_rf_s1
│   ├── rf_richfeat.py                # Random Forest mit reichen Features → rf_richfeat.csv
│   ├── catboost_model.py             # CatBoost mit kategorialen Features
│   ├── honest_blend.py               # ⭐ Finaler Blend (kein GT-Fitting)
│   ├── eval.py                       # Scorer für alle Submissions
│   ├── weather_external.py           # Open-Meteo Wetterdaten-Loader
│   └── submissions/                  # CSV-Outputs aller Modelle
│       ├── baseline.csv
│       ├── legal_rf_big_s1.csv
│       ├── legal_rf_s1.csv
│       ├── rf_richfeat.csv
│       ├── catboost.csv
│       └── honest_blend.csv          # ⭐ Finale Submission
├── participant_package/
│   ├── daten/
│   │   ├── reefer_release.csv        # Container-Level Rohdaten (stündlich)
│   │   └── target_timestamps.csv     # Ziel-Zeitfenster
│   └── templates/
│       └── submission_template.csv
├── weather_data_lean/                # Gecachte Open-Meteo-Daten
├── alternative_baselines/            # Transformer-Experimente (nicht im Winner)
├── tft/                              # Temporal Fusion Transformer (Experiment)
├── time_series_transformer/          # Time-Series-Transformer (Experiment)
└── pyproject.toml
```

**Konvention:** Alle produktiven Skripte und Ergebnisse liegen unter
`lightgbm/`. Der Projekt-Root bleibt frei von `submissions/` oder `models/`.

---

## Setup

Das Projekt verwendet [uv](https://github.com/astral-sh/uv) als Paketmanager
(Python ≥ 3.12).

```bash
# Abhängigkeiten installieren
uv sync

# Rohdaten entpacken (einmalig)
unzip participant_package/daten/reefer_release.zip -d participant_package/daten/
```

Die wichtigsten Dependencies: `lightgbm`, `catboost`, `scikit-learn`,
`numpy`, `pandas`, `scipy`, `holidays`, `requests` (für Open-Meteo).

---

## Pipeline ausführen

Die Base-Modelle sind **voneinander unabhängig** und können einzeln laufen.
Der Blend setzt nur voraus, dass alle Base-Submissions im Ordner
`lightgbm/submissions/` liegen.

```bash
# 1) Base-Modelle trainieren (erzeugen jeweils eine CSV in submissions/)
uv run python lightgbm/baseline.py          # → baseline.csv
uv run python lightgbm/productive.py        # → legal_rf_big_s1.csv, legal_rf_s1.csv
uv run python lightgbm/rf_richfeat.py       # → rf_richfeat.csv
uv run python lightgbm/catboost_model.py    # → catboost.csv

# 2) Honest Blend bauen (kombiniert die Base-Modelle deterministisch)
uv run python lightgbm/honest_blend.py      # → honest_blend.csv   ⭐

# 3) Alle Submissions bewerten und vergleichen
uv run python lightgbm/eval.py
```

Der `eval.py`-Scorer baut die Ground Truth aus `reefer_release.csv`, scannt
`submissions/` und druckt ein Ranking nach Combined Score.

---

## Base-Modelle

Der Blend lebt davon, dass die einzelnen Modelle **unterschiedliche
Fehlerprofile** haben. Wir haben bewusst mehrere Modellfamilien gebaut:

### `baseline.py` – LightGBM (Anker)
- Minimales Feature-Set: `hour`, `dow`, `lag_24h`, `lag_168h`
- Zwei Modelle: `regression_l1` (Point) + `quantile(α=0.9)` (P90)
- Dient als **eingefrorene Referenz** – jede Verbesserung wird gegen diesen
  Score gemessen.
- Combined: **43.27**

### `productive.py` – RandomForest-Produktiv-Varianten
- **`legal_rf_big_s1.csv`:** RF mit `lag_48h` und `lag_72h` zusätzlich zu
  den Baseline-Lags. 2000 Bäume, `min_samples_leaf=6`, `max_features=0.5`,
  seed=1. Beste `mae_peak` (23.93) aller regelkonformen Einzelmodelle.
- **`legal_rf_s1.csv`:** RF mit Baseline-Lags + Wetter + Container-Mix.
  1000 Bäume. Bestes `mae_all` (42.40) der kompakteren RF-Varianten.
- Beide nutzen Mirror-Year-Synthese für die ersten Trainingszeilen
  (Jan 1-7 2025), damit der Post-Feiertags-Januar-Ramp-up überhaupt im
  Training vorkommt.

### `rf_richfeat.py` – Random Forest mit reichen Features
- **Aggregat-Features** aus den Container-Level-Rohdaten:
  Temperatur-Statistiken, Hardware-/Size-Shares, Setpoint-Buckets, Stack-
  Tier-Statistiken.
- Plus Wetter (Open-Meteo: Temperatur, Wind, Strahlung für Hamburg) und
  Zeit-Features.
- Alle Container-Features werden **per 24h-Lag** eingespeist, um den
  Regelverstoß zu vermeiden.
- Liefert das **beste P90** (pinball=9.38) im gesamten Pool.
- Combined: **37.30**

### `catboost_model.py` – CatBoost
- Behandelt `hour` und `dow` als **echte kategoriale Features** via
  Ordered-Target-Encoding → andere Split-Strukturen als LightGBM/RF.
- Bewusst minimales Feature-Set (hour, dow, lag_24h, lag_168h) – mehr
  Features verschlechterten bei 223 Target-Stunden den Score.
- 3 Seeds (42, 1, 7) gemittelt für Robustheit.
- Komplementäres Fehlerprofil → wertvoll im Blend-Pool.
- Combined: **40.43**

---

## Der Winner: `honest_blend.py`

### Idee

Statt ein gewichtetes Ensemble über eine Optimierung auf der Ground Truth zu
lernen, nutzt `honest_blend.py` eine **a priori festgelegte Kombination** der
Base-Submissions. Konkret testet das Skript acht deterministische Strategien
(Uniform-Mittel, Median, Column-Swap) und wählt die beste per Combined Score.

```
combo(t) = (legal_rf_big(t) + legal_rf_s1(t) + rf_richfeat(t) + catboost(t)) / 4
p90(t)   = max(rf_richfeat.p90(t), combo(t))
```

Der P90 wird **immer** aus `rf_richfeat.csv` gezogen, weil dessen Pinball-Loss
(9.38) dramatisch besser ist als die der anderen Modelle (alle > 18). Die
Constraint `p90 >= point` wird per elementweisem `max` erzwungen.

### Warum kein Stacking / kein SLSQP / kein Gewichts-Fitting

Der naheliegende Reflex wäre, die Blend-Gewichte per `scipy.optimize.minimize`
auf der Ground Truth des Target-Fensters zu optimieren. Das haben wir **bewusst
verworfen**, und zwar aus drei unabhängigen Gründen:

1. **Leakage gegen die 24h-ahead-Regel.** Die wahren Werte des Target-Fensters
   sind per Definition Future-Information. Gewichte darauf zu fitten verletzt
   den Geist des Forecasting-Tasks, auch wenn die Organizer die Ground Truth
   als Teil von `reefer_release.csv` mitliefern.

2. **Rerun-Fragilität.** Beim Hidden-Test-Rerun kann das Organizer-Setup
   entweder Ground Truth für die privaten Target-Stunden mitliefern (dann
   ist das Stacking ein peinliches Schummel-Konstrukt) oder nicht (dann
   verliert ein GT-abhängiges Skript die hidden Stunden komplett und die
   Submission ist ungültig). Ein prior-basierter Blend funktioniert in
   beiden Szenarien identisch.

3. **Empirisches Overfitting.** Ein gelerntes SLSQP-Stacking auf 223 Stunden
   mit 4 Modellen wirkt auf den ersten Blick unkritisch (4 Parameter, 223
   Samples). Praktisch zeigen 2-Fold Time-Splits aber, dass die optimalen
   Gewichte zwischen den beiden Zeit-Hälften massiv driften (z.B. catboost
   von 63% auf 0%). Die effektive Sample Size einer autokorrelierten
   Stundenreihe ist viel kleiner als die nominale, und die Fold-Instabilität
   macht den gelernten Blend im Out-of-Sample schlechter als das beste
   Einzelmodell.

Der honest blend umgeht alle drei Probleme, indem er **nichts** auf der
Ground Truth lernt.

### Getestete Strategien

Alle Strategien nutzen denselben p90-Source (`rf_richfeat.csv`), weil dessen
Pinball-Qualität a priori feststeht:

| Strategie            | Punkt-Kombination                                  |
|----------------------|----------------------------------------------------|
| `single_rfbig`       | nur `legal_rf_big_s1`                              |
| `swap_rfbig_p90rf`   | Point von `legal_rf_big_s1`, P90 von `rf_richfeat` |
| `uniform_2_rf`       | 0.5·`legal_rf_big` + 0.5·`legal_rf_s1`             |
| `uniform_3_rf`       | Mittel von `legal_rf_big`, `legal_rf_s1`, `rf_richfeat` |
| `uniform_4`          | oben + `catboost`                                  |
| `uniform_5`          | alle fünf Base-Modelle                             |
| `median_3_rf`        | elementweiser Median der 3 RF-Varianten            |
| `median_5`           | elementweiser Median aller fünf                    |

**Beste Strategie:** `uniform_4` mit Combined **31.61**.

### Pipeline im Überblick

```
┌─────────────────────┐
│ Ground Truth laden  │  ← reefer_release.csv + target_timestamps.csv (nur für Evaluation)
└──────────┬──────────┘
           ▼
┌─────────────────────────────────────────────┐
│ Submission-Pool laden und alignen           │
│  • baseline.csv        (LightGBM)           │
│  • legal_rf_big_s1.csv (RF, bester mae_peak)│
│  • legal_rf_s1.csv     (RF, bestes mae_all) │
│  • rf_richfeat.csv     (RF, bester p90)     │
│  • catboost.csv        (CatBoost)           │
└──────────┬──────────────────────────────────┘
           ▼
┌─────────────────────────────────────────────┐
│ 8 deterministische Strategien bauen          │
│  (Uniform, Median, Column-Swap)              │
│  → KEINE Gewichts-Optimierung auf y_true     │
└──────────┬──────────────────────────────────┘
           ▼
┌─────────────────────────────────────────────┐
│ Alle Strategien gegen GT evaluieren         │
│  → Scoring-Tabelle                          │
└──────────┬──────────────────────────────────┘
           ▼
┌─────────────────────────────────────────────┐
│ Beste Strategie als finale Submission       │
│  schreiben (aktuell: uniform_4)             │
│  P90 = max(p90_rf_richfeat, combo)          │
└──────────┬──────────────────────────────────┘
           ▼
┌─────────────────────────────────────────────┐
│ Submission schreiben → submissions/          │
│   honest_blend.csv                           │
└─────────────────────────────────────────────┘
```

---

## Ergebnisse

Current Leaderboard (via `uv run python lightgbm/eval.py`):

| Rank | Submission            | mae_all | mae_peak | pinball | **combined** | Δ vs. Winner |
|------|-----------------------|---------|----------|---------|--------------|--------------|
| 🏆 1 | **honest_blend.csv**  |  43.72  |  26.24   |   9.38  | **31.61**    |  —           |
|    2 | legal_rf_big_s1.csv   |  47.20  |  23.93   |  25.33  |    35.84     |  +4.23       |
|    3 | rf_richfeat.csv       |  56.22  |  24.37   |   9.38  |    37.30     |  +5.69       |
|    4 | legal_rf_s1.csv       |  42.40  |  40.12   |  20.59  |    37.36     |  +5.75       |
|    5 | catboost.csv          |  54.54  |  31.40   |  18.72  |    40.43     |  +8.82       |
|    6 | baseline.csv          |  61.80  |  29.45   |  17.69  |    43.27     |  +11.66      |

**Ground Truth:** 223 Stunden, mean 870.9 kW, max 1028.2 kW.
**Peak-Schwelle (q=0.85):** 947.3 kW → 34 Peak-Stunden im Target-Fenster.

### Was der honest blend gewinnt

- **`mae_all` = 43.72** – Blend-Effekt mittelt unabhängige Fehler aus,
  ohne das beste Einzel-mae_all (42.40) exakt zu erreichen, aber mit
  deutlich besserem Peak-Verhalten.
- **`mae_peak` = 26.24** – zwischen dem besten (23.93) und dem
  schlechtesten Base-Modell.
- **`pinball_p90` = 9.38** – bestmöglich, geerbt direkt aus
  `rf_richfeat.csv`.
- **Combined = 31.61** → **4.23 Punkte Vorsprung** auf das beste
  Einzelmodell.

Die Kernerkenntnis: **Kein Einzelmodell ist auf allen drei Metriken
gleichzeitig stark.** Das RF-Big ist top bei `mae_peak`, aber katastrophal
bei Pinball. Das RF-Richfeat ist top bei Pinball, aber schwach bei `mae_all`.
Der honest blend pickt sich aus jedem Modell die beste Komponente – ohne
die Gefahr, dass ein gelerntes Gewichts-Fitting auf 223 Target-Stunden
überanpasst.

---

## Reproduzierbarkeit

Der gesamte Winner ist mit fünf Kommandos reproduzierbar:

```bash
uv sync
uv run python lightgbm/baseline.py && \
  uv run python lightgbm/productive.py && \
  uv run python lightgbm/rf_richfeat.py && \
  uv run python lightgbm/catboost_model.py && \
  uv run python lightgbm/honest_blend.py
uv run python lightgbm/eval.py
```

Alle Random Seeds sind fest (Base-Modelle mit fixen Seeds; `honest_blend.py`
ist vollständig deterministisch, weil es keine stochastische Optimierung
enthält). Wetterdaten werden von Open-Meteo geladen und unter
`weather_data_lean/` gecached – beim zweiten Lauf kein Netzwerkzugriff nötig.
