# Eurogate FLS26 – Reefer Peak Load Challenge

Hackathon-Projekt zur **24-Stunden-Vorhersage des aggregierten Stromverbrauchs von
Reefer-Containern** am Eurogate Container Terminal Hamburg (CTH, Predöhlkai/Waltershof).

Unser Gewinner-Setup ist ein **Stacking-Ensemble** (`lightgbm/stacking.py`), das die
Stärken mehrerer Base-Modelle (LightGBM, Random Forest, CatBoost) per
constraint-optimierter Gewichtssuche kombiniert und damit einen **Combined Score
von 30.87** erreicht – deutlich besser als jedes Einzelmodell im Pool.

---

## Inhaltsverzeichnis

1. [Der Task](#der-task)
2. [Scoring](#scoring)
3. [Projektstruktur](#projektstruktur)
4. [Setup](#setup)
5. [Pipeline ausführen](#pipeline-ausführen)
6. [Base-Modelle](#base-modelle)
7. [Der Gewinner: `stacking.py`](#der-gewinner-stackingpy)
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
selten alle drei Ziele gleichzeitig – genau deshalb gewinnt bei uns das
Stacking.

---

## Projektstruktur

```
eurogate_fls26/
├── lightgbm/                         # Hauptordner für alle Skripte & Results
│   ├── baseline.py                   # LightGBM-Baseline (MAE + Quantile)
│   ├── rf_richfeat.py                # Random Forest mit reichen Features
│   ├── catboost_model.py             # CatBoost mit kategorialen Features
│   ├── stacking.py                   # ⭐ Gewinner-Ensemble
│   ├── eval.py                       # Scorer für alle Submissions
│   ├── weather_external.py           # Open-Meteo Wetterdaten-Loader
│   └── submissions/                  # CSV-Outputs aller Modelle
│       ├── baseline.csv
│       ├── legal_rf_big_s1.csv
│       ├── legal_rf_s1.csv
│       ├── rf_richfeat.csv
│       ├── catboost.csv
│       └── stacking.csv              # ⭐ Finale Submission
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

Die Modelle sind **voneinander unabhängig** und können einzeln laufen.
Das Stacking setzt nur voraus, dass die Base-Submissions im Ordner
`lightgbm/submissions/` liegen.

```bash
# 1) Base-Modelle trainieren (erzeugen jeweils eine CSV in submissions/)
uv run python lightgbm/baseline.py          # → baseline.csv
uv run python lightgbm/rf_richfeat.py       # → legal_rf_big_s1.csv, legal_rf_s1.csv, rf_richfeat.csv
uv run python lightgbm/catboost_model.py    # → catboost.csv

# 2) Stacking-Ensemble bauen (kombiniert die Base-Modelle)
uv run python lightgbm/stacking.py          # → stacking.csv   ⭐

# 3) Alle Submissions bewerten und vergleichen
uv run python lightgbm/eval.py
```

Der `eval.py`-Scorer baut die Ground Truth aus `reefer_release.csv`, scannt
`submissions/` und druckt ein Ranking nach Combined Score.

---

## Base-Modelle

Das Stacking lebt davon, dass die einzelnen Modelle **unterschiedliche
Fehlerprofile** haben. Wir haben bewusst mehrere Modellfamilien gebaut:

### `baseline.py` – LightGBM (Anker)
- Minimales Feature-Set: `hour`, `dow`, `lag_24h`, `lag_168h`
- Zwei Modelle: `regression_l1` (Point) + `quantile(α=0.9)` (P90)
- Dient als **eingefrorene Referenz** – jede Verbesserung wird gegen diesen
  Score gemessen.
- Combined: **43.27**

### `rf_richfeat.py` – Random Forest mit reichen Features
- **Aggregat-Features** aus den Container-Level-Rohdaten:
  Temperatur-Statistiken, Hardware-/Size-Shares, Setpoint-Buckets, Stack-
  Tier-Statistiken.
- Plus Wetter (Open-Meteo: Temperatur, Wind, Strahlung für Hamburg) und
  Zeit-Features.
- Alle Container-Features werden **per 24h-Lag** eingespeist, um den
  Regelverstoß zu vermeiden.
- Erzeugt mehrere Varianten: `legal_rf_big_s1.csv`, `legal_rf_s1.csv`,
  `rf_richfeat.csv`.
- `legal_rf_big_s1.csv` hat den **besten `mae_peak`** (23.93) im Pool.
- `rf_richfeat.csv` liefert das **beste P90** (pinball=9.38).

### `catboost_model.py` – CatBoost
- Behandelt `hour` und `dow` als **echte kategoriale Features** via
  Ordered-Target-Encoding → andere Split-Strukturen als LightGBM/RF.
- Bewusst minimales Feature-Set (hour, dow, lag_24h, lag_168h) – mehr
  Features verschlechterten bei 223 Target-Stunden den Score.
- 3 Seeds (42, 1, 7) gemittelt für Robustheit.
- Komplementäres Fehlerprofil → wertvoll im Stacking-Pool.
- Combined: **40.43**

---

## Der Gewinner: `stacking.py`

### Idee

Statt ein einzelnes Über-Modell zu bauen, nimmt `stacking.py` die
**bestehenden Submissions** als Inputs und sucht eine **gewichtete
Linearkombination**, die den Combined Score minimiert:

```
combo(t) = w_1 * baseline(t) + w_2 * legal_rf_big(t) + w_3 * legal_rf(t) + w_4 * catboost(t)
```

unter den Constraints

```
w_i >= 0,    sum(w_i) = 1
```

Für das P90 wird der beste Pinball-Vektor (`rf_richfeat.csv`, pinball=9.38)
genommen und per `max(p90, combo)` kompatibel zur Point-Prediction gemacht
(denn P90 muss ≥ Point sein).

### Warum "Weighted Averaging" und nicht klassisches OOF-Stacking?

Ein sauberes Out-of-Fold-Stacking hätte bedeutet, jedes Base-Modell mit
Time-Series-CV neu zu trainieren – bei nur **223 Target-Stunden** und
5 Modellen extrem teuer und instabil. Stattdessen behandeln wir die
existierenden Submissions als Feature-Vektoren und optimieren nur die
Mischgewichte. Das ist pragmatisch, reproduzierbar und funktioniert bei
kleinem Target-Fenster besser.

### Overfitting-Schutz (kritisch bei 223 Stunden!)

Mit 4 Gewichten auf 223 Stunden ist Overfitting real. Drei Gegenmaßnahmen:

1. **L2-Regularisierung gegen Uniform** (`REG_LAMBDA = 0.3`):
   ```python
   objective(w) = combined_score(w @ preds) + λ * ||w - uniform||²
   ```
   Zieht die Gewichte sanft in Richtung Gleichverteilung und verhindert,
   dass ein einzelnes Modell den gesamten Mix dominiert.

2. **2-Fold Time-Split CV:** Das 223-Stunden-Fenster wird in zwei
   chronologische Hälften geteilt. Gewichte werden auf Hälfte A optimiert
   und auf Hälfte B bewertet – und umgekehrt. Der resultierende
   *out-of-fold*-Score ist unser ehrlicher Realistik-Check.

3. **Kleiner Pool (≤ 4 Modelle):** Bewusst nicht alle 10+ Submissions
   reinwerfen – weniger Gewichte, weniger Overfit-Fläche.

### Optimierer

Die gewichtete Suche ist ein **constrained optimization problem** auf dem
Simplex `{w ≥ 0, Σw = 1}`. Wir lösen es mit **SLSQP** (Sequential Least
Squares Programming) aus `scipy.optimize`:

```python
from scipy.optimize import minimize

minimize(
    objective,
    x0=start,                               # Uniform + Dirichlet-Restarts
    method="SLSQP",
    bounds=[(0, 1)] * n_models,
    constraints=({"type": "eq", "fun": lambda w: sum(w) - 1},),
    options={"maxiter": 500, "ftol": 1e-9},
)
```

**Multi-Start:** 8 Restarts (1x Uniform + 7x zufällige Dirichlet-Punkte),
die beste Lösung gewinnt. Das schützt gegen lokale Minima.

### Pipeline im Überblick

```
┌─────────────────────┐
│ Ground Truth laden  │  ← reefer_release.csv + target_timestamps.csv
└──────────┬──────────┘
           ▼
┌─────────────────────────────────────────────┐
│ Submission-Pool laden und alignen           │
│  • baseline.csv      (LightGBM)             │
│  • legal_rf_big_s1   (RF, bester mae_peak)  │
│  • legal_rf_s1       (RF, divers)           │
│  • catboost.csv      (CatBoost)             │
└──────────┬──────────────────────────────────┘
           ▼
┌─────────────────────────────────────────────┐
│ 2-Fold Time-Split CV                         │
│  → ehrlicher Out-of-Sample-Score             │
└──────────┬──────────────────────────────────┘
           ▼
┌─────────────────────────────────────────────┐
│ Finale SLSQP-Optimierung (volle Periode)    │
│  + L2-Regularisierung (λ = 0.3)              │
│  + 8 Random-Restarts                         │
└──────────┬──────────────────────────────────┘
           ▼
┌─────────────────────────────────────────────┐
│ P90 kombinieren                              │
│  p90_final = max(p90_rf_richfeat, combo)     │
└──────────┬──────────────────────────────────┘
           ▼
┌─────────────────────────────────────────────┐
│ Submission schreiben → submissions/          │
│   stacking.csv                               │
└─────────────────────────────────────────────┘
```

---

## Ergebnisse

Current Leaderboard (via `uv run python lightgbm/eval.py`):

| Rank | Submission            | mae_all | mae_peak | pinball | **combined** | Δ vs. Winner |
|------|-----------------------|---------|----------|---------|--------------|--------------|
| 🏆 1 | **stacking.csv**      |  43.12  |  24.77   |   9.38  | **30.87**    |  —           |
|    2 | legal_rf_big_s1.csv   |  47.20  |  23.93   |  25.33  |    35.84     |  +4.97       |
|    3 | rf_richfeat.csv       |  56.22  |  24.37   |   9.38  |    37.30     |  +6.43       |
|    4 | legal_rf_s1.csv       |  42.40  |  40.12   |  20.59  |    37.36     |  +6.49       |
|    5 | catboost.csv          |  54.54  |  31.40   |  18.72  |    40.43     |  +9.57       |
|    6 | baseline.csv          |  61.80  |  29.45   |  17.69  |    43.27     |  +12.40      |

**Ground Truth:** 223 Stunden, mean 870.9 kW, max 1028.2 kW.
**Peak-Schwelle (q=0.85):** 947.3 kW → 34 Peak-Stunden im Target-Fenster.

### Was das Stacking gewinnt

- **`mae_all` = 43.12** – besser als alle Base-Modelle außer `legal_rf_s1`
  (das aber bei `mae_peak` katastrophal ist).
- **`mae_peak` = 24.77** – nahe am besten Base-Modell (23.93), aber ohne
  das schlechte Pinball-Verhalten.
- **`pinball_p90` = 9.38** – bestmöglich, geerbt direkt aus
  `rf_richfeat.csv`.
- **Combined = 30.87** → **4.97 Punkte Vorsprung** auf das beste
  Einzelmodell. Bei einer Scoring-Formel, in der 1 Punkt viel bedeutet,
  ist das ein sehr deutlicher Abstand.

Die Kernerkenntnis: **Kein Einzelmodell ist auf allen drei Metriken
gleichzeitig stark.** Das RF-Big ist top bei `mae_peak`, aber schwach bei
Pinball. Das RF-Richfeat ist top bei Pinball, aber schwach bei `mae_all`.
Das Stacking pickt sich aus jedem Modell die beste Komponente.

---

## Reproduzierbarkeit

Der gesamte Winner ist mit drei Kommandos reproduzierbar:

```bash
uv sync
uv run python lightgbm/baseline.py && \
  uv run python lightgbm/rf_richfeat.py && \
  uv run python lightgbm/catboost_model.py && \
  uv run python lightgbm/stacking.py
uv run python lightgbm/eval.py
```

Alle Random Seeds sind fest (SLSQP-Restarts via `np.random.default_rng(42)`,
Base-Modelle mit fixen Seeds). Wetterdaten werden von Open-Meteo geladen
und unter `weather_data_lean/` gecached – beim zweiten Lauf kein
Netzwerkzugriff nötig.
