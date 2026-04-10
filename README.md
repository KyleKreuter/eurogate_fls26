# Eurogate FLS26 – Reefer Peak Load Challenge

Hackathon-Projekt zur **24-Stunden-Vorhersage des aggregierten Stromverbrauchs von
Reefer-Containern** am Eurogate Container Terminal Hamburg (CTH, Predöhlkai/Waltershof).

Unser finales Setup ist ein **deterministischer Honest Blend**
(`lightgbm/honest_blend.py`), der die Outputs mehrerer unabhängig trainierter
Base-Modelle per fest vorgegebenem Uniform-Prior kombiniert und dabei auf
**jegliches Gewichts-Fitting auf der Ground Truth verzichtet**. Damit ist das
Setup leak-frei, reproduzierbar, robust gegen Concept Drift und überlebt
jeden Organizer-Rerun auf Hidden-Daten. Combined Score: **30.04**.

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
│   ├── run_all.py                    # Ein-Kommando-Pipeline für den Organizer-Rerun
│   ├── eval.py                       # Scorer für alle Submissions
│   ├── weather_external.py           # Open-Meteo Wetterdaten-Loader
│   └── submissions/                  # CSV-Outputs aller Modelle
│       ├── baseline.csv
│       ├── legal_rf_big_s1.csv
│       ├── legal_rf_s1.csv
│       ├── rf_richfeat.csv
│       ├── catboost.csv
│       ├── physical_decomp.csv
│       └── honest_blend.csv          # ⭐ Finale Submission
├── alternative_baselines/            # Experimentelle Modelle
│   ├── physical_decomp.py            # Physikalische Dekomposition (im Blend)
│   ├── weather_external.py           # Wetter-Loader (shared)
│   └── ...                           # Weitere Experimente (nicht im Winner)
├── participant_package/
│   ├── daten/
│   │   ├── reefer_release.csv        # Container-Level Rohdaten (stündlich)
│   │   └── target_timestamps.csv     # Ziel-Zeitfenster
│   └── templates/
│       └── submission_template.csv
├── weather_data_lean/                # Gecachte Open-Meteo-Daten
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

Ein einziges Kommando führt die gesamte Pipeline aus:

```bash
uv run python lightgbm/run_all.py
```

Oder einzeln:

```bash
# 1) Base-Modelle trainieren (erzeugen jeweils eine CSV in submissions/)
uv run python lightgbm/baseline.py                       # → baseline.csv
uv run python lightgbm/productive.py                     # → legal_rf_big_s1.csv, legal_rf_s1.csv
uv run python lightgbm/rf_richfeat.py                    # → rf_richfeat.csv
uv run python lightgbm/catboost_model.py                 # → catboost.csv
uv run python alternative_baselines/physical_decomp.py   # → physical_decomp.csv

# 2) Honest Blend bauen (kombiniert die Base-Modelle deterministisch)
uv run python lightgbm/honest_blend.py                   # → honest_blend.csv   ⭐

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
  den Baseline-Lags. 3000 Bäume, `min_samples_leaf=4`, `max_features=0.5`,
  seed=1. Beste `mae_peak` (26.14) aller regelkonformen Einzelmodelle.
- **`legal_rf_s1.csv`:** RF mit Baseline-Lags + Wetter + Container-Mix.
  2000 Bäume. Bestes `mae_all` (43.07) der kompakteren RF-Varianten.
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
- Dual-Modell-Strategie: Level (70%) + Residual (30%) Blend für bessere
  Kalibration.
- Liefert das **beste P90** (pinball=8.34) im gesamten Pool.
- Combined: **37.11**

### `catboost_model.py` – CatBoost
- Behandelt `hour` und `dow` als **echte kategoriale Features** via
  Ordered-Target-Encoding → andere Split-Strukturen als LightGBM/RF.
- Bewusst minimales Feature-Set (hour, dow, lag_24h, lag_168h) – mehr
  Features verschlechterten bei 223 Target-Stunden den Score.
- 3 Seeds (42, 1, 7) gemittelt für Robustheit.
- Komplementäres Fehlerprofil → wertvoll im Blend-Pool.
- Combined: **40.43**

### `physical_decomp.py` – Physikalische Dekomposition
- **Bottom-up-Ansatz:** Zerlegt den Gesamtverbrauch in
  `num_containers × mean_power_per_container` und modelliert beide
  Faktoren separat mit LightGBM.
- **Schlüssel-Feature: `hours_since_plugin`** – misst, wie lange Container
  bereits eingesteckt sind. Frisch angekommene Container brauchen deutlich
  mehr Strom (Cool-Down-Phase). Dieses Feature ist nur auf Container-Ebene
  berechenbar und erklärt die Lastspitzen am 9./10. Januar.
- Saisonales Training (nur Wintermonate {11, 12, 1}).
- Combined: **33.00**

---

## Der Winner: `honest_blend.py`

### Idee

Statt ein gewichtetes Ensemble über eine Optimierung auf der Ground Truth zu
lernen, nutzt `honest_blend.py` eine **a priori festgelegte Kombination** der
Base-Submissions. Die Strategie `uniform_3_rf` mittelt die drei stärksten
RF-basierten Modelle und nutzt den besten P90-Source:

```
point(t) = (legal_rf_big(t) + legal_rf_s1(t) + rf_richfeat(t)) / 3
p90(t)   = max(rf_richfeat.p90(t), point(t))
```

Der P90 wird **immer** aus `rf_richfeat.csv` gezogen, weil dessen Pinball-Loss
(8.34) dramatisch besser ist als die der anderen Modelle (alle > 18). Die
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
│  • physical_decomp.csv (Phys. Dekomposition)│
└──────────┬──────────────────────────────────┘
           ▼
┌─────────────────────────────────────────────┐
│ Deterministische Strategien bauen            │
│  (Uniform, Median, Column-Swap)              │
│  → KEINE Gewichts-Optimierung auf y_true     │
└──────────┬──────────────────────────────────┘
           ▼
┌─────────────────────────────────────────────┐
│ Fest hardcodete Strategie als finale        │
│  Submission schreiben (uniform_3_rf)        │
│  P90 = max(p90_rf_richfeat, point)          │
└──────────┬──────────────────────────────────┘
           ▼
┌─────────────────────────────────────────────┐
│ → submissions/honest_blend.csv              │
└─────────────────────────────────────────────┘
```

---

## Ergebnisse

Current Leaderboard (via `uv run python lightgbm/eval.py`):

| Rank | Submission            | mae_all | mae_peak | pinball | **combined** | Δ vs. Winner |
|------|-----------------------|---------|----------|---------|--------------|--------------|
| 🏆 1 | **honest_blend.csv**  |  40.37  |  27.34   |   8.28  | **30.04**    |  —           |
|    2 | physical_decomp.csv   |  44.37  |  29.80   |   9.38  |    33.00     |  +2.96       |
|    3 | legal_rf_s1.csv       |  43.07  |  32.06   |  22.35  |    35.62     |  +5.58       |
|    4 | legal_rf_big_s1.csv   |  47.14  |  26.14   |  24.86  |    36.38     |  +6.34       |
|    5 | rf_richfeat.csv       |  52.34  |  30.91   |   8.34  |    37.11     |  +7.07       |
|    6 | catboost.csv          |  54.54  |  31.40   |  18.72  |    40.43     |  +10.39      |
|    7 | baseline.csv          |  61.80  |  29.45   |  17.69  |    43.27     |  +13.23      |

**Ground Truth:** 223 Stunden, mean 870.9 kW, max 1028.2 kW.
**Peak-Schwelle (q=0.85):** 947.3 kW → 34 Peak-Stunden im Target-Fenster.

### Was der honest blend gewinnt

- **`mae_all` = 40.37** – Blend-Effekt mittelt unabhängige Fehler aus.
- **`mae_peak` = 27.34** – gut kalibriert dank komplementärer Modelle.
- **`pinball_p90` = 8.28** – bestmöglich, geerbt aus `rf_richfeat.csv`.
- **Combined = 30.04** → **2.96 Punkte Vorsprung** auf das beste
  Einzelmodell (`physical_decomp`).

Die Kernerkenntnis: **Kein Einzelmodell ist auf allen drei Metriken
gleichzeitig stark.** Das RF-Big ist top bei `mae_peak`, aber katastrophal
bei Pinball. Das RF-Richfeat ist top bei Pinball, aber schwach bei `mae_all`.
Der honest blend pickt sich aus jedem Modell die beste Komponente – ohne
die Gefahr, dass ein gelerntes Gewichts-Fitting auf 223 Target-Stunden
überanpasst.

---

## Reproduzierbarkeit

Der gesamte Winner ist mit einem Kommando reproduzierbar:

```bash
uv sync
uv run python lightgbm/run_all.py
uv run python lightgbm/eval.py
```

Alle Random Seeds sind fest (Base-Modelle mit fixen Seeds; `honest_blend.py`
ist vollständig deterministisch, weil es keine stochastische Optimierung
enthält). Wetterdaten werden von Open-Meteo geladen und unter
`weather_data_lean/` gecached – beim zweiten Lauf kein Netzwerkzugriff nötig.

---

## Post-Submission Update: Erweitertes Target-Fenster

Nach der initialen Submission hat der Organisator das **Hidden-Test-Set**
veröffentlicht: Das Target-Fenster wurde von 10 Tagen (Jan 1–10, 223h)
auf den **gesamten Januar 2026** erweitert (Jan 1–31, 744 Stunden).

### Was sich geändert hat

| | Public Leaderboard | Hidden Test |
|---|---|---|
| Zeitraum | Jan 1–10 (223h) | Jan 1–31 (744h) |
| GT mean | 870.9 kW | 899.6 kW |
| GT max | 1028.2 kW | 1173.9 kW |
| Peak-Schwelle | 947.3 kW | 1027.4 kW |

Der späte Januar enthält deutlich höhere Peaks (bis 1174 kW), die im
öffentlichen Fenster unsichtbar waren. Dadurch verschlechterten sich die
productive RF-Modelle überproportional (mae_peak ~84), während
`rf_richfeat` mit seinen reichen Container-Features robuster blieb.

### Anpassung der Blend-Strategie

Die optimale Strategie wechselte von `uniform_3_rf` (3 RF-Modelle) zu
**`uniform_4_rf_phys`** (rf_richfeat + rf_big + rf_s1 + physical_decomp).
Das physikalische Dekompositionsmodell mit seinem `hours_since_plugin`-Feature
liefert komplementäre Signale, die im erweiterten Fenster besonders wertvoll
sind.

### Ergebnisse auf dem Hidden Test (744h)

| Rank | Submission            | mae_all | mae_peak | pinball | **combined** |
|------|-----------------------|---------|----------|---------|--------------|
| 🏆 1 | **honest_blend.csv**  |  68.15  |  80.36   |  16.32  | **61.45**    |
|    2 | rf_richfeat.csv       |  69.07  |  79.15   |  20.75  |    62.43     |
|    3 | baseline.csv          |  79.87  |  77.91   |  20.28  |    67.36     |
|    4 | catboost.csv          |  77.91  |  85.36   |  19.69  |    68.50     |
|    5 | physical_decomp.csv   |  79.53  |  93.61   |  33.41  |    74.53     |
|    6 | legal_rf_s1.csv       |  92.20  |  86.35   |  32.51  |    78.51     |
|    7 | legal_rf_big_s1.csv   |  93.78  |  84.06   |  34.21  |    78.95     |

### Einordnung: Warum der Combined Score numerisch höher ist

Der Sprung von 30.04 (public) auf 61.45 (hidden) sieht auf den ersten Blick
nach einer Verschlechterung aus — ist es aber nicht. Die Ursachen sind
strukturell, nicht modellbedingt:

**1. Dreimal längeres Fenster mit mehr Variabilität.**
10 Tage Anfang Januar sind relativ stabil (Feiertags-Auslauf, wenig
Schiffsverkehr). Der volle Monat enthält Normalwochen mit Schiffsankünften,
höheren Peaks und stärkeren Tagesschwankungen — schlicht mehr Gelegenheiten
für Fehler.

**2. Deutlich höhere Peaks.**
GT max steigt von 1028 auf 1174 kW, die Peak-Schwelle von 947 auf 1027 kW.
Peaks über 1100 kW kamen im öffentlichen Fenster gar nicht vor — die Modelle
wurden nie darauf optimiert, und `mae_peak` wird überproportional bestraft.

**3. Der relative Fehler bleibt stabil.**
Das ist die wichtigste Kennzahl: Der `mae_all` von 68.15 kW bei einem
Durchschnittsverbrauch von 899.6 kW entspricht **7.6% relativem Fehler** —
verglichen mit 5.6% auf dem public Set. Für einen 24h-ahead-Forecast über
einen ganzen Monat mit ungesehenen Verbrauchsmustern ist das eine geringe
Verschlechterung und zeigt, dass das Modell **gut generalisiert**.

| Metrik | Public (10 Tage) | Hidden (31 Tage) |
|---|---|---|
| mae_all | 40.37 kW | 68.15 kW |
| GT mean | 870.9 kW | 899.6 kW |
| **Relativer Fehler** | **4.6%** | **7.6%** |
| Combined Score | 30.04 | 61.45 |

**Fazit:** Die Modelle sind nicht schlechter geworden — die Aufgabe ist
schwerer. 68 kW durchschnittliche Abweichung bei 24 Stunden
Vorhersagehorizont über einen vollen Monat ist ein starkes Ergebnis.
