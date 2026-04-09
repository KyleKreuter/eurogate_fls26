"""Ein-Kommando-Pipeline fuer den Organizer-Rerun.

Fuehrt alle Base-Modelle und den Honest-Blend in der korrekten Reihenfolge
aus und schreibt am Ende genau eine finale Submission:

    lightgbm/submissions/honest_blend.csv

Schritte:
    1. baseline.py          -> baseline.csv
    2. productive.py        -> legal_rf_big_s1.csv, legal_rf_s1.csv
    3. rf_richfeat.py       -> rf_richfeat.csv
    4. catboost_model.py    -> catboost.csv
    5. honest_blend.py      -> honest_blend.csv  (SUBMIT_STRATEGY=uniform_3_rf)

Leakage-Schutz (wichtig fuer den Organizer-Rerun):
    - Alle Base-Scripts respektieren HARD_CUTOFF_TS aus baseline.py
      (Default: 2025-12-31 23:00 UTC). Reefer-Rohdaten nach diesem Zeitpunkt
      werden bereits beim CSV-Read verworfen - nichts davon landet jemals
      im Training oder in einem Feature.
    - Lag-Features fuer Target-Stunden werden aus der Mirror-Year-Synthese
      (364 Tage frueher, Wochentags-treu) gefuellt. Siehe
      baseline.extend_post_cutoff_with_mirror.
    - Nur eval.py ruft load_hourly_total mit einem spaeteren Cutoff auf
      und darf damit die echten Ground-Truth-Werte im Target-Fenster sehen.
      run_all.py ruft eval.py NICHT auf.

Fail-fast:
    Bricht sofort mit Exit-Code 1 ab, wenn ein Script failt. Stoppt vor dem
    Blend, wenn ein Base-Modell fehlt.

Ausfuehren (vom Projekt-Root):
    uv run python lightgbm/run_all.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
_SUBMISSIONS_DIR = _HERE / "submissions"


# Reihenfolge ist wichtig: honest_blend.py erwartet alle Base-Submissions.
PIPELINE_STEPS: list[tuple[str, Path, list[str]]] = [
    (
        "baseline",
        _HERE / "baseline.py",
        ["baseline.csv"],
    ),
    (
        "productive",
        _HERE / "productive.py",
        ["legal_rf_big_s1.csv", "legal_rf_s1.csv"],
    ),
    (
        "rf_richfeat",
        _HERE / "rf_richfeat.py",
        ["rf_richfeat.csv"],
    ),
    (
        "catboost_model",
        _HERE / "catboost_model.py",
        ["catboost.csv"],
    ),
    (
        "honest_blend",
        _HERE / "honest_blend.py",
        ["honest_blend.csv"],
    ),
]


def run_step(label: str, script: Path) -> float:
    """Fuehrt ein Script aus und gibt die Laufzeit in Sekunden zurueck."""
    print()
    print("=" * 78)
    print(f"[run_all] Starte {label}: {script.relative_to(_PROJECT_ROOT)}")
    print("=" * 78, flush=True)

    start = time.time()
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=_PROJECT_ROOT,
        check=False,
    )
    elapsed = time.time() - start

    if result.returncode != 0:
        print(
            f"\n[run_all] FEHLER: {label} ist mit Exit-Code "
            f"{result.returncode} abgebrochen.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"\n[run_all] {label} fertig in {elapsed:.1f}s")
    return elapsed


def verify_outputs(expected: list[str]) -> None:
    """Prueft, dass alle erwarteten Submission-CSVs existieren."""
    missing = [
        name for name in expected if not (_SUBMISSIONS_DIR / name).exists()
    ]
    if missing:
        print(
            f"\n[run_all] FEHLER: Erwartete Outputs fehlen: {missing}",
            file=sys.stderr,
        )
        sys.exit(1)


def main() -> None:
    print("[run_all] Eurogate FLS26 - End-to-End-Pipeline")
    print(f"[run_all] Projekt-Root: {_PROJECT_ROOT}")
    print(f"[run_all] Python:       {sys.executable}")
    print(f"[run_all] Submissions:  {_SUBMISSIONS_DIR.relative_to(_PROJECT_ROOT)}")
    print(
        "[run_all] Leakage-Schutz: HARD_CUTOFF_TS aktiv "
        "(keine Reefer-Daten > 2025-12-31 23:00 UTC im Training/Feature)."
    )

    total_start = time.time()
    timings: dict[str, float] = {}
    for label, script, outputs in PIPELINE_STEPS:
        elapsed = run_step(label, script)
        timings[label] = elapsed
        verify_outputs(outputs)

    total = time.time() - total_start

    print()
    print("=" * 78)
    print("[run_all] Pipeline komplett.")
    print("=" * 78)
    for label, elapsed in timings.items():
        print(f"  {label:<20} {elapsed:7.1f}s")
    print(f"  {'TOTAL':<20} {total:7.1f}s")
    print()
    final = _SUBMISSIONS_DIR / "honest_blend.csv"
    print(
        f"[run_all] Finale Submission: "
        f"{final.relative_to(_PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
