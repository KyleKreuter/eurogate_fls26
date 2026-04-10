"""GET /api/forecast - returns the last N hours of the Hamburg power forecast.

Reads ``backend/data/dashboard_data.csv`` using stdlib ``csv.DictReader`` (no
pandas) and slices to the last ``horizon`` rows. The parsed rows are cached in
a module-level dict for 60 seconds so repeated dashboard loads don't re-parse.
"""

from __future__ import annotations

import csv
import time
from enum import IntEnum

from fastapi import APIRouter, HTTPException

from ..config import settings
from ..models import ForecastPoint, ForecastResponse

router = APIRouter()


class Horizon(IntEnum):
    """Allowed forecast horizons (hours)."""

    day = 24
    fortnight = 336

_FORECAST_CSV = settings.data_dir / "dashboard_data.csv"
_CACHE_TTL = 60  # seconds
_cache: dict[str, tuple[float, list[ForecastPoint]]] = {}


def _load_forecast() -> list[ForecastPoint]:
    """Parse dashboard_data.csv into a list of ForecastPoint, with a 60s cache."""
    now = time.monotonic()
    cached = _cache.get("rows")
    if cached is not None and now - cached[0] < _CACHE_TTL:
        return cached[1]

    if not _FORECAST_CSV.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                f"Forecast data not available at {_FORECAST_CSV}. "
                "Run backend/scripts/generate_dashboard_data.py to regenerate."
            ),
        )

    rows: list[ForecastPoint] = []
    with _FORECAST_CSV.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                ForecastPoint(
                    timestamp_utc=row["timestamp_utc"],
                    pred_power_kw=float(row["pred_power_kw"]),
                    pred_p90_kw=float(row["pred_p90_kw"]),
                    history_lastyear_kw=float(row["history_lastyear_kw"]),
                )
            )
    _cache["rows"] = (now, rows)
    return rows


@router.get("/forecast", response_model=ForecastResponse)
async def get_forecast(horizon: Horizon = Horizon.fortnight) -> ForecastResponse:
    """Return the tail-``horizon`` rows of the forecast CSV as JSON."""
    rows = _load_forecast()
    n = int(horizon)
    sliced = rows[-n:] if len(rows) > n else rows
    return ForecastResponse(points=sliced, horizon=n)
