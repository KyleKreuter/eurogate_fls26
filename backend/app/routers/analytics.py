"""GET /api/overview-analytics — fleet-wide aggregations with TTL cache.

Ports the legacy ``compute_analytics()`` function from
``backend/legacy/server.py``. The legacy implementation used a module-level
``_analytics_cache`` that lived for the entire process lifetime; this router
upgrades that to a TTL-based cache driven by ``settings.analytics_cache_ttl``
so operators can refresh aggregations without bouncing the server.

The seven sections returned mirror the legacy shape exactly — raw SQL field
names (``hardware_type``, ``cnt``, ``avg_kw``, ``sp_bin``, ``size``, ``dow``)
are preserved so the existing React frontend can consume this endpoint with
zero key renames.
"""

from __future__ import annotations

import time
from threading import Lock

from fastapi import APIRouter

from ..config import settings
from ..db import get_connection
from ..models import (
    ActivePerDay,
    ContainerSizeRow,
    DurationHistogram,
    HardwareTypeEntry,
    HourlyHeatmapCell,
    MonthlyEnergyRow,
    OverviewAnalytics,
    SetpointBin,
)

router = APIRouter()

# ---------------------------------------------------------------------------
# In-process TTL cache
# ---------------------------------------------------------------------------
# The cache holds a single key ("analytics") mapped to
# ``(monotonic_timestamp, OverviewAnalytics)``. Access is guarded by a
# ``threading.Lock`` because FastAPI may dispatch concurrent requests to
# threadpool workers for non-async handlers, and concurrent expirations
# would otherwise race on dict writes.
_cache: dict[str, tuple[float, OverviewAnalytics]] = {}
_cache_lock = Lock()


# ---------------------------------------------------------------------------
# Legacy aggregation — 7 queries, single connection
# ---------------------------------------------------------------------------
# Fixed visit-duration bins copied verbatim from legacy.server.compute_analytics.
# Labels use en-dash (U+2013), matching what the frontend expects.
_DURATION_BINS: list[tuple[int, int, str]] = [
    (0, 12, "<12h"),
    (12, 24, "12\u201324h"),
    (24, 72, "1\u20133d"),
    (72, 168, "3\u20137d"),
    (168, 336, "1\u20132w"),
    (336, 720, "2\u20134w"),
    (720, 9999, ">4w"),
]


def _compute() -> OverviewAnalytics:
    """Run the seven legacy aggregations under a single SQLite connection.

    This is deliberately slow (~5-30s on the production DB) and is meant to
    be called at most once per TTL window.
    """
    with get_connection() as conn:
        cur = conn.cursor()

        # 1. Active container-visits per day -------------------------------
        cur.execute(
            """
            SELECT date(EventTime) as day, COUNT(DISTINCT container_visit_uuid) as active
            FROM events
            GROUP BY day
            ORDER BY day
            """
        )
        rows = cur.fetchall()
        active_per_day = ActivePerDay(
            dates=[r["day"] for r in rows],
            counts=[r["active"] for r in rows],
        )

        # 2. Hardware type distribution (count + mean kW) ------------------
        cur.execute(
            """
            SELECT hardware_type, COUNT(*) as cnt,
                   ROUND(AVG(avg_power_kw), 2) as avg_kw
            FROM visit_stats WHERE hardware_type IS NOT NULL
            GROUP BY hardware_type ORDER BY cnt DESC
            """
        )
        hardware_types = [
            HardwareTypeEntry(
                hardware_type=r["hardware_type"],
                cnt=r["cnt"],
                avg_kw=r["avg_kw"],
            )
            for r in cur.fetchall()
        ]

        # 3. Visit duration histogram (fixed bins, Python-side bucketing) --
        bin_counts: dict[str, int] = {lbl: 0 for *_, lbl in _DURATION_BINS}
        cur.execute(
            "SELECT duration_hours FROM visit_stats WHERE duration_hours IS NOT NULL"
        )
        for row in cur.fetchall():
            h = row["duration_hours"]
            for lo, hi, lbl in _DURATION_BINS:
                if lo <= h < hi:
                    bin_counts[lbl] += 1
                    break
        duration_hist = DurationHistogram(
            labels=[lbl for *_, lbl in _DURATION_BINS],
            counts=[bin_counts[lbl] for *_, lbl in _DURATION_BINS],
        )

        # 4. Monthly energy totals (MWh) -----------------------------------
        # total_mwh may be NULL for empty months — pass through unchanged.
        cur.execute(
            """
            SELECT strftime('%Y-%m', EventTime) as month,
                   ROUND(SUM(AvPowerCons) / 1000000.0, 2) as total_mwh
            FROM events WHERE EventTime IS NOT NULL
            GROUP BY month ORDER BY month
            """
        )
        monthly_energy = [
            MonthlyEnergyRow(month=r["month"], total_mwh=r["total_mwh"])
            for r in cur.fetchall()
        ]

        # 5. Setpoint temperature distribution (5-degree bins, -50..35) ----
        cur.execute(
            """
            SELECT CAST(ROUND(TemperatureSetPoint / 5.0) * 5 AS INTEGER) as sp_bin,
                   COUNT(*) as cnt
            FROM events
            WHERE TemperatureSetPoint IS NOT NULL
              AND TemperatureSetPoint > -50 AND TemperatureSetPoint < 35
            GROUP BY sp_bin ORDER BY sp_bin
            """
        )
        setpoint_dist = [
            SetpointBin(sp_bin=r["sp_bin"], cnt=r["cnt"]) for r in cur.fetchall()
        ]

        # 6. Container size vs avg power -----------------------------------
        cur.execute(
            """
            SELECT CAST(ContainerSize AS INTEGER) as size, COUNT(*) as cnt,
                   ROUND(AVG(AvPowerCons) / 1000.0, 2) as avg_power_kw
            FROM events WHERE ContainerSize IS NOT NULL
            GROUP BY size ORDER BY size
            """
        )
        container_sizes = [
            ContainerSizeRow(
                size=r["size"], cnt=r["cnt"], avg_power_kw=r["avg_power_kw"]
            )
            for r in cur.fetchall()
        ]

        # 7. Hour x day-of-week heatmap (active visits per slot) -----------
        # dow follows SQLite strftime('%w', ...) => 0 = Sunday.
        cur.execute(
            """
            SELECT CAST(strftime('%H', EventTime) AS INTEGER) as hour,
                   CAST(strftime('%w', EventTime) AS INTEGER) as dow,
                   COUNT(DISTINCT container_visit_uuid) as count
            FROM events
            WHERE EventTime IS NOT NULL
            GROUP BY hour, dow
            ORDER BY dow, hour
            """
        )
        hourly_heatmap = [
            HourlyHeatmapCell(hour=r["hour"], dow=r["dow"], count=r["count"])
            for r in cur.fetchall()
        ]

    return OverviewAnalytics(
        active_per_day=active_per_day,
        hardware_types=hardware_types,
        duration_hist=duration_hist,
        monthly_energy=monthly_energy,
        setpoint_dist=setpoint_dist,
        container_sizes=container_sizes,
        hourly_heatmap=hourly_heatmap,
    )


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------
@router.get("/overview-analytics", response_model=OverviewAnalytics)
async def get_overview_analytics() -> OverviewAnalytics:
    """Return cached fleet analytics, recomputing when the TTL expires.

    The first call (or the first call after TTL expiry) runs ~7 SQL
    aggregations against ``reefer.db`` and can take several seconds.
    Subsequent calls within ``settings.analytics_cache_ttl`` seconds
    return the cached payload instantly.
    """
    now = time.monotonic()
    with _cache_lock:
        cached = _cache.get("analytics")
        if cached is not None:
            ts, data = cached
            if now - ts < settings.analytics_cache_ttl:
                return data

    # Expired or empty — compute fresh outside the lock so concurrent
    # readers of a still-fresh cache aren't blocked by a slow recompute.
    data = _compute()
    with _cache_lock:
        _cache["analytics"] = (time.monotonic(), data)
    return data
