"""Pydantic v2 response models for the Eurogate FLS26 dashboard API.

Field names mirror exactly what ``backend/legacy/server.py`` returns, so the
React frontend can be ported without any key renames. Router implementations
are expected to return dicts matching these shapes (or the models directly).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# GET /api/containers
# ---------------------------------------------------------------------------


class ContainerRow(BaseModel):
    """One row in the paginated container listing.

    The legacy server renames ``container_uuid`` -> ``uuid`` before returning,
    so we keep that convention here.
    """

    model_config = ConfigDict(extra="forbid")

    uuid: str
    num_visits: int
    total_connected_hours: float
    avg_visit_hours: float
    last_visit_start: str | None = None
    last_visit_end: str | None = None


class ContainersResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    containers: list[ContainerRow]
    total: int
    limit: int
    offset: int


# ---------------------------------------------------------------------------
# GET /api/data?uuid=...
# ---------------------------------------------------------------------------


class TimelinePoint(BaseModel):
    """A single event reading for a container.

    Note: the legacy DB has a typo (``RemperatureSupply``), but the server
    normalizes it to ``temp_supply`` for the API response. Power is divided
    by 1000 and rounded to 2 decimals (W -> kW).
    """

    model_config = ConfigDict(extra="forbid")

    time: str
    visit_uuid: str | None = None
    power_kw: float
    energy_hour: float | None = None
    energy_total: float | None = None
    setpoint: float | None = None
    ambient: float | None = None
    temp_return: float | None = None
    temp_supply: float | None = None
    hardware_type: str | None = None
    container_size: str | None = None
    stack_tier: int | None = None


class Visit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    visit_uuid: str
    visit_start: str
    visit_end: str
    duration_hours: float
    reading_count: int
    hardware_type: str | None = None
    container_size: str | None = None
    avg_power_kw: float


class ContainerDetail(BaseModel):
    """Response for /api/data — full drilldown for a single container uuid."""

    model_config = ConfigDict(extra="forbid")

    timeline: list[TimelinePoint]
    visits: list[Visit]
    num_visits: int
    total_connected_hours: float
    avg_visit_hours: float
    last_visit_start: str | None = None
    last_visit_end: str | None = None


# ---------------------------------------------------------------------------
# GET /api/overview-analytics
# ---------------------------------------------------------------------------


class ActivePerDay(BaseModel):
    """Distinct active container-visits grouped by calendar day."""

    model_config = ConfigDict(extra="forbid")

    dates: list[str]
    counts: list[int]


class HardwareTypeEntry(BaseModel):
    """Per-hardware-type visit count + mean power draw."""

    model_config = ConfigDict(extra="forbid")

    hardware_type: str
    cnt: int
    avg_kw: float | None = None


class DurationHistogram(BaseModel):
    """Visit duration distribution across fixed bins."""

    model_config = ConfigDict(extra="forbid")

    labels: list[str]
    counts: list[int]


class MonthlyEnergyRow(BaseModel):
    """Monthly total energy consumption in MWh (``SUM(AvPowerCons)/1e6``)."""

    model_config = ConfigDict(extra="forbid")

    month: str
    total_mwh: float | None = None


class SetpointBin(BaseModel):
    """Setpoint temperature histogram (5 degree bins, range -50..35)."""

    model_config = ConfigDict(extra="forbid")

    sp_bin: int
    cnt: int


class ContainerSizeRow(BaseModel):
    """Container size (ft) vs. mean power draw."""

    model_config = ConfigDict(extra="forbid")

    size: int
    cnt: int
    avg_power_kw: float | None = None


class HourlyHeatmapCell(BaseModel):
    """Hour-of-day x day-of-week activity heatmap cell.

    ``dow`` follows SQLite's ``strftime('%w', ...)`` convention: 0 = Sunday.
    """

    model_config = ConfigDict(extra="forbid")

    hour: int
    dow: int
    count: int


class OverviewAnalytics(BaseModel):
    """Cached fleet-wide analytics aggregation.

    Mirrors the seven sections produced by ``legacy.server.compute_analytics``.
    """

    model_config = ConfigDict(extra="forbid")

    active_per_day: ActivePerDay
    hardware_types: list[HardwareTypeEntry]
    duration_hist: DurationHistogram
    monthly_energy: list[MonthlyEnergyRow]
    setpoint_dist: list[SetpointBin]
    container_sizes: list[ContainerSizeRow]
    hourly_heatmap: list[HourlyHeatmapCell]


# ---------------------------------------------------------------------------
# GET /api/forecast?horizon=24|336
# ---------------------------------------------------------------------------


class ForecastPoint(BaseModel):
    """One row streamed from ``backend/data/dashboard_data.csv``."""

    model_config = ConfigDict(extra="forbid")

    timestamp_utc: str
    pred_power_kw: float
    pred_p90_kw: float
    history_lastyear_kw: float


class ForecastResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    points: list[ForecastPoint]
    horizon: int = Field(..., description="Number of hours returned (24 or 336)")
