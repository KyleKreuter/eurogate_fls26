"""GET /api/data?uuid=... — full drilldown for a single container.

Ported 1:1 from ``backend/legacy/server.py::handle_get_data``. Field names
and value transforms match the legacy response so the frontend works without
any key renames.

Note on column names in the ``events`` table:
  * ``RemperatureSupply`` is a real typo in the source schema — we select
    it literally and map it to ``temp_supply`` in the response.
  * ``AvPowerCons`` is in Watts; we divide by 1000 and round to 2 decimals
    to get ``power_kw``, handling NULL as 0 (matches legacy).
  * ``EventTime`` is the event timestamp column.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from ..db import get_connection
from ..models import ContainerDetail, TimelinePoint, Visit

router = APIRouter()


@router.get("/data", response_model=ContainerDetail)
async def get_container_data(
    uuid: str = Query(..., description="Container UUID to fetch drilldown data for"),
) -> ContainerDetail:
    """Return the full timeline, visits, and aggregate stats for a container.

    Raises:
        HTTPException 404: if ``uuid`` does not exist in ``container_stats``.
    """
    with get_connection() as conn:
        stats_row = conn.execute(
            """
            SELECT num_visits, total_connected_hours, avg_visit_hours,
                   last_visit_start, last_visit_end
              FROM container_stats
             WHERE container_uuid = ?
            """,
            (uuid,),
        ).fetchone()
        if stats_row is None:
            raise HTTPException(status_code=404, detail=f"Container {uuid} not found")

        # NOTE: the events table can be very large for high-activity containers.
        # Legacy does not paginate/limit, and we intentionally preserve that
        # behaviour — callers should expect potentially large responses.
        timeline_rows = conn.execute(
            """
            SELECT EventTime, container_visit_uuid,
                   AvPowerCons, TtlEnergyConsHour, TtlEnergyCons,
                   TemperatureSetPoint, TemperatureAmbient,
                   TemperatureReturn, RemperatureSupply,
                   HardwareType, ContainerSize, stack_tier
              FROM events
             WHERE container_uuid = ?
             ORDER BY EventTime ASC
            """,
            (uuid,),
        ).fetchall()

        visit_rows = conn.execute(
            """
            SELECT container_visit_uuid, visit_start, visit_end, duration_hours,
                   reading_count, hardware_type, container_size, avg_power_kw
              FROM visit_stats
             WHERE container_uuid = ?
             ORDER BY visit_start ASC
            """,
            (uuid,),
        ).fetchall()

    timeline = [
        TimelinePoint(
            time=r["EventTime"],
            visit_uuid=r["container_visit_uuid"],
            power_kw=round((r["AvPowerCons"] or 0) / 1000.0, 2),
            energy_hour=r["TtlEnergyConsHour"],
            energy_total=r["TtlEnergyCons"],
            setpoint=r["TemperatureSetPoint"],
            ambient=r["TemperatureAmbient"],
            temp_return=r["TemperatureReturn"],
            temp_supply=r["RemperatureSupply"],  # intentional typo: real column name
            hardware_type=r["HardwareType"],
            container_size=r["ContainerSize"],
            stack_tier=r["stack_tier"],
        )
        for r in timeline_rows
    ]

    visits = [
        Visit(
            visit_uuid=r["container_visit_uuid"],
            visit_start=r["visit_start"],
            visit_end=r["visit_end"],
            duration_hours=r["duration_hours"],
            reading_count=r["reading_count"],
            hardware_type=r["hardware_type"],
            container_size=r["container_size"],
            avg_power_kw=r["avg_power_kw"],
        )
        for r in visit_rows
    ]

    return ContainerDetail(
        timeline=timeline,
        visits=visits,
        num_visits=stats_row["num_visits"],
        total_connected_hours=stats_row["total_connected_hours"],
        avg_visit_hours=stats_row["avg_visit_hours"],
        last_visit_start=stats_row["last_visit_start"],
        last_visit_end=stats_row["last_visit_end"],
    )
