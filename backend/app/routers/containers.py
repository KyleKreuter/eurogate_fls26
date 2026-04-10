"""FastAPI router for ``GET /api/containers`` — paginated container listing.

Ports the legacy handler from ``backend/legacy/server.py`` verbatim (same SQL,
same column rename ``container_uuid`` -> ``uuid``). All user-supplied values
are bound as SQL parameters; sort column and direction are validated via
``Enum`` so they're safe to interpolate into the ``ORDER BY`` clause.
"""

from __future__ import annotations

from enum import Enum

from fastapi import APIRouter, Query

from ..db import get_connection
from ..models import ContainerRow, ContainersResponse

router = APIRouter()


class SortColumn(str, Enum):
    uuid = "uuid"
    num_visits = "num_visits"
    total_connected_hours = "total_connected_hours"
    avg_visit_hours = "avg_visit_hours"


class SortDirection(str, Enum):
    asc = "ASC"
    desc = "DESC"

    @classmethod
    def _missing_(cls, value: object) -> "SortDirection | None":
        if isinstance(value, str):
            upper = value.upper()
            for member in cls:
                if member.value == upper:
                    return member
        return None


_SORT_COL_MAP: dict[SortColumn, str] = {
    SortColumn.uuid: "container_uuid",
    SortColumn.num_visits: "num_visits",
    SortColumn.total_connected_hours: "total_connected_hours",
    SortColumn.avg_visit_hours: "avg_visit_hours",
}


@router.get("/containers", response_model=ContainersResponse)
async def list_containers(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    sort: SortColumn = Query(SortColumn.num_visits),
    dir: SortDirection = Query(SortDirection.desc),
    q: str | None = Query(None),
) -> ContainersResponse:
    """Return a paginated, optionally-filtered slice of ``container_stats``."""
    sql_col = _SORT_COL_MAP[sort]
    sql_dir = dir.value

    where_sql = ""
    params: list[object] = []
    if q:
        where_sql = "WHERE container_uuid LIKE ?"
        params.append(f"%{q}%")

    with get_connection() as conn:
        total = conn.execute(
            f"SELECT COUNT(*) AS c FROM container_stats {where_sql}",
            params,
        ).fetchone()["c"]

        rows = conn.execute(
            f"""
            SELECT container_uuid, num_visits, total_connected_hours,
                   avg_visit_hours, last_visit_start, last_visit_end
              FROM container_stats
              {where_sql}
              ORDER BY {sql_col} {sql_dir}
              LIMIT ? OFFSET ?
            """,
            params + [limit, offset],
        ).fetchall()

    containers = [
        ContainerRow(
            uuid=r["container_uuid"],
            num_visits=r["num_visits"],
            total_connected_hours=r["total_connected_hours"],
            avg_visit_hours=r["avg_visit_hours"],
            last_visit_start=r["last_visit_start"],
            last_visit_end=r["last_visit_end"],
        )
        for r in rows
    ]
    return ContainersResponse(
        containers=containers, total=total, limit=limit, offset=offset
    )
