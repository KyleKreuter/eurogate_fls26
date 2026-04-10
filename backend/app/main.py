"""FastAPI application entry point for the Eurogate FLS26 dashboard backend.

Run with::

    uvicorn app.main:app --reload --host 0.0.0.0 --port 8080

The app wires together:
  * CORS middleware (frontend dev servers)
  * /api/* routers (registered below by separate router modules)
  * /data/* static files (CSV/JSON used by the frontend)
  * SPA fallback serving the built React bundle from web/dist/
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import settings
from .routers import analytics, container_data, containers, forecast

app = FastAPI(
    title="Eurogate FLS26 Dashboard API",
    version="1.0.0",
    description="Reefer container telemetry, fleet analytics, and power forecast API.",
)

# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/api/health", tags=["meta"])
async def health() -> dict[str, str]:
    """Lightweight liveness probe."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# API routers
# ---------------------------------------------------------------------------
# Router agents will register their routers here. Each router module lives
# under app/routers/ and exposes a module-level ``router = APIRouter()``.
#
#   from .routers import containers, container_data, analytics, forecast
#   app.include_router(containers.router, prefix="/api", tags=["containers"])
#   app.include_router(container_data.router, prefix="/api", tags=["containers"])
#   app.include_router(analytics.router, prefix="/api", tags=["analytics"])
#   app.include_router(forecast.router, prefix="/api", tags=["forecast"])
app.include_router(containers.router, prefix="/api", tags=["containers"])
app.include_router(container_data.router, prefix="/api", tags=["container_data"])
app.include_router(analytics.router, prefix="/api", tags=["analytics"])
app.include_router(forecast.router, prefix="/api", tags=["forecast"])


# ---------------------------------------------------------------------------
# Static data files (served at /data/*)
# ---------------------------------------------------------------------------
if settings.data_dir.exists():
    app.mount(
        "/data",
        StaticFiles(directory=settings.data_dir),
        name="data",
    )


# ---------------------------------------------------------------------------
# SPA serving: web/dist/ with client-side routing fallback
# ---------------------------------------------------------------------------
if settings.dist_dir.exists():
    _assets_dir = settings.dist_dir / "assets"
    if _assets_dir.exists():
        app.mount(
            "/assets",
            StaticFiles(directory=_assets_dir),
            name="assets",
        )

    @app.get("/{full_path:path}", include_in_schema=False, response_model=None)
    async def spa_fallback(full_path: str) -> FileResponse | JSONResponse:
        """Serve files from web/dist/, falling back to index.html for client routes.

        /api/* requests are excluded so they return the router's real 404
        rather than the SPA shell.
        """
        if full_path.startswith("api/"):
            return JSONResponse(status_code=404, content={"detail": "Not Found"})

        candidate = settings.dist_dir / full_path
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(settings.dist_dir / "index.html")
