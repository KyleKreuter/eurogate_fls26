"""Application configuration via pydantic-settings.

All paths are resolved relative to this file (NOT the current working directory),
so the server behaves identically regardless of where uvicorn is launched from.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Directory layout anchors (script-relative):
#   _APP_DIR     = <repo>/backend/app
#   _BACKEND_DIR = <repo>/backend
#   _REPO_ROOT   = <repo>
_APP_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _APP_DIR.parent
_REPO_ROOT = _BACKEND_DIR.parent


class Settings(BaseSettings):
    """Runtime settings. Override via EUROGATE_* environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="EUROGATE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # SQLite database file (reassembled at runtime from reefer.db.zip.00*)
    db_path: Path = _BACKEND_DIR / "reefer.db"

    # Canonical data directory served as static files at /data/*
    data_dir: Path = _BACKEND_DIR / "data"

    # Built React SPA output (exists only after `npm run build` in web/)
    dist_dir: Path = _REPO_ROOT / "web" / "dist"

    # Cache TTL for overview analytics aggregation (seconds)
    analytics_cache_ttl: int = 3600

    # CORS allow-list for the frontend dev servers + legacy port
    cors_origins: list[str] = [
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:8080",
    ]


settings = Settings()
