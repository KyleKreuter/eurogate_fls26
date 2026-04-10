# Eurogate FLS26 Backend

FastAPI backend serving the Hamburg reefer container power forecast dashboard and container inspector API, reading from a SQLite database.

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (or `pip` + `venv`)
- Git LFS (required for pulling the `reefer_release.csv` source data used by the rebuild scripts)

## Quick Start

```bash
# 1. Install dependencies
cd backend
uv sync

# 2. Reassemble and unzip the SQLite database (first time only)
cat reefer.db.zip.0* > reefer.db.zip
unzip reefer.db.zip
rm reefer.db.zip

# 3. Start the server
uvicorn app.main:app --reload --port 8080

# 4. Open auto-docs
open http://localhost:8080/docs
```

## Layout

```
backend/
├── app/                  # FastAPI application
│   ├── main.py           # App factory, CORS, static mounts, SPA fallback
│   ├── config.py         # pydantic-settings (DB_PATH, DATA_DIR)
│   ├── db.py             # SQLite connection helper (check_same_thread=False)
│   ├── models.py         # Pydantic response models
│   └── routers/
│       ├── containers.py       # GET /api/containers
│       ├── container_data.py   # GET /api/data
│       ├── analytics.py        # GET /api/overview-analytics
│       └── forecast.py         # GET /api/forecast
├── data/                 # Canonical data files served at /data/*
│   ├── dashboard_data.csv
│   ├── historical_visualizations.json
│   └── predictions.csv
├── scripts/              # Data pipeline (see "Regenerating Data")
│   ├── build_database.py
│   ├── train_and_predict.py
│   ├── generate_dashboard_data.py
│   └── generate_historical_viz_data.py
├── legacy/               # Old Python dashboard, kept for QA diff until Phase 13
├── reefer.db             # gitignored; rebuilt from zips or source CSV
├── reefer.db.zip.001..004  # split SQLite database
└── pyproject.toml
```

## API Endpoints

All endpoints return JSON. FastAPI auto-docs live at `/docs` and `/redoc`.

### `GET /api/containers`

Paginated container list backed by the `container_stats` table.

Query params:

| Param    | Default                    | Description                                                         |
| -------- | -------------------------- | ------------------------------------------------------------------- |
| `limit`  | `50`                       | Page size                                                           |
| `offset` | `0`                        | Row offset                                                          |
| `sort`   | `total_connected_hours`    | One of `uuid`, `num_visits`, `total_connected_hours`, `avg_visit_hours` |
| `dir`    | `DESC`                     | `ASC` or `DESC`                                                     |
| `q`      | empty                      | Substring filter on `container_uuid`                                |

```bash
curl "http://localhost:8080/api/containers?limit=5&sort=num_visits&dir=DESC"
```

### `GET /api/data?uuid=<uuid>`

Returns a single container's full event timeline, per-visit aggregates, and container-level stats.

```bash
curl "http://localhost:8080/api/data?uuid=a3f2-8c91-4b7d-..."
```

Response shape: `{ timeline: [...], visits: [...], num_visits, total_connected_hours, avg_visit_hours, last_visit_start, last_visit_end }`.

### `GET /api/overview-analytics`

Fleet-wide analytics (7 aggregations: active-per-day, hardware types, duration histogram, monthly energy, setpoint distribution, container sizes, hour-DoW heatmap). Result is cached in memory (1h TTL).

```bash
curl "http://localhost:8080/api/overview-analytics"
```

### `GET /api/forecast?horizon=24|336`

Hamburg peak-load forecast for the next 24h or 14d (336h), read from `data/dashboard_data.csv` (produced by `generate_dashboard_data.py`).

```bash
curl "http://localhost:8080/api/forecast?horizon=24"
```

### Static + SPA

- `/data/*` serves files from `backend/data/` (CSV/JSON consumed directly by the frontend).
- `/` serves the built React SPA from `../web/dist/` in production, with a catch-all fallback for client-side routes.

## Development Workflow

During development the stack runs as two processes:

- **`backend/`** — FastAPI on `http://localhost:8080` (`uvicorn app.main:app --reload --port 8080`)
- **`web/`** — Vite React dev server on `http://localhost:5173` (`npm run dev`)

`web/vite.config.ts` proxies `/api/*` and `/data/*` to `:8080`, so the frontend uses relative URLs in both dev and prod.

## Regenerating Data

The data pipeline runs in order. Run each script from the repository root with the Python environment activated:

```bash
# 1. Build SQLite DB from the raw release CSV
python backend/scripts/build_database.py --source-csv participant_package/daten/reefer_release.csv

# 2. Train ML model and write predictions.csv
python backend/scripts/train_and_predict.py

# 3. Merge predictions into the dashboard dataset
python backend/scripts/generate_dashboard_data.py

# 4. Precompute historical aggregations for the analytics tab
python backend/scripts/generate_historical_viz_data.py
```

## Production Deploy

Single-process deployment: build the SPA, then let Uvicorn serve both the API and the static bundle.

```bash
cd web && npm run build        # → web/dist/
cd ../backend && uvicorn app.main:app --host 0.0.0.0 --port 8080
```

The FastAPI app mounts `../web/dist/` at `/` with an SPA fallback so client-side routes resolve correctly.

## Risks / Known Issues

- **`reefer.db.zip.001..004` are raw blobs, not Git LFS.** Despite what `.gitattributes` implies, these files contribute roughly 174 MB of repo bloat. LFS migration is pending (Phase 13).
- **Cold-start analytics.** The first call to `/api/overview-analytics` runs seven aggregation queries (~seconds). Subsequent calls are served from an in-memory cache with a 1 hour TTL.
- **SQLite under Uvicorn** requires `check_same_thread=False` (handled in `app/db.py`). A single writer / many readers pattern is fine for MVP load, but contention is possible under heavy concurrency.
- **`build_database.py` historically contained a hardcoded `/Users/rkohlbach` path.** The refactored script uses `--source-csv`/env vars; verify your invocation points at the correct CSV.
