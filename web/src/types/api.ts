/**
 * Shared TypeScript types for the Eurogate FLS26 backend API.
 *
 * These MUST mirror the Pydantic v2 response models in
 * `backend/app/models.py` exactly (same field names, same nullability).
 * When the backend shape changes, update this file in lock-step.
 */

// ────────────────────────────────────────────────────────────────────────────
// GET /api/containers
// ────────────────────────────────────────────────────────────────────────────

export type SortColumn =
  | "uuid"
  | "num_visits"
  | "total_connected_hours"
  | "avg_visit_hours";

export type SortDirection = "ASC" | "DESC";

export interface ContainerRow {
  uuid: string;
  num_visits: number;
  total_connected_hours: number;
  avg_visit_hours: number;
  last_visit_start: string | null;
  last_visit_end: string | null;
}

export interface ContainersResponse {
  containers: ContainerRow[];
  total: number;
  limit: number;
  offset: number;
}

export interface ContainersQuery {
  limit?: number;
  offset?: number;
  sort?: SortColumn;
  dir?: SortDirection;
  q?: string;
}

// ────────────────────────────────────────────────────────────────────────────
// GET /api/data?uuid=...
// ────────────────────────────────────────────────────────────────────────────

/**
 * One event reading for a container.
 *
 * `temp_supply` is the normalized version of the SQL column `RemperatureSupply`
 * (the typo is a real column name in the DB, preserved verbatim in SQL and
 * re-keyed in the response).
 */
export interface TimelinePoint {
  time: string;
  visit_uuid: string | null;
  power_kw: number;
  energy_hour: number | null;
  energy_total: number | null;
  setpoint: number | null;
  ambient: number | null;
  temp_return: number | null;
  temp_supply: number | null;
  hardware_type: string | null;
  container_size: string | null;
  stack_tier: number | null;
}

export interface Visit {
  visit_uuid: string;
  visit_start: string;
  visit_end: string;
  duration_hours: number;
  reading_count: number;
  hardware_type: string | null;
  container_size: string | null;
  avg_power_kw: number;
}

export interface ContainerDetail {
  timeline: TimelinePoint[];
  visits: Visit[];
  num_visits: number;
  total_connected_hours: number;
  avg_visit_hours: number;
  last_visit_start: string | null;
  last_visit_end: string | null;
}

// ────────────────────────────────────────────────────────────────────────────
// GET /api/overview-analytics
// ────────────────────────────────────────────────────────────────────────────

export interface ActivePerDay {
  dates: string[];
  counts: number[];
}

export interface HardwareTypeEntry {
  hardware_type: string;
  cnt: number;
  avg_kw: number | null;
}

export interface DurationHistogram {
  labels: string[];
  counts: number[];
}

export interface MonthlyEnergyRow {
  month: string;
  total_mwh: number | null;
}

export interface SetpointBin {
  sp_bin: number;
  cnt: number;
}

export interface ContainerSizeRow {
  size: number;
  cnt: number;
  avg_power_kw: number | null;
}

export interface HourlyHeatmapCell {
  hour: number;
  /** SQLite strftime('%w', ...) — 0 = Sunday */
  dow: number;
  count: number;
}

export interface OverviewAnalytics {
  active_per_day: ActivePerDay;
  hardware_types: HardwareTypeEntry[];
  duration_hist: DurationHistogram;
  monthly_energy: MonthlyEnergyRow[];
  setpoint_dist: SetpointBin[];
  container_sizes: ContainerSizeRow[];
  hourly_heatmap: HourlyHeatmapCell[];
}

// ────────────────────────────────────────────────────────────────────────────
// GET /api/forecast?horizon=24|336
// ────────────────────────────────────────────────────────────────────────────

export type Horizon = 24 | 336;

export interface ForecastPoint {
  timestamp_utc: string;
  pred_power_kw: number;
  pred_p90_kw: number;
  history_lastyear_kw: number;
}

export interface ForecastResponse {
  points: ForecastPoint[];
  horizon: number;
}
