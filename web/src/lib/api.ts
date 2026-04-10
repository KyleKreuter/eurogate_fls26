/**
 * Typed fetch wrappers for the FastAPI backend.
 *
 * All functions hit relative paths (`/api/...`) so they work via the Vite
 * dev proxy (→ 127.0.0.1:8080) and in production (same-origin SPA served by
 * the uvicorn backend itself). Override with `VITE_API_BASE` for split
 * deploys.
 */

import type {
  ContainersQuery,
  ContainersResponse,
  ContainerDetail,
  OverviewAnalytics,
  ForecastResponse,
  Horizon,
} from "@/types/api";

const API_BASE = import.meta.env.VITE_API_BASE ?? "";

export class ApiError extends Error {
  readonly status: number;
  readonly statusText: string;
  readonly body: string;

  constructor(status: number, statusText: string, body: string) {
    super(`${status} ${statusText}: ${body.slice(0, 120)}`);
    this.name = "ApiError";
    this.status = status;
    this.statusText = statusText;
    this.body = body;
  }
}

async function get<T>(path: string, signal?: AbortSignal): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { signal });
  if (!res.ok) {
    throw new ApiError(res.status, res.statusText, await res.text());
  }
  return res.json() as Promise<T>;
}

function buildContainerQuery(p: ContainersQuery): string {
  const params = new URLSearchParams();
  params.set("limit", String(p.limit ?? 50));
  params.set("offset", String(p.offset ?? 0));
  params.set("sort", p.sort ?? "num_visits");
  params.set("dir", p.dir ?? "DESC");
  if (p.q && p.q.trim().length > 0) {
    params.set("q", p.q.trim());
  }
  return params.toString();
}

export const api = {
  containers: (p: ContainersQuery, signal?: AbortSignal) =>
    get<ContainersResponse>(`/api/containers?${buildContainerQuery(p)}`, signal),

  containerDetail: (uuid: string, signal?: AbortSignal) =>
    get<ContainerDetail>(`/api/data?uuid=${encodeURIComponent(uuid)}`, signal),

  overviewAnalytics: (signal?: AbortSignal) =>
    get<OverviewAnalytics>(`/api/overview-analytics`, signal),

  forecast: (horizon: Horizon, signal?: AbortSignal) =>
    get<ForecastResponse>(`/api/forecast?horizon=${horizon}`, signal),

  health: (signal?: AbortSignal) =>
    get<{ status: string }>(`/api/health`, signal),
};
