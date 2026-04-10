import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type { Horizon } from "@/types/api";

/**
 * Fetch the Hamburg peak-load forecast from the FastAPI backend.
 *
 * `horizon` is 24 or 336 hours. Results are cached for 5 minutes — the
 * backend itself reads from a 60-second file cache, so this is just a
 * client-side de-duplication.
 */
export function useForecast(horizon: Horizon) {
  return useQuery({
    queryKey: ["forecast", horizon] as const,
    queryFn: ({ signal }) => api.forecast(horizon, signal),
    staleTime: 5 * 60_000,
  });
}
