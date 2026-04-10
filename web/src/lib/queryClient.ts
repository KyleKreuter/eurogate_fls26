import { QueryClient } from "@tanstack/react-query";

/**
 * Shared TanStack Query client for the Eurogate dashboard.
 *
 * - `staleTime: 60s` — most endpoints are aggregated reads that don't need
 *   aggressive refetching. Individual hooks override this where needed
 *   (e.g. `useOverviewAnalytics` uses `Infinity` to match the backend TTL).
 * - `refetchOnWindowFocus: false` — the dashboard is an operations tool,
 *   users alt-tab constantly; we don't want flicker.
 * - `retry: 1` — one retry on transient failures, then surface the error.
 */
export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 60_000,
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
});
