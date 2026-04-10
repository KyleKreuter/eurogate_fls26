import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";

/**
 * Backend caches this response for 1h already, so on the client we treat
 * it as effectively immutable for the session.
 */
export function useOverviewAnalytics() {
  return useQuery({
    queryKey: ["overview-analytics"] as const,
    queryFn: ({ signal }) => api.overviewAnalytics(signal),
    staleTime: Infinity,
    gcTime: Infinity,
  });
}
