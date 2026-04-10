import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";

/**
 * Full drilldown for a single container uuid. Only fires when `uuid` is
 * set (disabled otherwise, so the detail pane shows an empty state until
 * the user picks a row).
 */
export function useContainerDetail(uuid: string | null) {
  return useQuery({
    queryKey: ["container-detail", uuid] as const,
    queryFn: ({ signal }) => api.containerDetail(uuid as string, signal),
    enabled: !!uuid,
    staleTime: 5 * 60_000,
  });
}
