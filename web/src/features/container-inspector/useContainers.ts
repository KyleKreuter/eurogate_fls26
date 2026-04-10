import { useQuery, keepPreviousData } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type { ContainersQuery } from "@/types/api";

/**
 * Paginated, sortable, searchable container list. Uses
 * `keepPreviousData` so page/sort changes don't flash the table back
 * to a skeleton.
 */
export function useContainers(params: ContainersQuery) {
  return useQuery({
    queryKey: ["containers", params] as const,
    queryFn: ({ signal }) => api.containers(params, signal),
    placeholderData: keepPreviousData,
    staleTime: 30_000,
  });
}
