import { useState } from "react";
import { useSearchParams } from "react-router-dom";
import { GlassPanel } from "@/components/GlassPanel";
import { PageHeader } from "@/components/PageHeader";
import { ErrorState } from "@/components/ErrorState";
import { useDebounce } from "@/hooks/useDebounce";
import type { ContainerRow, SortColumn, SortDirection } from "@/types/api";
import { useContainers } from "./useContainers";
import { ContainerSearchInput } from "./ContainerSearchInput";
import { ContainerTable } from "./ContainerTable";
import { ContainerPagination } from "./ContainerPagination";
import { ContainerDetailPanel } from "./ContainerDetailPanel";

const LIMIT = 50;
const SORT_COLUMNS: SortColumn[] = [
  "uuid",
  "num_visits",
  "total_connected_hours",
  "avg_visit_hours",
];

function isSortColumn(v: string | null): v is SortColumn {
  return v != null && (SORT_COLUMNS as string[]).includes(v);
}

/**
 * Third tab of the Hamburg dashboard — container inspector.
 * 40/60 two-pane: table left, detail panel right. URL search params
 * drive sort/dir/q/offset; selected container uuid is local state.
 */
export function ContainerInspectorTab() {
  const [searchParams, setSearchParams] = useSearchParams();

  // Read URL params
  const sort: SortColumn = isSortColumn(searchParams.get("sort"))
    ? (searchParams.get("sort") as SortColumn)
    : "num_visits";
  const dirRaw = searchParams.get("dir");
  const dir: SortDirection = dirRaw === "ASC" ? "ASC" : "DESC";
  const q = searchParams.get("q") ?? "";
  const offset = Math.max(0, Number(searchParams.get("offset") ?? "0") || 0);

  // Debounced search input
  const [searchInput, setSearchInput] = useState(q);
  const debouncedSearch = useDebounce(searchInput, 300);

  // Selected container (ephemeral UI state)
  const [selectedUuid, setSelectedUuid] = useState<string | null>(null);

  // Write debounced search back to URL
  if (debouncedSearch !== q) {
    setSearchParams(
      (prev) => {
        const p = new URLSearchParams(prev);
        if (debouncedSearch) p.set("q", debouncedSearch);
        else p.delete("q");
        p.delete("offset");
        // keep tab param
        return p;
      },
      { replace: true },
    );
  }

  const updateParams = (patch: Record<string, string | null>) => {
    setSearchParams(
      (prev) => {
        const p = new URLSearchParams(prev);
        for (const [key, value] of Object.entries(patch)) {
          if (value == null) p.delete(key);
          else p.set(key, value);
        }
        return p;
      },
      { replace: true },
    );
  };

  const handleSortChange = (key: string) => {
    if (!isSortColumn(key)) return;
    if (key === sort) {
      updateParams({ dir: dir === "ASC" ? "DESC" : "ASC", offset: "0" });
    } else {
      updateParams({ sort: key, dir: "DESC", offset: "0" });
    }
  };

  const handleRowClick = (row: ContainerRow) => {
    setSelectedUuid(row.uuid);
  };

  const { data, isLoading, isFetching, isError, error, refetch } = useContainers({
    limit: LIMIT,
    offset,
    sort,
    dir,
    q: q || undefined,
  });

  return (
    <div className="flex flex-col gap-6 page-enter">
      <PageHeader
        title="Container Inspector"
        subtitle="Isolate a specific payload"
      />

      <div className="flex flex-col lg:flex-row gap-6">
        {/* Left pane — 40% on desktop */}
        <div className="flex flex-col gap-3 lg:w-[42%] min-w-0">
          <ContainerSearchInput
            value={searchInput}
            onChange={setSearchInput}
          />

          <GlassPanel variant="flush">
            {isError ? (
              <div className="p-6">
                <ErrorState
                  title="Failed to load containers"
                  description={(error as Error)?.message}
                  onRetry={() => refetch()}
                />
              </div>
            ) : (
              <ContainerTable
                rows={data?.containers ?? []}
                isLoading={isLoading || isFetching}
                sort={sort}
                dir={dir}
                onSortChange={handleSortChange}
                selectedUuid={selectedUuid}
                onRowClick={handleRowClick}
              />
            )}
          </GlassPanel>

          {data && (
            <ContainerPagination
              total={data.total}
              limit={LIMIT}
              offset={offset}
              onChange={(next) => updateParams({ offset: String(next) })}
            />
          )}
        </div>

        {/* Right pane — 60% */}
        <div className="flex-1 min-w-0">
          <ContainerDetailPanel
            uuid={selectedUuid}
            onClose={() => setSelectedUuid(null)}
          />
        </div>
      </div>
    </div>
  );
}
