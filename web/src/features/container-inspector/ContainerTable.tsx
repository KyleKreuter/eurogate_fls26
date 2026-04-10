import { DataTable } from "@/components/DataTable";
import { EmptyState } from "@/components/EmptyState";
import type { ContainerRow, SortColumn, SortDirection } from "@/types/api";
import { formatInt, formatHours, formatUuidShort } from "@/lib/format";

interface Props {
  rows: ContainerRow[];
  isLoading: boolean;
  sort: SortColumn;
  dir: SortDirection;
  onSortChange: (key: string) => void;
  selectedUuid: string | null;
  onRowClick: (row: ContainerRow) => void;
}

export function ContainerTable({
  rows,
  isLoading,
  sort,
  dir,
  onSortChange,
  selectedUuid,
  onRowClick,
}: Props) {
  return (
    <DataTable<ContainerRow>
      rows={rows}
      rowKey={(r) => r.uuid}
      selectedKey={selectedUuid ?? undefined}
      onRowClick={onRowClick}
      isLoading={isLoading}
      sortKey={sort}
      sortDir={dir}
      onSortChange={onSortChange}
      emptyState={
        <EmptyState
          title="No containers match your search"
          description="Try a shorter UUID fragment or clear the filter."
        />
      }
      columns={[
        {
          key: "uuid",
          label: "UUID",
          sortable: true,
          mono: true,
          render: (r) => formatUuidShort(r.uuid, 2),
        },
        {
          key: "num_visits",
          label: "Visits",
          align: "right",
          sortable: true,
          mono: true,
          render: (r) => formatInt(r.num_visits),
        },
        {
          key: "total_connected_hours",
          label: "Connected",
          align: "right",
          sortable: true,
          mono: true,
          render: (r) => formatHours(r.total_connected_hours),
        },
        {
          key: "avg_visit_hours",
          label: "Avg / Visit",
          align: "right",
          sortable: true,
          mono: true,
          render: (r) => formatHours(r.avg_visit_hours),
        },
      ]}
    />
  );
}
