import { StatPill } from "@/components/StatPill";
import { formatInt, formatHours, formatDateTime } from "@/lib/format";
import type { ContainerDetail } from "@/types/api";

interface Props {
  detail: ContainerDetail;
}

/** 4 stat pills summarising a single container's activity. */
export function ContainerStatsBanner({ detail }: Props) {
  return (
    <div className="flex flex-wrap gap-3">
      <StatPill
        highlight
        label="Total Visits"
        value={formatInt(detail.num_visits)}
      />
      <StatPill
        label="Total Connected"
        value={formatHours(detail.total_connected_hours)}
      />
      <StatPill
        label="Avg / Visit"
        value={formatHours(detail.avg_visit_hours)}
      />
      <StatPill
        label="Last Visit"
        value={formatDateTime(detail.last_visit_end ?? detail.last_visit_start)}
      />
    </div>
  );
}
