import { useMemo } from "react";
import { Heatmap } from "@/components/Heatmap";
import type { HourlyHeatmapCell } from "@/types/api";

interface Props {
  data: HourlyHeatmapCell[];
}

const DOW_LABELS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
const HOUR_LABELS = Array.from({ length: 24 }, (_, h) =>
  h.toString().padStart(2, "0"),
);

/**
 * Hour × Day-of-Week activity heatmap (24 × 7 grid).
 */
export function HourDowHeatmap({ data }: Props) {
  const grid = useMemo(() => {
    const rows: number[][] = Array.from({ length: 7 }, () =>
      new Array(24).fill(0),
    );
    for (const cell of data) {
      if (cell.dow >= 0 && cell.dow < 7 && cell.hour >= 0 && cell.hour < 24) {
        rows[cell.dow][cell.hour] = cell.count;
      }
    }
    return rows;
  }, [data]);

  return (
    <Heatmap
      data={grid}
      rowLabels={DOW_LABELS}
      colLabels={HOUR_LABELS}
      cellSize={16}
      cellGap={2}
      tooltip={(v, row, col) => `${row} ${col}:00 — ${v} active`}
      aria-label="Activity heatmap: hour of day vs day of week"
    />
  );
}
