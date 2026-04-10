import { useMemo } from "react";
import { Heatmap } from "@/components/Heatmap";
import type { ActivePerDay } from "@/types/api";

interface Props {
  /**
   * The backend only ships `active_per_day` — we synthesize the 52×7
   * calendar grid client-side since legacy has no calendar_heatmap endpoint.
   */
  activePerDay: ActivePerDay;
}

const DOW_LABELS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
const WEEK_LABELS = Array.from({ length: 53 }, (_, w) => `W${w + 1}`);

/**
 * Annual calendar heatmap (52 weeks × 7 days).
 * Each cell = one day's distinct-container-visit count.
 */
export function CalendarHeatmap({ activePerDay }: Props) {
  const grid = useMemo(() => {
    if (activePerDay.dates.length === 0) {
      return Array.from({ length: 7 }, () => new Array(53).fill(0));
    }

    // Anchor: the Sunday on or before the first date.
    const first = new Date(activePerDay.dates[0] + "T00:00:00Z");
    const firstDow = first.getUTCDay();
    const anchor = new Date(first);
    anchor.setUTCDate(first.getUTCDate() - firstDow);

    const rows: number[][] = Array.from({ length: 7 }, () =>
      new Array(53).fill(0),
    );

    for (let i = 0; i < activePerDay.dates.length; i++) {
      const d = new Date(activePerDay.dates[i] + "T00:00:00Z");
      const daysSinceAnchor = Math.floor(
        (d.getTime() - anchor.getTime()) / (1000 * 60 * 60 * 24),
      );
      const week = Math.floor(daysSinceAnchor / 7);
      const dow = daysSinceAnchor % 7;
      if (week >= 0 && week < 53 && dow >= 0 && dow < 7) {
        rows[dow][week] = activePerDay.counts[i] ?? 0;
      }
    }
    return rows;
  }, [activePerDay]);

  return (
    <Heatmap
      data={grid}
      rowLabels={DOW_LABELS}
      colLabels={WEEK_LABELS}
      cellSize={12}
      cellGap={2}
      tooltip={(v) => `${v} active containers`}
      aria-label="Annual calendar heatmap"
    />
  );
}
