import { useMemo } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Cell,
  ResponsiveContainer,
} from "recharts";
import type { DurationHistogram } from "@/types/api";
import { DATA_PALETTE } from "@/lib/palette";
import { chartTheme } from "@/lib/chart-theme";

interface Props {
  data: DurationHistogram;
}

export function DurationHistogramChart({ data }: Props) {
  const rows = useMemo(
    () => data.labels.map((l, i) => ({ label: l, count: data.counts[i] ?? 0 })),
    [data],
  );

  return (
    <div style={{ width: "100%", height: 220 }}>
      <ResponsiveContainer>
        <BarChart
          data={rows}
          layout="vertical"
          margin={{ top: 10, right: 20, left: 20, bottom: 0 }}
        >
          <CartesianGrid {...chartTheme.grid} vertical horizontal={false} />
          <XAxis type="number" {...chartTheme.axis} />
          <YAxis type="category" dataKey="label" {...chartTheme.axis} width={70} />
          <Tooltip
            contentStyle={chartTheme.tooltip.contentStyle}
            labelStyle={chartTheme.tooltip.labelStyle}
            itemStyle={chartTheme.tooltip.itemStyle}
          />
          <Bar dataKey="count" isAnimationActive={false} radius={[0, 3, 3, 0]}>
            {rows.map((_, i) => (
              <Cell key={i} fill={DATA_PALETTE[i % DATA_PALETTE.length]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
