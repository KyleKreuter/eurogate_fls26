import { useMemo } from "react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import type { ActivePerDay } from "@/types/api";
import { chartColors, chartTheme } from "@/lib/chart-theme";

interface Props {
  data: ActivePerDay;
}

export function ActivePerDayChart({ data }: Props) {
  const rows = useMemo(
    () => data.dates.map((d, i) => ({ date: d, count: data.counts[i] ?? 0 })),
    [data],
  );

  const tickInterval = Math.max(1, Math.floor(rows.length / 12));

  return (
    <div style={{ width: "100%", height: 220 }}>
      <ResponsiveContainer>
        <AreaChart data={rows} margin={{ top: 10, right: 20, left: -10, bottom: 0 }}>
          <defs>
            <linearGradient id="activeFill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={chartColors.forecast} stopOpacity={0.25} />
              <stop offset="100%" stopColor={chartColors.forecast} stopOpacity={0.02} />
            </linearGradient>
          </defs>
          <CartesianGrid {...chartTheme.grid} />
          <XAxis dataKey="date" {...chartTheme.axis} interval={tickInterval} />
          <YAxis {...chartTheme.axis} />
          <Tooltip
            contentStyle={chartTheme.tooltip.contentStyle}
            labelStyle={chartTheme.tooltip.labelStyle}
            itemStyle={chartTheme.tooltip.itemStyle}
          />
          <Area
            type="monotone"
            dataKey="count"
            name="Active"
            stroke={chartColors.forecast}
            strokeWidth={2}
            fill="url(#activeFill)"
            isAnimationActive={false}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
