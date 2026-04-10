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
import type { SetpointBin } from "@/types/api";
import { chartTheme } from "@/lib/chart-theme";

interface Props {
  data: SetpointBin[];
}

/**
 * Setpoint temperature histogram (5° bins). Cold bins get blue,
 * warm bins get orange.
 */
export function SetpointDistChart({ data }: Props) {
  const rows = data.map((r) => ({
    bin: `${r.sp_bin}°`,
    cnt: r.cnt,
    raw: r.sp_bin,
  }));

  return (
    <div style={{ width: "100%", height: 220 }}>
      <ResponsiveContainer>
        <BarChart data={rows} margin={{ top: 10, right: 20, left: -10, bottom: 0 }}>
          <CartesianGrid {...chartTheme.grid} />
          <XAxis dataKey="bin" {...chartTheme.axis} />
          <YAxis {...chartTheme.axis} />
          <Tooltip
            contentStyle={chartTheme.tooltip.contentStyle}
            labelStyle={chartTheme.tooltip.labelStyle}
            itemStyle={chartTheme.tooltip.itemStyle}
          />
          <Bar dataKey="cnt" isAnimationActive={false} radius={[3, 3, 0, 0]}>
            {rows.map((r, i) => (
              <Cell
                key={i}
                fill={r.raw < 0 ? "var(--accent-primary)" : "var(--warning)"}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
