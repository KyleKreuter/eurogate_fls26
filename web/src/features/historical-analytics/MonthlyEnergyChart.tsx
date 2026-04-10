import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import type { MonthlyEnergyRow } from "@/types/api";
import { chartColors, chartTheme } from "@/lib/chart-theme";

interface Props {
  data: MonthlyEnergyRow[];
}

export function MonthlyEnergyChart({ data }: Props) {
  const rows = data.map((r) => ({
    month: r.month,
    mwh: r.total_mwh ?? 0,
  }));

  return (
    <div style={{ width: "100%", height: 220 }}>
      <ResponsiveContainer>
        <BarChart data={rows} margin={{ top: 10, right: 20, left: -10, bottom: 0 }}>
          <CartesianGrid {...chartTheme.grid} />
          <XAxis dataKey="month" {...chartTheme.axis} />
          <YAxis {...chartTheme.axis} />
          <Tooltip
            contentStyle={chartTheme.tooltip.contentStyle}
            labelStyle={chartTheme.tooltip.labelStyle}
            itemStyle={chartTheme.tooltip.itemStyle}
            formatter={(v) => [`${Number(v).toFixed(2)} MWh`, "Energy"]}
          />
          <Bar
            dataKey="mwh"
            name="Energy"
            fill={chartColors.forecast}
            isAnimationActive={false}
            radius={[3, 3, 0, 0]}
          />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
