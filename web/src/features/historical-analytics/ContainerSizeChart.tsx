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
import type { ContainerSizeRow } from "@/types/api";
import { DATA_PALETTE } from "@/lib/palette";
import { chartTheme } from "@/lib/chart-theme";

interface Props {
  data: ContainerSizeRow[];
}

export function ContainerSizeChart({ data }: Props) {
  const rows = data.map((r) => ({
    size: `${r.size}ft`,
    avg_kw: r.avg_power_kw ?? 0,
  }));

  return (
    <div style={{ width: "100%", height: 220 }}>
      <ResponsiveContainer>
        <BarChart
          data={rows}
          layout="vertical"
          margin={{ top: 10, right: 20, left: 10, bottom: 0 }}
        >
          <CartesianGrid {...chartTheme.grid} vertical horizontal={false} />
          <XAxis type="number" {...chartTheme.axis} />
          <YAxis type="category" dataKey="size" {...chartTheme.axis} width={50} />
          <Tooltip
            contentStyle={chartTheme.tooltip.contentStyle}
            labelStyle={chartTheme.tooltip.labelStyle}
            itemStyle={chartTheme.tooltip.itemStyle}
            formatter={(v) => [`${Number(v).toFixed(1)} kW`, "Avg Power"]}
          />
          <Bar dataKey="avg_kw" isAnimationActive={false} radius={[0, 3, 3, 0]}>
            {rows.map((_, i) => (
              <Cell key={i} fill={DATA_PALETTE[i % DATA_PALETTE.length]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
