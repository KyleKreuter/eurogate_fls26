import { useMemo } from "react";
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer, Legend } from "recharts";
import type { HardwareTypeEntry } from "@/types/api";
import { DATA_PALETTE } from "@/lib/palette";
import { chartTheme } from "@/lib/chart-theme";

interface Props {
  data: HardwareTypeEntry[];
}

/**
 * Top-7 hardware types + an "Other" bucket, rendered as a doughnut chart.
 */
export function HardwareDoughnut({ data }: Props) {
  const prepared = useMemo(() => {
    const sorted = [...data].sort((a, b) => b.cnt - a.cnt);
    const top = sorted.slice(0, 7);
    const rest = sorted.slice(7);
    const otherCnt = rest.reduce((sum, r) => sum + r.cnt, 0);
    const rows = top.map((r) => ({
      name: r.hardware_type || "unknown",
      value: r.cnt,
    }));
    if (otherCnt > 0) rows.push({ name: "Other", value: otherCnt });
    return rows;
  }, [data]);

  return (
    <div style={{ width: "100%", height: 220 }}>
      <ResponsiveContainer>
        <PieChart margin={{ top: 0, right: 0, left: 0, bottom: 0 }}>
          <Pie
            data={prepared}
            dataKey="value"
            nameKey="name"
            cx="38%"
            cy="50%"
            innerRadius={50}
            outerRadius={80}
            paddingAngle={1.5}
            stroke="var(--bg-card)"
            isAnimationActive={false}
          >
            {prepared.map((_, i) => (
              <Cell key={i} fill={DATA_PALETTE[i % DATA_PALETTE.length]} />
            ))}
          </Pie>
          <Tooltip
            contentStyle={chartTheme.tooltip.contentStyle}
            labelStyle={chartTheme.tooltip.labelStyle}
            itemStyle={chartTheme.tooltip.itemStyle}
          />
          <Legend
            layout="vertical"
            align="right"
            verticalAlign="middle"
            wrapperStyle={{
              ...chartTheme.legend.wrapperStyle,
              fontSize: "0.68rem",
              paddingLeft: "0.5rem",
            }}
          />
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
}
