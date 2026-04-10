import { useMemo } from "react";
import {
  ComposedChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import type { TimelinePoint } from "@/types/api";
import type { Granularity } from "./TimelineGranularityToggle";
import { chartColors, chartTheme } from "@/lib/chart-theme";
import { formatDateTime, formatDateShort } from "@/lib/format";

interface Props {
  timeline: TimelinePoint[];
  granularity: Granularity;
}

interface ChartRow {
  label: string;
  power: number | null;
  setpoint: number | null;
  ambient: number | null;
}

/**
 * Dual Y-axis timeline chart. Left axis = power (kW, blue), right axis =
 * temperature (°C, setpoint red dashed, ambient orange).
 */
export function ContainerTimelineChart({ timeline, granularity }: Props) {
  const rows: ChartRow[] = useMemo(() => {
    if (granularity === "hourly") {
      return timeline.map((p) => ({
        label: formatDateTime(p.time),
        power: Number.isFinite(p.power_kw) ? p.power_kw : null,
        setpoint: p.setpoint,
        ambient: p.ambient,
      }));
    }

    // Daily aggregation — average values per calendar date
    const buckets = new Map<
      string,
      { powerSum: number; powerN: number; spSum: number; spN: number; aSum: number; aN: number }
    >();
    for (const p of timeline) {
      const day = (p.time ?? "").slice(0, 10);
      const b = buckets.get(day) ?? {
        powerSum: 0,
        powerN: 0,
        spSum: 0,
        spN: 0,
        aSum: 0,
        aN: 0,
      };
      if (Number.isFinite(p.power_kw)) {
        b.powerSum += p.power_kw;
        b.powerN += 1;
      }
      if (p.setpoint != null && Number.isFinite(p.setpoint)) {
        b.spSum += p.setpoint;
        b.spN += 1;
      }
      if (p.ambient != null && Number.isFinite(p.ambient)) {
        b.aSum += p.ambient;
        b.aN += 1;
      }
      buckets.set(day, b);
    }
    return Array.from(buckets.entries()).map(([day, b]) => ({
      label: formatDateShort(`${day}T00:00:00Z`),
      power: b.powerN > 0 ? b.powerSum / b.powerN : null,
      setpoint: b.spN > 0 ? b.spSum / b.spN : null,
      ambient: b.aN > 0 ? b.aSum / b.aN : null,
    }));
  }, [timeline, granularity]);

  const tickInterval = Math.max(1, Math.floor(rows.length / 12));

  return (
    <div style={{ width: "100%", height: 360 }}>
      <ResponsiveContainer>
        <ComposedChart data={rows} margin={{ top: 10, right: 30, left: -10, bottom: 0 }}>
          <CartesianGrid {...chartTheme.grid} />
          <XAxis dataKey="label" {...chartTheme.axis} interval={tickInterval} />
          <YAxis yAxisId="power" {...chartTheme.axis} />
          <YAxis
            yAxisId="temp"
            orientation="right"
            {...chartTheme.axis}
            tickFormatter={(v) => `${v}°`}
          />
          <Tooltip
            contentStyle={chartTheme.tooltip.contentStyle}
            labelStyle={chartTheme.tooltip.labelStyle}
            itemStyle={chartTheme.tooltip.itemStyle}
          />
          <Legend
            wrapperStyle={chartTheme.legend.wrapperStyle}
            iconType="plainline"
          />
          <Line
            yAxisId="power"
            type="monotone"
            dataKey="power"
            name="Power (kW)"
            stroke={chartColors.forecast}
            strokeWidth={2}
            dot={false}
            connectNulls
            isAnimationActive={false}
          />
          <Line
            yAxisId="temp"
            type="monotone"
            dataKey="setpoint"
            name="Setpoint (°C)"
            stroke={chartColors.p90}
            strokeWidth={1.5}
            strokeDasharray="4 4"
            dot={false}
            connectNulls
            isAnimationActive={false}
          />
          <Line
            yAxisId="temp"
            type="monotone"
            dataKey="ambient"
            name="Ambient (°C)"
            stroke="#E67E22"
            strokeWidth={1.2}
            strokeOpacity={0.75}
            dot={false}
            connectNulls
            isAnimationActive={false}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
