import { useMemo } from "react";
import {
  ComposedChart,
  Area,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import type { ForecastPoint } from "@/types/api";
import { chartColors, chartTheme } from "@/lib/chart-theme";
import { formatTime, formatDateShort } from "@/lib/format";

interface PredictionChartProps {
  points: ForecastPoint[];
  /** Use compact hour-of-day ticks vs. date ticks. */
  compactLabels: boolean;
}

/**
 * Recharts ComposedChart with 3 series:
 *   • pred_power_kw     (forecast)  — solid blue line + gradient area fill
 *   • pred_p90_kw       (P90)       — solid red line
 *   • history_lastyear  (history)   — dashed gray line
 */
export function PredictionChart({ points, compactLabels }: PredictionChartProps) {
  const data = useMemo(
    () =>
      points.map((p) => ({
        label: compactLabels ? formatTime(p.timestamp_utc) : formatDateShort(p.timestamp_utc),
        forecast: p.pred_power_kw,
        p90: p.pred_p90_kw,
        history: p.history_lastyear_kw,
        _time: p.timestamp_utc,
      })),
    [points, compactLabels],
  );

  // Custom tick frequency so labels don't overlap
  const tickInterval = useMemo(() => {
    if (data.length <= 24) return Math.max(1, Math.floor(data.length / 6));
    return Math.max(1, Math.floor(data.length / 10));
  }, [data.length]);

  return (
    <div style={{ width: "100%", height: 420 }}>
      <ResponsiveContainer>
        <ComposedChart
          data={data}
          margin={{ top: 20, right: 20, left: -10, bottom: 10 }}
        >
          <defs>
            <linearGradient id="forecastFill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={chartColors.forecast} stopOpacity={0.25} />
              <stop offset="100%" stopColor={chartColors.forecast} stopOpacity={0.02} />
            </linearGradient>
          </defs>

          <CartesianGrid {...chartTheme.grid} />
          <XAxis
            dataKey="label"
            {...chartTheme.axis}
            interval={tickInterval}
            minTickGap={20}
          />
          <YAxis
            {...chartTheme.axis}
            domain={["auto", "auto"]}
            tickFormatter={(v) => `${Math.round(v as number)}`}
            label={{
              value: "kW",
              angle: -90,
              position: "insideLeft",
              style: {
                fontSize: 10,
                fontFamily: "var(--font-sans)",
                fill: "var(--ink-muted)",
                textTransform: "uppercase",
                letterSpacing: "0.08em",
              },
            }}
          />
          <Tooltip
            contentStyle={chartTheme.tooltip.contentStyle}
            labelStyle={chartTheme.tooltip.labelStyle}
            itemStyle={chartTheme.tooltip.itemStyle}
            formatter={(value, name) => {
              const num = typeof value === "number" ? value : Number(value);
              const key = String(name);
              const label =
                key === "forecast"
                  ? "Forecast"
                  : key === "p90"
                    ? "P90"
                    : key === "history"
                      ? "History"
                      : key;
              const display = Number.isFinite(num)
                ? `${Math.round(num).toLocaleString("en-US")} kW`
                : "—";
              return [display, label];
            }}
          />
          <Legend
            wrapperStyle={chartTheme.legend.wrapperStyle}
            iconType="plainline"
          />

          {/* History — dashed gray */}
          <Line
            type="monotone"
            dataKey="history"
            name="History"
            stroke={chartColors.history}
            strokeWidth={chartTheme.line.historyStrokeWidth}
            strokeDasharray={chartTheme.line.historyDash}
            dot={false}
            isAnimationActive={false}
          />

          {/* Forecast — blue area + solid line */}
          <Area
            type="monotone"
            dataKey="forecast"
            name="Forecast"
            stroke={chartColors.forecast}
            strokeWidth={chartTheme.line.forecastStrokeWidth}
            fill="url(#forecastFill)"
            isAnimationActive={false}
          />

          {/* P90 — red solid line */}
          <Line
            type="monotone"
            dataKey="p90"
            name="P90"
            stroke={chartColors.p90}
            strokeWidth={chartTheme.line.p90StrokeWidth}
            dot={false}
            isAnimationActive={false}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
