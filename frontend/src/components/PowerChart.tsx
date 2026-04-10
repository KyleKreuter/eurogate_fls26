import { useMemo, useCallback } from "react";
import {
  ComposedChart,
  Area,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import type { DataRow } from "@/lib/csv";

/* ─── Eurogate Chart Colors ─── */
const EG_BLUE = "#004494";
const EG_BLUE_BRIGHT = "#1a6fd4";
const EG_RED = "#e2001a";
const HISTORY_BLACK = "#1a1a1a";

interface ChartDatum {
  time: string;
  pred: number;
  p90: number;
  history: number;
}

function formatHour(d: Date): string {
  return d.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" });
}

function formatDayShort(d: Date): string {
  return d.toLocaleDateString("en-GB", { day: "2-digit", month: "short" });
}

/* ──── Tooltip ──── */

const SERIES_META: Record<string, { label: string; color: string }> = {
  pred: { label: "Forecast", color: EG_BLUE },
  p90: { label: "P90 Upper", color: EG_RED },
  history: { label: "Last Year", color: HISTORY_BLACK },
};

function CustomTooltip({
  active, payload, label,
}: {
  active?: boolean;
  payload?: Array<{ value: number; dataKey: string; color: string }>;
  label?: string;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: "var(--bg-white)",
      border: "1px solid var(--border-default)",
      borderRadius: 10,
      padding: "10px 14px",
      minWidth: 170,
      boxShadow: "0 4px 12px rgba(0,0,0,0.08)",
    }}>
      <p className="data-value text-[10px] mb-2 pb-1.5" style={{
        color: "var(--text-muted)",
        borderBottom: "1px solid rgba(0,20,60,0.06)",
      }}>
        {label}
      </p>
      {payload.map((entry) => {
        const meta = SERIES_META[entry.dataKey];
        if (!meta) return null;
        return (
          <div key={entry.dataKey} className="flex justify-between items-center gap-6 py-[3px]">
            <span className="flex items-center gap-2 text-[11px]" style={{ color: "var(--text-secondary)" }}>
              <span className="inline-block w-1.5 h-1.5 rounded-full" style={{ background: meta.color }} />
              {meta.label}
            </span>
            <span className="data-value text-[11px] font-semibold" style={{ color: "var(--text-primary)" }}>
              {Math.round(entry.value).toLocaleString("en-US")} kW
            </span>
          </div>
        );
      })}
    </div>
  );
}

/* ──── Overview Chart ──── */

interface OverviewChartProps {
  data: DataRow[];
  selectedDay: string | null;
  onSelectDay: (day: string) => void;
}

export function OverviewChart({ data, onSelectDay }: OverviewChartProps) {
  const chartData: ChartDatum[] = useMemo(
    () => data.map((r) => ({
      time: `${formatDayShort(r.timestamp)} ${formatHour(r.timestamp)}`,
      pred: Math.round(r.predPower * 10) / 10,
      p90: Math.round(r.predP90 * 10) / 10,
      history: Math.round(r.historyLastYear * 10) / 10,
    })),
    [data]
  );

  const dayBoundaries = useMemo(() => {
    const bounds: number[] = [];
    let lastDay = "";
    data.forEach((r, i) => {
      const day = r.timestamp.toISOString().slice(0, 10);
      if (day !== lastDay) { bounds.push(i); lastDay = day; }
    });
    return bounds;
  }, [data]);

  const handleClick = useCallback(
    (state: { activeLabel?: string | number | undefined } | null) => {
      if (!state?.activeLabel) return;
      const label = String(state.activeLabel);
      const match = data.find(
        (r) => `${formatDayShort(r.timestamp)} ${formatHour(r.timestamp)}` === label
      );
      if (match) onSelectDay(match.timestamp.toISOString().slice(0, 10));
    },
    [data, onSelectDay]
  );

  return (
    <ResponsiveContainer width="100%" height={200}>
      <ComposedChart data={chartData} margin={{ top: 8, right: 12, bottom: 0, left: 0 }} onClick={handleClick} style={{ cursor: "crosshair" }}>
        <defs>
          <linearGradient id="p90Grad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={EG_RED} stopOpacity={0.1} />
            <stop offset="100%" stopColor={EG_RED} stopOpacity={0.01} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 6" vertical={false} stroke="rgba(0,20,60,0.05)" />
        <XAxis dataKey="time" tick={{ fontSize: 9, fill: "#8a85a0" }} interval={23} tickLine={false} axisLine={{ stroke: "rgba(0,20,60,0.08)" }} />
        <YAxis tick={{ fontSize: 9, fill: "#8a85a0" }} tickLine={false} axisLine={false} domain={["auto", "auto"]} width={44} />
        <Tooltip content={<CustomTooltip />} />
        {dayBoundaries.slice(1).map((idx) => (
          <ReferenceLine key={idx} x={chartData[idx]?.time} stroke="rgba(0,20,60,0.06)" strokeDasharray="2 6" />
        ))}
        <Area type="monotone" dataKey="p90" stroke={`${EG_RED}60`} strokeWidth={1} fill="url(#p90Grad)" fillOpacity={1} dot={false} activeDot={false} isAnimationActive={false} />
        <Line type="monotone" dataKey="history" stroke={HISTORY_BLACK} strokeWidth={1.2} strokeDasharray="4 6" strokeOpacity={0.4} dot={false} activeDot={{ r: 2, fill: HISTORY_BLACK }} isAnimationActive={false} />
        <Line type="monotone" dataKey="pred" stroke={EG_BLUE} strokeWidth={2} dot={false} activeDot={{ r: 4, fill: "#fff", stroke: EG_BLUE, strokeWidth: 2 }} isAnimationActive={false} />
      </ComposedChart>
    </ResponsiveContainer>
  );
}

/* ──── Detail Chart ──── */

interface DetailChartProps {
  data: DataRow[];
  dayLabel: string;
}

export function DetailChart({ data, dayLabel }: DetailChartProps) {
  const chartData: ChartDatum[] = useMemo(
    () => data.map((r) => ({
      time: formatHour(r.timestamp),
      pred: Math.round(r.predPower * 10) / 10,
      p90: Math.round(r.predP90 * 10) / 10,
      history: Math.round(r.historyLastYear * 10) / 10,
    })),
    [data]
  );

  const peakIdx = useMemo(() => {
    let maxVal = -Infinity, idx = 0;
    data.forEach((r, i) => { if (r.predPower > maxVal) { maxVal = r.predPower; idx = i; } });
    return idx;
  }, [data]);

  return (
    <div>
      <div className="flex items-center justify-between mb-5">
        <div>
          <h3 className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>{dayLabel}</h3>
          <p className="text-[11px] mt-0.5" style={{ color: "var(--text-muted)" }}>Hourly detail — 24h view</p>
        </div>
        <div className="flex items-center gap-5 text-[11px]">
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-[2px] rounded-full" style={{ background: EG_BLUE }} />
            <span style={{ color: "var(--text-secondary)" }}>Forecast</span>
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-sm" style={{ background: `${EG_RED}14`, border: `1px solid ${EG_RED}60` }} />
            <span style={{ color: "var(--text-secondary)" }}>P90 Band</span>
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-[2px] rounded-full" style={{ background: HISTORY_BLACK, opacity: 0.4 }} />
            <span style={{ color: "var(--text-secondary)" }}>Last Year</span>
          </span>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={320}>
        <ComposedChart data={chartData} margin={{ top: 12, right: 12, bottom: 0, left: 0 }}>
          <defs>
            <linearGradient id="p90DetailGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={EG_RED} stopOpacity={0.12} />
              <stop offset="100%" stopColor={EG_RED} stopOpacity={0.01} />
            </linearGradient>
            <linearGradient id="forecastGlow" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={EG_BLUE} stopOpacity={0.08} />
              <stop offset="100%" stopColor={EG_BLUE} stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 6" vertical={false} stroke="rgba(0,20,60,0.05)" />
          <XAxis dataKey="time" tick={{ fontSize: 10, fill: "#8a85a0" }} interval={1} tickLine={false} axisLine={{ stroke: "rgba(0,20,60,0.08)" }} />
          <YAxis tick={{ fontSize: 10, fill: "#8a85a0" }} tickLine={false} axisLine={false} domain={["dataMin - 30", "dataMax + 30"]} width={44} />
          <Tooltip content={<CustomTooltip />} />
          {chartData[peakIdx] && (
            <ReferenceLine x={chartData[peakIdx].time} stroke={`${EG_BLUE}40`} strokeDasharray="3 4"
              label={{ value: `Peak ${Math.round(data[peakIdx].predPower)} kW`, position: "top", fill: EG_BLUE, fontSize: 10, fontFamily: "var(--font-mono)" }}
            />
          )}
          <Area type="monotone" dataKey="p90" stroke={`${EG_RED}50`} strokeWidth={1} fill="url(#p90DetailGrad)" fillOpacity={1} dot={false} activeDot={false} />
          <Area type="monotone" dataKey="pred" stroke="none" fill="url(#forecastGlow)" fillOpacity={1} dot={false} activeDot={false} />
          <Line type="monotone" dataKey="history" stroke={HISTORY_BLACK} strokeWidth={1.5} strokeDasharray="5 6" strokeOpacity={0.35} dot={false} activeDot={{ r: 3, fill: HISTORY_BLACK, stroke: "#fff", strokeWidth: 1 }} />
          <Line type="monotone" dataKey="pred" stroke={EG_BLUE} strokeWidth={2.5} dot={false} activeDot={{ r: 5, fill: "#fff", stroke: EG_BLUE_BRIGHT, strokeWidth: 2 }} />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
