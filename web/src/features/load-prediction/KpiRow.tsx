import { Zap, Activity, TrendingDown, Gauge } from "lucide-react";
import { KpiCard } from "@/components/KpiCard";
import type { ForecastPoint } from "@/types/api";

interface KpiRowProps {
  points: ForecastPoint[];
}

interface KpiStats {
  peak: number;
  avg: number;
  min: number;
  p90: number;
  peakHistory: number;
  avgHistory: number;
  minHistory: number;
}

function computeKpis(points: ForecastPoint[]): KpiStats | null {
  if (points.length === 0) return null;
  let peak = -Infinity;
  let min = Infinity;
  let sum = 0;
  let p90 = -Infinity;
  let peakH = -Infinity;
  let minH = Infinity;
  let sumH = 0;
  for (const p of points) {
    if (p.pred_power_kw > peak) peak = p.pred_power_kw;
    if (p.pred_power_kw < min) min = p.pred_power_kw;
    sum += p.pred_power_kw;
    if (p.pred_p90_kw > p90) p90 = p.pred_p90_kw;
    if (p.history_lastyear_kw > peakH) peakH = p.history_lastyear_kw;
    if (p.history_lastyear_kw < minH) minH = p.history_lastyear_kw;
    sumH += p.history_lastyear_kw;
  }
  const n = points.length;
  return {
    peak,
    avg: sum / n,
    min,
    p90,
    peakHistory: peakH,
    avgHistory: sumH / n,
    minHistory: minH,
  };
}

function ratio(current: number, history: number): number | undefined {
  if (!history || !Number.isFinite(history)) return undefined;
  return (current - history) / history;
}

/** 4 KPI cards showing Peak / Avg / Min / P90 vs. history. */
export function KpiRow({ points }: KpiRowProps) {
  const k = computeKpis(points);
  if (!k) return null;

  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-5">
      <KpiCard
        label="Peak Load"
        value={k.peak}
        unit="kW"
        deltaRatio={ratio(k.peak, k.peakHistory)}
        deltaLabel="vs prev. year"
        icon={<Zap size={32} strokeWidth={2} />}
      />
      <KpiCard
        label="Avg Load"
        value={k.avg}
        unit="kW"
        deltaRatio={ratio(k.avg, k.avgHistory)}
        deltaLabel="vs prev. year"
        icon={<Activity size={32} strokeWidth={2} />}
      />
      <KpiCard
        label="Min Load"
        value={k.min}
        unit="kW"
        deltaRatio={ratio(k.min, k.minHistory)}
        deltaLabel="vs prev. year"
        icon={<TrendingDown size={32} strokeWidth={2} />}
      />
      <KpiCard
        label="P90 Peak"
        value={k.p90}
        unit="kW"
        deltaRatio={ratio(k.p90, k.peak)}
        deltaLabel="vs forecast peak"
        icon={<Gauge size={32} strokeWidth={2} />}
      />
    </div>
  );
}
