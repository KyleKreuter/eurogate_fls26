import Papa from "papaparse";
import csvUrl from "@/shared/dashboard_data.csv?url";

export interface DataRow {
  timestamp: Date;
  predPower: number;
  predP90: number;
  historyLastYear: number;
}

interface RawRow {
  timestamp_utc: string;
  pred_power_kw: string;
  pred_p90_kw: string;
  history_lastyear_kw: string;
}

export async function loadDashboardData(): Promise<DataRow[]> {
  const response = await fetch(csvUrl);
  const text = await response.text();

  const { data } = Papa.parse<RawRow>(text, {
    header: true,
    skipEmptyLines: true,
  });

  return data.map((row) => ({
    timestamp: new Date(row.timestamp_utc),
    predPower: parseFloat(row.pred_power_kw),
    predP90: parseFloat(row.pred_p90_kw),
    historyLastYear: parseFloat(row.history_lastyear_kw),
  }));
}

export function groupByDay(rows: DataRow[]): Map<string, DataRow[]> {
  const groups = new Map<string, DataRow[]>();
  for (const row of rows) {
    const key = row.timestamp.toISOString().slice(0, 10);
    const arr = groups.get(key) ?? [];
    arr.push(row);
    groups.set(key, arr);
  }
  return groups;
}

export interface KpiStats {
  peak: number;
  avg: number;
  min: number;
  peakHistory: number;
  avgHistory: number;
  minHistory: number;
  deltaPeakPct: number;
  deltaAvgPct: number;
  deltaMinPct: number;
  p90Max: number;
}

export function computeKpis(rows: DataRow[]): KpiStats {
  const preds = rows.map((r) => r.predPower);
  const hists = rows.map((r) => r.historyLastYear);
  const p90s = rows.map((r) => r.predP90);

  const peak = Math.max(...preds);
  const avg = preds.reduce((a, b) => a + b, 0) / preds.length;
  const min = Math.min(...preds);

  const peakHistory = Math.max(...hists);
  const avgHistory = hists.reduce((a, b) => a + b, 0) / hists.length;
  const minHistory = Math.min(...hists);

  return {
    peak,
    avg,
    min,
    peakHistory,
    avgHistory,
    minHistory,
    deltaPeakPct: ((peak - peakHistory) / peakHistory) * 100,
    deltaAvgPct: ((avg - avgHistory) / avgHistory) * 100,
    deltaMinPct: ((min - minHistory) / minHistory) * 100,
    p90Max: Math.max(...p90s),
  };
}