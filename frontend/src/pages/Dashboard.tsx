import { useEffect, useState, useMemo } from "react";
import { Zap, BarChart3, TrendingDown, Activity } from "lucide-react";
import { loadDashboardData, groupByDay, computeKpis } from "@/lib/csv";
import type { DataRow, KpiStats } from "@/lib/csv";
import { KpiCard } from "@/components/KpiCard";
import { OverviewChart, DetailChart } from "@/components/PowerChart";
import { Navigation } from "@/components/Navigation";

export function Dashboard() {
  const [data, setData] = useState<DataRow[]>([]);
  const [selectedDay, setSelectedDay] = useState<string | null>(null);

  useEffect(() => {
    loadDashboardData().then((rows) => {
      setData(rows);
      if (rows.length > 0) setSelectedDay(rows[0].timestamp.toISOString().slice(0, 10));
    });
  }, []);

  const dayGroups = useMemo(() => groupByDay(data), [data]);
  const days = useMemo(() => Array.from(dayGroups.keys()), [data, dayGroups]);
  const kpis: KpiStats | null = useMemo(() => (data.length > 0 ? computeKpis(data) : null), [data]);
  const selectedDayData = useMemo(() => (selectedDay ? dayGroups.get(selectedDay) ?? [] : []), [selectedDay, dayGroups]);
  const selectedDayLabel = useMemo(() => {
    if (!selectedDay || selectedDayData.length === 0) return "";
    return selectedDayData[0].timestamp.toLocaleDateString("en-GB", {
      weekday: "long", day: "numeric", month: "long", year: "numeric",
    });
  }, [selectedDay, selectedDayData]);

  if (data.length === 0) {
    return (
      <div className="min-h-screen flex flex-col" style={{ background: "var(--bg-page)" }}>
        <Navigation variant="light" showBackLink centerLabel="Hamburg · CTH" />
        <div className="flex-1 flex items-center justify-center">
          <span className="text-sm" style={{ color: "var(--text-muted)" }}>Loading...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex flex-col page-enter">
      <Navigation variant="light" showBackLink centerLabel="Hamburg · CTH" />

      <div className="flex-1 px-5 md:px-10 py-8 mx-auto w-full" style={{ maxWidth: 1360 }}>
        {/* ── Page Header ── */}
        <header className="mb-8 flex flex-col md:flex-row md:items-end md:justify-between gap-3">
          <div>
            <h1 className="text-xl font-bold tracking-tight" style={{ color: "var(--eg-navy)" }}>
              Reefer Peak Load Forecast
            </h1>
            <p className="text-sm mt-1" style={{ color: "var(--text-muted)" }}>
              Container Terminal Hamburg — 10-day power demand forecast
            </p>
          </div>
          <div className="data-value text-[11px] px-4 py-2 rounded-full text-center md:text-left" style={{
            background: "rgba(0,68,148,0.05)",
            border: "1px solid rgba(0,68,148,0.1)",
            color: "var(--text-secondary)",
          }}>
            Jan 1 – 10, 2026
          </div>
        </header>

        {/* ── KPI Row ── */}
        {kpis && (
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-5 mb-8">
            <KpiCard
              label="Peak Load"
              value={kpis.peak}
              unit="kW"
              deltaPct={kpis.deltaPeakPct}
              icon={<Zap size={15} strokeWidth={2.5} />}
              accent="#e2001a"
            />
            <KpiCard
              label="Avg Load"
              value={kpis.avg}
              unit="kW"
              deltaPct={kpis.deltaAvgPct}
              icon={<BarChart3 size={15} strokeWidth={2.5} />}
              accent="#004494"
            />
            <KpiCard
              label="Min Load"
              value={kpis.min}
              unit="kW"
              deltaPct={kpis.deltaMinPct}
              icon={<TrendingDown size={15} strokeWidth={2.5} />}
              accent="#34d399"
            />
            <KpiCard
              label="P90 Peak"
              value={kpis.p90Max}
              unit="kW"
              deltaPct={((kpis.p90Max - kpis.peak) / kpis.peak) * 100}
              icon={<Activity size={15} strokeWidth={2.5} />}
              accent="#7c5cfc"
            />
          </div>
        )}

        {/* ── Overview Chart ── */}
        <div className="card-panel p-4 md:p-7 mb-6">
          <div className="flex items-center justify-between mb-5">
            <div>
              <h2 className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
                10-Day Overview
              </h2>
              <p className="text-[11px] mt-1" style={{ color: "var(--text-muted)" }}>
                Click a section to view hourly detail
              </p>
            </div>
            <div className="flex items-center gap-5 text-[11px]">
              <span className="flex items-center gap-1.5">
                <span className="w-3 h-[2px] rounded-full" style={{ background: "#004494" }} />
                <span style={{ color: "var(--text-secondary)" }}>Forecast</span>
              </span>
              <span className="flex items-center gap-1.5">
                <span className="w-3 h-3 rounded-sm" style={{ background: "rgba(226,0,26,0.08)", border: "1px solid rgba(226,0,26,0.4)" }} />
                <span style={{ color: "var(--text-secondary)" }}>P90 Band</span>
              </span>
              <span className="flex items-center gap-1.5">
                <span className="w-3 h-[2px] rounded-full" style={{ background: "#1a1a1a", opacity: 0.4 }} />
                <span style={{ color: "var(--text-secondary)" }}>Last Year</span>
              </span>
            </div>
          </div>

          <OverviewChart data={data} selectedDay={selectedDay} onSelectDay={setSelectedDay} />

          <div className="glow-divider mt-5 mb-5" />

          <div className="flex flex-wrap gap-2 justify-center">
            {days.map((day) => {
              const d = new Date(day + "T00:00:00Z");
              const label = d.toLocaleDateString("en-GB", { weekday: "short", day: "numeric", month: "short" });
              return (
                <button
                  key={day}
                  className="day-pill"
                  data-active={selectedDay === day}
                  onClick={() => setSelectedDay(day)}
                  aria-label={`Select ${label}`}
                  aria-pressed={selectedDay === day}
                >
                  {label}
                </button>
              );
            })}
          </div>
        </div>

        {/* ── Detail Chart ── */}
        {selectedDayData.length > 0 && (
          <div className="card-panel p-4 md:p-7">
            <DetailChart data={selectedDayData} dayLabel={selectedDayLabel} />
          </div>
        )}

        {/* ── Footer ── */}
        <footer className="mt-10 pb-6 text-center">
          <span className="data-value text-[11px]" style={{ color: "var(--text-muted)" }}>
            {data.length} data points — {days.length} days
          </span>
        </footer>
      </div>
    </div>
  );
}
