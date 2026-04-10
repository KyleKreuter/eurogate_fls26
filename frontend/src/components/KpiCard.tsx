import { TrendingUp, TrendingDown } from "lucide-react";

interface KpiCardProps {
  label: string;
  value: number;
  unit: string;
  deltaPct: number;
  icon: React.ReactNode;
  accent: string;
}

export function KpiCard({ label, value, unit, deltaPct, icon, accent }: KpiCardProps) {
  const isUp = deltaPct >= 0;

  return (
    <div
      className="kpi-card"
      style={{ "--kpi-accent": accent, cursor: "default" } as React.CSSProperties}
    >
      <div className="relative z-10">
        <div className="flex items-center justify-between mb-4">
          <span
            className="text-[10px] font-semibold tracking-[0.14em] uppercase"
            style={{ color: "var(--text-muted)" }}
          >
            {label}
          </span>
          <div
            className="flex items-center justify-center w-8 h-8 rounded-full"
            style={{ background: `${accent}10`, color: accent }}
          >
            {icon}
          </div>
        </div>

        <div className="flex items-baseline gap-2 mb-3">
          <span className="data-value text-3xl font-bold leading-none" style={{ color: "var(--eg-navy)" }}>
            {Math.round(value).toLocaleString("en-US")}
          </span>
          <span className="data-value text-sm font-medium" style={{ color: "var(--text-muted)" }}>
            {unit}
          </span>
        </div>

        <div className="flex items-center gap-2">
          <div
            className="inline-flex items-center gap-1 px-2 py-[3px] rounded-full text-[11px] font-medium"
            style={{
              background: isUp ? "var(--positive-bg)" : "var(--negative-bg)",
              color: isUp ? "var(--positive)" : "var(--negative)",
            }}
          >
            {isUp ? <TrendingUp size={11} strokeWidth={2.5} /> : <TrendingDown size={11} strokeWidth={2.5} />}
            <span className="data-value">{isUp ? "+" : ""}{deltaPct.toFixed(1)}%</span>
          </div>
          <span className="text-[10px]" style={{ color: "var(--text-muted)" }}>vs prev. year</span>
        </div>
      </div>
    </div>
  );
}
