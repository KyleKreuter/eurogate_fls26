import type { ReactNode } from "react";
import { TrendingUp, TrendingDown } from "lucide-react";
import { formatDeltaPct, formatInt } from "@/lib/format";

interface KpiCardProps {
  label: string;
  /** Value already in target unit (e.g. kW) */
  value: number;
  unit?: string;
  /** Ratio, not percent — pass `0.042` for +4.2%. Optional. */
  deltaRatio?: number;
  /** Optional "vs prev. year" style note next to the delta chip */
  deltaLabel?: string;
  icon: ReactNode;
}

/**
 * Apple-style KPI card (large lucide icon top-left, no accent bar).
 * Value renders in JetBrains Mono, label in Outfit uppercase, delta
 * picks color from `--success`/`--danger` based on sign.
 */
export function KpiCard({
  label,
  value,
  unit,
  deltaRatio,
  deltaLabel,
  icon,
}: KpiCardProps) {
  const hasDelta = deltaRatio != null && !Number.isNaN(deltaRatio);
  const isUp = (deltaRatio ?? 0) >= 0;

  return (
    <div className="kpi-card">
      <div className="kpi-card__icon">{icon}</div>
      <div className="kpi-card__label">{label}</div>
      <div className="kpi-card__value">
        {formatInt(value)}
        {unit && <span className="kpi-card__unit">{unit}</span>}
      </div>
      {hasDelta && (
        <div
          className={`kpi-card__delta ${
            isUp ? "kpi-card__delta--pos" : "kpi-card__delta--neg"
          }`}
        >
          {isUp ? (
            <TrendingUp size={14} strokeWidth={2} aria-hidden="true" />
          ) : (
            <TrendingDown size={14} strokeWidth={2} aria-hidden="true" />
          )}
          <span>{formatDeltaPct(deltaRatio)}</span>
          {deltaLabel && (
            <span
              style={{
                fontFamily: "var(--font-sans)",
                fontSize: "0.72rem",
                fontWeight: 400,
                color: "var(--ink-muted)",
                textTransform: "uppercase",
                letterSpacing: "0.05em",
                marginLeft: "0.5rem",
              }}
            >
              {deltaLabel}
            </span>
          )}
        </div>
      )}
    </div>
  );
}
