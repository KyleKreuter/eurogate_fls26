import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

interface StatPillProps {
  label: ReactNode;
  value: ReactNode;
  highlight?: boolean;
  className?: string;
}

/**
 * Label + value chip. `highlight` variant for the hero stat in a row
 * (e.g. the primary Container Inspector stat).
 */
export function StatPill({ label, value, highlight, className }: StatPillProps) {
  return (
    <div className={cn("stat-pill", highlight && "stat-pill--highlight", className)}>
      <div className="stat-pill__label">{label}</div>
      <div className="stat-pill__value">{value}</div>
    </div>
  );
}
