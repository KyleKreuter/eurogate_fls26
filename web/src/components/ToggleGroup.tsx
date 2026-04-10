import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

export interface ToggleItem<T extends string | number> {
  value: T;
  label: ReactNode;
}

interface ToggleGroupProps<T extends string | number> {
  items: ToggleItem<T>[];
  value: T;
  onChange: (next: T) => void;
  className?: string;
  "aria-label"?: string;
}

/**
 * Segmented-control. Used for the 24h/14d horizon toggle on the Load
 * Prediction chart and the hourly/daily toggle on the Container Inspector
 * timeline.
 */
export function ToggleGroup<T extends string | number>({
  items,
  value,
  onChange,
  className,
  "aria-label": ariaLabel,
}: ToggleGroupProps<T>) {
  return (
    <div
      role="radiogroup"
      aria-label={ariaLabel}
      className={cn("toggle-group", className)}
    >
      {items.map((item) => {
        const active = item.value === value;
        return (
          <button
            key={String(item.value)}
            role="radio"
            aria-checked={active}
            data-active={active ? "true" : "false"}
            className="toggle-btn"
            onClick={() => onChange(item.value)}
            type="button"
          >
            {item.label}
          </button>
        );
      })}
    </div>
  );
}
