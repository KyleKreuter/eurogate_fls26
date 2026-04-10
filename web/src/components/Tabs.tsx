import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

export interface TabItem<T extends string = string> {
  value: T;
  label: ReactNode;
}

interface TabsProps<T extends string> {
  items: TabItem<T>[];
  value: T;
  onChange: (next: T) => void;
  className?: string;
  "aria-label"?: string;
}

/**
 * Underline-style tab bar (3px navy slide with `ease-snap` overshoot).
 * Controlled — owner holds `value` and handles `onChange`. The tab panel
 * content is rendered by the caller (usually via conditional render driven
 * by URL search params).
 */
export function Tabs<T extends string>({
  items,
  value,
  onChange,
  className,
  "aria-label": ariaLabel,
}: TabsProps<T>) {
  return (
    <div
      role="tablist"
      aria-label={ariaLabel}
      className={cn("primary-tabs", className)}
    >
      {items.map((item) => {
        const active = item.value === value;
        return (
          <button
            key={item.value}
            role="tab"
            aria-selected={active}
            data-active={active ? "true" : "false"}
            className="tab-nav-btn"
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
