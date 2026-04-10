import { ToggleGroup } from "@/components/ToggleGroup";
import type { Horizon } from "@/types/api";

interface HorizonToggleProps {
  value: Horizon;
  onChange: (next: Horizon) => void;
}

/** 24h vs 14d horizon selector for the Load Prediction chart. */
export function HorizonToggle({ value, onChange }: HorizonToggleProps) {
  return (
    <ToggleGroup<Horizon>
      aria-label="Forecast horizon"
      items={[
        { value: 24, label: "24 Hours" },
        { value: 336, label: "14 Days" },
      ]}
      value={value}
      onChange={onChange}
    />
  );
}
