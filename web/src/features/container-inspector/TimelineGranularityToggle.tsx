import { ToggleGroup } from "@/components/ToggleGroup";

export type Granularity = "hourly" | "daily";

interface Props {
  value: Granularity;
  onChange: (next: Granularity) => void;
}

export function TimelineGranularityToggle({ value, onChange }: Props) {
  return (
    <ToggleGroup<Granularity>
      aria-label="Timeline granularity"
      items={[
        { value: "hourly", label: "Hourly" },
        { value: "daily", label: "Daily" },
      ]}
      value={value}
      onChange={onChange}
    />
  );
}
