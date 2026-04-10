/**
 * Ordinal data palette — matches legacy Python dashboard's chart color wheel.
 * Used by Recharts Cell arrays and the Heatmap primitive's fallback color scale.
 *
 * Access via CSS var for theme-aware usage: `var(--data-0)`, etc.
 * These hard-coded values are for JavaScript code paths where a CSS var
 * string isn't directly consumable (e.g. a Recharts `fill` prop on a
 * `<Cell>` when the chart renderer can't resolve custom properties).
 */
export const DATA_PALETTE = [
  "#2980B9", // data-0 — primary
  "#27AE60", // data-1
  "#8E44AD", // data-2
  "#E67E22", // data-3
  "#E74C3C", // data-4
  "#1ABC9C", // data-5
  "#F39C12", // data-6
  "#D35400", // data-7
] as const;

export const HEAT_RAMP = [
  "#F1F5F9", // empty
  "#D6EAF8", // low
  "#5DADE2", // mid
  "#1A5490", // high
] as const;

/**
 * Assign a color index to a series, cycling through the ordinal palette.
 */
export function paletteColor(index: number): string {
  return DATA_PALETTE[index % DATA_PALETTE.length];
}

/**
 * Map a numeric value to a 4-bucket quantile color from the heat ramp.
 * Returns the `empty` color if `max === 0`.
 */
export function heatColor(value: number, max: number): string {
  if (max <= 0) return HEAT_RAMP[0];
  const ratio = value / max;
  if (ratio === 0) return HEAT_RAMP[0];
  if (ratio < 0.33) return HEAT_RAMP[1];
  if (ratio < 0.66) return HEAT_RAMP[2];
  return HEAT_RAMP[3];
}
