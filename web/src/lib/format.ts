/**
 * Locale-agnostic, tabular-friendly formatters for the dashboard.
 *
 * All functions are pure and can be called in render paths without side
 * effects. Timestamps are treated as UTC strings (matches backend).
 */

const integerFmt = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });
const oneDecimalFmt = new Intl.NumberFormat("en-US", {
  minimumFractionDigits: 1,
  maximumFractionDigits: 1,
});
const signedPctFmt = new Intl.NumberFormat("en-US", {
  minimumFractionDigits: 1,
  maximumFractionDigits: 1,
  signDisplay: "exceptZero",
});

/** Integer with thousand separators. `formatInt(1043.7) -> "1,044"`. */
export function formatInt(n: number | null | undefined): string {
  if (n == null || Number.isNaN(n)) return "—";
  return integerFmt.format(Math.round(n));
}

/** One decimal place, no thousand separators. `formatKw(892.3) -> "892.3"`. */
export function formatKw(n: number | null | undefined): string {
  if (n == null || Number.isNaN(n)) return "—";
  return oneDecimalFmt.format(n);
}

/** Signed percentage with one decimal. `formatDeltaPct(0.042) -> "+4.2%"`. */
export function formatDeltaPct(ratio: number | null | undefined): string {
  if (ratio == null || Number.isNaN(ratio)) return "—";
  return `${signedPctFmt.format(ratio * 100)}%`;
}

/** ISO UTC timestamp → `Jan 03, 14:00`. */
export function formatDateTime(iso: string | null | undefined): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const mon = d.toLocaleDateString("en-US", { month: "short", timeZone: "UTC" });
  const day = d.getUTCDate().toString().padStart(2, "0");
  const hh = d.getUTCHours().toString().padStart(2, "0");
  const mm = d.getUTCMinutes().toString().padStart(2, "0");
  return `${mon} ${day}, ${hh}:${mm}`;
}

/** ISO UTC timestamp → `Jan 03`. */
export function formatDateShort(iso: string | null | undefined): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const mon = d.toLocaleDateString("en-US", { month: "short", timeZone: "UTC" });
  const day = d.getUTCDate().toString().padStart(2, "0");
  return `${mon} ${day}`;
}

/** ISO UTC timestamp → `14:00`. */
export function formatTime(iso: string | null | undefined): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const hh = d.getUTCHours().toString().padStart(2, "0");
  const mm = d.getUTCMinutes().toString().padStart(2, "0");
  return `${hh}:${mm}`;
}

/** Hours → compact string. `formatHours(512) -> "512h (21d)"`. */
export function formatHours(hours: number | null | undefined): string {
  if (hours == null || Number.isNaN(hours)) return "—";
  const h = Math.round(hours);
  if (h < 48) return `${h}h`;
  const days = (h / 24).toFixed(1).replace(/\.0$/, "");
  return `${h}h (${days}d)`;
}

/** Truncate a UUID to the first N hex chunks. */
export function formatUuidShort(uuid: string, chunks = 2): string {
  return uuid.split("-").slice(0, chunks).join("-");
}
