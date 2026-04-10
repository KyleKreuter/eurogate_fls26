import { heatColor } from "@/lib/palette";

interface HeatmapProps {
  /**
   * 2D grid of values. Outer array = rows, inner array = columns.
   * e.g. for Hour×DoW: rows[dow][hour], 7 × 24.
   */
  data: number[][];
  rowLabels: string[];
  colLabels: string[];
  cellSize?: number;
  cellGap?: number;
  /** Formatter for the native `<title>` tooltip. */
  tooltip?: (value: number, rowLabel: string, colLabel: string) => string;
  "aria-label"?: string;
}

/**
 * Custom SVG heatmap — used for both the Hour×DoW (24×7) and the Annual
 * Calendar (52×7) visualizations. Colors via the 4-bucket quantile ramp
 * (`--heat-empty`/`--heat-low`/`--heat-mid`/`--heat-high`).
 */
export function Heatmap({
  data,
  rowLabels,
  colLabels,
  cellSize = 14,
  cellGap = 2,
  tooltip,
  "aria-label": ariaLabel,
}: HeatmapProps) {
  if (data.length === 0 || data[0].length === 0) return null;

  const rows = data.length;
  const cols = data[0].length;

  // Find max across the grid for quantile bucketing
  let max = 0;
  for (const row of data) {
    for (const v of row) {
      if (v > max) max = v;
    }
  }

  // Layout math
  const labelColWidth = 32; // reserved for row labels on the left
  const labelRowHeight = 18; // reserved for col labels on top
  const step = cellSize + cellGap;
  const width = labelColWidth + cols * step;
  const height = labelRowHeight + rows * step;

  return (
    <svg
      role="img"
      aria-label={ariaLabel}
      width="100%"
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="xMidYMid meet"
      style={{ display: "block" }}
    >
      {/* Column labels (top) */}
      {colLabels.map((lbl, c) => {
        // Only render every other label for dense grids to avoid overlap
        if (cols > 20 && c % Math.ceil(cols / 12) !== 0) return null;
        return (
          <text
            key={`col-${c}`}
            x={labelColWidth + c * step + cellSize / 2}
            y={labelRowHeight - 6}
            textAnchor="middle"
            style={{
              fontSize: 9,
              fontFamily: 'var(--font-sans)',
              fill: 'var(--ink-muted)',
              textTransform: 'uppercase',
              letterSpacing: '0.04em',
            }}
          >
            {lbl}
          </text>
        );
      })}

      {/* Row labels (left) */}
      {rowLabels.map((lbl, r) => (
        <text
          key={`row-${r}`}
          x={labelColWidth - 6}
          y={labelRowHeight + r * step + cellSize / 2 + 3}
          textAnchor="end"
          style={{
            fontSize: 9,
            fontFamily: 'var(--font-sans)',
            fill: 'var(--ink-muted)',
            textTransform: 'uppercase',
            letterSpacing: '0.04em',
          }}
        >
          {lbl}
        </text>
      ))}

      {/* Cells */}
      {data.flatMap((row, r) =>
        row.map((v, c) => {
          const fill = heatColor(v, max);
          const x = labelColWidth + c * step;
          const y = labelRowHeight + r * step;
          return (
            <rect
              key={`${r}-${c}`}
              x={x}
              y={y}
              width={cellSize}
              height={cellSize}
              rx={2}
              ry={2}
              fill={fill}
            >
              {tooltip && (
                <title>
                  {tooltip(v, rowLabels[r] ?? "", colLabels[c] ?? "")}
                </title>
              )}
            </rect>
          );
        }),
      )}
    </svg>
  );
}
