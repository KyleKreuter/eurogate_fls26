/**
 * Shared Recharts theme — imported by every chart in the app so the visual
 * language stays consistent. Values are the same tokens defined in
 * `src/index.css` (`var(--...)`), exposed here as plain strings so Recharts
 * components (which can't always resolve CSS vars in inline props) can use
 * them directly.
 *
 * When you change a color in `index.css`, update this file in lock-step.
 */

export const chartColors = {
  forecast: "#2980B9",           // var(--accent-primary)
  p90: "#C0392B",                // var(--danger)
  history: "#95A5A6",            // var(--neutral)
  gridStroke: "#E2E8F0",         // var(--rule-soft)
  axisStroke: "#E2E8F0",         // var(--rule-soft)
  tickFill: "#7F8C8D",           // var(--ink-secondary)
  labelFill: "#7F8C8D",          // var(--ink-secondary)
  tooltipBg: "#FFFFFF",          // var(--bg-card)
  tooltipBorder: "#0b0222",      // var(--eg-navy)
  areaFillAlpha: 0.12,
} as const;

export const chartTheme = {
  /** Forecast / P90 / History line config */
  line: {
    forecastStrokeWidth: 2.5,
    p90StrokeWidth: 2,
    historyStrokeWidth: 1.5,
    historyDash: "4 4",
  },

  /** Cartesian grid — dashed minor gridlines, no major ticks */
  grid: {
    stroke: chartColors.gridStroke,
    strokeDasharray: "2 4",
    vertical: false,
  },

  /** X-axis / Y-axis shared props */
  axis: {
    stroke: chartColors.axisStroke,
    tickLine: false,
    axisLine: false,
    tick: {
      fill: chartColors.tickFill,
      fontSize: 11,
      fontFamily:
        '"JetBrains Mono Variable", ui-monospace, "SF Mono", Menlo, monospace',
    },
  },

  /** Recharts <Tooltip contentStyle={...} /> — white pill w/ navy hairline */
  tooltip: {
    contentStyle: {
      background: chartColors.tooltipBg,
      border: `1px solid ${chartColors.tooltipBorder}`,
      borderRadius: "8px",
      boxShadow: "0 8px 24px rgba(11, 2, 34, 0.06)",
      padding: "0.75rem 1rem",
      fontFamily:
        '"JetBrains Mono Variable", ui-monospace, "SF Mono", Menlo, monospace',
      fontSize: "0.8rem",
    } as const,
    labelStyle: {
      fontFamily:
        '"Outfit Variable", -apple-system, system-ui, sans-serif',
      fontSize: "0.72rem",
      fontWeight: 600,
      letterSpacing: "0.08em",
      textTransform: "uppercase" as const,
      color: "#7F8C8D",
      marginBottom: "0.35rem",
    },
    itemStyle: {
      color: "#2C3E50",
      padding: "2px 0",
    },
  },

  /** Legend (top center, uppercase tracking) */
  legend: {
    wrapperStyle: {
      fontFamily:
        '"Outfit Variable", -apple-system, system-ui, sans-serif',
      fontSize: "0.72rem",
      textTransform: "uppercase" as const,
      letterSpacing: "0.08em",
      color: "#7F8C8D",
      paddingBottom: "0.75rem",
    },
  },
} as const;
