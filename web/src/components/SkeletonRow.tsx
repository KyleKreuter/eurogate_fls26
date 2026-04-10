interface SkeletonRowProps {
  height?: number;
  widthPct?: number;
  className?: string;
}

/**
 * Single animated placeholder bar used inside `animate-pulse` skeleton
 * layouts. Useful when you want to compose skeletons outside of DataTable.
 */
export function SkeletonRow({ height = 10, widthPct = 100, className }: SkeletonRowProps) {
  return (
    <div
      className={`animate-pulse rounded ${className ?? ""}`.trim()}
      style={{
        height: `${height}px`,
        width: `${widthPct}%`,
        background: "var(--rule-soft)",
      }}
    />
  );
}
