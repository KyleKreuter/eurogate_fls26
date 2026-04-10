import type { ReactNode } from "react";
import { ChevronUp, ChevronDown, ChevronsUpDown } from "lucide-react";
import { cn } from "@/lib/utils";

export interface DataTableColumn<T> {
  key: string;
  label: ReactNode;
  align?: "left" | "right" | "center";
  sortable?: boolean;
  /** Render a raw number/timestamp/uuid in monospace */
  mono?: boolean;
  render: (row: T) => ReactNode;
}

interface DataTableProps<T> {
  columns: DataTableColumn<T>[];
  rows: T[];
  rowKey: (row: T) => string;
  sortKey?: string;
  sortDir?: "ASC" | "DESC";
  onSortChange?: (key: string) => void;
  selectedKey?: string;
  onRowClick?: (row: T) => void;
  isLoading?: boolean;
  skeletonRows?: number;
  emptyState?: ReactNode;
  className?: string;
}

/**
 * Generic sortable table built on `.premium-table` CSS. Stateless — the
 * caller owns sort state, selection, and row clicks.
 */
export function DataTable<T>({
  columns,
  rows,
  rowKey,
  sortKey,
  sortDir,
  onSortChange,
  selectedKey,
  onRowClick,
  isLoading,
  skeletonRows = 12,
  emptyState,
  className,
}: DataTableProps<T>) {
  return (
    <div className={cn("overflow-auto", className)}>
      <table className="premium-table">
        <thead>
          <tr>
            {columns.map((col) => {
              const isActive = sortKey === col.key;
              const Chevron =
                !col.sortable
                  ? null
                  : !isActive
                    ? ChevronsUpDown
                    : sortDir === "ASC"
                      ? ChevronUp
                      : ChevronDown;
              return (
                <th
                  key={col.key}
                  data-sortable={col.sortable ? "true" : undefined}
                  onClick={
                    col.sortable && onSortChange
                      ? () => onSortChange(col.key)
                      : undefined
                  }
                  style={{
                    textAlign: col.align ?? "left",
                  }}
                >
                  <span className="inline-flex items-center gap-1">
                    {col.label}
                    {Chevron && (
                      <Chevron
                        size={14}
                        style={{
                          color: isActive
                            ? "var(--eg-navy)"
                            : "var(--ink-muted)",
                        }}
                      />
                    )}
                  </span>
                </th>
              );
            })}
          </tr>
        </thead>
        <tbody>
          {isLoading && rows.length === 0
            ? Array.from({ length: skeletonRows }).map((_, i) => (
                <tr key={`skeleton-${i}`}>
                  {columns.map((col) => (
                    <td key={col.key}>
                      <div
                        className="animate-pulse rounded"
                        style={{
                          height: "0.8rem",
                          background: "var(--rule-soft)",
                          width: `${40 + ((i * 7) % 40)}%`,
                        }}
                      />
                    </td>
                  ))}
                </tr>
              ))
            : rows.length === 0
              ? (
                <tr>
                  <td colSpan={columns.length} style={{ padding: "3rem 1rem" }}>
                    {emptyState ?? (
                      <div
                        style={{
                          textAlign: "center",
                          color: "var(--ink-secondary)",
                          fontSize: "0.85rem",
                        }}
                      >
                        No results
                      </div>
                    )}
                  </td>
                </tr>
              )
              : rows.map((row) => {
                  const key = rowKey(row);
                  const isSelected = selectedKey === key;
                  return (
                    <tr
                      key={key}
                      data-selected={isSelected ? "true" : undefined}
                      onClick={onRowClick ? () => onRowClick(row) : undefined}
                      style={{
                        cursor: onRowClick ? "pointer" : undefined,
                      }}
                    >
                      {columns.map((col) => (
                        <td
                          key={col.key}
                          data-variant={col.mono ? "mono" : undefined}
                          style={{ textAlign: col.align ?? "left" }}
                        >
                          {col.render(row)}
                        </td>
                      ))}
                    </tr>
                  );
                })}
        </tbody>
      </table>
    </div>
  );
}
