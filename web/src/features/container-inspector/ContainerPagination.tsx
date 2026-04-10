import { ChevronLeft, ChevronRight } from "lucide-react";
import { formatInt } from "@/lib/format";

interface Props {
  total: number;
  limit: number;
  offset: number;
  onChange: (nextOffset: number) => void;
}

export function ContainerPagination({ total, limit, offset, onChange }: Props) {
  const page = Math.floor(offset / limit) + 1;
  const pageCount = Math.max(1, Math.ceil(total / limit));
  const canPrev = offset > 0;
  const canNext = offset + limit < total;

  const btn = (
    label: string,
    disabled: boolean,
    handler: () => void,
    icon: React.ReactNode,
  ) => (
    <button
      type="button"
      onClick={handler}
      disabled={disabled}
      aria-label={label}
      style={{
        display: "inline-flex",
        alignItems: "center",
        justifyContent: "center",
        width: 32,
        height: 32,
        background: "var(--bg-card)",
        border: "1px solid var(--rule-soft)",
        borderRadius: "var(--r-btn)",
        cursor: disabled ? "default" : "pointer",
        color: disabled ? "var(--ink-muted)" : "var(--eg-navy)",
        opacity: disabled ? 0.5 : 1,
      }}
    >
      {icon}
    </button>
  );

  return (
    <div className="flex items-center justify-between gap-3 px-1 py-2">
      <span
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: "0.78rem",
          color: "var(--ink-secondary)",
          fontVariantNumeric: "tabular-nums",
        }}
      >
        Page {page} of {pageCount} · {formatInt(total)} containers
      </span>
      <div className="flex items-center gap-2">
        {btn("Previous page", !canPrev, () => onChange(Math.max(0, offset - limit)), <ChevronLeft size={16} />)}
        {btn("Next page", !canNext, () => onChange(offset + limit), <ChevronRight size={16} />)}
      </div>
    </div>
  );
}
