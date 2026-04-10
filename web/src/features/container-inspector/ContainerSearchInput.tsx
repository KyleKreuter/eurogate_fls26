import { Search, X } from "lucide-react";

interface Props {
  value: string;
  onChange: (next: string) => void;
  placeholder?: string;
}

/**
 * Text input with leading lucide Search icon and trailing clear button.
 * Parent owns the value and can apply its own debounce (see useDebounce).
 */
export function ContainerSearchInput({ value, onChange, placeholder }: Props) {
  return (
    <div
      className="relative flex items-center"
      style={{
        background: "var(--bg-card)",
        border: "1px solid var(--rule-soft)",
        borderRadius: "var(--r-panel)",
      }}
    >
      <Search
        size={16}
        strokeWidth={2}
        style={{
          position: "absolute",
          left: "0.9rem",
          color: "var(--ink-muted)",
          pointerEvents: "none",
        }}
      />
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder ?? "Search by UUID"}
        className="w-full"
        style={{
          padding: "0.7rem 2.2rem 0.7rem 2.4rem",
          background: "transparent",
          border: 0,
          outline: "none",
          fontFamily: "var(--font-mono)",
          fontSize: "0.82rem",
          color: "var(--ink-primary)",
        }}
      />
      {value && (
        <button
          type="button"
          onClick={() => onChange("")}
          aria-label="Clear search"
          style={{
            position: "absolute",
            right: "0.6rem",
            background: "transparent",
            border: 0,
            padding: "0.35rem",
            cursor: "pointer",
            color: "var(--ink-muted)",
            display: "inline-flex",
            alignItems: "center",
          }}
        >
          <X size={14} strokeWidth={2} />
        </button>
      )}
    </div>
  );
}
