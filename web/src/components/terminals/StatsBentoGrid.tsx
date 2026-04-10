import { TERMINAL_STATS } from "@/data/terminals";

const BG_LEVELS = [
  "var(--td-surface-container-low)",
  "var(--td-surface-container)",
  "var(--td-surface-container-high)",
];

export function StatsBentoGrid() {
  return (
    <section className="mt-24 grid grid-cols-1 md:grid-cols-3 gap-[1px]">
      {TERMINAL_STATS.map((stat, i) => (
        <div
          key={stat.label}
          className="p-10"
          style={{
            background: BG_LEVELS[i],
            border: "1px solid rgba(255,255,255,0.06)",
          }}
        >
          <span
            className="text-4xl font-black text-white"
            style={{ fontFamily: "var(--font-display)" }}
          >
            {stat.value}
          </span>
          <p
            className="text-[0.6875rem] font-bold uppercase tracking-[0.2em] mt-2"
            style={{
              fontFamily: "var(--font-display)",
              color: "var(--td-on-surface-variant)",
            }}
          >
            {stat.label}
          </p>
        </div>
      ))}
    </section>
  );
}
