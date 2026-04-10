const FOOTER_LINKS = [
  { label: "Legal Notice", href: "#" },
  { label: "Privacy Policy", href: "#" },
  { label: "Compliance", href: "#" },
  { label: "Sustainability Report", href: "#" },
];

export function TerminalFooter() {
  return (
    <footer
      className="w-full flex flex-col md:flex-row justify-between items-center px-8 py-6 z-50"
      style={{
        background: "var(--td-surface-dim)",
        borderTop: "1px solid rgba(255,255,255,0.06)",
      }}
    >
      <span
        className="text-[0.6875rem] uppercase tracking-widest"
        style={{
          fontFamily: "var(--font-display)",
          color: "var(--td-on-surface-variant)",
        }}
      >
        &copy; 2024 EUROGATE GmbH &amp; Co. KGaA, Bremerhaven
      </span>

      <div className="flex gap-8 mt-4 md:mt-0">
        {FOOTER_LINKS.map((link) => (
          <a
            key={link.label}
            href={link.href}
            className="text-[0.6875rem] uppercase tracking-widest transition-all"
            style={{
              fontFamily: "var(--font-display)",
              color: "var(--td-on-surface-variant)",
            }}
            onMouseEnter={(e) => {
              (e.currentTarget as HTMLAnchorElement).style.color =
                "var(--td-secondary)";
            }}
            onMouseLeave={(e) => {
              (e.currentTarget as HTMLAnchorElement).style.color =
                "var(--td-on-surface-variant)";
            }}
          >
            {link.label}
          </a>
        ))}
      </div>
    </footer>
  );
}
