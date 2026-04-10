import { Globe, Search } from "lucide-react";
import { EurogateLogo } from "@/components/EurogateLogo";

const NAV_LINKS = [
  { label: "Terminals", href: "#", active: true },
  { label: "Services", href: "#", active: false },
  { label: "Investor Relations", href: "#", active: false },
  { label: "News", href: "#", active: false },
];

export function TerminalNavBar() {
  return (
    <nav
      className="fixed top-0 w-full z-50 flex justify-between items-center px-8 h-20"
      style={{
        background: "var(--td-surface-container-low)",
        borderBottom: "1px solid rgba(255,255,255,0.06)",
      }}
    >
      {/* Left: Logo + Nav Links */}
      <div className="flex items-center gap-12">
        <EurogateLogo height={28} className="brightness-0 invert" />

        <div className="hidden md:flex gap-8">
          {NAV_LINKS.map((link) => (
            <a
              key={link.label}
              href={link.href}
              className="uppercase tracking-[0.05em] text-[0.6875rem] font-bold transition-colors"
              style={{
                fontFamily: "var(--font-display)",
                color: link.active
                  ? "var(--td-secondary)"
                  : "var(--td-on-surface-variant)",
                borderBottom: link.active
                  ? "2px solid var(--td-secondary)"
                  : "2px solid transparent",
                paddingBottom: "4px",
              }}
              onMouseEnter={(e) => {
                if (!link.active) {
                  (e.currentTarget as HTMLAnchorElement).style.color = "#fff";
                }
              }}
              onMouseLeave={(e) => {
                if (!link.active) {
                  (e.currentTarget as HTMLAnchorElement).style.color =
                    "var(--td-on-surface-variant)";
                }
              }}
            >
              {link.label}
            </a>
          ))}
        </div>
      </div>

      {/* Right: Icons */}
      <div className="flex items-center gap-6">
        <Globe
          size={20}
          className="cursor-pointer transition-colors"
          style={{ color: "var(--td-on-surface-variant)" }}
          onMouseEnter={(e) => {
            (e.currentTarget as SVGSVGElement).style.color = "#fff";
          }}
          onMouseLeave={(e) => {
            (e.currentTarget as SVGSVGElement).style.color =
              "var(--td-on-surface-variant)";
          }}
        />
        <Search
          size={20}
          className="cursor-pointer transition-colors"
          style={{ color: "var(--td-on-surface-variant)" }}
          onMouseEnter={(e) => {
            (e.currentTarget as SVGSVGElement).style.color = "#fff";
          }}
          onMouseLeave={(e) => {
            (e.currentTarget as SVGSVGElement).style.color =
              "var(--td-on-surface-variant)";
          }}
        />
      </div>
    </nav>
  );
}
