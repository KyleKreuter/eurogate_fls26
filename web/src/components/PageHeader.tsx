import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

interface PageHeaderProps {
  title: ReactNode;
  subtitle?: ReactNode;
  actions?: ReactNode;
  className?: string;
}

/**
 * Signature page header — h1 + uppercase subtitle + 3px × 48px navy rank-bar.
 * The rank-bar is the app's small intentional visual fingerprint.
 */
export function PageHeader({ title, subtitle, actions, className }: PageHeaderProps) {
  return (
    <div className={cn("page-header flex items-start justify-between gap-6", className)}>
      <div>
        <h1 className="page-header__h1">{title}</h1>
        {subtitle && <div className="page-header__subtitle">{subtitle}</div>}
        <div className="rank-bar" aria-hidden="true" />
      </div>
      {actions && <div className="shrink-0">{actions}</div>}
    </div>
  );
}
