import { Navigation } from "@/components/Navigation";
import { LoadingSpinner } from "@/components/LoadingSpinner";

/**
 * Placeholder while Phase 8/9 build the HamburgDashboard tab shell that
 * replaces this page. Kept to satisfy the existing `/dashboard/hamburg`
 * route — will be removed in Phase 9.
 */
export function Dashboard() {
  return (
    <div className="min-h-screen flex flex-col">
      <Navigation backTo="/" centerLabel="Hamburg Dashboard" />
      <div className="flex-1 flex items-center justify-center">
        <LoadingSpinner label="Bootstrapping" />
      </div>
    </div>
  );
}
