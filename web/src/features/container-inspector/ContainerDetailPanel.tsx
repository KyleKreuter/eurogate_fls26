import { useState } from "react";
import { Box, X } from "lucide-react";
import { GlassPanel } from "@/components/GlassPanel";
import { LoadingSpinner } from "@/components/LoadingSpinner";
import { EmptyState } from "@/components/EmptyState";
import { ErrorState } from "@/components/ErrorState";
import { useContainerDetail } from "./useContainerDetail";
import { ContainerStatsBanner } from "./ContainerStatsBanner";
import { ContainerTimelineChart } from "./ContainerTimelineChart";
import {
  TimelineGranularityToggle,
  type Granularity,
} from "./TimelineGranularityToggle";

interface Props {
  uuid: string | null;
  onClose: () => void;
}

export function ContainerDetailPanel({ uuid, onClose }: Props) {
  const [granularity, setGranularity] = useState<Granularity>("daily");
  const { data, isLoading, isError, error, refetch } = useContainerDetail(uuid);

  if (!uuid) {
    return (
      <GlassPanel>
        <EmptyState
          icon={<Box size={48} strokeWidth={1.5} />}
          title="Select a container to view details"
          description="Click any row in the table to see visit stats and a timeline of its power consumption."
        />
      </GlassPanel>
    );
  }

  return (
    <GlassPanel>
      {/* Header */}
      <div className="flex items-start justify-between gap-4 mb-4">
        <div className="flex items-center gap-3">
          <Box
            size={24}
            strokeWidth={1.8}
            style={{ color: "var(--accent-primary)" }}
          />
          <div>
            <div
              style={{
                fontSize: "0.72rem",
                textTransform: "uppercase",
                letterSpacing: "0.08em",
                fontWeight: 600,
                color: "var(--ink-secondary)",
              }}
            >
              Container
            </div>
            <div
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: "0.95rem",
                fontWeight: 600,
                color: "var(--ink-primary)",
              }}
            >
              {uuid}
            </div>
          </div>
        </div>
        <button
          type="button"
          onClick={onClose}
          aria-label="Close detail panel"
          style={{
            background: "transparent",
            border: 0,
            padding: "0.4rem",
            cursor: "pointer",
            color: "var(--ink-secondary)",
          }}
        >
          <X size={18} strokeWidth={2} />
        </button>
      </div>

      {isLoading ? (
        <LoadingSpinner label="Loading container data" />
      ) : isError || !data ? (
        <ErrorState
          title="Failed to load container"
          description={(error as Error)?.message}
          onRetry={() => refetch()}
        />
      ) : (
        <>
          <ContainerStatsBanner detail={data} />
          <div className="mt-5 flex items-center justify-between">
            <div
              style={{
                fontSize: "0.78rem",
                textTransform: "uppercase",
                letterSpacing: "0.06em",
                color: "var(--ink-secondary)",
                fontWeight: 600,
              }}
            >
              Timeline
            </div>
            <TimelineGranularityToggle
              value={granularity}
              onChange={setGranularity}
            />
          </div>
          <div className="mt-3">
            <ContainerTimelineChart
              timeline={data.timeline}
              granularity={granularity}
            />
          </div>
        </>
      )}
    </GlassPanel>
  );
}
