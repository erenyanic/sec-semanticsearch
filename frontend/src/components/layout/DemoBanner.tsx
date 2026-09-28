"use client";

import { useStatus } from "@/hooks/useStatus";

/**
 * Persistent banner shown when the API is running in demo mode.
 *
 * Fetches the status via `useStatus()` (cache-shared with Dashboard
 * and other pages via React Query's `["status"]` key). Renders a
 * yellow/amber banner at the top of the page when `demo_mode` is true.
 *
 * Non-blocking: always renders children (the app layout) regardless
 * of the status fetch result. The banner is purely informational.
 */
export function DemoBanner({ children }: { children: React.ReactNode }) {
  const { data } = useStatus();

  return (
    <>
      {data?.demo_mode && (
        <div
          role="status"
          className="border-b border-warn/30 bg-warn/10 px-4 py-2 text-center text-sm font-medium text-warn"
        >
          Demo mode · Data is temporary and may be cleared at any time
        </div>
      )}
      {children}
    </>
  );
}
