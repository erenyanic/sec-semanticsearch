/**
 * Filings page — view, filter, sort, and delete ingested SEC filings.
 *
 * ## Architecture: two data sources + URL synchronisation
 *
 * This page uses two hooks:
 *
 *   1. **`useStatus()`** — fetched on mount. Provides available tickers
 *      and form types for the filter dropdowns, and the global filing
 *      count to distinguish "database empty" from "filters match nothing".
 *
 *   2. **`useFilings(params)`** — parameterised query that refetches
 *      whenever filter/sort params change. Also provides three delete
 *      mutations with optimistic cache updates.
 *
 * ## URL query parameter synchronisation
 *
 * The Dashboard navigates to `/filings?ticker=AAPL` or
 * `/filings?form_type=10-K`. This page reads those params on mount
 * (via `useSearchParams()`) and uses them as initial filter values.
 * When the user changes filters, the URL updates via `router.replace()`
 * so the page is bookmarkable. Default values are omitted from the URL
 * for cleanliness.
 *
 * ## State machine
 *
 *   statusLoading      → FilingTableSkeleton (content-shaped placeholder)
 *   statusError        → EmptyState (error)
 *   filing_count === 0 → EmptyState ("No filings", link to Ingest)
 *   filings.length==0  → filters + table with "no matches" message
 *   otherwise          → filters + bulk actions + table + delete dialog
 *
 * ## State ownership
 *
 *   - `params` (FilingQueryParams): filter + sort (synced with URL) and
 *     page + page size (local; reset to page 0 when filters or sort change)
 *   - `selected` (Set<string>): row checkboxes on the current page, cleared
 *     whenever the visible rows are re-queried (filter, sort or page change)
 *   - `deleteTarget` (DeleteTarget | null): controls the delete dialog
 */

"use client";

import { Suspense, useCallback, useEffect, useState } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import Link from "next/link";
import { Database, Upload } from "lucide-react";
import { useStatus } from "@/hooks/useStatus";
import { useAdminSession } from "@/hooks/useAdminSession";
import {
  useFilings,
  DEFAULT_QUERY_PARAMS,
  type FilingQueryParams,
} from "@/hooks/useFilings";
import { extractApiError } from "@/lib/api";
import { Button, EmptyState, FullPageSpinner, useToast } from "@/components/ui";
import {
  FilingTable,
  FilingTableSkeleton,
  FilingFilters,
  BulkActions,
  DeleteDialog,
  type DeleteTarget,
} from "@/components/filings";

// ---------------------------------------------------------------------------
// Inner component (needs useSearchParams, which requires Suspense boundary)
// ---------------------------------------------------------------------------

function FilingsContent() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const { addToast } = useToast();

  // ---- State: filter/sort params (initialised from URL) ----
  const [params, setParams] = useState<FilingQueryParams>(() => ({
    ticker: searchParams.get("ticker") ?? DEFAULT_QUERY_PARAMS.ticker,
    formType: searchParams.get("form_type") ?? DEFAULT_QUERY_PARAMS.formType,
    sortBy:
      (searchParams.get("sort_by") as FilingQueryParams["sortBy"]) ??
      DEFAULT_QUERY_PARAMS.sortBy,
    order:
      (searchParams.get("order") as FilingQueryParams["order"]) ??
      DEFAULT_QUERY_PARAMS.order,
    page: DEFAULT_QUERY_PARAMS.page,
    pageSize: DEFAULT_QUERY_PARAMS.pageSize,
  }));

  // ---- State: row selection ----
  const [selected, setSelected] = useState<Set<string>>(new Set());

  // ---- State: delete dialog ----
  const [deleteTarget, setDeleteTarget] = useState<DeleteTarget | null>(null);

  // ---- Data sources ----
  const {
    data: status,
    isLoading: isStatusLoading,
    isError: isStatusError,
  } = useStatus();
  const { isAdmin } = useAdminSession();

  const filing = useFilings(params);

  // ---- Sync state → URL ----
  // Uses router.replace (not push) to avoid polluting browser history.
  // Default values are omitted from the URL for cleanliness.
  useEffect(() => {
    const sp = new URLSearchParams();
    if (params.ticker) sp.set("ticker", params.ticker);
    if (params.formType) sp.set("form_type", params.formType);
    if (params.sortBy !== DEFAULT_QUERY_PARAMS.sortBy)
      sp.set("sort_by", params.sortBy);
    if (params.order !== DEFAULT_QUERY_PARAMS.order)
      sp.set("order", params.order);

    const qs = sp.toString();
    router.replace(`/filings${qs ? `?${qs}` : ""}`, { scroll: false });
  }, [params.ticker, params.formType, params.sortBy, params.order, router]);

  // ---- Clear selection when the visible rows are re-queried ----
  // Pages come from the server, so a selection can only cover the rows
  // on screen; it would otherwise reference filings the page no longer
  // holds (and the delete dialog could not count their chunks).
  useEffect(() => {
    setSelected(new Set());
  }, [
    params.ticker,
    params.formType,
    params.sortBy,
    params.order,
    params.page,
    params.pageSize,
  ]);

  /** Step back one page when a delete empties the current one. */
  function afterDelete(deletedOnPage: number) {
    if (params.page > 0 && deletedOnPage >= filing.filings.length) {
      setParams({ ...params, page: params.page - 1 });
    }
  }

  // ---- Delete handlers ----
  const handleDeleteSingle = useCallback((filing: { ticker: string; form_type: string; accession_number: string }) => {
    // Find the full filing object from the current list
    const full = filing as import("@/lib/types").Filing;
    setDeleteTarget({ kind: "single", filing: full });
  }, []);

  function handleDeleteSelected() {
    const selectedFilings = filing.filings.filter((f) =>
      selected.has(f.accession_number),
    );
    const totalChunks = selectedFilings.reduce(
      (sum, f) => sum + f.chunk_count,
      0,
    );
    setDeleteTarget({
      kind: "selected",
      count: selectedFilings.length,
      totalChunks,
    });
  }

  function handleDeleteAll() {
    setDeleteTarget({ kind: "all", count: filing.total });
  }

  async function executeDelete() {
    if (!deleteTarget) return;

    try {
      switch (deleteTarget.kind) {
        case "single": {
          await filing.deleteSingle(deleteTarget.filing.accession_number);
          // Also remove from selection if it was selected
          if (selected.has(deleteTarget.filing.accession_number)) {
            const next = new Set(selected);
            next.delete(deleteTarget.filing.accession_number);
            setSelected(next);
          }
          afterDelete(1);
          addToast(
            "success",
            `Deleted ${deleteTarget.filing.ticker} ${deleteTarget.filing.form_type}`,
          );
          break;
        }
        case "selected": {
          await filing.deleteSelected([...selected]);
          afterDelete(selected.size);
          setSelected(new Set());
          addToast("success", `Deleted ${deleteTarget.count} filings`);
          break;
        }
        case "all": {
          await filing.clearAll();
          setParams({ ...params, page: 0 });
          setSelected(new Set());
          addToast("success", "All filings deleted");
          break;
        }
      }
    } catch (err) {
      addToast("error", extractApiError(err).message);
    } finally {
      setDeleteTarget(null);
    }
  }

  // ---- State machine: loading ----
  if (isStatusLoading) {
    return <FilingTableSkeleton />;
  }

  // ---- State machine: error ----
  if (isStatusError) {
    return (
      <EmptyState
        icon={Database}
        title="Failed to load filings"
        description="Could not connect to the database. Is the API server running?"
        action={
          <Button onClick={() => window.location.reload()}>Retry</Button>
        }
      />
    );
  }

  // ---- State machine: database empty ----
  if (status && status.filing_count === 0) {
    return (
      <EmptyState
        icon={Database}
        title="No filings ingested"
        description="Ingest SEC filings first, then manage them here."
        action={
          <Link href="/ingest">
            <Button>
              <Upload className="mr-2 h-4 w-4" />
              Ingest Filings
            </Button>
          </Link>
        }
      />
    );
  }

  // ---- State machine: data loaded ----
  const totalFilings = filing.total;
  const tickerCount = status?.tickers.length ?? 0;

  return (
    <div className="space-y-8 [animation:fade-in_300ms_ease-out]">
      {/* ---- Page header ---- */}
      <div className="space-y-3">
        <h1 className="text-3xl font-semibold tracking-tight text-fg sm:text-4xl">
          Filings
        </h1>
        <p className="text-base text-fg-muted">
          <span className="font-semibold text-fg">
            {totalFilings.toLocaleString()}
          </span>{" "}
          filing{totalFilings !== 1 && "s"} across{" "}
          <span className="font-semibold text-fg">
            {tickerCount.toLocaleString()}
          </span>{" "}
          ticker{tickerCount !== 1 && "s"}
          {selected.size > 0 && (
            <>
              {" · "}
              <span className="font-semibold text-accent">
                {selected.size.toLocaleString()} selected
              </span>
            </>
          )}
        </p>
      </div>

      {/* ---- Toolbar: filters + bulk actions ---- */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <FilingFilters
          ticker={params.ticker}
          formType={params.formType}
          onTickerChange={(ticker) => setParams({ ...params, ticker, page: 0 })}
          onFormTypeChange={(formType) =>
            setParams({ ...params, formType, page: 0 })
          }
          availableTickers={status?.tickers ?? []}
          availableFormTypes={Object.keys(status?.form_breakdown ?? {})}
        />
        <BulkActions
          selectedCount={selected.size}
          totalFilings={filing.total}
          onDeleteSelected={handleDeleteSelected}
          onDeleteAll={handleDeleteAll}
          isDeleting={filing.isDeleting}
          isAdmin={isAdmin}
        />
      </div>

      {/* Table */}
      <FilingTable
        filings={filing.filings}
        total={filing.total}
        page={params.page}
        pageSize={params.pageSize}
        onPageChange={(page) => setParams({ ...params, page })}
        onPageSizeChange={(pageSize) =>
          setParams({ ...params, pageSize, page: 0 })
        }
        sortBy={params.sortBy}
        order={params.order}
        onSortChange={(sortBy, order) =>
          setParams({ ...params, sortBy, order, page: 0 })
        }
        selected={selected}
        onSelectionChange={setSelected}
        onDeleteFiling={handleDeleteSingle}
        isDeleting={filing.isDeleting}
      />

      {/* Delete confirmation dialog */}
      <DeleteDialog
        target={deleteTarget}
        onClose={() => setDeleteTarget(null)}
        onConfirm={executeDelete}
        isDeleting={filing.isDeleting}
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Page export (wraps content in Suspense for useSearchParams)
// ---------------------------------------------------------------------------

/**
 * Next.js App Router requires a Suspense boundary around components that
 * use `useSearchParams()`. Without it, the build fails with a static
 * rendering error. The fallback shows the page spinner while the URL
 * params are being read.
 */
export default function FilingsPage() {
  return (
    <Suspense fallback={<FullPageSpinner />}>
      <FilingsContent />
    </Suspense>
  );
}