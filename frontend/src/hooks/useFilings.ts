/**
 * useFilings — data hook for the Filings page.
 *
 * Bundles a parameterised query (one page of filings with filters/sort)
 * and three delete mutations (single, multi-select, clear-all) behind a
 * single hook interface. The page never touches React Query directly.
 *
 * ## Query key
 *
 *   `["filings", ticker, formType, sortBy, order, page, pageSize]`
 *
 * The API filters, sorts and slices in SQLite and returns one page plus
 * the total match count, so each page is its own cache entry. The
 * previous page stays on screen while the next one loads.
 *
 * ## Cache strategy after deletions
 *
 *   - **Single / multi-select delete:** remove the rows from the cached
 *     page at once for instant feedback, decrement `total`, then
 *     invalidate every `["filings"]` entry so the page refills from the
 *     next one and other pages re-sync.
 *   - **Clear all:** set every cached page to `{ filings: [], total: 0 }`.
 *
 * All mutations also invalidate `["status"]` so the Dashboard's counts
 * update without a manual refresh.
 */

"use client";

import {
  keepPreviousData,
  useMutation,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query";
import type {
  Filing,
  FilingListResponse,
  DeleteResponse,
  DeleteByIdsResponse,
  ClearAllResponse,
} from "@/lib/types";
import {
  getFilings,
  deleteFiling,
  deleteFilingsByIds,
  clearAllFilings,
  type FilingListParams,
} from "@/lib/api";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** Parameters controlling which page of filings to fetch and how to sort it. */
export interface FilingQueryParams {
  ticker: string;
  formType: string;
  sortBy: NonNullable<FilingListParams["sort_by"]>;
  order: "asc" | "desc";
  /** Zero-based page index. */
  page: number;
  /** Rows per page. At most 50, the delete-by-ids batch limit. */
  pageSize: number;
}

export const DEFAULT_QUERY_PARAMS: FilingQueryParams = {
  ticker: "",
  formType: "",
  sortBy: "filing_date",
  order: "desc",
  page: 0,
  pageSize: 10,
};

export interface UseFilingsReturn {
  /** The current page of filings matching the filters, server-sorted. */
  filings: Filing[];
  /** Total count of filings matching the current filters, across pages. */
  total: number;
  /** True while the filing list is loading. */
  isLoading: boolean;
  /** True if the fetch failed. */
  isError: boolean;
  /** Error object if the fetch failed. */
  error: Error | null;

  /** Delete a single filing by accession number. */
  deleteSingle: (accessionNumber: string) => Promise<DeleteResponse>;
  /** Delete multiple filings by accession number (single batch request). */
  deleteSelected: (accessionNumbers: string[]) => Promise<DeleteByIdsResponse>;
  /** Delete ALL filings in the database. */
  clearAll: () => Promise<ClearAllResponse>;
  /** True while any deletion is in progress. */
  isDeleting: boolean;
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function useFilings(params: FilingQueryParams): UseFilingsReturn {
  const queryClient = useQueryClient();

  // The query key encodes every parameter so React Query refetches
  // automatically when the user changes a filter, sort column or page.
  const queryKey = [
    "filings",
    params.ticker,
    params.formType,
    params.sortBy,
    params.order,
    params.page,
    params.pageSize,
  ];

  // ---- Query: fetch one page ----
  const { data, isLoading, isError, error } = useQuery<FilingListResponse>({
    queryKey,
    queryFn: () =>
      getFilings({
        ticker: params.ticker || undefined,
        form_type: params.formType || undefined,
        sort_by: params.sortBy,
        order: params.order,
        limit: params.pageSize,
        offset: params.page * params.pageSize,
      }),
    // Keep the previous page on screen while the next one loads.
    placeholderData: keepPreviousData,
  });

  /** Remove deleted rows from the cached page and refetch every page. */
  function removeFromCache(accessionNumbers: string[]) {
    const deleted = new Set(accessionNumbers);
    queryClient.setQueryData<FilingListResponse>(queryKey, (old) => {
      if (!old) return old;
      const filtered = old.filings.filter((f) => !deleted.has(f.accession_number));
      const removed = old.filings.length - filtered.length;
      return { filings: filtered, total: Math.max(0, old.total - removed) };
    });
    // Rows from later pages move up; other cached pages are now stale.
    queryClient.invalidateQueries({ queryKey: ["filings"] });
    // Dashboard counts should update too.
    queryClient.invalidateQueries({ queryKey: ["status"] });
  }

  // ---- Mutation: delete a single filing ----
  const singleDelete = useMutation<DeleteResponse, Error, string>({
    mutationFn: deleteFiling,
    onSuccess: (_result, accessionNumber) => removeFromCache([accessionNumber]),
    onError: () => {
      // Filing may have been evicted (demo mode FIFO) — refetch so the
      // stale row disappears from the table.
      queryClient.invalidateQueries({ queryKey: ["filings"] });
      queryClient.invalidateQueries({ queryKey: ["status"] });
    },
  });

  // ---- Mutation: clear all filings ----
  const clearMutation = useMutation<ClearAllResponse, Error, void>({
    mutationFn: clearAllFilings,
    onSuccess: () => {
      // Everything is gone: empty every cached page immediately.
      queryClient.setQueriesData<FilingListResponse>(
        { queryKey: ["filings"] },
        { filings: [], total: 0 },
      );
      queryClient.invalidateQueries({ queryKey: ["status"] });
    },
  });

  // Multi-select delete: single batch request via POST /api/filings/delete-by-ids.
  const batchDelete = useMutation<DeleteByIdsResponse, Error, string[]>({
    mutationFn: deleteFilingsByIds,
    onSuccess: (_result, accessionNumbers) => removeFromCache(accessionNumbers),
  });

  const isDeleting =
    singleDelete.isPending || batchDelete.isPending || clearMutation.isPending;

  return {
    filings: data?.filings ?? [],
    total: data?.total ?? 0,
    isLoading,
    isError,
    error: error ?? null,

    deleteSingle: (accession) => singleDelete.mutateAsync(accession),
    deleteSelected: (accessions) => batchDelete.mutateAsync(accessions),
    clearAll: () => clearMutation.mutateAsync(),
    isDeleting,
  };
}
