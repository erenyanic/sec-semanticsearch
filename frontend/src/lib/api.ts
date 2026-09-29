/**
 * Centralised API client for the SEC Semantic Search backend.
 *
 * All HTTP communication with the FastAPI backend flows through this
 * module.  It provides:
 *   1. A pre-configured Axios instance with the base URL and error
 *      interceptor.
 *   2. One function per API endpoint, fully typed with the interfaces
 *      from `types.ts`.
 *
 * Components never call `axios.get()` directly — they use these
 * functions, which keeps API details (URLs, method, body shape) in
 * one place.
 */

import axios, { AxiosError } from "axios";
import type {
  AdminSessionResponse,
  ApiError,
  ClearAllResponse,
  DeleteByIdsRequest,
  DeleteByIdsResponse,
  DeleteResponse,
  FilingListResponse,
  IngestRequest,
  ParentSegment,
  SearchRequest,
  SearchResponse,
  SearchResponseWire,
  StatusResponse,
  TaskListResponse,
  TaskResponse,
} from "./types";

// ---------------------------------------------------------------------------
// Axios instance
// ---------------------------------------------------------------------------

/**
 * Pre-configured Axios instance.
 *
 * `baseURL` is empty because the Next.js dev server proxies `/api/*`
 * to the FastAPI backend (see `next.config.ts` rewrites).  In
 * production, the same relative paths work when FastAPI serves the
 * frontend or both sit behind a reverse proxy.
 */
const client = axios.create({
  headers: {
    "Content-Type": "application/json",
    ...(process.env.NEXT_PUBLIC_API_KEY
      ? { "X-API-Key": process.env.NEXT_PUBLIC_API_KEY }
      : {}),
  },
});

// ---------------------------------------------------------------------------
// EDGAR session credential interceptor
// ---------------------------------------------------------------------------

/**
 * Attach `X-Edgar-Name` and `X-Edgar-Email` headers from sessionStorage
 * to every outgoing request.  The backend uses these for EDGAR API calls
 * (ingest routes only), but attaching them globally is harmless — the
 * backend ignores them on non-ingest endpoints.
 *
 * Reading from `sessionStorage` directly (rather than importing the hook)
 * keeps this module free of React dependencies.
 */
client.interceptors.request.use((config) => {
  if (typeof window !== "undefined") {
    const name = sessionStorage.getItem("edgar_name");
    const email = sessionStorage.getItem("edgar_email");
    if (name && email) {
      config.headers["X-Edgar-Name"] = name;
      config.headers["X-Edgar-Email"] = email;
    }
  }
  return config;
});

// ---------------------------------------------------------------------------
// Error interceptor
// ---------------------------------------------------------------------------

/**
 * Extract a structured `ApiError` from an Axios error.
 *
 * The FastAPI backend always returns `{ error, message, details?, hint? }`
 * on 4xx/5xx responses.  If the response doesn't match that shape (e.g.
 * network failure), we construct a fallback error.
 */
export function extractApiError(err: unknown): ApiError {
  if (err instanceof AxiosError && err.response?.data) {
    const data = err.response.data;
    // The backend returns our ErrorResponse schema.
    if (typeof data === "object" && "message" in data) {
      return data as ApiError;
    }
    // FastAPI HTTPException wraps our ErrorResponse in `detail`.
    if (typeof data === "object" && "detail" in data) {
      const detail = data.detail;
      // Our structured ErrorResponse inside `detail`.
      if (typeof detail === "object" && detail !== null && "message" in detail) {
        return detail as ApiError;
      }
      // FastAPI validation errors (422) have an array shape.
      return {
        error: "ValidationError",
        message: Array.isArray(detail)
          ? detail.map((d: { msg: string }) => d.msg).join("; ")
          : String(detail),
      };
    }
  }

  // Network error or unexpected shape.
  const message =
    err instanceof Error ? err.message : "An unexpected error occurred";
  return { error: "NetworkError", message };
}

// ---------------------------------------------------------------------------
// Status
// ---------------------------------------------------------------------------

/** Fetch database overview. */
export async function getStatus(): Promise<StatusResponse> {
  const { data } = await client.get<StatusResponse>("/api/status/");
  return data;
}

// ---------------------------------------------------------------------------
// Filings
// ---------------------------------------------------------------------------

export interface FilingListParams {
  ticker?: string;
  form_type?: string;
  sort_by?: "filing_date" | "ticker" | "form_type" | "chunk_count" | "ingested_at";
  order?: "asc" | "desc";
  /** Page size (1–200). */
  limit?: number;
  /** Rows to skip. */
  offset?: number;
}

/** List one page of filings; `total` counts every match across pages. */
export async function getFilings(
  params?: FilingListParams,
): Promise<FilingListResponse> {
  const { data } = await client.get<FilingListResponse>("/api/filings/", {
    params,
  });
  return data;
}

/** Delete a single filing. */
export async function deleteFiling(
  accessionNumber: string,
): Promise<DeleteResponse> {
  const { data } = await client.delete<DeleteResponse>(
    `/api/filings/${encodeURIComponent(accessionNumber)}`,
  );
  return data;
}

/** Delete specific filings by accession numbers in a single request. */
export async function deleteFilingsByIds(
  accessionNumbers: string[],
): Promise<DeleteByIdsResponse> {
  const { data } = await client.post<DeleteByIdsResponse>(
    "/api/filings/delete-by-ids",
    { accession_numbers: accessionNumbers } satisfies DeleteByIdsRequest,
  );
  return data;
}

/** Clear all filings (requires confirm=true). */
export async function clearAllFilings(): Promise<ClearAllResponse> {
  const { data } = await client.delete<ClearAllResponse>("/api/admin/filings", {
    params: { confirm: true },
  });
  return data;
}

// ---------------------------------------------------------------------------
// Admin session
// ---------------------------------------------------------------------------

/** Check whether the current browser has an active admin session. */
export async function getAdminSession(): Promise<AdminSessionResponse> {
  const { data } = await client.get<AdminSessionResponse>("/api/admin/session");
  return data;
}

/** Start an admin session using the server-side admin key validator. */
export async function loginAdminSession(adminKey: string): Promise<void> {
  await client.post("/api/admin/session", { admin_key: adminKey });
}

/** Clear the current admin session cookie. */
export async function logoutAdminSession(): Promise<void> {
  await client.delete("/api/admin/session");
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

/**
 * Resolve each result's parent segment.
 *
 * The API sends a parent once, on the first result that cites it, and
 * only `parent_key` on later ones. Components read `parent_content`
 * per result, so the lookup happens here, once, at the API boundary.
 */
export function hydrateSearchResponse(wire: SearchResponseWire): SearchResponse {
  const parents = new Map<string, ParentSegment>();
  const results = wire.results.map(({ parent_key, parent, ...rest }) => {
    if (parent_key && parent) parents.set(parent_key, parent);
    const resolved = parent_key ? parents.get(parent_key) : undefined;
    return {
      ...rest,
      parent_content: resolved?.content ?? null,
      parent_truncated_start: resolved?.truncated_start ?? false,
      parent_truncated_end: resolved?.truncated_end ?? false,
    };
  });
  return {
    results,
    total_results: wire.total_results,
    search_time_ms: wire.search_time_ms,
  };
}

/** Execute a semantic search query. */
export async function search(body: SearchRequest): Promise<SearchResponse> {
  const { data } = await client.post<SearchResponseWire>("/api/search/", body);
  return hydrateSearchResponse(data);
}

// ---------------------------------------------------------------------------
// Ingest
// ---------------------------------------------------------------------------

/** Start a single-ticker ingestion task. */
export async function ingestAdd(body: IngestRequest): Promise<TaskResponse> {
  const { data } = await client.post<TaskResponse>("/api/ingest/add", body);
  return data;
}

/** Start a multi-ticker batch ingestion task. */
export async function ingestBatch(body: IngestRequest): Promise<TaskResponse> {
  const { data } = await client.post<TaskResponse>("/api/ingest/batch", body);
  return data;
}

/** List all ingestion tasks (active + recent). */
export async function getTasks(): Promise<TaskListResponse> {
  const { data } = await client.get<TaskListResponse>("/api/ingest/tasks");
  return data;
}

/** Cancel a running task. */
export async function cancelTask(taskId: string): Promise<void> {
  await client.delete(`/api/ingest/tasks/${encodeURIComponent(taskId)}`);
}
