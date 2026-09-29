/**
 * TypeScript type definitions matching the FastAPI Pydantic schemas.
 *
 * Each interface corresponds to a response or request model in
 * `sec_semantic_search/api/schemas.py`.  Keep these in sync when the
 * API contract changes.
 *
 * Naming convention mirrors the Python side:
 *   Python: `StatusResponse`   → TypeScript: `StatusResponse`
 *   Python: `FilingSchema`     → TypeScript: `Filing`
 *   Python: `SearchRequest`    → TypeScript: `SearchRequest`
 */

// ---------------------------------------------------------------------------
// Shared / error
// ---------------------------------------------------------------------------

/** Structured error returned by all 4xx/5xx API responses. */
export interface ApiError {
  error: string;
  message: string;
  details?: string | null;
  hint?: string | null;
}

// ---------------------------------------------------------------------------
// Status
// ---------------------------------------------------------------------------

/** Per-ticker breakdown in the status response. */
export interface TickerBreakdown {
  ticker: string;
  filings: number;
  chunks: number;
  forms: string[];
}

/** GET /api/status/ */
export interface StatusResponse {
  filing_count: number;
  max_filings: number;
  chunk_count: number;
  tickers: string[];
  form_breakdown: Record<string, number>;
  ticker_breakdown: TickerBreakdown[];
  /** True when frontend must show Welcome screen for EDGAR credentials. */
  edgar_session_required: boolean;
  /** True when DEMO_MODE is enabled — frontend shows banner. */
  demo_mode: boolean;
  /** True when the request carries a valid admin key, or no admin key is configured. */
  is_admin: boolean;
}

// ---------------------------------------------------------------------------
// Filings
// ---------------------------------------------------------------------------

/** A single filing record (mirrors FilingSchema in Python). */
export interface Filing {
  ticker: string;
  form_type: string;
  filing_date: string;
  accession_number: string;
  chunk_count: number;
  ingested_at: string;
}

/** GET /api/filings/ */
export interface FilingListResponse {
  filings: Filing[];
  total: number;
}

/** DELETE /api/filings/{accession} */
export interface DeleteResponse {
  accession_number: string;
  chunks_deleted: number;
}

/** POST /api/filings/bulk-delete — request body */
export interface BulkDeleteRequest {
  ticker?: string | null;
  form_type?: string | null;
}

/** POST /api/filings/bulk-delete — response */
export interface BulkDeleteResponse {
  filings_deleted: number;
  chunks_deleted: number;
  tickers_affected: string[];
}

/** POST /api/filings/delete-by-ids — request body */
export interface DeleteByIdsRequest {
  accession_numbers: string[];
}

/** POST /api/filings/delete-by-ids — response */
export interface DeleteByIdsResponse {
  filings_deleted: number;
  chunks_deleted: number;
  not_found: string[];
}

/** DELETE /api/filings/?confirm=true */
export interface ClearAllResponse {
  filings_deleted: number;
  chunks_deleted: number;
}

/** GET /api/admin/session */
export interface AdminSessionResponse {
  admin_required: boolean;
  is_admin: boolean;
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

/** Parent segment text as sent by `POST /api/search/`.
 *
 * Segments over the API's length cap arrive as an excerpt around the
 * matched chunks; the flags say which ends were cut.
 */
export interface ParentSegment {
  content: string;
  truncated_start: boolean;
  truncated_end: boolean;
}

/** A search result as sent over the wire.
 *
 * Each parent segment travels once, on the first result that cites it;
 * later results citing the same segment carry only `parent_key`.
 * `hydrateSearchResponse()` in `lib/api.ts` resolves it per result.
 */
export interface SearchResultWire {
  content: string;
  path: string;
  content_type: "text" | "textsmall" | "table";
  ticker: string;
  form_type: string;
  similarity: number;
  filing_date?: string | null;
  accession_number?: string | null;
  chunk_id?: string | null;
  segment_index?: number | null;
  parent_key?: string | null;
  parent?: ParentSegment | null;
}

/** A single search result, with its parent segment resolved.
 *
 * `content` is the small chunk that scored against the query vector.
 * `parent_content` is the broader segment that chunk came from — use
 * it as the display body and highlight `content` inside it. Falls back
 * to `content` when `parent_content` is null (single-chunk segment or
 * legacy data). The truncation flags are set when the parent is an
 * excerpt of a longer segment.
 */
export interface SearchResult extends Omit<SearchResultWire, "parent_key" | "parent"> {
  parent_content?: string | null;
  parent_truncated_start?: boolean;
  parent_truncated_end?: boolean;
}

/** POST /api/search/ — request body */
export interface SearchRequest {
  query: string;
  top_k?: number;
  ticker?: string[] | null;
  form_type?: string[] | null;
  min_similarity?: number;
  accession_number?: string[] | null;
  start_date?: string | null;
  end_date?: string | null;
}

/** POST /api/search/ — response as sent (query intentionally omitted; see §F4). */
export interface SearchResponseWire {
  results: SearchResultWire[];
  total_results: number;
  search_time_ms: number;
}

/** POST /api/search/ — response after `hydrateSearchResponse()`. */
export interface SearchResponse {
  results: SearchResult[];
  total_results: number;
  search_time_ms: number;
}

// ---------------------------------------------------------------------------
// Ingest
// ---------------------------------------------------------------------------

/** Progress snapshot for a running ingestion task. */
export interface TaskProgress {
  current_ticker: string | null;
  current_form_type: string | null;
  step_label: string;
  step_index: number;
  step_total: number;
  filings_done: number;
  filings_total: number;
  filings_skipped: number;
  filings_failed: number;
}

/** Result for a single successfully ingested filing. */
export interface IngestResult {
  ticker: string;
  form_type: string;
  filing_date: string;
  accession_number: string;
  segment_count: number;
  chunk_count: number;
  duration_seconds: number;
}

/** Task status values (matches Python TaskState enum). */
export type TaskState =
  | "pending"
  | "running"
  | "completed"
  | "failed"
  | "cancelled";

/** Full status of an ingestion task. */
export interface TaskStatus {
  task_id: string;
  status: TaskState;
  tickers: string[];
  form_types: string[];
  progress: TaskProgress;
  results: IngestResult[];
  error?: string | null;
  started_at?: string | null;
  completed_at?: string | null;
}

/** POST /api/ingest/add or /api/ingest/batch — request body */
export interface IngestRequest {
  tickers: string[];
  form_types?: string[];
  count_mode?: "latest" | "total" | "per_form";
  count?: number | null;
  year?: number | null;
  start_date?: string | null;
  end_date?: string | null;
}

/** Immediate response when a task is created. */
export interface TaskResponse {
  task_id: string;
  status: string;
  websocket_url: string;
}

/** GET /api/ingest/tasks */
export interface TaskListResponse {
  tasks: TaskStatus[];
  total: number;
}

// ---------------------------------------------------------------------------
// GPU / Resources
// ---------------------------------------------------------------------------

/** GET /api/resources/gpu */
export interface GPUStatusResponse {
  model_loaded: boolean;
  device: string | null;
  model_name: string;
  approximate_vram_mb: number | null;
}

/** DELETE /api/resources/gpu */
export interface GPUUnloadResponse {
  status: "unloaded" | "already_unloaded";
}

// ---------------------------------------------------------------------------
// WebSocket message types (server → client)
// ---------------------------------------------------------------------------

/**
 * Sent on connect — current state snapshot for reconnection.
 *
 * `seq` is the sequence number of the last message the snapshot already
 * reflects. Messages queued while no client was connected follow it; any
 * with `seq <= snapshot.seq` must not be counted again.
 */
export interface WsSnapshot {
  type: "snapshot";
  task_id: string;
  status: string;
  seq?: number;
  progress: TaskProgress;
  results: WsFilingResult[];
}

/** Per-task sequence number carried by every streamed message. */
interface WsSequenced {
  seq?: number;
}

/**
 * Pipeline step progress. `step_number` is the 0-based index into the
 * five stepper steps (0 Fetching, 1 Parsing, 2 Chunking, 3 Embedding,
 * 4 Storing), the same scale as `TaskProgress.step_index`.
 */
export interface WsStep extends WsSequenced {
  type: "step";
  step: string;
  step_number: number;
  total_steps: number;
  ticker?: string;
  form_type?: string;
}

/** Filing successfully ingested. */
export interface WsFilingDone extends WsSequenced {
  type: "filing_done";
  ticker: string;
  form_type: string;
  filing_date: string;
  accession_number: string;
  segments: number;
  chunks: number;
  time: number;
}

/** Filing skipped (duplicate). */
export interface WsFilingSkipped extends WsSequenced {
  type: "filing_skipped";
  ticker: string;
  form_type: string;
  accession_number?: string;
  reason: string;
}

/** Filing processing failed. */
export interface WsFilingFailed extends WsSequenced {
  type: "filing_failed";
  ticker: string;
  form_type: string;
  accession_number?: string;
  error: string;
}

/** Task completed successfully. */
export interface WsCompleted {
  type: "completed";
  results: WsFilingResult[];
  summary: {
    total: number;
    succeeded: number;
    skipped: number;
    failed: number;
  };
}

/** Task failed. */
export interface WsFailed {
  type: "failed";
  error: string;
  details?: string;
}

/** Task cancelled. */
export interface WsCancelled {
  type: "cancelled";
}

/** FIFO eviction occurred during demo mode ingest. */
export interface WsEviction extends WsSequenced {
  type: "eviction";
  filings_evicted: number;
  chunks_evicted: number;
  tickers_affected: string[];
}

/** WebSocket error (e.g., task not found). */
export interface WsError {
  type: "error";
  error: string;
}

/** Filing result as sent via WebSocket snapshot/completed messages. */
export interface WsFilingResult {
  ticker: string;
  form_type: string;
  filing_date: string;
  accession_number: string;
  segments: number;
  chunks: number;
  time: number;
}

/** Union of all possible WebSocket messages. */
export type WsMessage =
  | WsSnapshot
  | WsStep
  | WsFilingDone
  | WsFilingSkipped
  | WsFilingFailed
  | WsEviction
  | WsCompleted
  | WsFailed
  | WsCancelled
  | WsError;