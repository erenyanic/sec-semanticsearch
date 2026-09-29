/**
 * Hook managing the full ingestion lifecycle.
 *
 * ## Architecture: useReducer + WebSocket ref
 *
 * Ingestion state is complex — a single WebSocket message can update
 * progress counters, append to the filing event list, and push a new
 * result simultaneously.  Using `useReducer` keeps these updates atomic
 * (one dispatch = one render) and avoids the stale-closure problem that
 * arises with multiple `useState` setters inside a WebSocket callback.
 *
 * The `IngestWebSocket` instance is stored in a `useRef` because it is
 * a mutable, long-lived object that should not trigger re-renders.
 *
 * ## Lifecycle
 *
 *   idle → startIngest() → pending → (WS) running → (WS) completed | failed | cancelled
 *                                                 ↳ cancel() → (WS) cancelled
 *   completed | failed | cancelled → reset() → idle
 *
 * ## Active task recovery
 *
 * On mount, the hook calls `GET /api/ingest/tasks` to check for a
 * running or pending task.  If found, it dispatches `RESUME` with the
 * task's current state and opens a WebSocket for continued streaming.
 * This allows the user to navigate away and come back without losing
 * progress.
 *
 * ## Reconnection and replayed messages
 *
 * Every streamed message carries a per-task `seq`, and the snapshot sent
 * on (re)connect carries the `seq` of the last message it already
 * reflects. Messages the server queued while no client was connected are
 * streamed after the snapshot; those with `seq <= lastSeq` are replays —
 * their counters are already in the snapshot, so the reducer only adds a
 * missing event row and never counts them again.
 */

import { useCallback, useEffect, useReducer, useRef } from "react";
import { useQueryClient } from "@tanstack/react-query";
import {
  ingestAdd,
  ingestBatch,
  getTasks,
  cancelTask,
  extractApiError,
} from "@/lib/api";
import { IngestWebSocket } from "@/lib/websocket";
import type {
  IngestRequest,
  TaskProgress,
  TaskStatus,
  WsFilingResult,
  WsMessage,
} from "@/lib/types";

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/**
 * A chronological event for a single filing during ingestion.
 *
 * The `type` discriminant lets the ProgressTracker and IngestSummary
 * render each event with the appropriate icon and detail fields.
 */
export interface FilingEvent {
  type: "done" | "skipped" | "failed" | "eviction";
  ticker: string;
  form_type: string;
  filing_date?: string;
  accession_number?: string;
  // done-only fields
  segments?: number;
  chunks?: number;
  time?: number;
  // skipped-only
  reason?: string;
  // failed-only
  error?: string;
  // eviction-only fields
  filings_evicted?: number;
  chunks_evicted?: number;
  tickers_affected?: string[];
}

// ---------------------------------------------------------------------------
// Reducer state and actions
// ---------------------------------------------------------------------------

interface IngestState {
  taskId: string | null;
  status: "idle" | "pending" | "running" | "completed" | "failed" | "cancelled";
  progress: TaskProgress;
  results: WsFilingResult[];
  filingEvents: FilingEvent[];
  summary: { total: number; succeeded: number; skipped: number; failed: number } | null;
  error: string | null;
  startedAt: Date | null;
  completedAt: Date | null;
  /** `seq` of the last message whose effect is in this state (see above). */
  lastSeq: number;
}

/** Exported for direct unit testing. */
export const DEFAULT_PROGRESS: TaskProgress = {
  current_ticker: null,
  current_form_type: null,
  step_label: "",
  step_index: 0,
  step_total: 5,
  filings_done: 0,
  filings_total: 0,
  filings_skipped: 0,
  filings_failed: 0,
};

/** Exported for direct unit testing. */
export const INITIAL_STATE: IngestState = {
  taskId: null,
  status: "idle",
  progress: DEFAULT_PROGRESS,
  results: [],
  filingEvents: [],
  summary: null,
  error: null,
  startedAt: null,
  completedAt: null,
  lastSeq: 0,
};

type IngestAction =
  | { type: "START"; taskId: string }
  | { type: "SNAPSHOT"; status: string; progress: TaskProgress; results: WsFilingResult[]; seq?: number }
  | { type: "STEP"; step: string; step_number: number; total_steps: number; ticker?: string; form_type?: string; seq?: number }
  | { type: "FILING_DONE"; ticker: string; form_type: string; filing_date: string; accession_number: string; segments: number; chunks: number; time: number; seq?: number }
  | { type: "FILING_SKIPPED"; ticker: string; form_type: string; reason: string; accession_number?: string; seq?: number }
  | { type: "FILING_FAILED"; ticker: string; form_type: string; error: string; accession_number?: string; seq?: number }
  | { type: "EVICTION"; filings_evicted: number; chunks_evicted: number; tickers_affected: string[]; seq?: number }
  | { type: "COMPLETED"; results: WsFilingResult[]; summary: { total: number; succeeded: number; skipped: number; failed: number } }
  | { type: "FAILED"; error: string; details?: string }
  | { type: "CANCELLED" }
  | { type: "ERROR"; error: string }
  | { type: "RESUME"; taskStatus: TaskStatus }
  | { type: "RESET" };

/** A message already reflected in the last snapshot (see module doc). */
function isReplay(state: IngestState, seq: number | undefined): boolean {
  return seq !== undefined && seq <= state.lastSeq;
}

/** Whether a replayed filing message's event row is already present. */
function hasFilingEvent(
  events: FilingEvent[],
  type: FilingEvent["type"],
  accession: string | undefined,
): boolean {
  return (
    accession !== undefined &&
    events.some((e) => e.type === type && e.accession_number === accession)
  );
}

/** Add a filing event row. */
function withEvent(state: IngestState, event: FilingEvent): IngestState {
  return { ...state, filingEvents: [...state.filingEvents, event] };
}

/**
 * Exported for direct unit testing — not part of the public hook API.
 *
 * Plain immutable updates: a batch holds at most a few hundred filings,
 * so copying the arrays per message costs nothing measurable, and the
 * ingest page no longer ships a draft-state library for one reducer.
 */
export function reducer(state: IngestState, action: IngestAction): IngestState {
  switch (action.type) {
    case "START":
      return {
        ...INITIAL_STATE,
        taskId: action.taskId,
        status: "pending",
        startedAt: new Date(),
      };

    case "SNAPSHOT":
      return {
        ...state,
        status:
          action.status === "pending" || action.status === "running"
            ? action.status
            : state.status,
        progress: action.progress,
        results: action.results,
        lastSeq: action.seq ?? 0,
      };

    case "STEP":
      // A replayed step is older than the snapshot's own step.
      if (isReplay(state, action.seq)) return state;
      return {
        ...state,
        status: "running",
        progress: {
          ...state.progress,
          step_label: action.step,
          step_index: action.step_number, // already 0-based
          step_total: action.total_steps,
          current_ticker: action.ticker ?? state.progress.current_ticker,
          current_form_type: action.form_type ?? state.progress.current_form_type,
        },
        lastSeq: action.seq ?? state.lastSeq,
      };

    case "FILING_DONE": {
      const event: FilingEvent = {
        type: "done",
        ticker: action.ticker,
        form_type: action.form_type,
        filing_date: action.filing_date,
        accession_number: action.accession_number,
        segments: action.segments,
        chunks: action.chunks,
        time: action.time,
      };
      if (isReplay(state, action.seq)) {
        return hasFilingEvent(state.filingEvents, "done", action.accession_number)
          ? state
          : withEvent(state, event);
      }
      return withEvent(
        {
          ...state,
          progress: { ...state.progress, filings_done: state.progress.filings_done + 1 },
          results: [
            ...state.results,
            {
              ticker: action.ticker,
              form_type: action.form_type,
              filing_date: action.filing_date,
              accession_number: action.accession_number,
              segments: action.segments,
              chunks: action.chunks,
              time: action.time,
            },
          ],
          lastSeq: action.seq ?? state.lastSeq,
        },
        event,
      );
    }

    case "FILING_SKIPPED": {
      const event: FilingEvent = {
        type: "skipped",
        ticker: action.ticker,
        form_type: action.form_type,
        accession_number: action.accession_number,
        reason: action.reason,
      };
      if (isReplay(state, action.seq)) {
        return hasFilingEvent(state.filingEvents, "skipped", action.accession_number)
          ? state
          : withEvent(state, event);
      }
      return withEvent(
        {
          ...state,
          progress: { ...state.progress, filings_skipped: state.progress.filings_skipped + 1 },
          lastSeq: action.seq ?? state.lastSeq,
        },
        event,
      );
    }

    case "FILING_FAILED": {
      const event: FilingEvent = {
        type: "failed",
        ticker: action.ticker,
        form_type: action.form_type,
        accession_number: action.accession_number,
        error: action.error,
      };
      if (isReplay(state, action.seq)) {
        return hasFilingEvent(state.filingEvents, "failed", action.accession_number)
          ? state
          : withEvent(state, event);
      }
      return withEvent(
        {
          ...state,
          progress: { ...state.progress, filings_failed: state.progress.filings_failed + 1 },
          lastSeq: action.seq ?? state.lastSeq,
        },
        event,
      );
    }

    case "EVICTION": {
      const event: FilingEvent = {
        type: "eviction",
        ticker: action.tickers_affected.join(", ") || "—",
        form_type: "",
        filings_evicted: action.filings_evicted,
        chunks_evicted: action.chunks_evicted,
        tickers_affected: action.tickers_affected,
      };
      if (isReplay(state, action.seq)) {
        const seen = state.filingEvents.some(
          (e) =>
            e.type === "eviction" &&
            e.filings_evicted === action.filings_evicted &&
            e.chunks_evicted === action.chunks_evicted,
        );
        return seen ? state : withEvent(state, event);
      }
      return withEvent({ ...state, lastSeq: action.seq ?? state.lastSeq }, event);
    }

    case "COMPLETED":
      return {
        ...state,
        status: "completed",
        results: action.results,
        summary: action.summary,
        completedAt: new Date(),
      };

    case "FAILED":
      return {
        ...state,
        status: "failed",
        error: action.details ? `${action.error}: ${action.details}` : action.error,
        completedAt: new Date(),
      };

    case "CANCELLED":
      return {
        ...state,
        status: "cancelled",
        summary: {
          total:
            state.progress.filings_done +
            state.progress.filings_skipped +
            state.progress.filings_failed,
          succeeded: state.progress.filings_done,
          skipped: state.progress.filings_skipped,
          failed: state.progress.filings_failed,
        },
        completedAt: new Date(),
      };

    case "ERROR":
      return { ...state, status: "failed", error: action.error, completedAt: new Date() };

    case "RESUME": {
      const ts = action.taskStatus;
      // Convert IngestResult[] from REST API to WsFilingResult[] shape.
      const wsResults: WsFilingResult[] = ts.results.map((r) => ({
        ticker: r.ticker,
        form_type: r.form_type,
        filing_date: r.filing_date,
        accession_number: r.accession_number,
        segments: r.segment_count,
        chunks: r.chunk_count,
        time: r.duration_seconds,
      }));
      // Reconstruct filing events from results (done events only —
      // skip/fail events are not in the REST task status; the WebSocket
      // replays any still queued, see "Reconnection" above).
      const events: FilingEvent[] = ts.results.map((r) => ({
        type: "done" as const,
        ticker: r.ticker,
        form_type: r.form_type,
        filing_date: r.filing_date,
        accession_number: r.accession_number,
        segments: r.segment_count,
        chunks: r.chunk_count,
        time: r.duration_seconds,
      }));
      return {
        ...INITIAL_STATE,
        taskId: ts.task_id,
        status: ts.status === "pending" || ts.status === "running" ? ts.status : "idle",
        progress: ts.progress,
        results: wsResults,
        filingEvents: events,
        startedAt: ts.started_at ? new Date(ts.started_at) : new Date(),
      };
    }

    case "RESET":
      return INITIAL_STATE;

    default:
      return state;
  }
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export interface UseIngestReturn extends IngestState {
  isActive: boolean;
  startIngest: (request: IngestRequest) => Promise<void>;
  cancel: () => Promise<void>;
  reset: () => void;
}

export function useIngest(): UseIngestReturn {
  const [state, dispatch] = useReducer(reducer, INITIAL_STATE);
  const wsRef = useRef<IngestWebSocket | null>(null);
  const queryClient = useQueryClient();

  // Derived: is an ingest actively running (pending or running)?
  const isActive = state.status === "pending" || state.status === "running";

  // ---- WebSocket message handler ----
  const handleMessage = useCallback(
    (msg: WsMessage) => {
      switch (msg.type) {
        case "snapshot":
          dispatch({
            type: "SNAPSHOT",
            status: msg.status,
            progress: msg.progress,
            results: msg.results,
            seq: msg.seq,
          });
          break;

        case "step":
          dispatch({
            type: "STEP",
            step: msg.step,
            step_number: msg.step_number,
            total_steps: msg.total_steps,
            ticker: msg.ticker,
            form_type: msg.form_type,
            seq: msg.seq,
          });
          break;

        case "filing_done":
          dispatch({
            type: "FILING_DONE",
            ticker: msg.ticker,
            form_type: msg.form_type,
            filing_date: msg.filing_date,
            accession_number: msg.accession_number,
            segments: msg.segments,
            chunks: msg.chunks,
            time: msg.time,
            seq: msg.seq,
          });
          break;

        case "filing_skipped":
          dispatch({
            type: "FILING_SKIPPED",
            ticker: msg.ticker,
            form_type: msg.form_type,
            accession_number: msg.accession_number,
            reason: msg.reason,
            seq: msg.seq,
          });
          break;

        case "filing_failed":
          dispatch({
            type: "FILING_FAILED",
            ticker: msg.ticker,
            form_type: msg.form_type,
            accession_number: msg.accession_number,
            error: msg.error,
            seq: msg.seq,
          });
          break;

        case "eviction":
          dispatch({
            type: "EVICTION",
            filings_evicted: msg.filings_evicted,
            chunks_evicted: msg.chunks_evicted,
            tickers_affected: msg.tickers_affected,
            seq: msg.seq,
          });
          // Invalidate status cache — filing counts have changed.
          queryClient.invalidateQueries({ queryKey: ["status"] });
          break;

        case "completed":
          dispatch({
            type: "COMPLETED",
            results: msg.results,
            summary: msg.summary,
          });
          // Invalidate the status cache so Dashboard shows new filings.
          queryClient.invalidateQueries({ queryKey: ["status"] });
          break;

        case "failed":
          dispatch({
            type: "FAILED",
            error: msg.error,
            details: msg.details,
          });
          break;

        case "cancelled":
          dispatch({ type: "CANCELLED" });
          // Invalidate in case some filings were ingested before cancel.
          queryClient.invalidateQueries({ queryKey: ["status"] });
          break;

        case "error":
          dispatch({ type: "ERROR", error: msg.error });
          break;
      }
    },
    [queryClient],
  );

  // ---- Connect WebSocket to a task ----
  const connectWs = useCallback(
    (taskId: string) => {
      // Close any existing connection first.
      wsRef.current?.close();
      const ws = new IngestWebSocket(taskId, handleMessage);
      wsRef.current = ws;
      ws.connect();
    },
    [handleMessage],
  );

  // ---- Start a new ingestion ----
  const startIngest = useCallback(
    async (request: IngestRequest) => {
      // Guard: don't start if already running.
      if (isActive) return;

      try {
        // Use /add for single ticker, /batch for multiple.
        const apiFn = request.tickers.length === 1 ? ingestAdd : ingestBatch;
        const response = await apiFn(request);

        dispatch({ type: "START", taskId: response.task_id });
        connectWs(response.task_id);
      } catch (err) {
        dispatch({ type: "ERROR", error: extractApiError(err).message });
      }
    },
    [isActive, connectWs],
  );

  // ---- Cancel the active task ----
  const cancel = useCallback(async () => {
    if (!state.taskId) return;
    // We don't dispatch here — the WebSocket will deliver a
    // "cancelled" message, which triggers the state transition.
    await cancelTask(state.taskId);
  }, [state.taskId]);

  // ---- Reset to idle ----
  const reset = useCallback(() => {
    wsRef.current?.close();
    wsRef.current = null;
    dispatch({ type: "RESET" });
  }, []);

  // ---- Active task recovery on mount ----
  useEffect(() => {
    let cancelled = false;

    async function checkForActiveTask() {
      try {
        const { tasks } = await getTasks();
        const active = tasks.find(
          (t) => t.status === "pending" || t.status === "running",
        );
        if (active && !cancelled) {
          dispatch({ type: "RESUME", taskStatus: active });
          connectWs(active.task_id);
        }
      } catch {
        // If the API is unreachable, stay idle — the user can
        // start a new ingest when the backend comes back.
      }
    }

    checkForActiveTask();

    return () => {
      cancelled = true;
    };
    // Only run on mount.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ---- Cleanup WebSocket on unmount ----
  useEffect(() => {
    return () => {
      wsRef.current?.close();
    };
  }, []);

  return {
    ...state,
    isActive,
    startIngest,
    cancel,
    reset,
  };
}