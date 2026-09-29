import { reducer, INITIAL_STATE, DEFAULT_PROGRESS } from "../useIngest";
import type { TaskProgress, TaskStatus, WsFilingResult } from "@/lib/types";

describe("useIngest reducer", () => {
  describe("START", () => {
    it("transitions to pending with taskId", () => {
      const next = reducer(INITIAL_STATE, { type: "START", taskId: "task-1" });
      expect(next.status).toBe("pending");
      expect(next.taskId).toBe("task-1");
      expect(next.startedAt).toBeInstanceOf(Date);
    });

    it("resets prior state", () => {
      const dirty = { ...INITIAL_STATE, status: "failed" as const, error: "old error" };
      const next = reducer(dirty, { type: "START", taskId: "task-2" });
      expect(next.error).toBeNull();
      expect(next.results).toEqual([]);
      expect(next.filingEvents).toEqual([]);
    });
  });

  describe("SNAPSHOT", () => {
    it("maps pending/running status from backend", () => {
      const progress: TaskProgress = { ...DEFAULT_PROGRESS, filings_done: 3 };
      const results: WsFilingResult[] = [];
      const next = reducer(
        { ...INITIAL_STATE, status: "pending" },
        { type: "SNAPSHOT", status: "running", progress, results },
      );
      expect(next.status).toBe("running");
      expect(next.progress.filings_done).toBe(3);
    });

    it("ignores unknown backend status strings", () => {
      const next = reducer(
        { ...INITIAL_STATE, status: "pending" },
        { type: "SNAPSHOT", status: "completed", progress: DEFAULT_PROGRESS, results: [] },
      );
      // Should keep the current state status, not map "completed"
      expect(next.status).toBe("pending");
    });
  });

  describe("STEP", () => {
    it("transitions to running and updates step info", () => {
      const next = reducer(INITIAL_STATE, {
        type: "STEP",
        step: "Fetching filings",
        step_number: 2,
        total_steps: 5,
        ticker: "AAPL",
        form_type: "10-K",
      });
      expect(next.status).toBe("running");
      expect(next.progress.step_label).toBe("Fetching filings");
      // step_number is already the 0-based stepper index (2 = Chunking),
      // the same scale as the snapshot's progress.step_index.
      expect(next.progress.step_index).toBe(2);
      expect(next.progress.step_total).toBe(5);
      expect(next.progress.current_ticker).toBe("AAPL");
      expect(next.progress.current_form_type).toBe("10-K");
    });

    it("preserves previous ticker when not provided", () => {
      const state = {
        ...INITIAL_STATE,
        progress: { ...DEFAULT_PROGRESS, current_ticker: "MSFT" },
      };
      const next = reducer(state, {
        type: "STEP",
        step: "Parsing",
        step_number: 3,
        total_steps: 5,
      });
      expect(next.progress.current_ticker).toBe("MSFT");
    });
  });

  describe("FILING_DONE", () => {
    it("increments filings_done and appends result + event", () => {
      const next = reducer(INITIAL_STATE, {
        type: "FILING_DONE",
        ticker: "AAPL",
        form_type: "10-K",
        filing_date: "2024-01-15",
        accession_number: "0001-24-000001",
        segments: 354,
        chunks: 357,
        time: 31.2,
      });
      expect(next.progress.filings_done).toBe(1);
      expect(next.results).toHaveLength(1);
      expect(next.results[0].ticker).toBe("AAPL");
      expect(next.filingEvents).toHaveLength(1);
      expect(next.filingEvents[0].type).toBe("done");
    });
  });

  describe("FILING_SKIPPED", () => {
    it("increments filings_skipped and appends skip event", () => {
      const next = reducer(INITIAL_STATE, {
        type: "FILING_SKIPPED",
        ticker: "AAPL",
        form_type: "10-K",
        reason: "Already ingested",
      });
      expect(next.progress.filings_skipped).toBe(1);
      expect(next.filingEvents).toHaveLength(1);
      expect(next.filingEvents[0].type).toBe("skipped");
      expect(next.filingEvents[0].reason).toBe("Already ingested");
    });
  });

  describe("FILING_FAILED", () => {
    it("increments filings_failed and appends fail event", () => {
      const next = reducer(INITIAL_STATE, {
        type: "FILING_FAILED",
        ticker: "AAPL",
        form_type: "10-K",
        error: "Parse error",
      });
      expect(next.progress.filings_failed).toBe(1);
      expect(next.filingEvents).toHaveLength(1);
      expect(next.filingEvents[0].type).toBe("failed");
      expect(next.filingEvents[0].error).toBe("Parse error");
    });
  });

  describe("COMPLETED", () => {
    it("sets completed status with results and summary", () => {
      const summary = { total: 5, succeeded: 3, skipped: 1, failed: 1 };
      const results: WsFilingResult[] = [
        { ticker: "AAPL", form_type: "10-K", filing_date: "2024-01-15", accession_number: "acc1", segments: 10, chunks: 12, time: 5.0 },
      ];
      const next = reducer(INITIAL_STATE, { type: "COMPLETED", results, summary });
      expect(next.status).toBe("completed");
      expect(next.results).toEqual(results);
      expect(next.summary).toEqual(summary);
      expect(next.completedAt).toBeInstanceOf(Date);
    });
  });

  describe("FAILED", () => {
    it("sets failed status with error message", () => {
      const next = reducer(INITIAL_STATE, { type: "FAILED", error: "GPU OOM" });
      expect(next.status).toBe("failed");
      expect(next.error).toBe("GPU OOM");
      expect(next.completedAt).toBeInstanceOf(Date);
    });

    it("concatenates error and details when details present", () => {
      const next = reducer(INITIAL_STATE, {
        type: "FAILED",
        error: "IngestError",
        details: "CUDA out of memory",
      });
      expect(next.error).toBe("IngestError: CUDA out of memory");
    });
  });

  describe("CANCELLED", () => {
    it("builds summary from progress counters", () => {
      const state = {
        ...INITIAL_STATE,
        status: "running" as const,
        progress: {
          ...DEFAULT_PROGRESS,
          filings_done: 2,
          filings_skipped: 1,
          filings_failed: 0,
        },
      };
      const next = reducer(state, { type: "CANCELLED" });
      expect(next.status).toBe("cancelled");
      expect(next.summary).toEqual({ total: 3, succeeded: 2, skipped: 1, failed: 0 });
      expect(next.completedAt).toBeInstanceOf(Date);
    });
  });

  describe("ERROR", () => {
    it("transitions to failed with error", () => {
      const next = reducer(INITIAL_STATE, { type: "ERROR", error: "WS disconnected" });
      expect(next.status).toBe("failed");
      expect(next.error).toBe("WS disconnected");
    });
  });

  describe("RESUME", () => {
    it("reconstructs state from REST API task status", () => {
      const taskStatus: TaskStatus = {
        task_id: "task-99",
        status: "running",
        tickers: ["AAPL"],
        form_types: ["10-K"],
        progress: { ...DEFAULT_PROGRESS, filings_done: 1, filings_total: 3 },
        results: [
          {
            ticker: "AAPL",
            form_type: "10-K",
            filing_date: "2024-01-15",
            accession_number: "acc1",
            segment_count: 100,
            chunk_count: 110,
            duration_seconds: 12.5,
          },
        ],
        started_at: "2024-06-01T12:00:00Z",
      };
      const next = reducer(INITIAL_STATE, { type: "RESUME", taskStatus });
      expect(next.taskId).toBe("task-99");
      expect(next.status).toBe("running");
      expect(next.results).toHaveLength(1);
      expect(next.results[0].segments).toBe(100); // mapped from segment_count
      expect(next.results[0].time).toBe(12.5); // mapped from duration_seconds
      expect(next.filingEvents).toHaveLength(1);
      expect(next.filingEvents[0].type).toBe("done");
      expect(next.startedAt).toEqual(new Date("2024-06-01T12:00:00Z"));
    });

    it("maps completed backend status to idle (not resumable)", () => {
      const taskStatus: TaskStatus = {
        task_id: "task-100",
        status: "completed",
        tickers: ["AAPL"],
        form_types: ["10-K"],
        progress: DEFAULT_PROGRESS,
        results: [],
      };
      const next = reducer(INITIAL_STATE, { type: "RESUME", taskStatus });
      expect(next.status).toBe("idle");
    });
  });

  describe("RESET", () => {
    it("returns initial state", () => {
      const dirty = {
        ...INITIAL_STATE,
        taskId: "task-1",
        status: "completed" as const,
        error: "some error",
      };
      const next = reducer(dirty, { type: "RESET" });
      expect(next).toEqual(INITIAL_STATE);
    });
  });

  describe("unknown action", () => {
    it("returns state unchanged", () => {
      // @ts-expect-error — testing runtime safety for unknown action
      const next = reducer(INITIAL_STATE, { type: "UNKNOWN_ACTION" });
      expect(next).toBe(INITIAL_STATE);
    });
  });

  describe("cumulative event tracking", () => {
    it("accumulates mixed events in order", () => {
      let state = INITIAL_STATE;
      state = reducer(state, {
        type: "FILING_DONE",
        ticker: "AAPL", form_type: "10-K", filing_date: "2024-01-15",
        accession_number: "acc1", segments: 10, chunks: 12, time: 5.0,
      });
      state = reducer(state, {
        type: "FILING_SKIPPED",
        ticker: "MSFT", form_type: "10-Q", reason: "Duplicate",
      });
      state = reducer(state, {
        type: "FILING_FAILED",
        ticker: "GOOG", form_type: "10-K", error: "Parse error",
      });

      expect(state.filingEvents).toHaveLength(3);
      expect(state.filingEvents.map((e) => e.type)).toEqual(["done", "skipped", "failed"]);
      expect(state.progress.filings_done).toBe(1);
      expect(state.progress.filings_skipped).toBe(1);
      expect(state.progress.filings_failed).toBe(1);
    });
  });
  describe("reconnect replays (seq)", () => {
    const done = (n: number, seq?: number) => ({
      type: "FILING_DONE" as const,
      ticker: "AAPL",
      form_type: "10-K",
      filing_date: "2024-11-01",
      accession_number: `acc-${n}`,
      segments: 1,
      chunks: 1,
      time: 0.1,
      seq,
    });
    const result = (n: number): WsFilingResult => ({
      ticker: "AAPL",
      form_type: "10-K",
      filing_date: "2024-11-01",
      accession_number: `acc-${n}`,
      segments: 1,
      chunks: 1,
      time: 0.1,
    });
    const snapshot = (doneCount: number, seq?: number) => ({
      type: "SNAPSHOT" as const,
      status: "running",
      progress: { ...DEFAULT_PROGRESS, filings_done: doneCount, filings_total: 5 },
      results: Array.from({ length: doneCount }, (_, i) => result(i + 1)),
      seq,
    });

    it("stores the snapshot's seq", () => {
      const next = reducer(INITIAL_STATE, snapshot(2, 7));
      expect(next.lastSeq).toBe(7);
    });

    it("does not count a replayed filing_done again", () => {
      let state = reducer(INITIAL_STATE, snapshot(2, 2));
      state = reducer(state, done(2, 2));
      expect(state.progress.filings_done).toBe(2);
      expect(state.results).toHaveLength(2);
      // The event row was missing (no prior connection) and is added once.
      expect(state.filingEvents.map((e) => e.accession_number)).toEqual(["acc-2"]);
      state = reducer(state, done(2, 2));
      expect(state.filingEvents).toHaveLength(1);
    });

    it("counts a message newer than the snapshot and advances lastSeq", () => {
      let state = reducer(INITIAL_STATE, snapshot(2, 2));
      state = reducer(state, done(3, 3));
      expect(state.progress.filings_done).toBe(3);
      expect(state.results).toHaveLength(3);
      expect(state.lastSeq).toBe(3);
    });

    it("does not count replayed skips or failures", () => {
      let state = reducer(INITIAL_STATE, {
        ...snapshot(0, 5),
        progress: { ...DEFAULT_PROGRESS, filings_skipped: 1, filings_failed: 1, filings_done: 2 },
      });
      state = reducer(state, {
        type: "FILING_SKIPPED",
        ticker: "A",
        form_type: "10-K",
        reason: "duplicate",
        accession_number: "acc-s",
        seq: 3,
      });
      state = reducer(state, {
        type: "FILING_FAILED",
        ticker: "A",
        form_type: "10-K",
        error: "x",
        accession_number: "acc-f",
        seq: 4,
      });
      expect(state.progress.filings_skipped).toBe(1);
      expect(state.progress.filings_failed).toBe(1);
      expect(state.filingEvents.map((e) => e.type)).toEqual(["skipped", "failed"]);
    });

    it("ignores a replayed step so the stepper cannot move backwards", () => {
      let state = reducer(INITIAL_STATE, {
        ...snapshot(0, 10),
        progress: { ...DEFAULT_PROGRESS, step_index: 3, step_label: "Embedding" },
      });
      state = reducer(state, { type: "STEP", step: "Parsing", step_number: 1, total_steps: 5, seq: 9 });
      expect(state.progress.step_index).toBe(3);
      state = reducer(state, { type: "STEP", step: "Complete", step_number: 4, total_steps: 5, seq: 11 });
      expect(state.progress.step_index).toBe(4);
      expect(state.lastSeq).toBe(11);
    });

    it("adds a replayed eviction once", () => {
      const eviction = {
        type: "EVICTION" as const,
        filings_evicted: 501,
        chunks_evicted: 9000,
        tickers_affected: ["MSFT"],
        seq: 1,
      };
      let state = reducer(INITIAL_STATE, snapshot(0, 1));
      state = reducer(state, eviction);
      state = reducer(state, eviction);
      expect(state.filingEvents.filter((e) => e.type === "eviction")).toHaveLength(1);
    });

    it("applies messages without seq as before (older server)", () => {
      let state = reducer(INITIAL_STATE, snapshot(2));
      state = reducer(state, done(3));
      expect(state.progress.filings_done).toBe(3);
    });

    it("reload mid-ingest: counters match the server exactly (audit repro)", () => {
      // 5-filing ingest; the tab closed after filing 2 and filings 3–4
      // finished with no client connected. The page reloads.
      const taskStatus: TaskStatus = {
        task_id: "t1",
        status: "running",
        tickers: ["AAPL"],
        form_types: ["10-K"],
        progress: { ...DEFAULT_PROGRESS, filings_done: 4, filings_total: 5 },
        results: [1, 2, 3, 4].map((n) => ({
          ticker: "AAPL",
          form_type: "10-K",
          filing_date: "2024-11-01",
          accession_number: `acc-${n}`,
          segment_count: 1,
          chunk_count: 1,
          duration_seconds: 0.1,
        })),
        error: null,
        started_at: null,
        completed_at: null,
      };
      let state = reducer(INITIAL_STATE, { type: "RESUME", taskStatus });
      state = reducer(state, snapshot(4, 4));
      state = reducer(state, done(3, 3)); // replay
      state = reducer(state, done(4, 4)); // replay
      state = reducer(state, done(5, 5)); // new
      expect(state.progress.filings_done).toBe(5);
      expect(state.results.map((r) => r.accession_number)).toEqual([
        "acc-1",
        "acc-2",
        "acc-3",
        "acc-4",
        "acc-5",
      ]);
      expect(state.filingEvents.filter((e) => e.type === "done")).toHaveLength(5);
    });
  });
});