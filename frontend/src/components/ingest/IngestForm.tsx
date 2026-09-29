/**
 * Form for configuring and starting an ingestion task.
 *
 * ## State ownership
 *
 * Unlike the Search page (where query + filters are lifted to the page
 * for persistence across searches), the ingest form manages its own
 * local state. The values are one-shot — submit builds an
 * `IngestRequest` and the form disappears, replaced by the progress
 * tracker. No reason to burden the page with ephemeral form state.
 *
 * ## Sub-sections
 *
 *   1. Ticker tag input  — Enter/comma adds, X removes
 *   2. Form type chips   — toggleable (10-K / 10-Q selected by default)
 *   3. Count mode radio  — latest / total / per_form
 *   4. Date filters      — collapsible year, start_date, end_date
 *   5. Submit button     — disabled until valid
 */

"use client";

import { type SubmitEvent, type KeyboardEvent, useId, useState } from "react";
import { Upload, X, ChevronDown, ChevronUp, Calendar } from "lucide-react";
import { Button, useToast } from "@/components/ui";
import type { IngestRequest } from "@/lib/types";
import {
  CHIP_ACTIVE,
  CHIP_BASE,
  CHIP_INACTIVE,
  FIELD_LABEL,
  FORM_TYPES,
  INPUT_CLASS,
} from "@/lib/ui-constants";

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

interface IngestFormProps {
  /** Called when the user submits the form with a valid request. */
  onSubmit: (request: IngestRequest) => void;
  /** True while the API call is in-flight (disables the submit button). */
  isSubmitting: boolean;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Descriptions shown beneath each count mode radio option. */
const COUNT_MODE_INFO: Record<string, { label: string; description: string }> = {
  latest: {
    label: "Latest",
    description: "1 filing per form type per ticker",
  },
  total: {
    label: "Total",
    description: "N filings shared across all form types",
  },
  per_form: {
    label: "Per form",
    description: "N filings per form type",
  },
};

// ---------------------------------------------------------------------------
// Styles (shared parts in lib/ui-constants.ts)
// ---------------------------------------------------------------------------

// Numbers and dates line up; chips are larger than the search filters'.
const INPUT = INPUT_CLASS + " tabular-nums";

const SECTION_HEADING = "text-base font-semibold text-fg";

const CHIP = CHIP_BASE + " gap-1 px-3.5 py-2 tabular-nums";

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function IngestForm({ onSubmit, isSubmitting }: IngestFormProps) {
  const { addToast } = useToast();

  // ---- Ticker tag state ----
  const [tickers, setTickers] = useState<string[]>([]);
  const [tickerInput, setTickerInput] = useState("");

  // ---- Form type chips ----
  const [formTypes, setFormTypes] = useState<Set<string>>(
    new Set(["10-K", "10-Q"]),
  );

  // ---- Count mode ----
  const [countMode, setCountMode] = useState<"latest" | "total" | "per_form">(
    "latest",
  );
  const [count, setCount] = useState("");

  // ---- Date filters (collapsible) ----
  const [showDateFilters, setShowDateFilters] = useState(false);
  const dateFiltersPanelId = useId();
  const [year, setYear] = useState("");
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");

  // When any date filter is active, count mode is suppressed — all matching
  // filings are fetched, consistent with CLI behaviour (BF-003).
  const hasDateFilter = !!(year || startDate || endDate);

  // ---- Ticker helpers ----

  function addTicker(raw: string) {
    const ticker = raw.toUpperCase().trim();
    if (!ticker) return;
    if (tickers.includes(ticker)) return;
    setTickers((prev) => [...prev, ticker]);
    setTickerInput("");
  }

  function removeTicker(ticker: string) {
    setTickers((prev) => prev.filter((t) => t !== ticker));
  }

  function handleTickerKeyDown(e: KeyboardEvent<HTMLInputElement>) {
    if (e.key === "Enter" || e.key === ",") {
      e.preventDefault();
      addTicker(tickerInput);
    }
    if (e.key === "Backspace" && !tickerInput && tickers.length > 0) {
      setTickers((prev) => prev.slice(0, -1));
    }
  }

  function handleTickerBlur() {
    if (tickerInput.trim()) {
      addTicker(tickerInput);
    }
  }

  // ---- Form type chip toggle ----

  function toggleFormType(formType: string) {
    // Guard before the state updater — calling addToast (which sets
    // ToastProvider state) inside setFormTypes's updater would violate
    // React's "no setState during another component's render" rule.
    if (formTypes.has(formType) && formTypes.size === 1) {
      addToast("info", "At least one form type must be selected.");
      return;
    }
    setFormTypes((prev) => {
      const next = new Set(prev);
      if (next.has(formType)) {
        next.delete(formType);
      } else {
        next.add(formType);
      }
      return next;
    });
  }

  // ---- Submit ----

  const canSubmit = tickers.length > 0 && formTypes.size > 0 && !isSubmitting;

  function handleSubmit(e: SubmitEvent) {
    e.preventDefault();
    if (!canSubmit) return;

    // When date filters are active, suppress count mode and fetch all matching
    // filings (count: null). This matches CLI behaviour (BF-003).
    const effectiveCountMode = hasDateFilter ? "latest" : countMode;
    const effectiveCount = hasDateFilter
      ? undefined
      : countMode === "latest"
        ? undefined
        : (Number(count) || undefined);

    const request: IngestRequest = {
      tickers,
      form_types: Array.from(formTypes),
      count_mode: effectiveCountMode,
      count: effectiveCount,
      year: Number(year) || undefined,
      start_date: startDate || undefined,
      end_date: endDate || undefined,
    };

    onSubmit(request);
  }

  return (
    <form
      onSubmit={handleSubmit}
      className="space-y-7 rounded-2xl border border-hairline bg-card/70 p-7 shadow-sm backdrop-blur-sm"
    >
      {/* ==================== TICKERS ==================== */}
      <section className="space-y-3">
        <div className={SECTION_HEADING}>Tickers</div>
        <div className="flex flex-wrap items-center gap-2 rounded-lg border border-hairline bg-card p-2.5 transition-colors focus-within:border-accent focus-within:ring-2 focus-within:ring-accent/25">
          {tickers.map((ticker) => (
            <span
              key={ticker}
              className="inline-flex items-center gap-1 rounded-md border border-accent/60 bg-accent/15 px-2.5 py-1 text-sm font-semibold tabular-nums text-accent"
            >
              {ticker}
              <button
                type="button"
                onClick={() => removeTicker(ticker)}
                className="rounded p-0.5 text-accent/80 transition-colors hover:bg-accent/20 hover:text-accent focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent"
                aria-label={`Remove ${ticker}`}
              >
                <X className="h-3.5 w-3.5" />
              </button>
            </span>
          ))}
          <input
            type="text"
            value={tickerInput}
            onChange={(e) => setTickerInput(e.target.value)}
            onKeyDown={handleTickerKeyDown}
            onBlur={handleTickerBlur}
            placeholder={
              tickers.length === 0 ? "Type a ticker and press Enter..." : ""
            }
            className="min-w-[160px] flex-1 border-0 bg-transparent p-1 text-sm tabular-nums text-fg outline-none placeholder:text-fg-subtle"
          />
        </div>
        <p className="text-sm text-fg-subtle">
          Press Enter or comma to add · Backspace removes the last tag
        </p>
      </section>

      {/* ==================== FORM TYPES ==================== */}
      <section className="space-y-3">
        <div className={SECTION_HEADING}>Form types</div>
        <div className="flex flex-wrap gap-2" role="group" aria-label="Form type filters">
          {FORM_TYPES.map((ft) => {
            const isActive = formTypes.has(ft);
            return (
              <button
                key={ft}
                type="button"
                onClick={() => toggleFormType(ft)}
                aria-pressed={isActive}
                className={`${CHIP} ${isActive ? CHIP_ACTIVE : CHIP_INACTIVE}`}
              >
                {ft}
              </button>
            );
          })}
        </div>
      </section>

      {/* ==================== COUNT MODE ==================== */}
      {hasDateFilter ? (
        <p className="text-sm text-fg-muted">
          When a date filter is active, all matching filings are fetched.
        </p>
      ) : (
        <section className="space-y-3">
          <div className={SECTION_HEADING}>Count mode</div>
          <div className="grid gap-3 sm:grid-cols-3">
            {(Object.keys(COUNT_MODE_INFO) as Array<"latest" | "total" | "per_form">).map(
              (mode) => {
                const info = COUNT_MODE_INFO[mode];
                const isSelected = countMode === mode;
                return (
                  <label
                    key={mode}
                    className={`flex cursor-pointer items-start gap-3 rounded-xl border p-4 transition-all ${
                      isSelected
                        ? "border-accent/60 bg-accent/10"
                        : "border-hairline bg-card hover:border-accent/40"
                    }`}
                  >
                    <input
                      type="radio"
                      name="countMode"
                      value={mode}
                      checked={isSelected}
                      onChange={() => setCountMode(mode)}
                      className="mt-1 accent-[var(--accent)]"
                    />
                    <div className="min-w-0">
                      <span className="block text-sm font-semibold text-fg">
                        {info.label}
                      </span>
                      <p className="mt-1 text-sm text-fg-subtle">
                        {info.description}
                      </p>
                    </div>
                  </label>
                );
              },
            )}
          </div>

          {/* Count input — visible only for total / per_form modes */}
          {countMode !== "latest" && (
            <label className="mt-2 block space-y-2">
              <span className={FIELD_LABEL}>Number of filings</span>
              <input
                type="number"
                min={1}
                max={20}
                value={count}
                onChange={(e) => setCount(e.target.value)}
                placeholder="e.g. 3"
                className={`${INPUT} w-36`}
              />
            </label>
          )}
        </section>
      )}

      {/* ==================== DATE FILTERS ==================== */}
      <section>
        <button
          type="button"
          onClick={() => setShowDateFilters(!showDateFilters)}
          aria-expanded={showDateFilters}
          aria-controls={dateFiltersPanelId}
          className="inline-flex items-center gap-2 rounded-lg text-sm font-medium text-fg-muted transition-colors hover:text-fg focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
        >
          <Calendar className="h-4 w-4" />
          Date filters
          {showDateFilters ? (
            <ChevronUp className="h-4 w-4" />
          ) : (
            <ChevronDown className="h-4 w-4" />
          )}
        </button>

        {showDateFilters && (
          <div
            id={dateFiltersPanelId}
            role="region"
            aria-label="Date filters"
            className="mt-3 grid gap-4 rounded-xl border border-hairline bg-card p-5 sm:grid-cols-3"
          >
            <label className="space-y-2">
              <span className={FIELD_LABEL}>Year</span>
              <input
                type="number"
                min={1993}
                max={new Date().getFullYear()}
                value={year}
                onChange={(e) => setYear(e.target.value)}
                placeholder="e.g. 2024"
                className={INPUT}
              />
            </label>

            <label className="space-y-2">
              <span className={FIELD_LABEL}>Start date</span>
              <input
                type="date"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
                className={INPUT}
              />
            </label>

            <label className="space-y-2">
              <span className={FIELD_LABEL}>End date</span>
              <input
                type="date"
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
                className={INPUT}
              />
            </label>
          </div>
        )}
      </section>

      {/* ==================== SUBMIT ==================== */}
      <div className="flex justify-end border-t border-hairline pt-5">
        <Button
          type="submit"
          size="lg"
          disabled={!canSubmit}
          loading={isSubmitting}
        >
          <Upload className="mr-2 h-4 w-4" />
          {tickers.length <= 1
            ? "Start Ingestion"
            : `Ingest ${tickers.length} Tickers`}
        </Button>
      </div>
    </form>
  );
}
