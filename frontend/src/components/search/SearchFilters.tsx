/**
 * Filter panel for search refinement.
 *
 * ## Two modes
 *
 *   - **Collapsible** (default) — "Filters" toggle button reveals a
 *     panel. Used in tight layouts and preserved so existing tests
 *     (which click the toggle button) continue to pass.
 *
 *   - **Rail** (`alwaysOpen`) — panel is always visible, with no
 *     toggle button. Used by the Search page's two-column layout,
 *     where the filter rail lives in a dedicated column.
 *
 * Both modes share the same `<FilterControls>` sub-component, so
 * restyling or adding a filter field automatically propagates.
 *
 * ## State ownership
 *
 * The **parent page** owns all filter values. This component is
 * presentational — it renders inputs and calls `onFiltersChange`
 * when any value changes. This makes the parent the single source
 * of truth for the search request, which simplifies debugging.
 *
 * ## Multi-select filters
 *
 * Ticker and form type use chip-based multi-select. Accession
 * numbers are entered as comma-separated text. All three filter
 * fields map to arrays in the API request.
 */

"use client";

import { useId, useState, type ReactNode } from "react";
import { SlidersHorizontal, X } from "lucide-react";
import { Badge } from "@/components/ui";
import {
  CHIP_ACTIVE,
  CHIP_BASE,
  CHIP_INACTIVE,
  FIELD_LABEL,
  FORM_TYPES,
  INPUT_CLASS,
} from "@/lib/ui-constants";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface SearchFilterValues {
  tickers: string[];
  formTypes: string[];
  topK: number;
  minSimilarity: number;
  accessionNumbers: string[];
  startDate: string;
  endDate: string;
}

interface SearchFiltersProps {
  filters: SearchFilterValues;
  onFiltersChange: (filters: SearchFilterValues) => void;
  /** Available tickers for the dropdown (from status endpoint). */
  availableTickers: string[];
  /**
   * When true, renders the filter panel inline (no toggle button).
   * Used by the two-column Search page layout. Default: false
   * (collapsible, backwards-compatible with existing tests).
   */
  alwaysOpen?: boolean;
}

// ---------------------------------------------------------------------------
// Defaults
// ---------------------------------------------------------------------------

export const DEFAULT_FILTERS: SearchFilterValues = {
  tickers: [],
  formTypes: [],
  topK: 5,
  minSimilarity: 0,
  accessionNumbers: [],
  startDate: "",
  endDate: "",
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Count how many filters differ from their defaults. */
export function countActiveFilters(filters: SearchFilterValues): number {
  let count = 0;
  if (filters.tickers.length > 0) count++;
  if (filters.formTypes.length > 0) count++;
  if (filters.topK !== DEFAULT_FILTERS.topK) count++;
  if (filters.minSimilarity !== DEFAULT_FILTERS.minSimilarity) count++;
  if (filters.accessionNumbers.length > 0) count++;
  if (filters.startDate) count++;
  if (filters.endDate) count++;
  return count;
}

// ---------------------------------------------------------------------------
// Styles (shared parts in lib/ui-constants.ts)
// ---------------------------------------------------------------------------

// Filter chips are compact.
const CHIP = CHIP_BASE + " gap-1.5 px-3 py-1.5";

// Group heading style (used in rail mode)
const GROUP_HEADING =
  "mb-3 text-sm font-semibold text-fg-muted";

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export function SearchFilters({
  filters,
  onFiltersChange,
  availableTickers,
  alwaysOpen = false,
}: SearchFiltersProps) {
  const [isOpen, setIsOpen] = useState(false);
  const panelId = useId();
  const activeCount = countActiveFilters(filters);

  /** Update a single field, preserving the rest via object spread. */
  function update(patch: Partial<SearchFilterValues>) {
    onFiltersChange({ ...filters, ...patch });
  }

  // -------------------------------------------------------------------
  // Rail mode — inline, always visible, grouped sections
  // -------------------------------------------------------------------
  if (alwaysOpen) {
    return (
      <aside
        aria-label="Search filters"
        className="sticky top-20 space-y-6 rounded-2xl border border-hairline bg-card/70 p-6 shadow-sm backdrop-blur-sm"
      >
        {/* Header: active count + clear */}
        <div className="flex items-center justify-between border-b border-hairline pb-4">
          <div className="flex items-center gap-2.5">
            <SlidersHorizontal className="h-4 w-4 text-fg-muted" />
            <span className="text-base font-semibold text-fg">Filters</span>
            {activeCount > 0 && (
              <span className="ml-1 inline-flex h-5 min-w-5 items-center justify-center rounded-full bg-accent px-1.5 text-xs font-semibold tabular-nums text-accent-fg">
                {activeCount}
              </span>
            )}
          </div>
          {activeCount > 0 && (
            <button
              type="button"
              onClick={() => onFiltersChange(DEFAULT_FILTERS)}
              className="flex items-center gap-1 rounded-md text-sm font-medium text-fg-muted transition-colors hover:text-fg focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
            >
              <X className="h-3.5 w-3.5" />
              Clear
            </button>
          )}
        </div>

        <FilterControls
          filters={filters}
          onChange={update}
          availableTickers={availableTickers}
          layout="rail"
        />
      </aside>
    );
  }

  // -------------------------------------------------------------------
  // Collapsible mode — legacy toggle-button layout (preserves tests)
  // -------------------------------------------------------------------
  return (
    <div>
      {/* ---- Toggle button ---- */}
      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={() => setIsOpen(!isOpen)}
          aria-expanded={isOpen}
          aria-controls={panelId}
          className="inline-flex items-center gap-2 rounded-lg border border-hairline bg-card px-3.5 py-2 text-sm font-medium text-fg transition-colors hover:border-accent/40 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
        >
          <SlidersHorizontal className="h-4 w-4" />
          Filters
          {activeCount > 0 && (
            <Badge variant="blue" className="ml-1">
              {activeCount}
            </Badge>
          )}
        </button>
        {activeCount > 0 && (
          <button
            type="button"
            onClick={() => onFiltersChange(DEFAULT_FILTERS)}
            className="inline-flex items-center gap-1 rounded-md px-2.5 py-1.5 text-sm font-medium text-fg-muted transition-colors hover:bg-card hover:text-fg focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
          >
            <X className="mr-0.5 h-4 w-4" />
            Clear
          </button>
        )}
      </div>

      {/* ---- Collapsible filter panel ---- */}
      {isOpen && (
        <div
          id={panelId}
          role="region"
          aria-label="Search filters"
          className="mt-4 rounded-2xl border border-hairline bg-card/70 p-5 backdrop-blur-sm"
        >
          <FilterControls
            filters={filters}
            onChange={update}
            availableTickers={availableTickers}
            layout="inline"
          />
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Shared filter controls (used by both modes)
// ---------------------------------------------------------------------------

interface FilterControlsProps {
  filters: SearchFilterValues;
  onChange: (patch: Partial<SearchFilterValues>) => void;
  availableTickers: string[];
  layout: "rail" | "inline";
}

function FilterControls({
  filters,
  onChange,
  availableTickers,
  layout,
}: FilterControlsProps) {
  function toggleTicker(ticker: string) {
    const next = filters.tickers.includes(ticker)
      ? filters.tickers.filter((t) => t !== ticker)
      : [...filters.tickers, ticker];
    onChange({ tickers: next });
  }

  function toggleFormType(formType: string) {
    const next = filters.formTypes.includes(formType)
      ? filters.formTypes.filter((f) => f !== formType)
      : [...filters.formTypes, formType];
    onChange({ formTypes: next });
  }

  function handleAccessionChange(value: string) {
    if (!value.trim()) {
      onChange({ accessionNumbers: [] });
      return;
    }
    const parsed = value
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean);
    onChange({ accessionNumbers: parsed });
  }

  const inlineGrid = layout === "inline";

  return (
    <div className={inlineGrid ? "space-y-4" : "space-y-6"}>
      {/* ==================== SCOPE ==================== */}
      <Section heading="Scope" showHeading={layout === "rail"}>
        {availableTickers.length > 0 && (
          <ChipGroup label="Tickers">
            <div className="flex flex-wrap gap-1.5" role="group" aria-label="Ticker filters">
              {availableTickers.map((t) => {
                const isActive = filters.tickers.includes(t);
                return (
                  <button
                    key={t}
                    type="button"
                    onClick={() => toggleTicker(t)}
                    className={`${CHIP} ${isActive ? CHIP_ACTIVE : CHIP_INACTIVE}`}
                    aria-pressed={isActive}
                  >
                    {t}
                    {isActive && <X className="h-3 w-3" />}
                  </button>
                );
              })}
            </div>
          </ChipGroup>
        )}

        <ChipGroup label="Form type">
          <div className="flex flex-wrap gap-1.5" role="group" aria-label="Form type filters">
            {FORM_TYPES.map((f) => {
              const isActive = filters.formTypes.includes(f);
              return (
                <button
                  key={f}
                  type="button"
                  onClick={() => toggleFormType(f)}
                  className={`${CHIP} ${isActive ? CHIP_ACTIVE : CHIP_INACTIVE}`}
                  aria-pressed={isActive}
                >
                  {f}
                  {isActive && <X className="h-3 w-3" />}
                </button>
              );
            })}
          </div>
        </ChipGroup>

        <Field label="Accession number(s)">
          <input
            type="text"
            value={filters.accessionNumbers.join(", ")}
            onChange={(e) => handleAccessionChange(e.target.value)}
            placeholder="e.g. 0000320193-24-000123, 0000320193-24-000456"
            className={INPUT_CLASS}
          />
        </Field>
      </Section>

      {/* ==================== RELEVANCE ==================== */}
      <Section heading="Relevance" showHeading={layout === "rail"}>
        <div className={inlineGrid ? "grid gap-4 sm:grid-cols-2" : "space-y-3"}>
          <Field label="Results (top K)">
            <input
              type="number"
              min={1}
              max={100}
              value={filters.topK}
              onChange={(e) =>
                onChange({
                  topK: Math.max(1, Math.min(100, Number(e.target.value) || 1)),
                })
              }
              className={INPUT_CLASS}
            />
          </Field>

          <Field label={`Min similarity: ${Math.round(filters.minSimilarity * 100)}%`}>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={filters.minSimilarity}
              onChange={(e) => onChange({ minSimilarity: Number(e.target.value) })}
              className="w-full accent-[var(--accent)]"
            />
          </Field>
        </div>
      </Section>

      {/* ==================== DATE RANGE ==================== */}
      <Section heading="Date range" showHeading={layout === "rail"}>
        <div className={inlineGrid ? "grid gap-4 sm:grid-cols-2" : "grid grid-cols-2 gap-3"}>
          <Field label="From date">
            <input
              type="date"
              value={filters.startDate}
              onChange={(e) => onChange({ startDate: e.target.value })}
              className={INPUT_CLASS}
            />
          </Field>

          <Field label="To date">
            <input
              type="date"
              value={filters.endDate}
              onChange={(e) => onChange({ endDate: e.target.value })}
              className={INPUT_CLASS}
            />
          </Field>
        </div>
      </Section>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Tiny layout helpers
// ---------------------------------------------------------------------------

function Section({
  heading,
  showHeading,
  children,
}: {
  heading: string;
  showHeading: boolean;
  children: ReactNode;
}) {
  return (
    <div>
      {showHeading && <div className={GROUP_HEADING}>{heading}</div>}
      <div className="space-y-3">{children}</div>
    </div>
  );
}

function Field({ label, children }: { label: string; children: ReactNode }) {
  // Used for real form inputs (number, text, date, range) — the
  // native `<label>` wrapping gives implicit labelling for a11y.
  return (
    <label className="block space-y-1.5">
      <span className={FIELD_LABEL}>{label}</span>
      {children}
    </label>
  );
}

function ChipGroup({ label, children }: { label: string; children: ReactNode }) {
  // Used for multi-button chip groups. A `<label>` would collapse
  // every chip's accessible name into the group label, so we use a
  // plain `<div>` with a visible caption instead.
  return (
    <div className="space-y-1.5">
      <span className={FIELD_LABEL}>{label}</span>
      {children}
    </div>
  );
}
