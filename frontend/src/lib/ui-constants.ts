/**
 * Form styling and options shared by the search filters and the ingest form.
 *
 * Class names stay complete literal strings so the Tailwind v4 scanner
 * finds them (AD#22). A component adds its own spacing to `CHIP_BASE` by
 * concatenating another literal string, never by building a class name.
 */

/** SEC form types the pipeline accepts — `SUPPORTED_FORMS` in config/constants.py. */
export const FORM_TYPES = ["8-K", "8-K/A", "10-K", "10-K/A", "10-Q", "10-Q/A"] as const;

/** Toggle chip: shape, type and behaviour. Callers add gap and padding. */
export const CHIP_BASE =
  "inline-flex items-center rounded-lg border text-sm font-medium " +
  "transition-all cursor-pointer select-none";

export const CHIP_ACTIVE =
  "border-accent/50 bg-accent/15 text-accent hover:bg-accent/20";

export const CHIP_INACTIVE =
  "border-hairline bg-card text-fg-muted hover:border-accent/40 hover:text-fg";

/** Text, number and date inputs. */
export const INPUT_CLASS =
  "w-full rounded-lg border border-hairline bg-card px-3.5 py-2.5 text-sm text-fg " +
  "placeholder:text-fg-subtle outline-none transition-colors " +
  "focus:border-accent focus:ring-2 focus:ring-accent/20";

export const FIELD_LABEL = "text-sm font-medium text-fg-muted";
