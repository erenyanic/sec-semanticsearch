/**
 * Small coloured label — for status indicators, form types, and tags.
 *
 * Five colour variants:
 *   - **gray**   — neutral, default
 *   - **blue**   — active / running / informational
 *   - **green**  — success / completed
 *   - **amber**  — warning / pending
 *   - **red**    — error / failed / cancelled
 *
 * The colour pattern (`bg-{color}-50 text-{color}-700` in light mode,
 * `dark:bg-{color}-950 dark:text-{color}-300` in dark mode) mirrors
 * the active nav link styling established in Navbar.tsx.
 *
 * No `"use client"` needed — pure class computation and rendering.
 */

import { type ReactNode } from "react";

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface BadgeProps {
  /** Colour variant. Defaults to `"gray"`. */
  variant?: "gray" | "blue" | "green" | "amber" | "red";
  /** Badge content (usually short text). */
  children: ReactNode;
  /** Additional CSS classes. */
  className?: string;
}

// ---------------------------------------------------------------------------
// Class mappings
// ---------------------------------------------------------------------------

/**
 * Base classes shared by all variants: inline layout, pill shape,
 * compact padding, and small bold text.
 */
const BASE =
  "inline-flex items-center rounded-md border px-2 py-0.5 text-xs font-medium tabular-nums";

/**
 * Each variant uses a tinted background + border + matching text tone,
 * driven by semantic CSS variables so light/dark mode swap cleanly.
 */
const VARIANT_CLASSES: Record<NonNullable<BadgeProps["variant"]>, string> = {
  gray: "border-hairline bg-surface text-fg-muted",
  blue: "border-accent/40 bg-accent/10 text-accent",
  green: "border-pos/40 bg-pos/10 text-pos",
  amber: "border-warn/40 bg-warn/10 text-warn",
  red: "border-neg/40 bg-neg/10 text-neg",
};

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function Badge({ variant = "gray", children, className }: BadgeProps) {
  const classes = [BASE, VARIANT_CLASSES[variant], className]
    .filter(Boolean)
    .join(" ");

  return <span className={classes}>{children}</span>;
}
