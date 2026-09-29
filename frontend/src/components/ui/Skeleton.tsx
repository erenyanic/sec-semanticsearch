/**
 * Skeleton loading placeholder — shows a shimmer animation where
 * content will appear.
 *
 * One export, `Skeleton`: a single block sized by its classes.
 *
 * ## How the shimmer works
 *
 * The element has a linear-gradient background with three colour stops:
 *   transparent → semi-white → transparent
 *
 * `background-size: 200% 100%` makes the gradient twice as wide as
 * the element, so the bright band can sweep from right to left.
 * The `shimmer` keyframe (defined in `globals.css`) animates
 * `background-position` from `200% 0` → `-200% 0`.
 *
 * ## Why no `"use client"`?
 *
 * Skeletons are pure presentational — no hooks, no state, no browser
 * APIs.  They render server-side (or client-side when placed inside
 * a Client Component parent) with zero JavaScript overhead.
 */

// ---------------------------------------------------------------------------
// Skeleton block
// ---------------------------------------------------------------------------

interface SkeletonProps {
  /** Additional classes for width/height (e.g. `"h-6 w-32"`). */
  className?: string;
}

/**
 * A single skeleton block with a shimmer animation.
 *
 * The caller controls dimensions via `className`:
 *
 * ```tsx
 * <Skeleton className="h-6 w-32" />          // badge placeholder
 * <Skeleton className="h-10 w-full" />        // full-width bar
 * <Skeleton className="h-40 w-full rounded-lg" /> // card placeholder
 * ```
 */
export function Skeleton({ className }: SkeletonProps) {
  return (
    <div
      className={[
        // Base shape: rounded, muted surface
        "rounded-md bg-card",
        // Shimmer gradient overlay
        "bg-gradient-to-r from-transparent via-surface to-transparent",
        // Animation: sweep the gradient across (defined in globals.css)
        "bg-[length:200%_100%] [animation:shimmer_1.5s_ease-in-out_infinite]",
        className,
      ]
        .filter(Boolean)
        .join(" ")}
    />
  );
}
