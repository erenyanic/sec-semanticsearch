"use client";

/**
 * Theme provider for dark/light mode.
 *
 * "use client" at the top marks this as a **Client Component**.  In
 * Next.js App Router, components are Server Components by default —
 * they render on the server and send HTML to the browser.  But server
 * code can't access `localStorage`, `window`, or React hooks like
 * `useState`.  The "use client" directive tells Next.js to render
 * this component (and its children that use hooks) in the browser.
 *
 * How it works:
 *   1. Before first paint, the root layout's inline script
 *      (`themeInitScript` in `lib/theme.ts`) sets the `dark` class on
 *      `<html>` from `localStorage`, falling back to the OS preference
 *      via `prefers-color-scheme`.  Tailwind's `dark:` variant uses
 *      this class to swap colours.
 *   2. After hydration, this provider reads the same value and keeps
 *      the class in step when the theme changes.
 *   3. Exposes `theme` and `toggleTheme` via React Context so any
 *      component can read or change the theme.
 */

import {
  createContext,
  useContext,
  useEffect,
  useSyncExternalStore,
  type ReactNode,
} from "react";
import { THEME_STORAGE_KEY as STORAGE_KEY, type Theme } from "@/lib/theme";

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

interface ThemeContextValue {
  theme: Theme;
  toggleTheme: () => void;
}

/**
 * React Context for theme state.
 *
 * `createContext` creates a "slot" that child components can read via
 * `useContext(ThemeContext)`.  The `null!` default is never used in
 * practice because we always wrap the app in `<ThemeProvider>`.
 */
const ThemeContext = createContext<ThemeContextValue>(null!);

// ---------------------------------------------------------------------------
// Provider component
// ---------------------------------------------------------------------------

/**
 * Read the theme from localStorage (the "external store").
 *
 * `useSyncExternalStore` is the React 19-recommended way to read
 * from an external source (like localStorage) without triggering
 * the "setState in effect" warning.  It takes three arguments:
 *
 *   1. `subscribe` — called once; should set up a listener that
 *      calls the provided callback when the store changes.
 *   2. `getSnapshot` — returns the current value from the store.
 *   3. `getServerSnapshot` — returns the value during SSR (where
 *      localStorage doesn't exist).
 */
function subscribe(callback: () => void): () => void {
  // Listen for changes made in other tabs/windows.
  window.addEventListener("storage", callback);
  return () => window.removeEventListener("storage", callback);
}

function getSnapshot(): Theme {
  const stored = localStorage.getItem(STORAGE_KEY);
  if (stored === "light" || stored === "dark") return stored;
  return window.matchMedia("(prefers-color-scheme: dark)").matches
    ? "dark"
    : "light";
}

function getServerSnapshot(): Theme {
  return "light";
}

export function ThemeProvider({ children }: { children: ReactNode }) {
  // `useSyncExternalStore` reads the theme from localStorage without
  // causing a cascading render.  React calls `getSnapshot()` during
  // render and re-renders when `subscribe`'s callback fires.
  const theme = useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);

  // Keep the `dark` class on <html> in step with the theme.  This
  // effect ONLY writes to the DOM — it doesn't call setState, so it's
  // perfectly fine.
  //
  // It applies the store's value, not `theme`: during hydration `theme`
  // is the server snapshot ("light"), and applying that strips the class
  // the inline script set; the follow-up render restores it.  React
  // happens to do both before the next paint, but each toggle restyles
  // the whole document, and nothing guarantees the timing.
  useEffect(() => {
    document.documentElement.classList.toggle("dark", getSnapshot() === "dark");
  }, [theme]);

  function toggleTheme() {
    const next = theme === "light" ? "dark" : "light";
    localStorage.setItem(STORAGE_KEY, next);
    // Dispatch a storage event so `useSyncExternalStore` picks up
    // the change (the native `storage` event only fires in other
    // tabs, so we trigger a manual re-render via `dispatchEvent`).
    window.dispatchEvent(new StorageEvent("storage", { key: STORAGE_KEY }));
  }

  return (
    <ThemeContext.Provider value={{ theme, toggleTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

/**
 * Read the current theme and get a toggle function.
 *
 * Usage:
 *   const { theme, toggleTheme } = useTheme();
 */
export function useTheme(): ThemeContextValue {
  return useContext(ThemeContext);
}