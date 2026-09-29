/**
 * Theme storage key and the pre-hydration theme script.
 *
 * Kept out of `ThemeProvider.tsx`: that module is a Client Component, and
 * a Server Component (the root layout) importing a plain value from a
 * "use client" module gets a client reference, not the value.
 */

export type Theme = "light" | "dark";

/** localStorage key holding the user's explicit choice. */
export const THEME_STORAGE_KEY = "sec-search-theme";

/**
 * Inline script run from `<head>` before the body paints: applies the
 * saved theme (or the OS preference) to `<html>`, so dark-mode users do
 * not see a light frame until React hydrates. Same rule as
 * `ThemeProvider`'s snapshot. localStorage can throw (blocked storage);
 * the page then starts light, as before.
 */
export const themeInitScript = `(function(){try{var t=localStorage.getItem(${JSON.stringify(
  THEME_STORAGE_KEY,
)});if(t!=="light"&&t!=="dark")t=matchMedia("(prefers-color-scheme: dark)").matches?"dark":"light";document.documentElement.classList.toggle("dark",t==="dark")}catch(e){}})()`;
