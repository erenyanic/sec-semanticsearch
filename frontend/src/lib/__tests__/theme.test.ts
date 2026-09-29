/**
 * The pre-hydration theme script (lib/theme.ts) sets the same class
 * ThemeProvider would, before React runs.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { THEME_STORAGE_KEY, themeInitScript } from "@/lib/theme";

function prefersDark(dark: boolean) {
  vi.stubGlobal(
    "matchMedia",
    vi.fn((query: string) => ({ matches: dark && query === "(prefers-color-scheme: dark)" })),
  );
}

function run() {
  // The browser runs the string as a classic inline script.
  new Function(themeInitScript)();
}

describe("themeInitScript", () => {
  beforeEach(() => {
    localStorage.clear();
    document.documentElement.className = "";
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("applies a saved dark choice over a light OS preference", () => {
    prefersDark(false);
    localStorage.setItem(THEME_STORAGE_KEY, "dark");
    run();
    expect(document.documentElement.classList.contains("dark")).toBe(true);
  });

  it("applies a saved light choice over a dark OS preference", () => {
    prefersDark(true);
    localStorage.setItem(THEME_STORAGE_KEY, "light");
    document.documentElement.classList.add("dark");
    run();
    expect(document.documentElement.classList.contains("dark")).toBe(false);
  });

  it.each([
    [true, true],
    [false, false],
  ])("falls back to the OS preference (dark=%s)", (osDark, expected) => {
    prefersDark(osDark);
    run();
    expect(document.documentElement.classList.contains("dark")).toBe(expected);
  });

  it("ignores an unexpected stored value", () => {
    prefersDark(false);
    localStorage.setItem(THEME_STORAGE_KEY, "<img src=x>");
    run();
    expect(document.documentElement.className).toBe("");
  });

  it("does not throw when storage is blocked", () => {
    prefersDark(true);
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new DOMException("denied", "SecurityError");
    });
    expect(run).not.toThrow();
    expect(document.documentElement.className).toBe("");
  });

  it("keeps other classes on <html>", () => {
    prefersDark(true);
    document.documentElement.className = "font-sans";
    run();
    expect(document.documentElement.className).toBe("font-sans dark");
  });

  it("reads the key ThemeProvider writes", () => {
    expect(themeInitScript).toContain(JSON.stringify(THEME_STORAGE_KEY));
    expect(THEME_STORAGE_KEY).toBe("sec-search-theme");
  });
});
