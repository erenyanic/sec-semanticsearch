/**
 * Tests for the page Content-Security-Policy (lib/csp.ts).
 */
import { describe, it, expect } from "vitest";

import { buildContentSecurityPolicy, generateNonce } from "@/lib/csp";

function directives(policy: string): Map<string, string> {
  return new Map(
    policy.split("; ").map((d) => {
      const [name, ...values] = d.split(" ");
      return [name, values.join(" ")];
    }),
  );
}

describe("buildContentSecurityPolicy", () => {
  it("allows scripts only by nonce, with trust passed to the chunks they load", () => {
    const scripts = directives(buildContentSecurityPolicy("abc123", false)).get("script-src");
    expect(scripts).toBe("'self' 'nonce-abc123' 'strict-dynamic'");
  });

  it("never allows inline or eval'd scripts in production", () => {
    const scripts = directives(buildContentSecurityPolicy("n", false)).get("script-src")!;
    expect(scripts).not.toContain("'unsafe-inline'");
    expect(scripts).not.toContain("'unsafe-eval'");
  });

  it("adds 'unsafe-eval' in development only", () => {
    const scripts = directives(buildContentSecurityPolicy("n", true)).get("script-src")!;
    expect(scripts).toContain("'unsafe-eval'");
    expect(scripts).not.toContain("'unsafe-inline'");
  });

  it("keeps the framing, base, form and plugin restrictions", () => {
    const policy = directives(buildContentSecurityPolicy("n", false));
    expect(policy.get("default-src")).toBe("'self'");
    expect(policy.get("frame-ancestors")).toBe("'none'");
    expect(policy.get("base-uri")).toBe("'self'");
    expect(policy.get("form-action")).toBe("'self'");
    expect(policy.get("object-src")).toBe("'none'");
  });

  it("loads nothing from third-party origins", () => {
    const policy = buildContentSecurityPolicy("n", false);
    expect(policy).not.toMatch(/https?:/);
  });
});

describe("generateNonce", () => {
  it("is 128 bits of base64", () => {
    const nonce = generateNonce();
    expect(nonce).toMatch(/^[A-Za-z0-9+/]{22}==$/);
    expect(atob(nonce)).toHaveLength(16);
  });

  it("is new every time", () => {
    const nonces = new Set(Array.from({ length: 100 }, generateNonce));
    expect(nonces.size).toBe(100);
  });
});
