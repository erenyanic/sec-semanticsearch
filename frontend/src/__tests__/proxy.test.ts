// @vitest-environment node
/**
 * The proxy gives every page request its own CSP nonce, hands it to the
 * renderer, and runs only for pages.
 */
import { describe, expect, it } from "vitest";
import { NextRequest } from "next/server";
import { unstable_doesMiddlewareMatch } from "next/experimental/testing/server";

import { config, proxy } from "../proxy";

function nonceOf(policy: string | null): string {
  const match = policy?.match(/'nonce-([^']+)'/);
  expect(match).not.toBeNull();
  return match![1];
}

describe("proxy", () => {
  it("sends the policy with the page and the same nonce to the renderer", () => {
    const response = proxy(new NextRequest("http://localhost/search"));
    const policy = response.headers.get("content-security-policy");
    const nonce = nonceOf(policy);

    // NextResponse.next({ request }) forwards overridden request headers
    // as x-middleware-request-* on the response.
    expect(response.headers.get("x-middleware-request-x-nonce")).toBe(nonce);
    expect(response.headers.get("x-middleware-request-content-security-policy")).toBe(policy);
  });

  it("uses a new nonce for every request", () => {
    const nonces = new Set(
      Array.from({ length: 20 }, () =>
        nonceOf(proxy(new NextRequest("http://localhost/")).headers.get("content-security-policy")),
      ),
    );
    expect(nonces.size).toBe(20);
  });

  it("ignores a nonce the client sends", () => {
    const response = proxy(
      new NextRequest("http://localhost/", { headers: { "x-nonce": "attacker" } }),
    );
    expect(response.headers.get("x-middleware-request-x-nonce")).not.toBe("attacker");
    expect(response.headers.get("content-security-policy")).not.toContain("attacker");
  });
});

describe("proxy matcher", () => {
  const matches = (url: string, headers: Record<string, string> = {}) =>
    unstable_doesMiddlewareMatch({ config, url, headers });

  it.each(["/", "/search", "/ingest", "/filings", "/admin"])("runs for page %s", (path) => {
    expect(matches(path)).toBe(true);
  });

  it.each([
    "/api/admin/session",
    "/api/search",
    "/_next/static/chunks/main.js",
    "/_next/image?url=x",
    "/favicon.ico",
  ])("skips %s", (path) => {
    expect(matches(path)).toBe(false);
  });

  it("skips prefetches", () => {
    expect(matches("/search", { "next-router-prefetch": "1" })).toBe(false);
    expect(matches("/search", { purpose: "prefetch" })).toBe(false);
  });
});
