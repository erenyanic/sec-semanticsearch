import { NextResponse, type NextRequest } from "next/server";

import { buildContentSecurityPolicy, generateNonce } from "@/lib/csp";

/**
 * Per-request CSP nonce for HTML pages (see `lib/csp.ts`).
 *
 * The nonce travels to the renderer in the request's
 * `Content-Security-Policy` header (Next.js reads it from there and tags
 * its own scripts) and in `x-nonce` for the layout's inline theme script;
 * the browser gets the same policy on the response.
 */
export function proxy(request: NextRequest): NextResponse {
  const nonce = generateNonce();
  const policy = buildContentSecurityPolicy(nonce, process.env.NODE_ENV === "development");

  const requestHeaders = new Headers(request.headers);
  requestHeaders.set("x-nonce", nonce);
  requestHeaders.set("Content-Security-Policy", policy);

  const response = NextResponse.next({ request: { headers: requestHeaders } });
  response.headers.set("Content-Security-Policy", policy);
  return response;
}

export const config = {
  matcher: [
    {
      // Pages only: not API routes, build assets or the favicon, and not
      // prefetches (their payload is used by a page that already has a nonce).
      source: "/((?!api|_next/static|_next/image|favicon.ico).*)",
      missing: [
        { type: "header", key: "next-router-prefetch" },
        { type: "header", key: "purpose", value: "prefetch" },
      ],
    },
  ],
};
