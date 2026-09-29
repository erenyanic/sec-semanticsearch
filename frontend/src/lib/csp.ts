/**
 * Content-Security-Policy for the app's HTML pages.
 *
 * Next.js inlines its RSC payload as `<script>` tags, so a policy without
 * `'unsafe-inline'` must carry a per-request nonce: `proxy.ts` generates
 * one, and Next.js adds it to every script it renders. `'strict-dynamic'`
 * extends that trust to the chunks those scripts load. Styles keep
 * `'unsafe-inline'`: Tailwind and the chart library set inline style
 * attributes, which nonces cannot cover.
 *
 * nginx and FastAPI set their own policy for `/api/` responses only.
 */
export function buildContentSecurityPolicy(nonce: string, isDev: boolean): string {
  return [
    "default-src 'self'",
    // React's dev tooling needs eval; production never does.
    `script-src 'self' 'nonce-${nonce}' 'strict-dynamic'${isDev ? " 'unsafe-eval'" : ""}`,
    "style-src 'self' 'unsafe-inline'",
    "img-src 'self' data: blob:",
    "font-src 'self' data:",
    "connect-src 'self' ws: wss:",
    "object-src 'none'",
    "base-uri 'self'",
    "form-action 'self'",
    "frame-ancestors 'none'",
  ].join("; ");
}

/** A fresh, unguessable nonce (128 bits, base64). */
export function generateNonce(): string {
  const bytes = new Uint8Array(16);
  crypto.getRandomValues(bytes);
  return btoa(String.fromCharCode(...bytes));
}
