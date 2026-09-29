import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { connection } from "next/server";
import { Providers } from "@/components/Providers";
import { Navbar, Footer, WelcomeGate, DemoBanner } from "@/components/layout";
import "./globals.css";

/**
 * Google's Geist fonts — loaded via Next.js's built-in font optimiser.
 *
 * `next/font/google` downloads the font at build time and serves it
 * locally (no external requests to Google Fonts at runtime).  The
 * `variable` option creates a CSS custom property (e.g. `--font-geist-sans`)
 * that Tailwind can reference.
 */
const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

/**
 * Page metadata — used by Next.js for the `<title>` and `<meta>`
 * tags in the HTML `<head>`.  Individual pages can override this.
 */
export const metadata: Metadata = {
  title: "SEC Semantic Search",
  description:
    "Semantic search over SEC filings (8-K, 10-K, 10-Q) using vector similarity",
};

/**
 * Root layout — wraps every page in the application.
 *
 * Structure:
 *   <html>
 *     <body>
 *       <Providers>                 ← React Query + Theme context
 *         <a>Skip to content</a>   ← Keyboard a11y (WCAG 2.1 AA)
 *         <Navbar />                ← Always visible
 *         <main id>{page}</main>   ← Current page content
 *         <Footer />                ← Always visible
 *       </Providers>
 *     </body>
 *   </html>
 *
 * ## Skip-to-content link
 *
 * WCAG 2.1 Success Criterion 2.4.1 requires a way to bypass repeated
 * navigation. The skip link is visually hidden (`sr-only`) but becomes
 * visible when focused via keyboard Tab. It jumps focus to `<main>`,
 * skipping the navbar entirely.
 *
 * This is a Server Component (no "use client").  The `<Providers>`
 * wrapper handles the client-side boundary.
 *
 * ## Dynamic rendering
 *
 * Every page is rendered per request: the CSP nonce (`proxy.ts`) is new
 * for each response, and Next.js can only add it to its inline scripts
 * while rendering. A prerendered page would carry scripts the policy
 * blocks.
 */
export default async function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  await connection();
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={`${geistSans.variable} ${geistMono.variable} font-sans antialiased
          bg-bg text-fg`}
      >
        <Providers>
          <WelcomeGate>
            <DemoBanner>
              {/* Skip-to-content — first focusable element on the page.
                  Hidden until focused (sr-only → not-sr-only on focus). */}
              <a
                href="#main-content"
                className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-4 focus:z-50 focus:rounded-lg focus:bg-accent focus:px-4 focus:py-2 focus:text-sm focus:font-medium focus:text-accent-fg focus:outline-none"
              >
                Skip to content
              </a>
              <div className="flex min-h-screen flex-col">
                <Navbar />
                <main
                  id="main-content"
                  className="mx-auto w-full max-w-[1440px] flex-1 px-6 py-10 sm:px-8 lg:px-12"
                >
                  {children}
                </main>
                <Footer />
              </div>
            </DemoBanner>
          </WelcomeGate>
        </Providers>
      </body>
    </html>
  );
}