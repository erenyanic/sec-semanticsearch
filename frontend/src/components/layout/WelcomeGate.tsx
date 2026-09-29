"use client";

/**
 * Welcome screen gate — wraps the page content inside `<main>`.
 *
 * Only the ingest endpoints use EDGAR credentials, so only the ingest
 * page is gated.  When the backend requires per-session credentials
 * (`edgar_session_required: true` in the status response) and this tab
 * has none, the gate renders a form in place of that page.  Once the
 * user provides their SEC EDGAR name and email, the credentials are
 * stored in `sessionStorage` and the page is shown.
 *
 * The gate never blocks first paint: the page renders while
 * `/api/status/` loads (the API may be cold-starting), if the request
 * fails, and whenever credentials are **not** required (Scenario A with
 * server-side env vars).  The navbar and footer sit outside it.
 */

import { type SubmitEvent, useState, type ReactNode } from "react";
import { usePathname } from "next/navigation";
import { useEdgarSession } from "@/hooks/useEdgarSession";
import { useStatus } from "@/hooks/useStatus";
import { Button } from "@/components/ui";

/** Routes whose API calls need EDGAR credentials. */
const GATED_ROUTES = ["/ingest"];

function needsEdgarCredentials(pathname: string | null): boolean {
  return GATED_ROUTES.some(
    (route) => pathname === route || pathname?.startsWith(`${route}/`),
  );
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

interface WelcomeGateProps {
  children: ReactNode;
}

export function WelcomeGate({ children }: WelcomeGateProps) {
  const pathname = usePathname();
  const { data: status } = useStatus();
  const { isAuthenticated, login } = useEdgarSession();

  // Show the form only once the server has said credentials are needed
  // (`status` is undefined while loading and after an error — the pages
  // have their own error handling), and only on a gated route.
  if (
    needsEdgarCredentials(pathname) &&
    status?.edgar_session_required === true &&
    !isAuthenticated
  ) {
    return <WelcomeForm onLogin={login} />;
  }

  return <>{children}</>;
}

// ---------------------------------------------------------------------------
// Welcome form (internal)
// ---------------------------------------------------------------------------

interface WelcomeFormProps {
  onLogin: (name: string, email: string) => void;
}

function WelcomeForm({ onLogin }: WelcomeFormProps) {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");

  function handleSubmit(e: SubmitEvent) {
    e.preventDefault();
    if (name.trim() && email.trim()) {
      onLogin(name, email);
    }
  }

  return (
    <div className="flex min-h-[60vh] items-center justify-center">
      <div className="mx-auto w-full max-w-md space-y-8 rounded-2xl border border-hairline bg-card/80 p-8 shadow-2xl backdrop-blur-xl">
        {/* Header */}
        <div className="space-y-3 text-center">
          <div className="flex items-center justify-center">
            <span
              className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-accent to-accent/70 text-accent-fg shadow-lg shadow-accent/20"
              aria-hidden="true"
            >
              <span className="text-base font-bold">S</span>
            </span>
          </div>
          <h1 className="text-2xl font-semibold tracking-tight text-fg">
            EDGAR credentials required
          </h1>
          <p className="text-sm text-fg-muted">
            The SEC requires a name and email in every EDGAR request. Please
            enter your details to continue.
          </p>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="space-y-4">
          <label htmlFor="edgar-name" className="block space-y-2">
            <span className="text-sm font-medium text-fg">Full name</span>
            <input
              id="edgar-name"
              type="text"
              required
              autoComplete="name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Jane Smith"
              className="block w-full rounded-lg border border-hairline bg-card px-4 py-2.5 text-sm text-fg outline-none transition-colors placeholder:text-fg-subtle focus:border-accent focus:ring-2 focus:ring-accent/25"
            />
          </label>
          <label htmlFor="edgar-email" className="block space-y-2">
            <span className="text-sm font-medium text-fg">Email address</span>
            <input
              id="edgar-email"
              type="email"
              required
              autoComplete="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="jane@example.com"
              className="block w-full rounded-lg border border-hairline bg-card px-4 py-2.5 text-sm text-fg outline-none transition-colors placeholder:text-fg-subtle focus:border-accent focus:ring-2 focus:ring-accent/25"
            />
          </label>
          <Button type="submit" size="lg" className="w-full">
            Continue
          </Button>
        </form>

        {/* Privacy notice */}
        <p className="border-t border-hairline pt-4 text-center text-xs text-fg-subtle">
          Credentials stay in this tab · Never saved on the server
        </p>
      </div>
    </div>
  );
}
