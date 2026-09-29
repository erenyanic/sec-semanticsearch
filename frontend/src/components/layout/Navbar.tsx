"use client";

/**
 * Top navigation bar — present on every page.
 *
 * Modern translucent chrome with a subtle accent brand mark, mixed-case
 * sans-serif nav links, and a cluster of session/theme controls.
 */

import Image from "next/image";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { type SubmitEvent, useState } from "react";
import {
  LayoutDashboard,
  Search,
  Upload,
  FileText,
  Sun,
  Moon,
  LogOut,
  Shield,
  ShieldCheck,
} from "lucide-react";
import { useTheme } from "./ThemeProvider";
import { useEdgarSession } from "@/hooks/useEdgarSession";
import { useAdminSession } from "@/hooks/useAdminSession";
import { Modal, useToast } from "@/components/ui";

const NAV_ITEMS = [
  { href: "/", label: "Dashboard", icon: LayoutDashboard },
  { href: "/search", label: "Search", icon: Search },
  { href: "/ingest", label: "Ingest", icon: Upload },
  { href: "/filings", label: "Filings", icon: FileText },
] as const;

const ICON_BUTTON =
  "flex h-10 w-10 items-center justify-center rounded-lg border border-hairline bg-card/50 text-fg-muted " +
  "transition-all hover:border-accent/40 hover:bg-card hover:text-fg " +
  "focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent";

const PILL_BUTTON =
  "inline-flex items-center gap-2 rounded-lg border border-hairline bg-card/50 px-3.5 py-2 " +
  "text-sm font-medium text-fg-muted transition-all " +
  "hover:border-accent/40 hover:bg-card hover:text-fg " +
  "focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent";

export function Navbar() {
  const pathname = usePathname();
  const { theme, toggleTheme } = useTheme();
  const { isAuthenticated, logout } = useEdgarSession();
  const {
    adminRequired,
    isAdmin,
    login,
    logout: logoutAdmin,
    isPending,
  } = useAdminSession();
  const { addToast } = useToast();
  const [showAdminModal, setShowAdminModal] = useState(false);
  const [adminKey, setAdminKey] = useState("");

  async function submitAdminLogin() {
    try {
      await login(adminKey);
      setAdminKey("");
      setShowAdminModal(false);
      addToast("success", "Admin session enabled");
    } catch {
      addToast("error", "Invalid admin key");
    }
  }

  async function handleAdminLogin(event: SubmitEvent) {
    event.preventDefault();
    await submitAdminLogin();
  }

  async function handleAdminLogout() {
    await logoutAdmin();
    addToast("info", "Admin session cleared");
  }

  return (
    <nav className="sticky top-0 z-20 border-b border-hairline bg-bg/75 backdrop-blur-xl">
      <div className="mx-auto flex h-16 max-w-[1440px] items-center gap-8 px-6 sm:px-8 lg:px-12">
        {/* ---- Brand mark ---- */}
        <Link
          href="/"
          className="group flex items-center gap-2.5 text-base font-semibold tracking-tight text-fg transition-colors"
        >
          <span
            className="flex h-7 w-7 items-center justify-center rounded-lg bg-gradient-to-br from-accent to-accent/70 text-accent-fg shadow-lg shadow-accent/20 transition-transform group-hover:scale-105"
            aria-hidden="true"
          >
            <span className="text-xs font-bold">S</span>
          </span>
          <span className="hidden sm:inline">SEC Semantic Search</span>
        </Link>

        {/* ---- Centre: navigation links ---- */}
        <div className="flex items-center gap-1">
          {NAV_ITEMS.map(({ href, label, icon: Icon }) => {
            const isActive =
              href === "/" ? pathname === "/" : pathname.startsWith(href);
            return (
              <Link
                key={href}
                href={href}
                aria-current={isActive ? "page" : undefined}
                className={`
                  group flex items-center gap-2 rounded-lg px-3.5 py-2 text-sm font-medium transition-all
                  ${
                    isActive
                      ? "bg-accent/10 text-accent"
                      : "text-fg-muted hover:bg-card hover:text-fg"
                  }
                `}
              >
                <Icon className="h-4 w-4" />
                <span className="hidden md:inline">{label}</span>
              </Link>
            );
          })}
        </div>

        {/* ---- Right: status + controls ---- */}
        <div className="ml-auto flex items-center gap-2">
          {adminRequired && !isAdmin && (
            <button
              onClick={() => setShowAdminModal(true)}
              className={PILL_BUTTON}
              aria-label="Open admin access dialog"
            >
              <Shield className="h-4 w-4" />
              <span className="hidden sm:inline">Admin</span>
            </button>
          )}

          {adminRequired && isAdmin && (
            <button
              onClick={handleAdminLogout}
              className="inline-flex items-center gap-2 rounded-lg border border-pos/40 bg-pos/10 px-3.5 py-2 text-sm font-medium text-pos transition-all hover:bg-pos/15 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-pos"
              aria-label="Clear admin session"
            >
              <ShieldCheck className="h-4 w-4" />
              <span className="hidden sm:inline">Admin Active</span>
            </button>
          )}

          {isAuthenticated && (
            <button
              onClick={logout}
              className={PILL_BUTTON}
              aria-label="Clear EDGAR credentials and return to Welcome screen"
            >
              <LogOut className="h-4 w-4" />
              <span className="hidden sm:inline">Logout</span>
            </button>
          )}

          <span className="mx-1 h-6 w-px bg-hairline" aria-hidden="true" />

          <a
            href="https://github.com/ErenYanic"
            target="_blank"
            rel="noopener noreferrer"
            className={ICON_BUTTON}
            aria-label="GitHub profile"
          >
            <Image
              src="/GitHub_Invertocat_White.png"
              alt=""
              width={18}
              height={18}
              className="h-[18px] w-[18px] object-contain brightness-0 dark:brightness-100"
            />
          </a>
          <a
            href="https://www.linkedin.com/in/erenyanic/"
            target="_blank"
            rel="noopener noreferrer"
            className={ICON_BUTTON}
            aria-label="LinkedIn profile"
          >
            <Image
              src="/LI-In-Bug.png"
              alt=""
              width={20}
              height={20}
              className="h-[18px] w-[18px] object-contain"
            />
          </a>

          <button
            onClick={toggleTheme}
            className={ICON_BUTTON}
            aria-label={`Switch to ${theme === "light" ? "dark" : "light"} mode`}
          >
            {theme === "light" ? (
              <Moon className="h-4 w-4" />
            ) : (
              <Sun className="h-4 w-4" />
            )}
          </button>
        </div>
      </div>

      <Modal
        open={showAdminModal}
        onClose={() => {
          setShowAdminModal(false);
          setAdminKey("");
        }}
        onConfirm={() => {
          void submitAdminLogin();
        }}
        title="Admin Access"
        confirmLabel="Unlock"
        confirmDisabled={adminKey.trim().length === 0}
        confirmLoading={isPending}
      >
        <form onSubmit={handleAdminLogin} className="space-y-4">
          <p className="text-sm text-fg-muted">
            Enter the admin key to enable destructive operations in this
            browser session.
          </p>
          <label className="block space-y-2">
            <span className="text-sm font-medium text-fg">Admin key</span>
            <input
              type="password"
              value={adminKey}
              onChange={(event) => setAdminKey(event.target.value)}
              className="w-full rounded-lg border border-hairline bg-card px-4 py-2.5 text-sm text-fg outline-none transition-colors focus:border-accent focus:ring-2 focus:ring-accent/25"
              autoComplete="current-password"
            />
          </label>
        </form>
      </Modal>
    </nav>
  );
}
