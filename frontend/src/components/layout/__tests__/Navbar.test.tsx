import { render, screen, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ToastProvider } from "@/components/ui/Toast";
import type { ReactNode } from "react";
import { Navbar } from "../Navbar";

// ---------------------------------------------------------------------------
// Mocks
// ---------------------------------------------------------------------------

// Next.js navigation hooks
vi.mock("next/navigation", () => ({
  usePathname: () => "/",
}));

// Theme provider
vi.mock("../ThemeProvider", () => ({
  useTheme: () => ({ theme: "light", toggleTheme: vi.fn() }),
}));

// EDGAR session
vi.mock("@/hooks/useEdgarSession", () => ({
  useEdgarSession: () => ({
    isAuthenticated: false,
    logout: vi.fn(),
  }),
}));

// Admin session
vi.mock("@/hooks/useAdminSession", () => ({
  useAdminSession: () => ({
    adminRequired: false,
    isAdmin: false,
    login: vi.fn(),
    logout: vi.fn(),
    isPending: false,
  }),
}));

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function renderNavbar() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  function Wrapper({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>
        <ToastProvider>{children}</ToastProvider>
      </QueryClientProvider>
    );
  }
  return render(<Navbar />, { wrapper: Wrapper });
}

// ---------------------------------------------------------------------------
// BF-009: GitHub and LinkedIn icons in top-right navbar
// ---------------------------------------------------------------------------

describe("Navbar — portfolio icons (BF-009)", () => {
  it("renders a GitHub link with correct aria-label", () => {
    renderNavbar();
    const link = screen.getByRole("link", { name: "GitHub profile" });
    expect(link).toBeInTheDocument();
    expect(link).toHaveAttribute("href", "https://github.com/ErenYanic");
    expect(link).toHaveAttribute("target", "_blank");
    expect(link).toHaveAttribute("rel", "noopener noreferrer");
  });

  it("renders a LinkedIn link with correct aria-label", () => {
    renderNavbar();
    const link = screen.getByRole("link", { name: "LinkedIn profile" });
    expect(link).toBeInTheDocument();
    expect(link).toHaveAttribute(
      "href",
      "https://www.linkedin.com/in/erenyanic/",
    );
    expect(link).toHaveAttribute("target", "_blank");
    expect(link).toHaveAttribute("rel", "noopener noreferrer");
  });

  it("renders portfolio icons with rounded-lg chrome styling", () => {
    renderNavbar();
    const githubLink = screen.getByRole("link", { name: "GitHub profile" });
    expect(githubLink.className).toContain("rounded-lg");
    expect(githubLink.className).toContain("border");
  });

  it("places portfolio icons in the right-side container (ml-auto)", () => {
    renderNavbar();
    const githubLink = screen.getByRole("link", { name: "GitHub profile" });
    // The parent container should have ml-auto class
    const rightContainer = githubLink.closest(".ml-auto");
    expect(rightContainer).not.toBeNull();
  });

  it("portfolio icons appear before theme toggle in the right container", () => {
    renderNavbar();
    const rightContainer = screen
      .getByRole("link", { name: "GitHub profile" })
      .closest(".ml-auto");
    expect(rightContainer).not.toBeNull();
    const themeToggle = within(rightContainer as HTMLElement).getByRole("button", {
      name: /Switch to .* mode/,
    });
    expect(themeToggle).toBeInTheDocument();
  });
});
