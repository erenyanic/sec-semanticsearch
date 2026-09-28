import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ToastProvider } from "@/components/ui/Toast";
import type { ReactNode } from "react";
import type { Filing } from "@/lib/types";
import FilingsPage from "../filings/page";

// Mock hooks (page tests mock hooks, never the API client — AD#27).
vi.mock("@/hooks/useStatus");
vi.mock("@/hooks/useFilings", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/hooks/useFilings")>();
  return { ...actual, useFilings: vi.fn() };
});
vi.mock("@/hooks/useAdminSession", () => ({
  useAdminSession: () => ({ isAdmin: false }),
}));

const replace = vi.fn();
vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace }),
  useSearchParams: () => new URLSearchParams(),
}));

vi.mock("next/link", () => ({
  default: ({ children, href }: { children: ReactNode; href: string }) => (
    <a href={href}>{children}</a>
  ),
}));

import { useStatus } from "@/hooks/useStatus";
import { useFilings, type FilingQueryParams } from "@/hooks/useFilings";
const mockUseStatus = vi.mocked(useStatus);
const mockUseFilings = vi.mocked(useFilings);

function wrapper({ children }: { children: ReactNode }) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return (
    <QueryClientProvider client={queryClient}>
      <ToastProvider>{children}</ToastProvider>
    </QueryClientProvider>
  );
}

function filingAt(i: number, ticker = "AAPL"): Filing {
  return {
    ticker,
    form_type: "10-K",
    filing_date: "2024-01-15",
    accession_number: `0000320193-24-${String(i).padStart(6, "0")}`,
    chunk_count: 10,
    ingested_at: "2024-02-01T12:00:00Z",
  };
}

const deleteSingle = vi.fn().mockResolvedValue({ accession_number: "x", chunks_deleted: 10 });

function mockPage(filings: Filing[], total: number) {
  mockUseFilings.mockReturnValue({
    filings,
    total,
    isLoading: false,
    isError: false,
    error: null,
    deleteSingle,
    deleteSelected: vi.fn(),
    clearAll: vi.fn(),
    isDeleting: false,
  });
}

/** The params passed to useFilings on the most recent render. */
function lastParams(): FilingQueryParams {
  return mockUseFilings.mock.calls.at(-1)![0];
}

describe("FilingsPage — server-side pagination (F-06)", () => {
  beforeEach(() => {
    mockUseStatus.mockReturnValue({
      data: {
        filing_count: 30,
        max_filings: 2500,
        chunk_count: 300,
        tickers: ["AAPL", "MSFT"],
        form_breakdown: { "10-K": 30 },
        ticker_breakdown: [],
      },
      isLoading: false,
      isError: false,
    } as unknown as ReturnType<typeof useStatus>);
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("starts on the first page of ten", () => {
    mockPage([filingAt(1)], 30);
    render(<FilingsPage />, { wrapper });
    expect(lastParams()).toMatchObject({ page: 0, pageSize: 10 });
  });

  it("moves to the next page on request", () => {
    mockPage(Array.from({ length: 10 }, (_, i) => filingAt(i)), 30);
    render(<FilingsPage />, { wrapper });
    fireEvent.click(screen.getByRole("button", { name: "Next page" }));
    expect(lastParams().page).toBe(1);
  });

  it("returns to the first page when a filter changes", () => {
    mockPage(Array.from({ length: 10 }, (_, i) => filingAt(i)), 30);
    render(<FilingsPage />, { wrapper });
    fireEvent.click(screen.getByRole("button", { name: "Next page" }));
    expect(lastParams().page).toBe(1);

    fireEvent.change(screen.getByLabelText("Filter by ticker"), { target: { value: "MSFT" } });
    expect(lastParams()).toMatchObject({ ticker: "MSFT", page: 0 });
  });

  it("returns to the first page when the sort changes", () => {
    mockPage(Array.from({ length: 10 }, (_, i) => filingAt(i)), 30);
    render(<FilingsPage />, { wrapper });
    fireEvent.click(screen.getByRole("button", { name: "Next page" }));
    fireEvent.click(screen.getByRole("button", { name: /Ticker/ }));
    expect(lastParams()).toMatchObject({ sortBy: "ticker", page: 0 });
  });

  it("clears the selection when the page changes", () => {
    mockPage(Array.from({ length: 10 }, (_, i) => filingAt(i)), 30);
    render(<FilingsPage />, { wrapper });
    const box = screen.getAllByRole("checkbox", { name: /^Select AAPL/ })[0];
    fireEvent.click(box);
    expect(screen.getByText("1 selected")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Next page" }));
    expect(screen.queryByText("1 selected")).not.toBeInTheDocument();
  });

  it("steps back a page when a delete empties the current one", async () => {
    mockPage(Array.from({ length: 10 }, (_, i) => filingAt(i)), 11);
    render(<FilingsPage />, { wrapper });

    // Page 2 holds the eleventh filing only.
    mockPage([filingAt(10)], 11);
    fireEvent.click(screen.getByRole("button", { name: "Next page" }));
    expect(lastParams().page).toBe(1);

    fireEvent.click(screen.getByRole("button", { name: /^Delete AAPL 10-K$/ }));
    // The modal backdrop is aria-hidden, so the dialog needs hidden: true.
    fireEvent.click(screen.getByRole("button", { name: "Delete", hidden: true }));

    await waitFor(() => expect(deleteSingle).toHaveBeenCalled());
    await waitFor(() => expect(lastParams().page).toBe(0));
  });
});
