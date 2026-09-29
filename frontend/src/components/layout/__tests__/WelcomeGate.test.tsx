import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { WelcomeGate } from "../WelcomeGate";

// ---------------------------------------------------------------------------
// Mocks
// ---------------------------------------------------------------------------

vi.mock("@/hooks/useStatus", () => ({
  useStatus: vi.fn(),
}));

vi.mock("@/hooks/useEdgarSession", () => ({
  useEdgarSession: vi.fn(),
}));

vi.mock("next/navigation", () => ({
  usePathname: vi.fn(),
}));

import { usePathname } from "next/navigation";
import { useStatus } from "@/hooks/useStatus";
import { useEdgarSession } from "@/hooks/useEdgarSession";

const mockUseStatus = vi.mocked(useStatus);
const mockUseEdgarSession = vi.mocked(useEdgarSession);
const mockUsePathname = vi.mocked(usePathname);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
  }
  return Wrapper;
}

function renderGate(children: ReactNode = <div data-testid="app-content">App</div>) {
  const Wrapper = createWrapper();
  return render(
    <Wrapper>
      <WelcomeGate>{children}</WelcomeGate>
    </Wrapper>,
  );
}

const statusRequiringSession = {
  isLoading: false,
  isError: false,
  data: {
    filing_count: 0,
    max_filings: 500,
    chunk_count: 0,
    tickers: [],
    form_breakdown: {},
    ticker_breakdown: [],
    edgar_session_required: true,
  },
} as unknown as ReturnType<typeof useStatus>;

// Default mock values
const defaultSession = {
  name: null,
  email: null,
  isAuthenticated: false,
  login: vi.fn(),
  logout: vi.fn(),
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("WelcomeGate", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    // The gated route; the route tests below override it.
    mockUsePathname.mockReturnValue("/ingest");
  });

  it("renders the page while status is loading (never blocks first paint)", () => {
    mockUseStatus.mockReturnValue({
      isLoading: true,
      isError: false,
      data: undefined,
    } as ReturnType<typeof useStatus>);
    mockUseEdgarSession.mockReturnValue(defaultSession);

    renderGate();
    expect(screen.getByTestId("app-content")).toBeInTheDocument();
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
    expect(screen.queryByText("EDGAR credentials required")).not.toBeInTheDocument();
  });

  it("swaps the ingest page for the form once status requires a session", () => {
    mockUseEdgarSession.mockReturnValue(defaultSession);
    mockUseStatus.mockReturnValue({
      isLoading: true,
      isError: false,
      data: undefined,
    } as ReturnType<typeof useStatus>);

    const { rerender } = renderGate();
    expect(screen.getByTestId("app-content")).toBeInTheDocument();

    mockUseStatus.mockReturnValue(statusRequiringSession);
    rerender(
      <QueryClientProvider client={new QueryClient()}>
        <WelcomeGate>
          <div data-testid="app-content">App</div>
        </WelcomeGate>
      </QueryClientProvider>,
    );
    expect(screen.queryByTestId("app-content")).not.toBeInTheDocument();
    expect(screen.getByText("EDGAR credentials required")).toBeInTheDocument();
  });

  it.each(["/", "/search", "/filings"])(
    "never gates %s, which needs no EDGAR credentials",
    (path) => {
      mockUsePathname.mockReturnValue(path);
      mockUseStatus.mockReturnValue(statusRequiringSession);
      mockUseEdgarSession.mockReturnValue(defaultSession);

      renderGate();
      expect(screen.getByTestId("app-content")).toBeInTheDocument();
      expect(screen.queryByText("EDGAR credentials required")).not.toBeInTheDocument();
    },
  );

  it("gates nested ingest routes but not look-alike paths", () => {
    mockUseStatus.mockReturnValue(statusRequiringSession);
    mockUseEdgarSession.mockReturnValue(defaultSession);

    mockUsePathname.mockReturnValue("/ingest/history");
    const { unmount } = renderGate();
    expect(screen.getByText("EDGAR credentials required")).toBeInTheDocument();
    unmount();

    mockUsePathname.mockReturnValue("/ingestion");
    renderGate();
    expect(screen.getByTestId("app-content")).toBeInTheDocument();
  });

  it("renders children when status has error (fallback)", () => {
    mockUseStatus.mockReturnValue({
      isLoading: false,
      isError: true,
      data: undefined,
    } as ReturnType<typeof useStatus>);
    mockUseEdgarSession.mockReturnValue(defaultSession);

    renderGate();
    expect(screen.getByTestId("app-content")).toBeInTheDocument();
  });

  it("renders children when edgar_session_required is false", () => {
    mockUseStatus.mockReturnValue({
      isLoading: false,
      isError: false,
      data: {
        filing_count: 0,
        max_filings: 500,
        chunk_count: 0,
        tickers: [],
        form_breakdown: {},
        ticker_breakdown: [],
        edgar_session_required: false,
      },
    } as unknown as ReturnType<typeof useStatus>);
    mockUseEdgarSession.mockReturnValue(defaultSession);

    renderGate();
    expect(screen.getByTestId("app-content")).toBeInTheDocument();
  });

  it("renders children when already authenticated", () => {
    mockUseStatus.mockReturnValue({
      isLoading: false,
      isError: false,
      data: {
        filing_count: 0,
        max_filings: 500,
        chunk_count: 0,
        tickers: [],
        form_breakdown: {},
        ticker_breakdown: [],
        edgar_session_required: true,
      },
    } as unknown as ReturnType<typeof useStatus>);
    mockUseEdgarSession.mockReturnValue({
      ...defaultSession,
      isAuthenticated: true,
      name: "Jane",
      email: "jane@example.com",
    });

    renderGate();
    expect(screen.getByTestId("app-content")).toBeInTheDocument();
  });

  it("shows welcome form when session required and not authenticated", () => {
    mockUseStatus.mockReturnValue({
      isLoading: false,
      isError: false,
      data: {
        filing_count: 0,
        max_filings: 500,
        chunk_count: 0,
        tickers: [],
        form_breakdown: {},
        ticker_breakdown: [],
        edgar_session_required: true,
      },
    } as unknown as ReturnType<typeof useStatus>);
    mockUseEdgarSession.mockReturnValue(defaultSession);

    renderGate();
    expect(screen.queryByTestId("app-content")).not.toBeInTheDocument();
    expect(screen.getByText("EDGAR credentials required")).toBeInTheDocument();
    expect(screen.getByLabelText("Full name")).toBeInTheDocument();
    expect(screen.getByLabelText("Email address")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Continue" })).toBeInTheDocument();
  });

  it("calls login() on form submit", async () => {
    const mockLogin = vi.fn();
    mockUseStatus.mockReturnValue({
      isLoading: false,
      isError: false,
      data: {
        filing_count: 0,
        max_filings: 500,
        chunk_count: 0,
        tickers: [],
        form_breakdown: {},
        ticker_breakdown: [],
        edgar_session_required: true,
      },
    } as unknown as ReturnType<typeof useStatus>);
    mockUseEdgarSession.mockReturnValue({ ...defaultSession, login: mockLogin });

    renderGate();

    const nameInput = screen.getByLabelText("Full name");
    const emailInput = screen.getByLabelText("Email address");
    const submitButton = screen.getByRole("button", { name: "Continue" });

    fireEvent.change(nameInput, { target: { value: "Jane Smith" } });
    fireEvent.change(emailInput, { target: { value: "jane@example.com" } });
    fireEvent.click(submitButton);

    await waitFor(() => {
      expect(mockLogin).toHaveBeenCalledWith("Jane Smith", "jane@example.com");
    });
  });

  it("shows privacy notice in welcome form", () => {
    mockUseStatus.mockReturnValue({
      isLoading: false,
      isError: false,
      data: {
        filing_count: 0,
        max_filings: 500,
        chunk_count: 0,
        tickers: [],
        form_breakdown: {},
        ticker_breakdown: [],
        edgar_session_required: true,
      },
    } as unknown as ReturnType<typeof useStatus>);
    mockUseEdgarSession.mockReturnValue(defaultSession);

    renderGate();
    expect(
      screen.getByText(/never saved on the server/i),
    ).toBeInTheDocument();
  });
});
