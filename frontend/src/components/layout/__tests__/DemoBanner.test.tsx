import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { DemoBanner } from "../DemoBanner";

// ---------------------------------------------------------------------------
// Mocks
// ---------------------------------------------------------------------------

vi.mock("@/hooks/useStatus", () => ({
  useStatus: vi.fn(),
}));

import { useStatus } from "@/hooks/useStatus";

const mockUseStatus = vi.mocked(useStatus);

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

function renderBanner(children: ReactNode = <div data-testid="app-content">App</div>) {
  const Wrapper = createWrapper();
  return render(
    <Wrapper>
      <DemoBanner>{children}</DemoBanner>
    </Wrapper>,
  );
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("DemoBanner", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("shows banner when demo_mode is true", () => {
    mockUseStatus.mockReturnValue({
      isLoading: false,
      isError: false,
      data: { demo_mode: true },
    } as ReturnType<typeof useStatus>);

    renderBanner();
    expect(screen.getByText(/Data is temporary and may be cleared at any time/)).toBeInTheDocument();
  });

  it("hides banner when demo_mode is false", () => {
    mockUseStatus.mockReturnValue({
      isLoading: false,
      isError: false,
      data: { demo_mode: false },
    } as ReturnType<typeof useStatus>);

    renderBanner();
    expect(screen.queryByText(/Data is temporary/)).not.toBeInTheDocument();
  });

  it("always renders children regardless of demo_mode", () => {
    mockUseStatus.mockReturnValue({
      isLoading: false,
      isError: false,
      data: { demo_mode: true },
    } as ReturnType<typeof useStatus>);

    renderBanner();
    expect(screen.getByTestId("app-content")).toBeInTheDocument();
  });

  it("renders children when status is loading (no banner)", () => {
    mockUseStatus.mockReturnValue({
      isLoading: true,
      isError: false,
      data: undefined,
    } as ReturnType<typeof useStatus>);

    renderBanner();
    expect(screen.getByTestId("app-content")).toBeInTheDocument();
    expect(screen.queryByText(/Data is temporary/)).not.toBeInTheDocument();
  });

  it("renders children when status has error (no banner)", () => {
    mockUseStatus.mockReturnValue({
      isLoading: false,
      isError: true,
      data: undefined,
    } as ReturnType<typeof useStatus>);

    renderBanner();
    expect(screen.getByTestId("app-content")).toBeInTheDocument();
    expect(screen.queryByText(/Data is temporary/)).not.toBeInTheDocument();
  });

  it("banner has role=status for accessibility", () => {
    mockUseStatus.mockReturnValue({
      isLoading: false,
      isError: false,
      data: { demo_mode: true },
    } as ReturnType<typeof useStatus>);

    renderBanner();
    expect(screen.getByRole("status")).toBeInTheDocument();
  });
});
