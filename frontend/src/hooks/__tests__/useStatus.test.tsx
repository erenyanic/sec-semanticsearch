import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import type { StatusResponse } from "@/lib/types";
import { useStatus } from "../useStatus";

// Mock the API module at the boundary — not axios, not the network
vi.mock("@/lib/api", () => ({
  getStatus: vi.fn(),
}));

import { getStatus } from "@/lib/api";
const mockGetStatus = vi.mocked(getStatus);

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
  }
  return Wrapper;
}

describe("useStatus", () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("returns data on success", async () => {
    const mockData: StatusResponse = {
      filing_count: 5,
      max_filings: 200,
      chunk_count: 1500,
      tickers: ["AAPL"],
      form_breakdown: { "10-K": 3, "10-Q": 2 },
      ticker_breakdown: [],
      edgar_session_required: false,
      demo_mode: false,
      is_admin: true,
    };
    mockGetStatus.mockResolvedValue(mockData);

    const { result } = renderHook(() => useStatus(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data).toEqual(mockData);
  });

  it("returns error state on failure", async () => {
    mockGetStatus.mockRejectedValue(new Error("Network error"));

    const { result } = renderHook(() => useStatus(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isError).toBe(true));
    expect(result.current.error).toBeTruthy();
  });

  it("starts in loading state", () => {
    mockGetStatus.mockReturnValue(new Promise(() => {})); // never resolves

    const { result } = renderHook(() => useStatus(), {
      wrapper: createWrapper(),
    });

    expect(result.current.isLoading).toBe(true);
    expect(result.current.data).toBeUndefined();
  });
});