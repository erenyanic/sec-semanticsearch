import { fireEvent } from "@testing-library/react";
import { renderWithProviders, screen } from "@/test/utils";
import { FilingTable } from "../FilingTable";
import type { Filing } from "@/lib/types";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const FILING: Filing = {
  ticker: "AAPL",
  form_type: "10-K",
  filing_date: "2024-01-15",
  accession_number: "0000320193-24-000006",
  chunk_count: 42,
  ingested_at: "2024-02-01T12:00:00Z",
};

const noop = () => {};

interface PageProps {
  total?: number;
  page?: number;
  pageSize?: number;
  onPageChange?: (page: number) => void;
  onPageSizeChange?: (pageSize: number) => void;
}

function renderTable(filings: Filing[] = [FILING], paging: PageProps = {}) {
  return renderWithProviders(
    <FilingTable
      filings={filings}
      total={paging.total ?? filings.length}
      page={paging.page ?? 0}
      pageSize={paging.pageSize ?? 10}
      onPageChange={paging.onPageChange ?? noop}
      onPageSizeChange={paging.onPageSizeChange ?? noop}
      sortBy="filing_date"
      order="desc"
      onSortChange={noop}
      selected={new Set()}
      onSelectionChange={noop}
      onDeleteFiling={noop}
      isDeleting={false}
    />,
  );
}

// ---------------------------------------------------------------------------
// BF-007: Accession number column
// ---------------------------------------------------------------------------

describe("FilingTable — Accession column (BF-007)", () => {
  it("renders the Accession No. column header", () => {
    renderTable();
    expect(screen.getByText("Accession No.")).toBeInTheDocument();
  });

  it("displays the accession number in the row", () => {
    renderTable();
    expect(screen.getByText("0000320193-24-000006")).toBeInTheDocument();
  });

  it("renders the accession number in monospace font", () => {
    renderTable();
    const accessionEl = screen.getByText("0000320193-24-000006");
    expect(accessionEl.className).toContain("font-mono");
  });

  it("renders a copy button with aria-label for each filing", () => {
    renderTable();
    expect(
      screen.getByRole("button", {
        name: "Copy accession number 0000320193-24-000006",
      }),
    ).toBeInTheDocument();
  });

  it("accession column is not sortable (no button in header)", () => {
    renderTable();
    const headerText = screen.getByText("Accession No.");
    // The header text should be plain text, not wrapped in a button
    expect(headerText.tagName).not.toBe("BUTTON");
    expect(headerText.closest("button")).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// F-06: server-side pagination
// ---------------------------------------------------------------------------

function filingAt(i: number): Filing {
  return { ...FILING, accession_number: `0000320193-24-${String(i).padStart(6, "0")}` };
}

describe("FilingTable — server-side pagination (F-06)", () => {
  it("renders every row it is given without slicing", () => {
    const page = Array.from({ length: 10 }, (_, i) => filingAt(i));
    renderTable(page, { total: 37, pageSize: 10 });
    expect(screen.getAllByRole("checkbox", { name: /^Select AAPL/ })).toHaveLength(10);
  });

  it("shows the range and the total across pages", () => {
    const page = Array.from({ length: 10 }, (_, i) => filingAt(i + 10));
    renderTable(page, { total: 37, page: 1, pageSize: 10 });
    const footer = screen.getByText("37").closest("span")!.parentElement!;
    expect(footer.textContent).toContain("11–20 of 37 filings");
    expect(screen.getByText("2")).toBeInTheDocument(); // current page
    expect(screen.getByText("4")).toBeInTheDocument(); // total pages
  });

  it("asks the parent for the next and previous page", () => {
    const onPageChange = vi.fn();
    renderTable([filingAt(1)], { total: 25, page: 1, pageSize: 10, onPageChange });
    fireEvent.click(screen.getByRole("button", { name: "Next page" }));
    expect(onPageChange).toHaveBeenLastCalledWith(2);
    fireEvent.click(screen.getByRole("button", { name: "Previous page" }));
    expect(onPageChange).toHaveBeenLastCalledWith(0);
  });

  it("disables next on the last page", () => {
    renderTable([filingAt(1)], { total: 21, page: 2, pageSize: 10 });
    expect(screen.getByRole("button", { name: "Next page" })).toBeDisabled();
  });

  it("reports page size changes to the parent", () => {
    const onPageSizeChange = vi.fn();
    renderTable([FILING], { onPageSizeChange });
    fireEvent.change(screen.getByLabelText("Rows"), { target: { value: "50" } });
    expect(onPageSizeChange).toHaveBeenCalledWith(50);
  });

  it("offers no page size above the delete-by-ids limit", () => {
    renderTable();
    const sizes = Array.from(
      (screen.getByLabelText("Rows") as HTMLSelectElement).options,
    ).map((o) => Number(o.value));
    expect(Math.max(...sizes)).toBeLessThanOrEqual(50);
  });

  it("keeps the footer on a page emptied elsewhere so the user can step back", () => {
    const onPageChange = vi.fn();
    renderTable([], { total: 12, page: 3, pageSize: 10, onPageChange });
    expect(screen.getByText("No filings on this page")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Previous page" }));
    expect(onPageChange).toHaveBeenCalledWith(1);
  });

  it("shows the no-match state only when nothing matches at all", () => {
    renderTable([], { total: 0 });
    expect(screen.getByText("No filings match the current filters")).toBeInTheDocument();
  });
});
