import { hydrateSearchResponse } from "../api";
import type { SearchResultWire } from "../types";

const base: SearchResultWire = {
  content: "chunk",
  path: "Part II > Item 7",
  content_type: "text",
  ticker: "AAPL",
  form_type: "10-K",
  similarity: 0.5,
  accession_number: "0000320193-24-000001",
  segment_index: 1,
  parent_key: null,
  parent: null,
};

function wire(results: SearchResultWire[]) {
  return { results, total_results: results.length, search_time_ms: 3.2 };
}

describe("hydrateSearchResponse", () => {
  it("resolves a parent sent on an earlier result by key", () => {
    const parent = { content: "chunk one. chunk two.", truncated_start: false, truncated_end: false };
    const out = hydrateSearchResponse(
      wire([
        { ...base, content: "chunk one.", parent_key: "A:1", parent },
        { ...base, content: "chunk two.", parent_key: "A:1", parent: null },
      ]),
    );
    expect(out.results.map((r) => r.parent_content)).toEqual([
      "chunk one. chunk two.",
      "chunk one. chunk two.",
    ]);
    expect(out.total_results).toBe(2);
    expect(out.search_time_ms).toBe(3.2);
  });

  it("leaves parent_content null without a key", () => {
    const out = hydrateSearchResponse(wire([{ ...base }]));
    expect(out.results[0].parent_content).toBeNull();
    expect(out.results[0].parent_truncated_start).toBe(false);
    expect(out.results[0].parent_truncated_end).toBe(false);
  });

  it("carries the excerpt flags to every result citing the parent", () => {
    const parent = { content: "excerpt", truncated_start: true, truncated_end: false };
    const out = hydrateSearchResponse(
      wire([
        { ...base, parent_key: "A:9", parent },
        { ...base, parent_key: "A:9", parent: null },
      ]),
    );
    for (const r of out.results) {
      expect(r.parent_truncated_start).toBe(true);
      expect(r.parent_truncated_end).toBe(false);
    }
  });

  it("does not leak wire-only fields into the hydrated result", () => {
    const out = hydrateSearchResponse(
      wire([{ ...base, parent_key: "A:1", parent: { content: "p", truncated_start: false, truncated_end: false } }]),
    );
    expect(out.results[0]).not.toHaveProperty("parent_key");
    expect(out.results[0]).not.toHaveProperty("parent");
  });

  it("keeps distinct parents apart", () => {
    const out = hydrateSearchResponse(
      wire([
        { ...base, parent_key: "A:1", parent: { content: "one", truncated_start: false, truncated_end: false } },
        { ...base, parent_key: "B:1", parent: { content: "two", truncated_start: false, truncated_end: false } },
        { ...base, parent_key: "A:1", parent: null },
      ]),
    );
    expect(out.results.map((r) => r.parent_content)).toEqual(["one", "two", "one"]);
  });
});
