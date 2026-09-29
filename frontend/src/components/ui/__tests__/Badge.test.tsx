import { render, screen } from "@testing-library/react";
import { Badge } from "../Badge";

describe("Badge", () => {
  it("renders children", () => {
    render(<Badge>10-K</Badge>);
    expect(screen.getByText("10-K")).toBeInTheDocument();
  });

  it("applies gray variant by default", () => {
    render(<Badge>Default</Badge>);
    expect(screen.getByText("Default").className).toContain("bg-surface");
  });

  it.each([
    ["blue", "bg-accent/10"],
    ["green", "bg-pos/10"],
    ["amber", "bg-warn/10"],
    ["red", "bg-neg/10"],
  ] as const)("applies %s variant", (variant, expectedClass) => {
    render(<Badge variant={variant}>Label</Badge>);
    expect(screen.getByText("Label").className).toContain(expectedClass);
  });

  it("merges custom className", () => {
    render(<Badge className="extra">Tag</Badge>);
    expect(screen.getByText("Tag").className).toContain("extra");
  });

  it("renders as a span", () => {
    render(<Badge>Tag</Badge>);
    expect(screen.getByText("Tag").tagName).toBe("SPAN");
  });
});
