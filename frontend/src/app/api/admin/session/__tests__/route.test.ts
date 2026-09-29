// @vitest-environment node
/**
 * Admin login route: the brute-force limit keys on the address the
 * reverse proxy appended, not on a value the client chose (F5).
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { NextRequest } from "next/server";

import { resetAdminLoginRateLimit } from "@/lib/adminAuth";

import { POST } from "../route";

function login(key: string, xff: string): Promise<Response> {
  return POST(
    new NextRequest("http://localhost/api/admin/session", {
      method: "POST",
      headers: { "content-type": "application/json", "x-forwarded-for": xff },
      body: JSON.stringify({ admin_key: key }),
    }),
  );
}

describe("POST /api/admin/session", () => {
  beforeEach(() => {
    vi.stubEnv("ADMIN_API_KEY", "correct-admin-key");
    vi.spyOn(console, "warn").mockImplementation(() => {});
    resetAdminLoginRateLimit();
  });

  afterEach(() => {
    vi.unstubAllEnvs();
    vi.restoreAllMocks();
  });

  it("blocks the sixth guess even when the client rotates X-Forwarded-For", async () => {
    for (let i = 0; i < 5; i++) {
      const response = await login("wrong", `198.51.100.${i}, 203.0.113.9`);
      expect(response.status).toBe(403);
    }
    const blocked = await login("correct-admin-key", "198.51.100.77, 203.0.113.9");
    expect(blocked.status).toBe(429);
    expect(blocked.headers.get("Retry-After")).not.toBeNull();
  });

  it("does not block a different client", async () => {
    for (let i = 0; i < 5; i++) {
      await login("wrong", "203.0.113.9");
    }
    const other = await login("correct-admin-key", "203.0.113.10");
    expect(other.status).toBe(200);
  });
});
