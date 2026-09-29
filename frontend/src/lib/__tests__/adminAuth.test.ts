/**
 * Tests for admin login rate limiter (F5 — brute-force protection).
 */
import { describe, it, expect, beforeEach } from "vitest";

import {
  adminLoginClientKey,
  checkAdminLoginRate,
  recordFailedAdminLogin,
  resetAdminLoginRateLimit,
} from "@/lib/adminAuth";

describe("Admin login rate limiter", () => {
  beforeEach(() => {
    resetAdminLoginRateLimit();
  });

  it("allows the first attempt", () => {
    const result = checkAdminLoginRate("192.168.1.1");
    expect(result.allowed).toBe(true);
    expect(result.retryAfter).toBe(0);
  });

  it("allows up to 5 failed attempts within a minute", () => {
    const ip = "10.0.0.1";
    for (let i = 0; i < 5; i++) {
      recordFailedAdminLogin(ip);
    }
    // The 5 failures are recorded; the next CHECK should be blocked.
    const result = checkAdminLoginRate(ip);
    expect(result.allowed).toBe(false);
    expect(result.retryAfter).toBeGreaterThanOrEqual(1);
  });

  it("blocks the 6th attempt after 5 failures", () => {
    const ip = "172.16.0.1";
    for (let i = 0; i < 5; i++) {
      recordFailedAdminLogin(ip);
    }
    const result = checkAdminLoginRate(ip);
    expect(result.allowed).toBe(false);
  });

  it("isolates rate limits per IP", () => {
    const attacker = "evil.attacker";
    const legitimate = "good.user";

    // Exhaust attacker's budget
    for (let i = 0; i < 5; i++) {
      recordFailedAdminLogin(attacker);
    }

    // Attacker is blocked
    expect(checkAdminLoginRate(attacker).allowed).toBe(false);
    // Legitimate user is not affected
    expect(checkAdminLoginRate(legitimate).allowed).toBe(true);
  });

  it("returns retry-after in seconds", () => {
    const ip = "10.0.0.2";
    for (let i = 0; i < 5; i++) {
      recordFailedAdminLogin(ip);
    }
    const result = checkAdminLoginRate(ip);
    expect(result.allowed).toBe(false);
    expect(typeof result.retryAfter).toBe("number");
    expect(result.retryAfter).toBeGreaterThanOrEqual(1);
    expect(result.retryAfter).toBeLessThanOrEqual(62); // window is 60s + 1
  });

  it("reset clears all tracked attempts", () => {
    const ip = "10.0.0.3";
    for (let i = 0; i < 5; i++) {
      recordFailedAdminLogin(ip);
    }
    expect(checkAdminLoginRate(ip).allowed).toBe(false);

    resetAdminLoginRateLimit();
    expect(checkAdminLoginRate(ip).allowed).toBe(true);
  });
});

describe("adminLoginClientKey", () => {
  const key = (xff?: string) =>
    adminLoginClientKey(new Headers(xff === undefined ? {} : { "x-forwarded-for": xff }));

  it("uses the entry appended by the nearest proxy (right-most)", () => {
    // nginx / Cloud Run append the real address after the client's value.
    expect(key("6.6.6.6, 203.0.113.9")).toBe("203.0.113.9");
    expect(key("1.1.1.1, 2.2.2.2 ,  203.0.113.9 ")).toBe("203.0.113.9");
  });

  it("uses a single entry as is", () => {
    expect(key("203.0.113.9")).toBe("203.0.113.9");
  });

  it("falls back to a shared key without the header", () => {
    expect(key()).toBe("unknown");
    expect(key(" , ")).toBe("unknown");
  });

  it("ignores X-Real-IP, which a client can set on Cloud Run", () => {
    const headers = new Headers({ "x-real-ip": "6.6.6.6", "x-forwarded-for": "203.0.113.9" });
    expect(adminLoginClientKey(headers)).toBe("203.0.113.9");
  });

  it("a rotating spoofed prefix cannot escape the limit", () => {
    resetAdminLoginRateLimit();
    for (let i = 0; i < 5; i++) {
      recordFailedAdminLogin(key(`10.9.9.${i}, 203.0.113.9`));
    }
    expect(checkAdminLoginRate(key("10.9.9.99, 203.0.113.9")).allowed).toBe(false);
  });
});
