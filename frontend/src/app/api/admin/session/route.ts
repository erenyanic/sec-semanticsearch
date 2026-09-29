import { timingSafeEqual } from "node:crypto";

import { NextRequest, NextResponse } from "next/server";

import {
  ADMIN_SESSION_COOKIE,
  adminLoginClientKey,
  buildAdminSessionValue,
  checkAdminLoginRate,
  getConfiguredAdminKey,
  hasValidAdminSession,
  recordFailedAdminLogin,
} from "@/lib/adminAuth";

export async function GET(request: NextRequest): Promise<NextResponse> {
  const adminKey = getConfiguredAdminKey();
  return NextResponse.json({
    admin_required: adminKey !== null,
    is_admin: hasValidAdminSession(request),
  });
}

export async function POST(request: NextRequest): Promise<NextResponse> {
  const expected = getConfiguredAdminKey();
  if (expected === null) {
    return NextResponse.json({ admin_required: false, is_admin: true });
  }

  // Brute-force protection: rate limit failed login attempts per IP (F5).
  const clientIp = adminLoginClientKey(request.headers);

  const rateCheck = checkAdminLoginRate(clientIp);
  if (!rateCheck.allowed) {
    console.warn(
      `[SECURITY] Admin login rate limit exceeded for IP ${clientIp}`,
    );
    return NextResponse.json(
      {
        error: "rate_limited",
        message: `Too many login attempts. Try again in ${rateCheck.retryAfter}s.`,
      },
      {
        status: 429,
        headers: { "Retry-After": String(rateCheck.retryAfter) },
      },
    );
  }

  const body = await request.json().catch(() => null);
  const provided = typeof body?.admin_key === "string" ? body.admin_key.trim() : "";

  const providedBytes = Buffer.from(provided, "utf-8");
  const expectedBytes = Buffer.from(expected, "utf-8");
  const keysMatch =
    providedBytes.length === expectedBytes.length &&
    timingSafeEqual(providedBytes, expectedBytes);

  if (!keysMatch) {
    recordFailedAdminLogin(clientIp);
    console.warn(
      `[SECURITY] Failed admin login attempt from IP ${clientIp}`,
    );
    return NextResponse.json(
      { error: "admin_required", message: "Invalid admin key." },
      { status: 403 },
    );
  }

  const response = NextResponse.json({ admin_required: true, is_admin: true });
  response.cookies.set({
    name: ADMIN_SESSION_COOKIE,
    value: buildAdminSessionValue(expected),
    httpOnly: true,
    sameSite: "lax",
    secure: process.env.NODE_ENV === "production",
    path: "/",
    maxAge: 3600,  // 1 hour — force re-authentication (F6)
  });
  return response;
}

export async function DELETE(): Promise<NextResponse> {
  const response = NextResponse.json({ admin_required: true, is_admin: false });
  response.cookies.set({
    name: ADMIN_SESSION_COOKIE,
    value: "",
    httpOnly: true,
    sameSite: "lax",
    secure: process.env.NODE_ENV === "production",
    path: "/",
    maxAge: 0,
  });
  return response;
}