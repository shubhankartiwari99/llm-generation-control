import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_API_URL ?? "http://127.0.0.1:8000";

export async function GET(req: NextRequest) {
  const { searchParams } = new URL(req.url);
  const limit = searchParams.get("limit") ?? "8";

  try {
    const response = await fetch(`${BACKEND_URL}/runs/recent?limit=${encodeURIComponent(limit)}`, {
      method: "GET",
      cache: "no-store",
    });

    const text = await response.text();
    return new NextResponse(text, {
      status: response.status,
      headers: { "Content-Type": response.headers.get("content-type") ?? "application/json" },
    });
  } catch {
    return NextResponse.json(
      { detail: "Frontend proxy failed to reach backend /runs/recent endpoint" },
      { status: 502 },
    );
  }
}

