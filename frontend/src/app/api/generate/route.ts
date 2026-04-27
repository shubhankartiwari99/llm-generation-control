import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = (process.env.BACKEND_API_URL ?? "http://127.0.0.1:8000").replace(/\/+$/, "");

export async function POST(req: NextRequest) {
  try {
    const payload = await req.json();
    const response = await fetch(`${BACKEND_URL}/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      cache: "no-store",
    });

    const text = await response.text();
    return new NextResponse(text, {
      status: response.status,
      headers: { "Content-Type": response.headers.get("content-type") ?? "application/json" },
    });
  } catch {
    return NextResponse.json(
      { detail: "Frontend proxy failed to reach backend /generate endpoint" },
      { status: 502 },
    );
  }
}

