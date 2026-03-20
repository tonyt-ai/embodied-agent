import { NextResponse } from "next/server";

async function createSession() {
  try {
    const apiKey = process.env.LIVEAVATAR_API_KEY;

    if (!apiKey) {
      return NextResponse.json(
        { step: "env", error: "Missing LIVEAVATAR_API_KEY in .env.local" },
        { status: 500 }
      );
    }

    const tokenPayload = {
      mode: "LITE",
      avatar_id: "38eedff2-9761-44a2-8d8f-0d674d0e6500", //"073b60a9-89a8-45aa-8902-c358f64d2852", //Katya Sitting
    };

    const tokenRes = await fetch("https://api.liveavatar.com/v1/sessions/token", {
      method: "POST",
      headers: {
        "X-API-KEY": apiKey,
        "accept": "application/json",
        "content-type": "application/json",
      },
      body: JSON.stringify(tokenPayload),
    });

    const tokenText = await tokenRes.text();
    let tokenData: any;
    try {
      tokenData = JSON.parse(tokenText);
    } catch {
      tokenData = { rawText: tokenText };
    }

    if (!tokenRes.ok) {
      return NextResponse.json(
        {
          step: "token",
          status: tokenRes.status,
          sent: tokenPayload,
          error: tokenData,
        },
        { status: tokenRes.status }
      );
    }

    const sessionToken =
      tokenData?.data?.session_token ?? tokenData?.session_token;
    const sessionId =
      tokenData?.data?.session_id ?? tokenData?.session_id;

    if (!sessionToken) {
      return NextResponse.json(
        {
          step: "token-parse",
          error: "Missing session_token in token response",
          tokenData,
        },
        { status: 500 }
      );
    }

    const startRes = await fetch("https://api.liveavatar.com/v1/sessions/start", {
      method: "POST",
      headers: {
        "accept": "application/json",
        "authorization": `Bearer ${sessionToken}`,
      },
    });

    const startText = await startRes.text();
    let startData: any;
    try {
      startData = JSON.parse(startText);
    } catch {
      startData = { rawText: startText };
    }

    if (!startRes.ok) {
      return NextResponse.json(
        {
          step: "start",
          status: startRes.status,
          sessionId,
          error: startData,
        },
        { status: startRes.status }
      );
    }

    return NextResponse.json({
      sessionId,
      startData,
    });
  } catch (error) {
    return NextResponse.json(
      {
        step: "catch",
        error: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 }
    );
  }
}

export async function GET() {
  return createSession();
}

export async function POST() {
  return createSession();
}