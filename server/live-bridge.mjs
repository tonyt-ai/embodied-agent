import { WebSocketServer, WebSocket } from "ws";

const GEMINI_API_KEY = process.env.GEMINI_API_KEY;

if (!GEMINI_API_KEY) {
  throw new Error("Missing GEMINI_API_KEY");
}

const PORT = 8081;

// Native audio output model for realtime speech-in / speech-out.
const MODEL = "models/gemini-2.5-flash-native-audio-preview-12-2025";

const GEMINI_URL =
  "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key=" +
  encodeURIComponent(GEMINI_API_KEY);

const wss = new WebSocketServer({ port: PORT });

console.log(`Local Gemini bridge listening on ws://localhost:${PORT}`);

wss.on("connection", (browser) => {
  console.log("Browser connected to local bridge");

  let geminiReady = false;

  const gemini = new WebSocket(GEMINI_URL);

  gemini.on("open", () => {
    console.log("Connected to Gemini Live");

    // IMPORTANT: first message must be { setup: { ... } }
    gemini.send(
      JSON.stringify({
        setup: {
          model: MODEL,
          generationConfig: {
            responseModalities: ["AUDIO"],
            speechConfig: {
              voiceConfig: {
                prebuiltVoiceConfig: {
                  voiceName: "Orus",
                },
              },
            },
          },
          inputAudioTranscription: {},
          outputAudioTranscription: {},
          systemInstruction: {
            parts: [
              {
                text:
                  "You are a concise realtime avatar assistant. Speak in a calm natural male voice.",
              },
            ],
          },
        },
      })
    );
  });

  gemini.on("message", (raw) => {
    try {
      const msg = JSON.parse(raw.toString());
      console.log("Gemini raw message:", JSON.stringify(msg));

      if (msg.setupComplete) {
        geminiReady = true;
        browser.send(JSON.stringify({ type: "gemini_ready" }));
        return;
      }

      const serverContent = msg.serverContent;
      if (!serverContent) {
        return;
      }

      // Input transcript from user's mic
      if (serverContent.inputTranscription?.text) {
        browser.send(
          JSON.stringify({
            type: "input_transcript",
            text: serverContent.inputTranscription.text,
          })
        );
      }

      // Output transcript from Gemini speech
      if (serverContent.outputTranscription?.text) {
        browser.send(
          JSON.stringify({
            type: "output_transcript",
            text: serverContent.outputTranscription.text,
          })
        );
      }

      // Audio chunks from Gemini
      const parts = serverContent.modelTurn?.parts ?? [];
      for (const part of parts) {
        if (part.inlineData?.data) {
          browser.send(
            JSON.stringify({
              type: "gemini_audio",
              data: part.inlineData.data, // base64 PCM 24kHz
            })
          );
        }
      }

      if (serverContent.interrupted) {
        browser.send(JSON.stringify({ type: "interrupted" }));
      }

      if (serverContent.turnComplete) {
        browser.send(JSON.stringify({ type: "turn_complete" }));
      }
    } catch (err) {
      console.error("Failed to parse Gemini message:", err);
      try {
        browser.send(
          JSON.stringify({
            type: "gemini_error",
            error: String(err),
          })
        );
      } catch {}
    }
  });

  gemini.on("close", (code, reason) => {
    console.log("Gemini socket closed:", code, reason.toString());
    try {
      browser.send(
        JSON.stringify({
          type: "gemini_closed",
          code,
          reason: reason.toString(),
        })
      );
    } catch {}
  });

  gemini.on("error", (err) => {
    console.error("Gemini socket error:", err);
    try {
      browser.send(
        JSON.stringify({
          type: "gemini_error",
          error: String(err),
        })
      );
    } catch {}
  });

  browser.on("message", (raw) => {
    try {
      const msg = JSON.parse(raw.toString());

      if (msg.type === "mic_audio") {
        if (!geminiReady) {
          return;
        }

        gemini.send(
          JSON.stringify({
            realtimeInput: {
              audio: {
                data: msg.data,
                mimeType: "audio/pcm;rate=16000",
              },
            },
          })
        );
        return;
      }

      if (msg.type === "end_audio") {
        if (!geminiReady) {
          return;
        }

        gemini.send(
          JSON.stringify({
            realtimeInput: {
              audioStreamEnd: true,
            },
          })
        );
        return;
      }

      if (msg.type === "text") {
        if (!geminiReady) {
          return;
        }

        gemini.send(
          JSON.stringify({
            realtimeInput: {
              text: msg.text,
            },
          })
        );
        return;
      }

      if (msg.type === "world_model_explanation") {
        if (!geminiReady) {
          return;
        }

        gemini.send(
          JSON.stringify({
            realtimeInput: {
              text: `Repeat exactly the following command and nothing else: ${msg.text}`,
            },
          })
        );
        return;
      }

      return;
    } catch (err) {
      console.error("Failed to parse browser message:", err);
    }
  });

  browser.on("close", () => {
    console.log("Browser disconnected from local bridge");
    gemini.close();
  });

  browser.on("error", (err) => {
    console.error("Browser socket error:", err);
    gemini.close();
  });
});