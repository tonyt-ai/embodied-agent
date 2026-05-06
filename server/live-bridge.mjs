import { WebSocketServer, WebSocket } from "ws";
import fs from "fs";
import path from "path";

function loadDotEnvLocal() {
  const envPath = path.resolve(process.cwd(), ".env.local");
  if (!fs.existsSync(envPath)) return;
  const text = fs.readFileSync(envPath, "utf8");
  for (const line of text.split(/\r?\n/)) {
    const m = line.match(/^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$/);
    if (!m) continue;
    const key = m[1];
    let value = m[2];
    if (
      (value.startsWith("'") && value.endsWith("'")) ||
      (value.startsWith('"') && value.endsWith('"'))
    ) {
      value = value.slice(1, -1);
    }
    if (!process.env[key]) process.env[key] = value;
  }
}

loadDotEnvLocal();

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

const LABEL_MODEL = process.env.GEMINI_LABEL_MODEL || "gemini-2.5-flash";
const DEBUG_GEMINI_RAW = new Set(["1", "true", "yes"]).has(
  String(process.env.DEBUG_GEMINI_RAW || "0").toLowerCase()
);
const LABEL_URL =
  "https://generativelanguage.googleapis.com/v1beta/models/" +
  encodeURIComponent(LABEL_MODEL) +
  ":generateContent?key=" +
  encodeURIComponent(GEMINI_API_KEY);

function normalizeLabel(label) {
  const text = String(label || "").trim().toLowerCase().replace(/_/g, " ").replace(/\s+/g, " ");
  const aliases = new Map([
    ["cake stand", "dish"],
    ["cake plate", "dish"],
    ["serving stand", "dish"],
    ["serving plate", "dish"],
    ["fruit stand", "dish"],
    ["fruit plate", "dish"],
    ["fruit bowl", "dish"],
  ]);
  return aliases.get(text) || text;
}

async function labelObjectsWithGemini(objects) {
  const compact = Array.isArray(objects) ? objects.slice(0, 8) : [];
  if (compact.length === 0) return [];

  const guidedPrompt = process.env.GEMINI_LABEL_PROMPT_MODE === "guided";
  const strictPrompt = !guidedPrompt;
  const sceneTargets = String(process.env.SCENE_TARGET_LABELS || "").split(",").map((x) => x.trim()).filter(Boolean);
  const sceneMovables = String(process.env.SCENE_MOVABLE_LABELS || "").split(",").map((x) => x.trim()).filter(Boolean);
  const sceneForbidden = String(process.env.SCENE_FORBIDDEN_LABELS || "").split(",").map((x) => x.trim()).filter(Boolean);
  const hasSceneVocabulary = sceneTargets.length > 0 || sceneMovables.length > 0;
  const objectParts = [];
  const promptIdToObjectId = new Map();
  for (const obj of compact) {
    if (!obj || typeof obj !== "object") continue;
    const id = String(obj.id || "");
    const mimeType = String(obj.mime_type || "image/jpeg");
    const data = String(obj.image_base64 || "");
    const hint = String(obj.label_hint || "unknown");
    if (!id || !data) continue;

    const promptId = strictPrompt ? `object_${promptIdToObjectId.size}` : id;
    promptIdToObjectId.set(promptId, id);
    objectParts.push({
      text: strictPrompt
        ? `Object id: ${promptId}.`
        : `Object id: ${id}. Current label hint: ${hint}.`,
    });
    objectParts.push({ inline_data: { mime_type: mimeType, data } });
  }

  if (objectParts.length === 0) return [];

  const sceneVocabularyPrompt = `You are labeling cropped tabletop objects for tracking in a hand-object interaction demo. Use only labels that are physically visible. Placement targets for this scene: ${sceneTargets.join(", ") || "target"}. Movable objects for this scene: ${sceneMovables.join(", ") || "object"}. For every object id provided, output exactly one short noun label (1-3 words) and confidence 0..1. If uncertain, use a low-confidence best visible guess from: object, ${[...sceneTargets, ...sceneMovables, "hand", "table"].join(", ")}. ${sceneForbidden.length ? `Do not use these labels in this scene: ${sceneForbidden.join(", ")}.` : ""} Return ONLY strict JSON array: [{"id":string,"label":string,"confidence":number}].`;
  const labelPrompt = hasSceneVocabulary
    ? sceneVocabularyPrompt
    : strictPrompt
    ? "You are labeling cropped tabletop objects for tracking. Use only what is visible in each crop and output exactly one short physical noun label (1-3 words) and confidence 0..1 for every object id. If the crop does not show enough object detail, use a low-confidence best guess such as object, tray, mat, bottle, toy, or hand. Return ONLY strict JSON array: [{\"id\":string,\"label\":string,\"confidence\":number}]."
    : "You are labeling tabletop objects for tracking in a hand-object interaction demo. Prefer useful physical labels from this vocabulary when visible: apple, banana, mug, cup, coaster, dish, plate, black mat, tray, white tray, plastic tray, bottle, baby bottle, toy giraffe, hand, table. The white raised scalloped plate/dessert stand that receives fruit should be labeled dish. A white plastic tray, including a tray with polka dots, should be labeled tray. Do not use teddy bear unless the crop is clearly a plush bear; for Sophie la girafe use toy giraffe. For every object id provided, output exactly one short noun label (1-3 words) and confidence 0..1. If uncertain, use your best visible guess; do not omit ids. Return ONLY strict JSON array: [{\"id\":string,\"label\":string,\"confidence\":number}].";

  const body = {
    contents: [
      {
        role: "user",
        parts: [
          {
            text: labelPrompt,
          },
          ...objectParts,
        ],
      },
    ],
    generationConfig: {
      temperature: 0.1,
      topP: 0.95,
      maxOutputTokens: 512,
      responseMimeType: "application/json",
    },
  };

  const resp = await fetch(LABEL_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!resp.ok) {
    const txt = await resp.text();
    throw new Error(`Gemini label HTTP ${resp.status}: ${txt.slice(0, 260)}`);
  }
  const json = await resp.json();
  const text =
    json?.candidates?.[0]?.content?.parts?.map((p) => p?.text || "").join("\n") || "[]";

  let parsed = [];
  try {
    parsed = JSON.parse(text);
  } catch {
    const m = text.match(/\[[\s\S]*\]/);
    if (m) parsed = JSON.parse(m[0]);
  }
  if (!Array.isArray(parsed)) return [];
  const labels = parsed
    .map((x) => ({
      id: promptIdToObjectId.get(String(x?.id || "")) || String(x?.id || ""),
      label: normalizeLabel(x?.label || "unknown"),
      confidence: Number(x?.confidence ?? 0),
    }))
    .filter((x) => x.id && x.label);
  if (labels.length === 0) {
    return fallbackLabelsFromHints(compact);
  }
  const byId = new Map(labels.map((x) => [x.id, x]));
  for (const obj of compact) {
    const id = String(obj?.id || "");
    if (id && !byId.has(id)) {
      byId.set(id, {
        id,
        label: normalizeLabel(obj?.label_hint || "unknown") || "unknown",
        confidence: 0.25,
      });
    }
  }
  return Array.from(byId.values()).map((item) => {
    const source = compact.find((obj) => String(obj?.id || "") === String(item.id || ""));
    return {
      ...item,
      label_hint: normalizeLabel(source?.label_hint || ""),
    };
  });
}

function fallbackLabelsFromHints(objects) {
  return (Array.isArray(objects) ? objects : [])
    .map((obj) => ({
      id: String(obj?.id || ""),
      label: normalizeLabel(obj?.label_hint || "unknown") || "unknown",
      confidence: 0.25,
    }))
    .filter((x) => x.id && x.label);
}

if (process.argv[2] === "--label-crops") {
  const cropPath = process.argv[3] || "world_model/data/gemini_label_crops.json";
  const objects = JSON.parse(fs.readFileSync(cropPath, "utf8"));
  labelObjectsWithGemini(objects)
    .then((labels) => {
      console.log(JSON.stringify({ ok: true, labels }, null, 2));
      process.exitCode = 0;
    })
    .catch((err) => {
      console.error(JSON.stringify({
        ok: false,
        labels: fallbackLabelsFromHints(objects),
        error: String(err),
      }, null, 2));
      process.exitCode = 1;
    });
} else {
const wss = new WebSocketServer({ port: PORT });

console.log(`Local Gemini bridge listening on ws://localhost:${PORT}`);

wss.on("connection", (browser) => {
  console.log("Browser connected to local bridge");

  let geminiReady = false;
  let worldSpeechBusy = false;
  let lastWorldSpeechText = "";
  let lastWorldSpeechAt = 0;
  let worldSpeechBusyTimer = null;

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
      if (DEBUG_GEMINI_RAW) {
        console.log("Gemini raw message:", JSON.stringify(msg));
      }

      if (msg.setupComplete || msg.setup_complete) {
        geminiReady = true;
        browser.send(JSON.stringify({ type: "gemini_ready" }));
        return;
      }

      const serverContent = msg.serverContent || msg.server_content;
      if (!serverContent) {
        return;
      }

      // Input transcript from user's mic
      const inputTranscript =
        serverContent.inputTranscription?.text ||
        serverContent.input_transcription?.text;
      if (inputTranscript) {
        browser.send(
          JSON.stringify({
            type: "input_transcript",
            text: inputTranscript,
          })
        );
      }

      // Output transcript from Gemini speech
      const outputTranscript =
        serverContent.outputTranscription?.text ||
        serverContent.output_transcription?.text;
      if (outputTranscript) {
        browser.send(
          JSON.stringify({
            type: "output_transcript",
            text: outputTranscript,
          })
        );
      }

      // Audio chunks from Gemini
      const parts = serverContent.modelTurn?.parts ?? serverContent.model_turn?.parts ?? [];
      for (const part of parts) {
        const audioData = part.inlineData?.data || part.inline_data?.data;
        if (audioData) {
          browser.send(
            JSON.stringify({
              type: "gemini_audio",
              data: audioData, // base64 PCM 24kHz
            })
          );
        }
      }

      if (serverContent.interrupted) {
        worldSpeechBusy = false;
        browser.send(JSON.stringify({ type: "interrupted" }));
      }

      if (serverContent.turnComplete || serverContent.turn_complete) {
        worldSpeechBusy = false;
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
          browser.send(
            JSON.stringify({
              type: "world_model_explanation_skipped",
              text: String(msg.text || ""),
              reason: "gemini-not-ready",
            })
          );
          return;
        }

        const text = String(msg.text || "");
        const now = Date.now();
        if (worldSpeechBusy && now - lastWorldSpeechAt > 1800) {
          worldSpeechBusy = false;
        }
        if (!text || text === lastWorldSpeechText || now - lastWorldSpeechAt < 300 || worldSpeechBusy) {
          browser.send(
            JSON.stringify({
              type: "world_model_explanation_skipped",
              text,
              reason: worldSpeechBusy ? "speech-busy" : "duplicate-or-too-soon",
            })
          );
          return;
        }
        worldSpeechBusy = true;
        lastWorldSpeechText = text;
        lastWorldSpeechAt = now;
        const prompt = `Speak exactly this short scene cue in neutral US English, no preface: ${text}`;
        if (worldSpeechBusyTimer) clearTimeout(worldSpeechBusyTimer);
        worldSpeechBusyTimer = setTimeout(() => {
          worldSpeechBusy = false;
        }, 2400);
        browser.send(
          JSON.stringify({
            type: "world_model_explanation_forwarded",
            text,
            prompt,
          })
        );
        gemini.send(
          JSON.stringify({
            clientContent: {
              turns: [
                {
                  role: "user",
                  parts: [{ text: prompt }],
                },
              ],
              turnComplete: true,
            },
          })
        );
        return;
      }

      if (msg.type === "world_label_request") {
        const requestId = String(msg.request_id || `${Date.now()}`);
        const objects = Array.isArray(msg.objects) ? msg.objects : [];
        labelObjectsWithGemini(objects)
          .then((labels) => {
            browser.send(
              JSON.stringify({
                type: "world_label_result",
                request_id: requestId,
                labels,
              })
            );
          })
          .catch((err) => {
            browser.send(
              JSON.stringify({
                type: "world_label_result",
                request_id: requestId,
                labels: fallbackLabelsFromHints(objects),
                error: String(err),
              })
            );
          });
        return;
      }

      return;
    } catch (err) {
      console.error("Failed to parse browser message:", err);
    }
  });

  browser.on("close", () => {
    console.log("Browser disconnected from local bridge");
    if (worldSpeechBusyTimer) {
      clearTimeout(worldSpeechBusyTimer);
      worldSpeechBusyTimer = null;
    }
    gemini.close();
  });

  browser.on("error", (err) => {
    console.error("Browser socket error:", err);
    gemini.close();
  });
});
}
