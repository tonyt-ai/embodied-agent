import fs from "fs";
import { WebSocket } from "ws";

const url = process.env.GEMINI_BRIDGE_URL || "ws://localhost:8081";
const cropPath = process.argv[2] || "world_model/data/gemini_label_crops.json";
const objects = JSON.parse(fs.readFileSync(cropPath, "utf8"));

const ws = new WebSocket(url);
const timeout = setTimeout(() => {
  console.error(JSON.stringify({ ok: false, error: "timeout waiting for bridge label result" }));
  try {
    ws.close();
  } catch {}
  process.exit(2);
}, 45000);

ws.on("open", () => {
  ws.send(JSON.stringify({
    type: "world_label_request",
    request_id: `cli_${Date.now()}`,
    objects,
  }));
});

ws.on("message", (raw) => {
  const msg = JSON.parse(raw.toString());
  if (msg.type !== "world_label_result") return;
  clearTimeout(timeout);
  console.log(JSON.stringify({ ok: !msg.error, ...msg }, null, 2));
  ws.close();
});

ws.on("error", (err) => {
  clearTimeout(timeout);
  console.error(JSON.stringify({ ok: false, error: String(err) }));
  process.exit(1);
});
