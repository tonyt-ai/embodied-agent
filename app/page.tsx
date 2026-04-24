"use client";

import { useEffect, useRef, useState } from "react";
import { Room, RoomEvent, Track } from "livekit-client";

// This file implements the main UI and orchestration for the Embodied Agent demo.
// - LiveAvatar connection (video + audio playback from server)
// - Gemini bridge for AI speech transcription + generation
// - World model live loop for camera observation, object detection, and planning
// - Local microphone capture and resampling for streaming to remote agent

// Convert float32 audio samples into signed 16-bit PCM for downstream processing.
function floatTo16BitPCM(float32Array: Float32Array) {
  const buffer = new ArrayBuffer(float32Array.length * 2);
  const view = new DataView(buffer);

  let offset = 0;
  for (let i = 0; i < float32Array.length; i++, offset += 2) {
    let s = Math.max(-1, Math.min(1, float32Array[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }

  return new Uint8Array(buffer);
}

// Encode raw binary bytes as a base64 string for WebSocket transport.
function bytesToBase64(bytes: Uint8Array) {
  let binary = "";
  for (let i = 0; i < bytes.length; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

// Resample Float32 PCM from one sample rate to another (linear interpolation).
function resampleFloat32(
  input: Float32Array,
  inputSampleRate: number,
  outputSampleRate: number
) {
  if (inputSampleRate === outputSampleRate) return input;

  const ratio = inputSampleRate / outputSampleRate;
  const newLength = Math.max(1, Math.round(input.length / ratio));
  const result = new Float32Array(newLength);

  for (let i = 0; i < newLength; i++) {
    const index = i * ratio;
    const indexFloor = Math.floor(index);
    const indexCeil = Math.min(indexFloor + 1, input.length - 1);
    const frac = index - indexFloor;

    const sample =
      input[indexFloor] * (1 - frac) + input[indexCeil] * frac;

    result[i] = sample;
  }

  return result;
}

// Generate an event id for agent speech turns / interruptions.
function randomEventId() {
  return crypto.randomUUID();
}

type CaptureMode = "social" | "embodied";
type AvatarMode = "ai" | "direct";
type EmbodiedVideoSource = "scene" | "camera";

type ButtonTone = "primary" | "secondary" | "danger";

// UI helper: card container reboot style
function cardStyle(): React.CSSProperties {
  return {
    background: "#ffffff",
    border: "1px solid #e2e8f0",
    borderRadius: 20,
    boxShadow: "0 8px 24px rgba(15, 23, 42, 0.06)",
  };
}

function actionButtonStyle(
  tone: ButtonTone,
  disabled = false
): React.CSSProperties {
  const base: React.CSSProperties = {
    borderRadius: 12,
    padding: "10px 14px",
    fontSize: 14,
    fontWeight: 600,
    border: "1px solid transparent",
    cursor: disabled ? "not-allowed" : "pointer",
    opacity: disabled ? 0.55 : 1,
    transition: "all 0.15s ease",
    minHeight: 42,
  };

  if (tone === "primary") {
    return {
      ...base,
      background: "#0f172a",
      color: "#ffffff",
      borderColor: "#0f172a",
    };
  }

  // Secondary and danger share a common base visual logic.

  if (tone === "danger") {
    return {
      ...base,
      background: "#fff1f2",
      color: "#be123c",
      borderColor: "#fecdd3",
    };
  }

  return {
    ...base,
    background: "#ffffff",
    color: "#0f172a",
    borderColor: "#cbd5e1",
  };
}

function toggleChipStyle(active: boolean, disabled = false): React.CSSProperties {
  return {
    padding: "10px 14px",
    borderRadius: 999,
    border: `1px solid ${active ? "#93c5fd" : "#cbd5e1"}`,
    background: active ? "#eff6ff" : "#ffffff",
    color: active ? "#1d4ed8" : "#334155",
    fontWeight: 600,
    fontSize: 14,
    cursor: disabled ? "not-allowed" : "pointer",
    opacity: disabled ? 0.55 : 1,
  };
}

function statusDotColor(active: boolean) {
  return active ? "#16a34a" : "#94a3b8";
}

function parseJsonSafe<T>(value: string, fallback: T): T {
  try {
    return value ? JSON.parse(value) : fallback;
  } catch {
    return fallback;
  }
}

// Main React component for the app page. Manages refs, state, connections,
// and UI rendering for the agent + world model + camera pipeline.
export default function Home() {
  const videoRef = useRef<HTMLVideoElement>(null);

  const roomRef = useRef<Room | null>(null);
  const avatarWsRef = useRef<WebSocket | null>(null);
  const geminiBridgeRef = useRef<WebSocket | null>(null);
  const remoteAudioElRef = useRef<HTMLAudioElement | null>(null);

  const mediaStreamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const workletNodeRef = useRef<AudioWorkletNode | null>(null);
  const silentGainRef = useRef<GainNode | null>(null);

  const currentAvatarEventIdRef = useRef<string | null>(null);
  const avatarTurnStartedRef = useRef(false);

  const [captureMode, setCaptureMode] = useState<CaptureMode>("embodied");
  const [avatarMode, setAvatarMode] = useState<AvatarMode>("ai");
  const [embodiedVideoSource, setEmbodiedVideoSource] = useState<EmbodiedVideoSource>("scene");
  const [status, setStatus] = useState("idle");
  const [avatarConnected, setAvatarConnected] = useState(false);
  const [bridgeConnected, setBridgeConnected] = useState(false);
  const [micOn, setMicOn] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [inputTranscript, setInputTranscript] = useState("");
  const [outputTranscript, setOutputTranscript] = useState("");
  const [directAudioMonitor, setDirectAudioMonitor] = useState(false);
  const [observedObjects, setObservedObjects] = useState<any[]>([]);

  const [videoDevices, setVideoDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedCameraId, setSelectedCameraId] = useState<string | null>(null);

  const worldModelWsRef = useRef<WebSocket | null>(null);
  const localCamRef = useRef<HTMLVideoElement>(null);
  const localCamStreamRef = useRef<MediaStream | null>(null);
  const frameIntervalRef = useRef<number | null>(null);
  const frameCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const worldFrameInFlightRef = useRef(false);
  const pendingWorldFrameRef = useRef<{
    image: string;
    timestamp: number;
    width: number;
    height: number;
  } | null>(null);
  const lastDebugTextUpdateRef = useRef<number>(0);

  const [worldStateText, setWorldStateText] = useState("");
  const [eventLog, setEventLog] = useState("");
  const [lastQueryResultText, setLastQueryResultText] = useState("");
  const [plannerSummary, setPlannerSummary] = useState("");
  const [plannerSimulations, setPlannerSimulations] = useState<any | null>(null);
  const [bestActionName, setBestActionName] = useState("");
  const [cameraPoseText, setCameraPoseText] = useState("");
  const [objects3dText, setObjects3dText] = useState("");
  const [sparseMapText, setSparseMapText] = useState("");
  const [sparseMapData, setSparseMapData] = useState<any[]>([]);
  const [processedFrameUrl, setProcessedFrameUrl] = useState("");
  const [processedFrameTimestamp, setProcessedFrameTimestamp] = useState<number | null>(null);
  const [processedFrameLandmarks, setProcessedFrameLandmarks] = useState<any[]>([]);
  const [processedFrameSize, setProcessedFrameSize] = useState({ width: 640, height: 360 });
  const [depthDebugSize, setDepthDebugSize] = useState({ width: 160, height: 90 });
  const [handsText, setHandsText] = useState("");
  const [worldDebugText, setWorldDebugText] = useState("");
  const [depthDebugUrl, setDepthDebugUrl] = useState("");

  const [autoMode, setAutoMode] = useState(false);
  const [showDebug, setShowDebug] = useState(false);
  const lastSpokenRef = useRef<string>("");
  const lastSpeakTimeRef = useRef<number>(0);
  const [useAvatarSpeech, setUseAvatarSpeech] = useState(false);

  const [frameAgeMs, setFrameAgeMs] = useState<number | null>(null);
  const [captureMs, setCaptureMs] = useState<number | null>(null);
  const [serverDecodeMs, setServerDecodeMs] = useState<number | null>(null);
  const [serverDetectMs, setServerDetectMs] = useState<number | null>(null);
  const [serverDepthMs, setServerDepthMs] = useState<number | null>(null);
  const [serverPoseMs, setServerPoseMs] = useState<number | null>(null);
  const [serverWorldMs, setServerWorldMs] = useState<number | null>(null);
  const [serverTotalMs, setServerTotalMs] = useState<number | null>(null);
  const [pipelineAgeMs, setPipelineAgeMs] = useState<number | null>(null);

  const isEmbodiedMode = captureMode === "embodied";
  const isSocialMode = captureMode === "social";
  const cameraPoseData = parseJsonSafe<any>(cameraPoseText, null);
  const persistentMapData = Array.isArray(cameraPoseData?.persistent_map)
    ? cameraPoseData.persistent_map
    : sparseMapData;
  const cameraPositionWorld = Array.isArray(cameraPoseData?.camera_position_world)
    ? cameraPoseData.camera_position_world
    : [0, 0, 0];
  const mapPoints3d = persistentMapData
    .map((point: any) => {
      const position = point?.position_world;
      if (!Array.isArray(position) || position.length < 3) return null;
      const x = Number(position[0]);
      const y = Number(position[1]);
      const z = Number(position[2]);
      if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) return null;
      return {
        id: point?.id,
        x,
        y,
        z,
        quality: Number(point?.quality ?? 0),
        hits: Number(point?.hits ?? 0),
        status: point?.status || "visible",
        isLocal: Boolean(point?.is_local_map),
      };
    })
    .filter(Boolean) as Array<{
      id: number | string;
      x: number;
      y: number;
      z: number;
      quality: number;
      hits: number;
      status: string;
      isLocal: boolean;
    }>;
  const cameraMapX = Number(cameraPositionWorld[0] || 0);
  const cameraMapY = Number(cameraPositionWorld[1] || 0);
  const cameraMapZ = Number(cameraPositionWorld[2] || 0);
  const landmarkLifecycle = cameraPoseData?.landmark_lifecycle || {};
  const descriptorBackend = cameraPoseData?.descriptor_backend || {};
  const featureBackend = cameraPoseData?.feature_backend || {};
  const mapExtent = Math.max(
    0.25,
    ...mapPoints3d.flatMap((point) => [
      Math.abs(point.x - cameraMapX),
      Math.abs(point.z - cameraMapZ),
    ]),
  );
  const mapScale = 44 / mapExtent;
  const sideMapExtent = Math.max(
    0.25,
    ...mapPoints3d.flatMap((point) => [
      Math.abs(point.z - cameraMapZ),
      Math.abs(point.y - cameraMapY),
    ]),
  );
  const sideMapScale = 44 / sideMapExtent;
  const pointXs = mapPoints3d.map((point) => point.x);
  const pointYs = mapPoints3d.map((point) => point.y);
  const pointZs = mapPoints3d.map((point) => point.z);
  const axisRange = (values: number[]) => (
    values.length > 0 ? Math.max(...values) - Math.min(...values) : 0
  );
  const mapRangeX = axisRange(pointXs);
  const mapRangeY = axisRange(pointYs);
  const mapRangeZ = axisRange(pointZs);
  const sideYExaggeration = mapRangeY > 0 && mapRangeY < mapRangeZ * 0.35 ? 3 : 1;
  const mapHealthNotes = [
    (cameraPoseData?.geometry_verified_landmark_count ?? 0) < 15
      ? "Few geometry-verified landmarks: 3D anchors are still noisy."
      : null,
    (cameraPoseData?.covisibility_edges ?? 0) < 3 && (cameraPoseData?.keyframes ?? 0) > 4
      ? "Weak covisibility graph: views are not sharing enough persistent points."
      : null,
    (landmarkLifecycle.pruned ?? 0) > Math.max(500, (landmarkLifecycle.reassociated ?? landmarkLifecycle.descriptor_reassociated ?? 0) * 8)
      ? "High landmark churn: features are being created and pruned faster than they reconnect."
      : null,
    (cameraPoseData?.local_visible_landmark_count ?? 0) < 12 && cameraPoseData?.pnp_anchor_scope === "local-map"
      ? "Local PnP has thin support: pose may work but remain fragile."
      : null,
    (cameraPoseData?.local_keyframe_baseline ?? 0) < 0.03 && (cameraPoseData?.keyframes ?? 0) > 2
      ? "Low local baseline: 3D triangulation has too little parallax."
      : null,
    mapRangeY > 0 && mapRangeY < mapRangeZ * 0.2
      ? "Flat side view: vertical spread is much smaller than depth spread."
      : null,
  ].filter(Boolean) as string[];

  const socialState = {
    inputTranscript,
    outputTranscript,
    directAudioMonitor,
    useAvatarSpeech,
    isSpeaking,
  };

  const embodiedState = {
    observedObjects,
    worldStateText,
    eventLog,
    lastQueryResultText,
    plannerSummary,
    plannerSimulations,
    bestActionName,
    cameraPoseText,
    objects3dText,
    sparseMapText,
    processedFrameUrl,
    processedFrameTimestamp,
    processedFrameLandmarks,
    processedFrameSize,
    depthDebugSize,
    handsText,
    worldDebugText,
    depthDebugUrl,
    autoMode,
    frameAgeMs,
    captureMs,
    serverDecodeMs,
    serverDetectMs,
    serverDepthMs,
    serverPoseMs,
    serverWorldMs,
    serverTotalMs,
    pipelineAgeMs,
  };

  function resetWorldUiState() {
    setWorldStateText("");
    setEventLog("");
    setLastQueryResultText("");
    setPlannerSummary("");
    setPlannerSimulations(null);
    setBestActionName("");
    setCameraPoseText("");
    setObjects3dText("");
    setSparseMapText("");
    setSparseMapData([]);
    setProcessedFrameUrl("");
    setProcessedFrameTimestamp(null);
    setProcessedFrameLandmarks([]);
    setProcessedFrameSize({ width: 640, height: 360 });
    setDepthDebugSize({ width: 160, height: 90 });
    setHandsText("");
    setWorldDebugText("");
    setDepthDebugUrl("");
    setInputTranscript("");
    setOutputTranscript("");
    setObservedObjects([]);
    setFrameAgeMs(null);
    setCaptureMs(null);
    setServerDecodeMs(null);
    setServerDetectMs(null);
    setServerDepthMs(null);
    setServerPoseMs(null);
    setServerWorldMs(null);
    setServerTotalMs(null);
    setPipelineAgeMs(null);
    lastSpokenRef.current = "";
    lastSpeakTimeRef.current = 0;
    setAutoMode(false);
    window.speechSynthesis?.cancel();
  }

  // Establish avatar session via backend API and prepare world model + speech pipeline.
  async function startAvatar() {
    try {
      setStatus("Creating LiveAvatar session...");
      setAvatarConnected(false);
      setBridgeConnected(false);
      setIsSpeaking(false);
      setInputTranscript("");
      setOutputTranscript("");
      currentAvatarEventIdRef.current = null;
      avatarTurnStartedRef.current = false;

      resetWorldUiState();
      stopAllConnections();

      const res = await fetch("/api/liveavatar/session", { method: "POST" });
      const data = await res.json();

      if (!res.ok) {
        setStatus("Error creating session");
        return;
      }

      const livekitUrl = data?.startData?.data?.livekit_url;
      const livekitClientToken = data?.startData?.data?.livekit_client_token;
      const wsUrl = data?.startData?.data?.ws_url;

      if (!livekitUrl || !livekitClientToken || !wsUrl) {
        setStatus("Missing connection info");
        return;
      }

      const room = new Room();

      room.on(RoomEvent.TrackSubscribed, (track) => {
        if (track.kind === Track.Kind.Video && videoRef.current) {
          track.attach(videoRef.current);
        }

        if (track.kind === Track.Kind.Audio) {
          if (remoteAudioElRef.current) {
            remoteAudioElRef.current.remove();
            remoteAudioElRef.current = null;
          }

          const shouldPlayAudio =
            avatarMode === "ai" || (avatarMode === "direct" && directAudioMonitor);

          if (shouldPlayAudio) {
            const audioEl = track.attach();
            audioEl.autoplay = true;
            audioEl.style.display = "none";
            document.body.appendChild(audioEl);
            remoteAudioElRef.current = audioEl;
          }
        }
      });

      room.on(RoomEvent.Disconnected, () => {
        setAvatarConnected(false);
      });

      await room.connect(livekitUrl, livekitClientToken);
      roomRef.current = room;

      const avatarWs = new WebSocket(wsUrl);

      avatarWs.onopen = async () => {
        avatarWsRef.current = avatarWs;
        setAvatarConnected(true);

        try {
          if (isEmbodiedMode) {
            connectWorldModel();
            await startLocalCamera();
            startSendingFrames();
          } else {
            await startLocalCamera();
          }
        } catch (err) {
          console.error("Camera/world-model startup failed:", err);
        }

        if (avatarMode === "ai" || useAvatarSpeech) {
          connectGeminiBridge();
        } else {
          setBridgeConnected(false);
        }

        setStatus(
          isEmbodiedMode
            ? (avatarMode === "ai" ? "Avatar ready (Embodied AI)" : "Avatar ready (Embodied Direct)")
            : (avatarMode === "ai" ? "Avatar ready (Social AI)" : "Avatar ready (Social Direct)")
        );
      };

      avatarWs.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);

          if (msg.type === "agent.speak_started") {
            setIsSpeaking(true);
          }

          if (msg.type === "agent.speak_ended") {
            setIsSpeaking(false);
            avatarTurnStartedRef.current = false;
            currentAvatarEventIdRef.current = null;
            setStatus(avatarMode === "ai" ? "Avatar finished speaking" : "Avatar finished direct speech");
          }

          if (msg.type === "agent.speak_interrupted") {
            setIsSpeaking(false);
            avatarTurnStartedRef.current = false;
            currentAvatarEventIdRef.current = null;
            setStatus("Avatar interrupted");
          }
        } catch {}
      };

      avatarWs.onerror = () => {
        setStatus("Avatar WebSocket error");
      };

      avatarWs.onclose = (event) => {
        setAvatarConnected(false);
        setIsSpeaking(false);
        setStatus(`Avatar WebSocket closed (${event.code})`);
      };
    } catch (error) {
      console.error("startAvatar error:", error);
      setStatus(error instanceof Error ? error.message : "Unknown connection error");
    }
  }

  async function startWorldModelOnly() {
    if (!isEmbodiedMode) {
      setStatus("World model only is available in Embodied mode");
      return;
    }

    try {
      setStatus("Starting world model only...");
      resetWorldUiState();
      stopAllConnections();

      connectWorldModel();

      if (useAvatarSpeech) {
        connectGeminiBridge();
      }

      await startLocalCamera();
      startSendingFrames();

      setStatus("World model running (no avatar)");
    } catch (err) {
      console.error("World model startup failed:", err);
      setStatus("World model error");
    }
  }

  function connectGeminiBridge() {
    if (geminiBridgeRef.current && geminiBridgeRef.current.readyState === WebSocket.OPEN) {
      return;
    }

    geminiBridgeRef.current?.close();
    const ws = new WebSocket("ws://localhost:8081");

    ws.onopen = () => {
      setBridgeConnected(true);
    };

    ws.onmessage = async (event) => {
      const msg = JSON.parse(event.data);

      if (msg.type === "input_transcript") {
        const text = (msg.text ?? "").trim();
        setInputTranscript(text);

        const normalized = text.toLowerCase();

        const startGuidancePhrases = [
          "start guidance",
          "enable guidance",
          "guide me",
          "start guiding me",
        ];

        const stopGuidancePhrases = [
          "stop guidance",
          "disable guidance",
          "stop guiding me",
        ];

        if (isEmbodiedMode && startGuidancePhrases.some((p) => normalized.includes(p))) {
          setAutoMode(true);
          setStatus("Auto guidance enabled by voice");
          return;
        }

        if (isEmbodiedMode && stopGuidancePhrases.some((p) => normalized.includes(p))) {
          setAutoMode(false);
          setStatus("Auto guidance disabled by voice");
          return;
        }
      }

      if (msg.type === "output_transcript") {
        setOutputTranscript((msg.text ?? "").trim());
      }

      if (msg.type === "gemini_audio") {
        if (!avatarWsRef.current || avatarWsRef.current.readyState !== WebSocket.OPEN) {
          return;
        }

        if (!avatarTurnStartedRef.current) {
          const eventId = randomEventId();
          currentAvatarEventIdRef.current = eventId;
          avatarTurnStartedRef.current = true;

          avatarWsRef.current.send(
            JSON.stringify({
              type: "agent.stop_listening",
              event_id: eventId,
            })
          );
        }

        avatarWsRef.current.send(
          JSON.stringify({
            type: "agent.speak",
            event_id: currentAvatarEventIdRef.current,
            audio: msg.data,
          })
        );
      }

      if (msg.type === "turn_complete") {
        if (
          avatarWsRef.current &&
          avatarWsRef.current.readyState === WebSocket.OPEN &&
          currentAvatarEventIdRef.current
        ) {
          avatarWsRef.current.send(
            JSON.stringify({
              type: "agent.speak_end",
              event_id: currentAvatarEventIdRef.current,
            })
          );
        }

        return;
      }

      if (msg.type === "interrupted") {
        if (
          avatarWsRef.current &&
          avatarWsRef.current.readyState === WebSocket.OPEN
        ) {
          avatarWsRef.current.send(JSON.stringify({ type: "agent.interrupt" }));
        }

        avatarTurnStartedRef.current = false;
        currentAvatarEventIdRef.current = null;
      }

      if (msg.type === "gemini_error") {
        setStatus(`Gemini error: ${msg.error}`);
      }
    };

    ws.onerror = () => {
      setBridgeConnected(false);
      setStatus("Gemini bridge error");
    };

    ws.onclose = () => {
      setBridgeConnected(false);
    };

    geminiBridgeRef.current = ws;
  }

  function connectWorldModel() {
    worldModelWsRef.current?.close();
    const ws = new WebSocket("ws://localhost:8090");

    ws.onopen = () => {
      ws.send(JSON.stringify({ type: "query", query: "reset_world_model" }));
      setEventLog((prev) =>
        [`[WORLD MODEL CONNECTED]`, prev].filter(Boolean).join("\n\n").slice(0, 4000)
      );
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);

        if (msg.type === "state_updated") {
          worldFrameInFlightRef.current = false;
          const objects = msg.objects || [];
          const cameraPose = msg.camera_pose || null;
          const objects3d = msg.objects_3d || [];
          const sparseMap = msg.sparse_map || [];
          const hands = msg.hands || [];
          const worldDebug = msg.world_debug || {};
          const depthDebug = msg.depth_debug || null;
          const now = performance.now();
          setObservedObjects(objects);
          setWorldStateText(JSON.stringify({ objects }, null, 2));
          const shouldUpdateDebugText = now - lastDebugTextUpdateRef.current > 500;
          if (shouldUpdateDebugText) {
            setCameraPoseText(JSON.stringify(cameraPose, null, 2));
            setObjects3dText(JSON.stringify(objects3d, null, 2));
            setSparseMapText(JSON.stringify(sparseMap, null, 2));
            setWorldDebugText(JSON.stringify(worldDebug, null, 2));
            setHandsText(JSON.stringify(hands, null, 2));
            lastDebugTextUpdateRef.current = now;
          }
          setSparseMapData(Array.isArray(sparseMap) ? sparseMap : []);
          if (
            pendingWorldFrameRef.current &&
            pendingWorldFrameRef.current.timestamp === msg.frame_timestamp
          ) {
            setProcessedFrameUrl(pendingWorldFrameRef.current.image);
            setProcessedFrameTimestamp(pendingWorldFrameRef.current.timestamp);
            setProcessedFrameLandmarks(Array.isArray(sparseMap) ? sparseMap : []);
            setProcessedFrameSize({
              width: pendingWorldFrameRef.current.width,
              height: pendingWorldFrameRef.current.height,
            });
            pendingWorldFrameRef.current = null;
          }
          setDepthDebugUrl(depthDebug?.image ? `data:${depthDebug.mime_type};base64,${depthDebug.image}` : "");
          if (depthDebug?.width && depthDebug?.height) {
            setDepthDebugSize({ width: depthDebug.width, height: depthDebug.height });
          }

          if (typeof msg.frame_timestamp === "number") {
            const age = Date.now() - msg.frame_timestamp;
            setFrameAgeMs(age);
            setPipelineAgeMs(age);
          }

          if (typeof msg.capture_ms === "number") setCaptureMs(msg.capture_ms);
          if (typeof msg.server_decode_ms === "number") setServerDecodeMs(msg.server_decode_ms);
          if (typeof msg.server_detect_ms === "number") setServerDetectMs(msg.server_detect_ms);
          if (typeof msg.server_depth_ms === "number") setServerDepthMs(msg.server_depth_ms);
          if (typeof msg.server_pose_ms === "number") setServerPoseMs(msg.server_pose_ms);
          if (typeof msg.server_world_ms === "number") setServerWorldMs(msg.server_world_ms);
          if (typeof msg.server_total_ms === "number") setServerTotalMs(msg.server_total_ms);
          return;
        }

        if (msg.type === "query_result") {
          if (msg.result?.simulations) {
            const sims = msg.result.simulations || {};
            const best = msg.result.best_action || "";
            const bestSeq = msg.result.best_sequence || [];

            setPlannerSimulations(sims || null);
            setBestActionName(best);
            setPlannerSummary(
              `Action: ${best || "(none)"}\nPlan: ${bestSeq.length ? bestSeq.join(" → ") : "(none)"}`
            );
          }

          if (msg.result?.explanation) {
            const text = msg.result.explanation;

            if (useAvatarSpeech && isSpeaking) {
              return;
            }

            const now = Date.now();
            const last = lastSpeakTimeRef.current;
            const lastBase = lastSpokenRef.current;
            const base = text;

            const shouldSkipRepeatedStop = base === "stop" && lastBase === "stop";
            const changed = base !== lastBase;
            const enoughTime = now - last > 2000;

            if (!shouldSkipRepeatedStop && (changed || enoughTime)) {
              maybeSpeakWorldModelExplanation(text);
              lastSpokenRef.current = base;
              lastSpeakTimeRef.current = now;
            }
          }

          if (msg.result?.goal_reached) {
            setAutoMode(false);
          }

          if (showDebug) {
            setLastQueryResultText(JSON.stringify(msg.result, null, 2));
            setEventLog((prev) =>
              [`[QUERY RESULT]`, JSON.stringify(msg.result, null, 2), prev]
                .filter(Boolean)
                .join("\n\n")
                .slice(0, 4000)
            );
          }
          return;
        }

        if (showDebug) {
          setEventLog((prev) =>
            [`[WORLD MODEL MESSAGE]`, JSON.stringify(msg, null, 2), prev]
              .filter(Boolean)
              .join("\n\n")
              .slice(0, 4000)
          );
        }
      } catch (err) {
        console.error("Invalid world model message", err);
      }
    };

    ws.onerror = () => {
      worldFrameInFlightRef.current = false;
      if (showDebug) {
        setEventLog((prev) =>
          [`[WORLD MODEL ERROR]`, prev].filter(Boolean).join("\n\n").slice(0, 4000)
        );
      }
    };

    ws.onclose = () => {
      worldFrameInFlightRef.current = false;
      if (showDebug) {
        setEventLog((prev) =>
          [`[WORLD MODEL CLOSED]`, prev].filter(Boolean).join("\n\n").slice(0, 4000)
        );
      }
    };

    worldModelWsRef.current = ws;
  }

  async function startMic() {
    try {
      if (avatarMode === "ai") {
        if (!geminiBridgeRef.current || geminiBridgeRef.current.readyState !== WebSocket.OPEN) {
          alert("Gemini bridge is not connected yet.");
          return;
        }
      }

      if (!avatarWsRef.current || avatarWsRef.current.readyState !== WebSocket.OPEN) {
        alert("Avatar WebSocket is not connected yet.");
        return;
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });

      mediaStreamRef.current = stream;

      const targetSampleRate = avatarMode === "ai" ? 16000 : 24000;
      const audioContext = new AudioContext({ sampleRate: targetSampleRate });
      audioContextRef.current = audioContext;

      await audioContext.resume();
      await audioContext.audioWorklet.addModule("/audio-processor.js");

      const source = audioContext.createMediaStreamSource(stream);
      sourceRef.current = source;

      const workletNode = new AudioWorkletNode(audioContext, "pcm-audio-processor");
      workletNodeRef.current = workletNode;

      const silentGain = audioContext.createGain();
      silentGain.gain.value = 0;
      silentGainRef.current = silentGain;

      workletNode.port.onmessage = (event) => {
        const floatChunk = new Float32Array(event.data);
        const sourceRate = audioContextRef.current?.sampleRate ?? 16000;

        if (avatarMode === "ai") {
          const resampled = resampleFloat32(floatChunk, sourceRate, 16000);
          const pcm16 = floatTo16BitPCM(resampled);
          const base64 = bytesToBase64(pcm16);

          geminiBridgeRef.current?.send(
            JSON.stringify({
              type: "mic_audio",
              data: base64,
            })
          );
          return;
        }

        const resampled24k = resampleFloat32(floatChunk, sourceRate, 24000);
        const pcm16 = floatTo16BitPCM(resampled24k);
        const base64 = bytesToBase64(pcm16);

        if (!avatarWsRef.current || avatarWsRef.current.readyState !== WebSocket.OPEN) {
          return;
        }

        if (!avatarTurnStartedRef.current) {
          const eventId = randomEventId();
          currentAvatarEventIdRef.current = eventId;
          avatarTurnStartedRef.current = true;

          if (isSpeaking) {
            avatarWsRef.current.send(JSON.stringify({ type: "agent.interrupt" }));
          }

          avatarWsRef.current.send(
            JSON.stringify({
              type: "agent.stop_listening",
              event_id: eventId,
            })
          );
        }

        avatarWsRef.current.send(
          JSON.stringify({
            type: "agent.speak",
            event_id: currentAvatarEventIdRef.current,
            audio: base64,
          })
        );
      };

      source.connect(workletNode);
      workletNode.connect(silentGain);
      silentGain.connect(audioContext.destination);

      setMicOn(true);
      setStatus(avatarMode === "ai" ? "Mic on (AI mode)" : "Mic on (Direct mode)");
    } catch (error) {
      console.error("startMic error:", error);
      setStatus(error instanceof Error ? error.message : "Mic error");
    }
  }

  function stopMic() {
    workletNodeRef.current?.disconnect();
    workletNodeRef.current = null;

    sourceRef.current?.disconnect();
    sourceRef.current = null;

    silentGainRef.current?.disconnect();
    silentGainRef.current = null;

    mediaStreamRef.current?.getTracks().forEach((t) => t.stop());
    mediaStreamRef.current = null;

    audioContextRef.current?.close();
    audioContextRef.current = null;

    if (avatarMode === "ai") {
      geminiBridgeRef.current?.send(JSON.stringify({ type: "end_audio" }));
    } else {
      if (
        avatarWsRef.current &&
        avatarWsRef.current.readyState === WebSocket.OPEN &&
        currentAvatarEventIdRef.current
      ) {
        avatarWsRef.current.send(
          JSON.stringify({
            type: "agent.speak_end",
            event_id: currentAvatarEventIdRef.current,
          })
        );
      }

      avatarTurnStartedRef.current = false;
      currentAvatarEventIdRef.current = null;
    }

    setMicOn(false);
    setStatus(avatarMode === "ai" ? "Mic off (AI mode)" : "Mic off (Direct mode)");
  }

  async function loadVideoDevices() {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const cams = devices.filter((d) => d.kind === "videoinput");
    setVideoDevices(cams);
    
    if (cams.length > 0 && !selectedCameraId) {
      setSelectedCameraId(cams[0].deviceId);
    }
  }

  async function switchCamera(deviceId: string) {
    try {
      setEmbodiedVideoSource("camera");

      // Stop frame sending first.
      if (frameIntervalRef.current) {
        clearInterval(frameIntervalRef.current);
        frameIntervalRef.current = null;
      }

      // Stop and release previous stream.
      if (localCamStreamRef.current) {
        localCamStreamRef.current.getTracks().forEach((t) => t.stop());
        localCamStreamRef.current = null;
      }

      // Clear the video element.
      if (localCamRef.current) {
        localCamRef.current.pause();
        localCamRef.current.srcObject = null;
        localCamRef.current.load();
      }

      // Small delay helps flaky USB cameras on Windows.
      await new Promise((resolve) => setTimeout(resolve, 250));

      // Re-open with soft constraints first.
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          deviceId: { exact: deviceId },
        },
        audio: false,
      });

      localCamStreamRef.current = stream;

      if (localCamRef.current) {
        localCamRef.current.srcObject = stream;
        await localCamRef.current.play();
      }

      startSendingFrames();
    } catch (err) {
      console.error("switchCamera failed:", err);
    }
  }

  async function startSceneVideo() {
    if (localCamStreamRef.current) {
      localCamStreamRef.current.getTracks().forEach((t) => t.stop());
      localCamStreamRef.current = null;
    }

    if (localCamRef.current) {
      localCamRef.current.pause();
      localCamRef.current.srcObject = null;
      localCamRef.current.src = "/scene.mp4";
      localCamRef.current.loop = false;
      localCamRef.current.muted = true;
      localCamRef.current.playsInline = true;
      localCamRef.current.onended = async () => {
        worldModelWsRef.current?.send(JSON.stringify({ type: "query", query: "reset_world_model" }));
        if (localCamRef.current) {
          localCamRef.current.currentTime = 0;
          await localCamRef.current.play();
        }
      };
      localCamRef.current.currentTime = 0;
      await localCamRef.current.play();
    }
  }

  async function startLocalCamera(sourceOverride: EmbodiedVideoSource = embodiedVideoSource) {
    if (isEmbodiedMode && sourceOverride === "scene") {
      await startSceneVideo();
      return;
    }

    // First call to unlock labels
    const initialStream = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: false,
    });

    // Now we can list devices with labels
    await loadVideoDevices();

    // Stop initial stream
    initialStream.getTracks().forEach((t) => t.stop());

    // Start with selected camera
    const stream = await navigator.mediaDevices.getUserMedia({
      video: selectedCameraId
        ? { deviceId: { exact: selectedCameraId } }
        : true,
      audio: false,
    });

    localCamStreamRef.current = stream;

    if (localCamRef.current) {
      localCamRef.current.pause();
      localCamRef.current.removeAttribute("src");
      localCamRef.current.loop = false;
      localCamRef.current.onended = null;
      localCamRef.current.srcObject = stream;
      await localCamRef.current.play();
    }
  }

  async function switchEmbodiedVideoSource(source: EmbodiedVideoSource) {
    setEmbodiedVideoSource(source);
    if (!isEmbodiedMode) return;

    const shouldRestart =
      !!worldModelWsRef.current ||
      !!localCamStreamRef.current ||
      Boolean(localCamRef.current?.src);

    if (!shouldRestart) return;

    stopLocalCamera();
    await startLocalCamera(source);
    if (worldModelWsRef.current?.readyState === WebSocket.OPEN) {
      startSendingFrames();
    }
  }

  function startSendingFrames() {
    if (!localCamRef.current) return;

    const video = localCamRef.current;
    const maxCaptureWidth = 640;

    if (!frameCanvasRef.current) {
      frameCanvasRef.current = document.createElement("canvas");
    }
    const canvas = frameCanvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    if (frameIntervalRef.current) {
      clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = null;
    }

    frameIntervalRef.current = window.setInterval(() => {
      if (!worldModelWsRef.current || worldModelWsRef.current.readyState !== WebSocket.OPEN) {
        worldFrameInFlightRef.current = false;
        return;
      }

      if (video.readyState < 2) {
        return;
      }

      if (worldFrameInFlightRef.current) {
        return;
      }

      const t0 = performance.now();
      const sourceWidth = video.videoWidth || 640;
      const sourceHeight = video.videoHeight || 360;
      const scale = Math.min(1, maxCaptureWidth / Math.max(sourceWidth, 1));
      const frameWidth = Math.max(1, Math.round(sourceWidth * scale));
      const frameHeight = Math.max(1, Math.round(sourceHeight * scale));

      if (canvas.width !== frameWidth || canvas.height !== frameHeight) {
        canvas.width = frameWidth;
        canvas.height = frameHeight;
      }

      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const dataUrl = canvas.toDataURL("image/jpeg", 0.8);
      const t1 = performance.now();
      const timestamp = Date.now();

      setCaptureMs(t1 - t0);

      worldFrameInFlightRef.current = true;
      pendingWorldFrameRef.current = {
        image: dataUrl,
        timestamp,
        width: frameWidth,
        height: frameHeight,
      };
      worldModelWsRef.current.send(
        JSON.stringify({
          type: "frame",
          image: dataUrl,
          timestamp,
          frame_width: frameWidth,
          frame_height: frameHeight,
          capture_ms: t1 - t0,
        })
      );
    }, 200);
  }

  function stopLocalCamera() {
    worldFrameInFlightRef.current = false;
    pendingWorldFrameRef.current = null;

    if (frameIntervalRef.current) {
      clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = null;
    }

    localCamStreamRef.current?.getTracks().forEach((t) => t.stop());
    localCamStreamRef.current = null;

    if (localCamRef.current) {
      localCamRef.current.pause();
      localCamRef.current.srcObject = null;
      localCamRef.current.removeAttribute("src");
      localCamRef.current.loop = false;
      localCamRef.current.onended = null;
      localCamRef.current.load();
    }
  }

  function stopAllConnections() {
    stopMic();

    geminiBridgeRef.current?.close();
    geminiBridgeRef.current = null;

    avatarWsRef.current?.close();
    avatarWsRef.current = null;

    roomRef.current?.disconnect();
    roomRef.current = null;

    if (remoteAudioElRef.current) {
      remoteAudioElRef.current.remove();
      remoteAudioElRef.current = null;
    }

    worldModelWsRef.current?.close();
    worldModelWsRef.current = null;

    stopLocalCamera();
    setBridgeConnected(false);
    setAvatarConnected(false);
    setIsSpeaking(false);
  }

  function sendWorldQuery(payload: any) {
    if (!worldModelWsRef.current || worldModelWsRef.current.readyState !== WebSocket.OPEN) {
      alert("World model is not connected.");
      return;
    }

    worldModelWsRef.current.send(
      JSON.stringify({
        type: "query",
        ...payload,
      })
    );
  }

  function askSimulateActions() {
    sendWorldQuery({
      query: "simulate_actions",
    });
  }

  function getObservedCupPosition() {
    const cup = observedObjects.find((o: any) => o.label === "cup");
    if (!cup) return null;
    return { x: cup.x, y: cup.y };
  }

  function maybeSpeakWorldModelExplanation(text: string) {
    if (useAvatarSpeech) {
      if (!geminiBridgeRef.current || geminiBridgeRef.current.readyState !== WebSocket.OPEN) {
        setStatus("Avatar speech requested, but Gemini bridge is not connected");
        return;
      }

      geminiBridgeRef.current.send(
        JSON.stringify({
          type: "world_model_explanation",
          text,
        })
      );
      return;
    }

    if (!("speechSynthesis" in window)) return;

    const utter = new SpeechSynthesisUtterance(text);
    utter.rate = 1.25;

    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(utter);
  }

  useEffect(() => {
    if (!isEmbodiedMode || !autoMode) return;

    const interval = setInterval(() => {
      if (!worldModelWsRef.current || worldModelWsRef.current.readyState !== WebSocket.OPEN) {
        return;
      }

      worldModelWsRef.current.send(JSON.stringify({
        type: "query",
        query: "simulate_actions"
      }));
    }, 1200);

    return () => clearInterval(interval);
  }, [autoMode, isEmbodiedMode]);

  useEffect(() => {
    if (!useAvatarSpeech) return;

    const needsBridge =
      !geminiBridgeRef.current ||
      geminiBridgeRef.current.readyState !== WebSocket.OPEN;

    if (needsBridge) {
      connectGeminiBridge();
    }
  }, [useAvatarSpeech]);

  useEffect(() => {
    if (!selectedCameraId) return;
    if (isEmbodiedMode && embodiedVideoSource === "scene") return;

    if (localCamStreamRef.current) {
      stopLocalCamera();
      startLocalCamera();
    }
  }, [selectedCameraId, embodiedVideoSource, isEmbodiedMode]);

  useEffect(() => {
    return () => {
      stopAllConnections();
    };
  }, []);

  const avatarModeLocked = avatarConnected || micOn;
  const captureModeLocked = avatarConnected || micOn || !!worldModelWsRef.current;

  const statusItems = [
    { id: "status", label: "Status", value: status, active: status !== "idle" },
    { id: "capture", label: "Capture", value: isEmbodiedMode ? "embodied" : "social", active: true },
    { id: "avatar-mode", label: "Avatar", value: avatarMode === "ai" ? "AI mode" : "Direct mode", active: true },
    { id: "avatar-connection", label: "Avatar", value: avatarConnected ? "connected" : "disconnected", active: avatarConnected },
    { id: "gemini-bridge", label: "Gemini bridge", value: bridgeConnected ? "connected" : "disconnected", active: bridgeConnected },
    { id: "mic", label: "Mic", value: micOn ? "on" : "off", active: micOn },
    { id: "avatar-speaking", label: "Avatar speaking", value: isSpeaking ? "yes" : "no", active: isSpeaking },
  ];

  return (
    <main
      style={{
        maxWidth: 1280,
        margin: "0 auto",
        padding: "32px 20px 56px",
        fontFamily: "Inter, Arial, sans-serif",
        color: "#0f172a",
        background: "#f8fafc",
        minHeight: "100vh",
      }}
    >
      <div style={{ display: "grid", gap: 24 }}>
        <section
          style={{
            ...cardStyle(),
            padding: 24,
            background:
              "linear-gradient(135deg, rgba(15,23,42,1) 0%, rgba(30,41,59,1) 55%, rgba(37,99,235,0.95) 100%)",
            color: "#ffffff",
            overflow: "hidden",
            position: "relative",
          }}
        >
          <div style={{ position: "relative", zIndex: 1 }}>
            <div
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: 8,
                padding: "6px 12px",
                borderRadius: 999,
                background: "rgba(255,255,255,0.12)",
                border: "1px solid rgba(255,255,255,0.18)",
                marginBottom: 14,
                fontSize: 13,
                fontWeight: 700,
                letterSpacing: 0.2,
              }}
            >
              🧠 Embodied Agent
            </div>
            <h1 style={{ fontSize: 36, lineHeight: 1.1, margin: "0 0 10px 0" }}>
              Real-Time World Model + Embodied Avatar
            </h1>
            <p style={{ margin: 0, fontSize: 16, color: "rgba(255,255,255,0.86)", maxWidth: 800 }}>
              A compact multimodal agent that observes the world, predicts futures,
              plans actions, and speaks through an embodied avatar.
            </p>
          </div>
          <div
            style={{
              position: "absolute",
              right: -70,
              top: -70,
              width: 260,
              height: 260,
              borderRadius: "50%",
              background: "rgba(147, 197, 253, 0.18)",
              filter: "blur(8px)",
            }}
          />
        </section>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "minmax(320px, 1fr) minmax(420px, 1.4fr)",
            gap: 20,
          }}
        >
          <section style={{ ...cardStyle(), padding: 20 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14 }}>
              <div>
                <h2 style={{ margin: 0, fontSize: 20 }}>System status</h2>
                <p style={{ margin: "6px 0 0 0", color: "#475569", fontSize: 14 }}>
                  Live connection state for the avatar, bridge, and world model loop.
                </p>
              </div>
            </div>

            <div style={{ display: "grid", gap: 10 }}>
              {statusItems.map((item) => (
                <div
                  key={item.id}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    gap: 16,
                    padding: "12px 14px",
                    borderRadius: 14,
                    background: "#f8fafc",
                    border: "1px solid #e2e8f0",
                  }}
                >
                  <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                    <span
                      style={{
                        width: 10,
                        height: 10,
                        borderRadius: "50%",
                        background: statusDotColor(item.active),
                        boxShadow: item.active ? `0 0 0 4px ${item.active ? "rgba(34,197,94,0.12)" : "rgba(148,163,184,0.12)"}` : "none",
                        flexShrink: 0,
                      }}
                    />
                    <span style={{ fontWeight: 600 }}>{item.label}</span>
                  </div>
                  <span style={{ color: "#475569", textTransform: "capitalize" }}>{item.value}</span>
                </div>
              ))}
            </div>
          </section>

          <section style={{ ...cardStyle(), padding: 20 }}>
            <div style={{ marginBottom: 16 }}>
              <h2 style={{ margin: 0, fontSize: 20 }}>Controls</h2>
              <p style={{ margin: "6px 0 0 0", color: "#475569", fontSize: 14 }}>
                Choose a capture pipeline, then start the avatar and the relevant perception loop.
              </p>
            </div>

            <div style={{ display: "grid", gap: 14 }}>
              <div>
                <div style={{ fontSize: 13, fontWeight: 700, color: "#64748b", marginBottom: 8 }}>
                  Capture mode
                </div>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 10 }}>
                  <button
                    onClick={() => setCaptureMode("social")}
                    disabled={captureModeLocked || isSocialMode}
                    style={toggleChipStyle(isSocialMode, captureModeLocked || isSocialMode)}
                  >
                    Social mode
                  </button>
                  <button
                    onClick={() => setCaptureMode("embodied")}
                    disabled={captureModeLocked || isEmbodiedMode}
                    style={toggleChipStyle(isEmbodiedMode, captureModeLocked || isEmbodiedMode)}
                  >
                    Embodied mode
                  </button>
                </div>
                <p style={{ margin: "8px 0 0 0", color: "#64748b", fontSize: 13 }}>
                  {isEmbodiedMode
                    ? "Egocentric capture for world modeling, 3D state, and task guidance."
                    : "Static webcam mode."}
                </p>
              </div>

              <div>
                <div style={{ fontSize: 13, fontWeight: 700, color: "#64748b", marginBottom: 8 }}>
                  Avatar mode
                </div>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 10 }}>
                  <button
                    onClick={() => setAvatarMode("ai")}
                    disabled={avatarModeLocked || avatarMode === "ai"}
                    style={toggleChipStyle(avatarMode === "ai", avatarModeLocked || avatarMode === "ai")}
                  >
                    AI mode
                  </button>
                  <button
                    onClick={() => setAvatarMode("direct")}
                    disabled={avatarModeLocked || avatarMode === "direct"}
                    style={toggleChipStyle(avatarMode === "direct", avatarModeLocked || avatarMode === "direct")}
                  >
                    Direct avatar mode
                  </button>
                </div>
              </div>

              <div>
                <div style={{ fontSize: 13, fontWeight: 700, color: "#64748b", marginBottom: 8 }}>
                  Options
                </div>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 10 }}>
                  {[
                    {
                      label: "Echo on",
                      checked: directAudioMonitor,
                      onChange: (checked: boolean) => setDirectAudioMonitor(checked),
                      disabled: avatarModeLocked && avatarMode !== "direct",
                    },
                    {
                      label: "Use Avatar Speech",
                      checked: useAvatarSpeech,
                      onChange: (checked: boolean) => setUseAvatarSpeech(checked),
                    },
                    {
                      label: "Show debug",
                      checked: showDebug,
                      onChange: (checked: boolean) => setShowDebug(checked),
                    },
                  ].map((option) => (
                    <label
                      key={option.label}
                      style={{
                        display: "inline-flex",
                        alignItems: "center",
                        gap: 10,
                        padding: "10px 12px",
                        borderRadius: 12,
                        border: "1px solid #e2e8f0",
                        background: "#f8fafc",
                        opacity: option.disabled ? 0.6 : 1,
                        cursor: option.disabled ? "not-allowed" : "pointer",
                        fontWeight: 600,
                        fontSize: 14,
                        color: "#334155",
                      }}
                    >
                      <input
                        type="checkbox"
                        checked={option.checked}
                        onChange={(e) => option.onChange(e.target.checked)}
                        disabled={option.disabled}
                      />
                      {option.label}
                    </label>
                  ))}
                </div>
              </div>

              <div style={{ display: "grid", gap: 10 }}>
                <div style={{ fontSize: 13, fontWeight: 700, color: "#64748b" }}>Actions</div>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 10 }}>
                  <button onClick={startAvatar} style={actionButtonStyle("primary")}>
                    Start avatar
                  </button>
                  {isEmbodiedMode ? (
                    <button onClick={startWorldModelOnly} style={actionButtonStyle("secondary")}>
                      World model only
                    </button>
                  ) : null}
                  <button onClick={stopAllConnections} style={actionButtonStyle("danger")}>
                    Stop all
                  </button>
                  <button
                    onClick={startMic}
                    disabled={!avatarConnected || micOn || (avatarMode === "ai" && !bridgeConnected)}
                    style={actionButtonStyle("secondary", !avatarConnected || micOn || (avatarMode === "ai" && !bridgeConnected))}
                  >
                    Start mic
                  </button>
                  <button onClick={stopMic} disabled={!micOn} style={actionButtonStyle("secondary", !micOn)}>
                    Stop mic
                  </button>
                  {isEmbodiedMode ? (
                    <button
                      onClick={() => setAutoMode((v) => !v)}
                      style={actionButtonStyle(autoMode ? "primary" : "secondary")}
                    >
                      {autoMode ? "Stop guidance" : "Start guidance"}
                    </button>
                  ) : null}
                  {isEmbodiedMode ? (
                    <button onClick={askSimulateActions} style={actionButtonStyle("secondary")}>
                      Simulate futures
                    </button>
                  ) : null}
                </div>
                <div style={{ marginBottom: 12 }}>
                  {isEmbodiedMode ? (
                    <>
                      <div style={{ display: "flex", flexWrap: "wrap", gap: 8, alignItems: "center", marginBottom: 8 }}>
                        <span style={{ color: "#64748b", fontWeight: 700, fontSize: 13 }}>Embodied input:</span>
                        <button
                          onClick={() => switchEmbodiedVideoSource("scene")}
                          style={toggleChipStyle(embodiedVideoSource === "scene")}
                        >
                          scene.mp4
                        </button>
                        <button
                          onClick={() => switchEmbodiedVideoSource("camera")}
                          style={toggleChipStyle(embodiedVideoSource === "camera")}
                        >
                          Live camera
                        </button>
                      </div>
                      {embodiedVideoSource === "camera" ? (
                        <div>
                          <label style={{ marginRight: 8 }}>Egocentric camera:</label>
                          <select
                            value={selectedCameraId || ""}
                            onChange={async (e) => {
                              const id = e.target.value;
                              setSelectedCameraId(id);
                              await switchCamera(id);
                            }}
                          >
                            {videoDevices.map((cam, i) => (
                              <option key={cam.deviceId} value={cam.deviceId}>
                                {cam.label || `Camera ${i + 1}`}
                              </option>
                            ))}
                          </select>
                        </div>
                      ) : (
                        <div style={{ color: "#64748b", fontSize: 13, fontWeight: 700 }}>
                          Using /scene.mp4 from public.
                        </div>
                      )}
                    </>
                  ) : (
                    <>
                      <label style={{ marginRight: 8 }}>Social webcam:</label>
                      <select
                        value={selectedCameraId || ""}
                        onChange={async (e) => {
                          const id = e.target.value;
                          setSelectedCameraId(id);
                          await switchCamera(id);
                        }}
                      >
                        {videoDevices.map((cam, i) => (
                          <option key={cam.deviceId} value={cam.deviceId}>
                            {cam.label || `Camera ${i + 1}`}
                          </option>
                        ))}
                      </select>
                    </>
                  )}
                </div>
              </div>
            </div>
          </section>
        </div>

        <section
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(2, minmax(0, 1fr))",
            gap: 20,
          }}
        >
          <div style={{ ...cardStyle(), padding: 16 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
              <div>
                <h2 style={{ margin: 0, fontSize: 18 }}>Avatar output</h2>
                <p style={{ margin: "6px 0 0 0", color: "#64748b", fontSize: 14 }}>
                  Remote LiveAvatar video stream.
                </p>
              </div>
              <span
                style={{
                  padding: "6px 10px",
                  borderRadius: 999,
                  background: avatarConnected ? "#ecfdf5" : "#f1f5f9",
                  color: avatarConnected ? "#166534" : "#475569",
                  border: `1px solid ${avatarConnected ? "#bbf7d0" : "#e2e8f0"}`,
                  fontWeight: 700,
                  fontSize: 12,
                }}
              >
                {avatarConnected ? "LIVE" : "OFFLINE"}
              </span>
            </div>
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              style={{
                width: "100%",
                aspectRatio: "16 / 9",
                objectFit: "cover",
                background: "linear-gradient(180deg, #020617 0%, #0f172a 100%)",
                borderRadius: 16,
                border: "1px solid #0f172a",
                display: "block",
              }}
            />
          </div>

          <div style={{ ...cardStyle(), padding: 16 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
              <div>
                <h2 style={{ margin: 0, fontSize: 18 }}>{isEmbodiedMode ? "Camera + planning overlay" : "Webcam preview"}</h2>
                <p style={{ margin: "6px 0 0 0", color: "#64748b", fontSize: 14 }}>
{isEmbodiedMode ? "Local camera, goal marker, observed object, and simulated futures." : "Static webcam feed."}
                </p>
              </div>
              <span
                style={{
                  padding: "6px 10px",
                  borderRadius: 999,
                  background: isEmbodiedMode ? (autoMode ? "#eff6ff" : "#f8fafc") : "#fff7ed",
                  color: isEmbodiedMode ? (autoMode ? "#1d4ed8" : "#475569") : "#9a3412",
                  border: `1px solid ${isEmbodiedMode ? (autoMode ? "#bfdbfe" : "#e2e8f0") : "#fed7aa"}`,
                  fontWeight: 700,
                  fontSize: 12,
                }}
              >
                {isEmbodiedMode ? (autoMode ? "GUIDANCE ON" : "GUIDANCE OFF") : "SOCIAL PREVIEW"}
              </span>
            </div>

            <div
              style={{
                position: "relative",
                width: "100%",
                aspectRatio: "16 / 9",
                flexShrink: 0,
                borderRadius: 16,
                overflow: "hidden",
                border: "1px solid #1e293b",
                background: "linear-gradient(180deg, #111827 0%, #1f2937 100%)",
              }}
            >
              <video
                ref={localCamRef}
                autoPlay
                playsInline
                muted
                style={{
                  width: "100%",
                  height: "100%",
                  objectFit: "cover",
                  display: "block",
                }}
              />

              {false && isEmbodiedMode
                ? sparseMapData.slice(0, 60).map((point: any) => {
                    const imageXY = point?.image_xy;
                    if (!Array.isArray(imageXY) || imageXY.length < 2) return null;

                    const px = Number(imageXY[0]);
                    const py = Number(imageXY[1]);
                    if (!Number.isFinite(px) || !Number.isFinite(py)) return null;

                    const hits = Number(point?.hits || 0);
                    const size = Math.max(4, Math.min(9, 3 + hits * 0.45));
                    const opacity = Math.max(0.28, Math.min(0.88, 0.25 + hits * 0.06));

                    return (
                      <div
                        key={`landmark-${point?.id ?? `${px}-${py}`}`}
                        style={{
                          position: "absolute",
                          left: `${(px / 640) * 100}%`,
                          top: `${(py / 360) * 100}%`,
                          width: size,
                          height: size,
                          borderRadius: "50%",
                          transform: "translate(-50%, -50%)",
                          background: "rgba(56, 189, 248, 0.95)",
                          border: "1px solid rgba(224, 242, 254, 0.95)",
                          boxShadow: "0 0 0 3px rgba(56, 189, 248, 0.12)",
                          opacity,
                          pointerEvents: "none",
                        }}
                        title={`Landmark ${point?.id ?? "?"} • hits ${hits}`}
                      />
                    );
                  })
                : null}

              {isEmbodiedMode ? (
              <div
                style={{
                  position: "absolute",
                  left: "50%",
                  top: "50%",
                  width: 18,
                  height: 18,
                  borderRadius: "50%",
                  background: "#ef4444",
                  border: "3px solid rgba(255,255,255,0.95)",
                  transform: "translate(-50%, -50%)",
                  boxShadow: "0 0 0 6px rgba(239,68,68,0.14), 0 0 12px rgba(239,68,68,0.8)",
                  pointerEvents: "none",
                }}
                title="Goal"
              />
              ) : null}

              {isEmbodiedMode ? (() => {
                const cup = getObservedCupPosition();
                if (!cup) return null;

                return (
                  <div
                    style={{
                      position: "absolute",
                      left: `${cup.x * 100}%`,
                      top: `${cup.y * 100}%`,
                      width: 14,
                      height: 14,
                      borderRadius: "50%",
                      background: "#ffffff",
                      border: "2px solid #020617",
                      transform: "translate(-50%, -50%)",
                      pointerEvents: "none",
                      boxShadow: "0 0 0 4px rgba(255,255,255,0.15), 0 0 8px rgba(255,255,255,0.9)",
                    }}
                    title="Current cup position"
                  />
                );
              })() : null}

              {false && isEmbodiedMode ? (
                <div
                  style={{
                    position: "absolute",
                    left: 14,
                    bottom: 14,
                    padding: "6px 10px",
                    borderRadius: 999,
                    background: "rgba(2, 6, 23, 0.72)",
                    color: "#e0f2fe",
                    border: "1px solid rgba(125, 211, 252, 0.35)",
                    fontSize: 12,
                    fontWeight: 700,
                    letterSpacing: "0.02em",
                    pointerEvents: "none",
                  }}
                >
                  Sparse landmarks: {sparseMapData.length}
                </div>
              ) : null}

              {isEmbodiedMode && embodiedState.plannerSimulations &&
                Object.entries(embodiedState.plannerSimulations).map(([sequenceKey, sim]: any) => {
                  const step1 = sim?.step1_state || [];
                  const step2 = sim?.predicted_state || [];
                  const sequence = sim?.sequence || [];

                  const x1 = step1[0];
                  const y1 = step1[1];
                  const x2 = step2[0];
                  const y2 = step2[1];

                  if (
                    typeof x1 !== "number" || typeof y1 !== "number" ||
                    typeof x2 !== "number" || typeof y2 !== "number"
                  ) return null;

                  if (
                    x1 < 0 || y1 < 0 || x1 > 1 || y1 > 1 ||
                    x2 < 0 || y2 < 0 || x2 > 1 || y2 > 1
                  ) return null;

                  const firstAction = sequence[0];
                  const isBest = firstAction === embodiedState.bestActionName;

                  let color = "#ffffff";
                  if (firstAction === "left") color = "#60a5fa";
                  if (firstAction === "right") color = "#34d399";
                  if (firstAction === "up") color = "#fbbf24";
                  if (firstAction === "down") color = "#fb7185";

                  return (
                    <svg
                      key={sequenceKey}
                      style={{
                        position: "absolute",
                        left: 0,
                        top: 0,
                        width: "100%",
                        height: "100%",
                        overflow: "visible",
                        pointerEvents: "none",
                      }}
                    >
                      <circle
                        cx={`${x1 * 100}%`}
                        cy={`${y1 * 100}%`}
                        r={isBest ? 5 : 3}
                        fill={color}
                        opacity={isBest ? 0.95 : 0.5}
                      />
                    </svg>
                  );
                })}
            </div>
          </div>
        </section>

        <section style={{ ...cardStyle(), padding: 20 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
            <div>
              <h2 style={{ margin: 0, fontSize: 20 }}>Planner</h2>
              <p style={{ margin: "6px 0 0 0", color: "#64748b", fontSize: 14 }}>
                Current best action and rollout summary.
              </p>
            </div>
            {isEmbodiedMode && embodiedState.bestActionName ? (
              <span
                style={{
                  padding: "7px 12px",
                  borderRadius: 999,
                  background: "#eff6ff",
                  color: "#1d4ed8",
                  border: "1px solid #bfdbfe",
                  fontWeight: 700,
                  fontSize: 12,
                  textTransform: "uppercase",
                }}
              >
                Best action: {embodiedState.bestActionName}
              </span>
            ) : null}
          </div>
          <pre
            style={{
              whiteSpace: "pre-wrap",
              fontSize: 14,
              lineHeight: 1.6,
              background: "#f8fafc",
              color: "#0f172a",
              padding: 16,
              borderRadius: 16,
              border: "1px solid #e2e8f0",
              margin: 0,
              minHeight: 88,
            }}
          >
            {isEmbodiedMode ? (embodiedState.plannerSummary || "(No planner decision yet)") : "Social mode scaffolded: static webcam perception will appear here next."}
          </pre>
        </section>

        {showDebug && (
          <section style={{ ...cardStyle(), padding: 20 }}>
            <h2 style={{ marginTop: 0, fontSize: 20 }}>Transcripts</h2>
            <div
              style={{
                background: "#f8fafc",
                padding: 16,
                borderRadius: 16,
                border: "1px solid #e2e8f0",
                fontSize: 14,
              }}
            >
              <p style={{ margin: "0 0 10px 0" }}>
                <strong>User:</strong> {socialState.inputTranscript || "(none)"}
              </p>
              <p style={{ margin: 0 }}>
                <strong>Agent:</strong> {socialState.outputTranscript || "(none)"}
              </p>
            </div>
          </section>
        )}


        {showDebug && isSocialMode && (
          <section style={{ ...cardStyle(), padding: 20 }}>
            <h2 style={{ marginTop: 0, fontSize: 20 }}>Webcam Mode</h2>
            <div
              style={{
                background: "#f8fafc",
                padding: 16,
                borderRadius: 16,
                border: "1px solid #e2e8f0",
                fontSize: 14,
                lineHeight: 1.6,
              }}
            >
              <p style={{ margin: "0 0 10px 0" }}>
                Static webcam mode is active.
              </p>
              <p style={{ margin: 0 }}>
                Current social state: avatar {avatarConnected ? "connected" : "disconnected"}, bridge {bridgeConnected ? "connected" : "disconnected"}, mic {micOn ? "on" : "off"}.
              </p>
            </div>
          </section>
        )}

        {showDebug && isEmbodiedMode && (
          <section
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(2, minmax(0, 1fr))",
              gap: 16,
            }}
          >
            <div style={{ ...cardStyle(), padding: 18 }}>
              <h2 style={{ marginTop: 0, fontSize: 18 }}>Observed world state</h2>
              <pre
                style={{
                  whiteSpace: "pre-wrap",
                  fontSize: 12,
                  height: 200,
                  overflowY: "auto",
                  background: "#f8fafc",
                  padding: 14,
                  borderRadius: 14,
                  border: "1px solid #e2e8f0",
                  margin: 0,
                }}
              >
                {embodiedState.worldStateText || "(No state yet)"}
              </pre>
            </div>

            <div style={{ ...cardStyle(), padding: 18 }}>
              <h2 style={{ marginTop: 0, fontSize: 18 }}>Last query result</h2>
              <div
                style={{
                  whiteSpace: "pre-wrap",
                  fontSize: 12,
                  height: 200,
                  overflowY: "auto",
                  background: "#f8fafc",
                  padding: 14,
                  borderRadius: 14,
                  border: "1px solid #e2e8f0",
                  margin: 0,
                }}
              >
                {embodiedState.lastQueryResultText || "(No query yet)"}
              </div>
            </div>

            <div style={{ ...cardStyle(), padding: 18 }}>
              <h2 style={{ marginTop: 0, fontSize: 18 }}>Profiling</h2>
              <div
                style={{
                  fontSize: 13,
                  lineHeight: 1.75,
                  background: "#f8fafc",
                  padding: 14,
                  borderRadius: 14,
                  border: "1px solid #e2e8f0",
                }}
              >
                <p><strong>Capture:</strong> {captureMs !== null ? `${captureMs.toFixed(1)} ms` : "n/a"}</p>
                <p><strong>Frame latency:</strong> {frameAgeMs !== null ? `${frameAgeMs} ms` : "n/a"}</p>
                <p><strong>Server decode:</strong> {serverDecodeMs !== null ? `${serverDecodeMs.toFixed(1)} ms` : "n/a"}</p>
                <p><strong>Server detect:</strong> {serverDetectMs !== null ? `${serverDetectMs.toFixed(1)} ms` : "n/a"}</p>
                <p><strong>Server depth:</strong> {serverDepthMs !== null ? `${serverDepthMs.toFixed(1)} ms` : "n/a"}</p>
                <p><strong>Server pose:</strong> {serverPoseMs !== null ? `${serverPoseMs.toFixed(1)} ms` : "n/a"}</p>
                <p><strong>Server world:</strong> {serverWorldMs !== null ? `${serverWorldMs.toFixed(1)} ms` : "n/a"}</p>
                <p><strong>Server total:</strong> {serverTotalMs !== null ? `${serverTotalMs.toFixed(1)} ms` : "n/a"}</p>
                <p style={{ marginBottom: 0 }}><strong>Pipeline latency:</strong> {pipelineAgeMs !== null ? `${pipelineAgeMs} ms` : "n/a"}</p>
              </div>
            </div>

            <div style={{ ...cardStyle(), padding: 18 }}>
              <h2 style={{ marginTop: 0, fontSize: 18 }}>Event log</h2>
              <pre
                style={{
                  whiteSpace: "pre-wrap",
                  fontSize: 12,
                  height: 200,
                  overflowY: "auto",
                  background: "#f8fafc",
                  padding: 14,
                  borderRadius: 14,
                  border: "1px solid #e2e8f0",
                  margin: 0,
                }}
              >
                {embodiedState.eventLog || "(No events yet)"}
              </pre>
            </div>
          </section>
        )}

        {showDebug && isEmbodiedMode && (
          <section
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(2, minmax(0, 1fr))",
              gap: 16,
            }}
          >
            <div style={{ ...cardStyle(), padding: 18 }}>
              <h2 style={{ marginTop: 0, fontSize: 18 }}>3D object state</h2>
              <pre
                style={{
                  whiteSpace: "pre-wrap",
                  fontSize: 12,
                  height: 200,
                  overflowY: "auto",
                  background: "#f8fafc",
                  padding: 14,
                  borderRadius: 14,
                  border: "1px solid #e2e8f0",
                  margin: 0,
                }}
              >
                {embodiedState.objects3dText || "(No 3D objects yet)"}
              </pre>
            </div>

            <div style={{ ...cardStyle(), padding: 18 }}>
              <h2 style={{ marginTop: 0, fontSize: 18 }}>Sparse 3D map</h2>
              <pre
                style={{
                  whiteSpace: "pre-wrap",
                  fontSize: 12,
                  height: 200,
                  overflowY: "auto",
                  background: "#f8fafc",
                  padding: 14,
                  borderRadius: 14,
                  border: "1px solid #e2e8f0",
                  margin: 0,
                }}
              >
                {embodiedState.sparseMapText || "(No sparse map yet)"}
              </pre>
            </div>

            <div style={{ ...cardStyle(), padding: 18 }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12, marginBottom: 10 }}>
                <h2 style={{ margin: 0, fontSize: 18 }}>3D map view</h2>
                <span
                  style={{
                    padding: "5px 9px",
                    borderRadius: 999,
                    background: "#f8fafc",
                    border: "1px solid #e2e8f0",
                    color: "#475569",
                    fontSize: 12,
                    fontWeight: 700,
                  }}
                >
                  {mapPoints3d.length} pts · {cameraPoseData?.pose_source || "no pose"}
                </span>
              </div>
              <div
                style={{
                  position: "relative",
                  height: 240,
                  overflow: "hidden",
                  borderRadius: 14,
                  border: "1px solid #1e293b",
                  background:
                    "radial-gradient(circle at 50% 50%, rgba(14,165,233,0.16), transparent 35%), linear-gradient(180deg, #020617 0%, #0f172a 100%)",
                }}
              >
                <div
                  style={{
                    position: "absolute",
                    inset: 0,
                    backgroundImage:
                      "linear-gradient(rgba(148,163,184,0.12) 1px, transparent 1px), linear-gradient(90deg, rgba(148,163,184,0.12) 1px, transparent 1px)",
                    backgroundSize: "24px 24px",
                  }}
                />
                <div
                  style={{
                    position: "absolute",
                    left: "50%",
                    top: "50%",
                    width: 12,
                    height: 12,
                    borderRadius: "50%",
                    transform: "translate(-50%, -50%)",
                    background: "#f97316",
                    border: "2px solid #ffedd5",
                    boxShadow: "0 0 0 6px rgba(249,115,22,0.18)",
                    zIndex: 3,
                  }}
                  title={`Camera x=${cameraMapX.toFixed(3)}, z=${cameraMapZ.toFixed(3)}`}
                />
                {mapPoints3d.slice(0, 160).map((point) => {
                  const left = 50 + (point.x - cameraMapX) * mapScale;
                  const top = 50 - (point.z - cameraMapZ) * mapScale;
                  if (left < -10 || left > 110 || top < -10 || top > 110) return null;

                  const isMissing = point.status === "missing";
                  const size = Math.max(3, Math.min(8, 3 + point.hits * 0.25));
                  const color = isMissing ? "#64748b" : point.isLocal ? "#facc15" : point.quality > 0.6 ? "#22c55e" : "#38bdf8";

                  return (
                    <div
                      key={`map-point-${point.id}`}
                      style={{
                        position: "absolute",
                        left: `${left}%`,
                        top: `${top}%`,
                        width: size,
                        height: size,
                        borderRadius: "50%",
                        transform: "translate(-50%, -50%)",
                        background: color,
                        opacity: isMissing ? 0.45 : 0.9,
                        border: point.isLocal ? "2px solid rgba(254,240,138,0.95)" : "1px solid rgba(255,255,255,0.6)",
                        pointerEvents: "none",
                      }}
                      title={`Landmark ${point.id}: x=${point.x.toFixed(3)}, y=${point.y.toFixed(3)}, z=${point.z.toFixed(3)}, q=${point.quality.toFixed(2)}, ${point.status}`}
                    />
                  );
                })}
                <div
                  style={{
                    position: "absolute",
                    left: 12,
                    bottom: 10,
                    color: "#cbd5e1",
                    fontSize: 12,
                    fontWeight: 700,
                    background: "rgba(2,6,23,0.68)",
                    border: "1px solid rgba(148,163,184,0.24)",
                    borderRadius: 999,
                    padding: "6px 10px",
                  }}
                >
                  Top-down X/Z · radius ~{mapExtent.toFixed(2)}
                </div>
              </div>
              <div
                style={{
                  position: "relative",
                  height: 190,
                  overflow: "hidden",
                  borderRadius: 14,
                  border: "1px solid #1e293b",
                  background:
                    "radial-gradient(circle at 50% 50%, rgba(250,204,21,0.14), transparent 35%), linear-gradient(180deg, #020617 0%, #111827 100%)",
                  marginTop: 10,
                }}
              >
                <div
                  style={{
                    position: "absolute",
                    inset: 0,
                    backgroundImage:
                      "linear-gradient(rgba(148,163,184,0.12) 1px, transparent 1px), linear-gradient(90deg, rgba(148,163,184,0.12) 1px, transparent 1px)",
                    backgroundSize: "24px 24px",
                  }}
                />
                <div
                  style={{
                    position: "absolute",
                    left: "50%",
                    top: "50%",
                    width: 12,
                    height: 12,
                    borderRadius: "50%",
                    transform: "translate(-50%, -50%)",
                    background: "#f97316",
                    border: "2px solid #ffedd5",
                    boxShadow: "0 0 0 6px rgba(249,115,22,0.18)",
                    zIndex: 3,
                  }}
                  title={`Camera y=${cameraMapY.toFixed(3)}, z=${cameraMapZ.toFixed(3)}`}
                />
                {mapPoints3d.slice(0, 160).map((point) => {
                  const left = 50 + (point.z - cameraMapZ) * sideMapScale;
                  const top = 50 - (point.y - cameraMapY) * sideMapScale * sideYExaggeration;
                  if (left < -10 || left > 110 || top < -10 || top > 110) return null;

                  const isMissing = point.status === "missing";
                  const size = Math.max(3, Math.min(8, 3 + point.hits * 0.25));
                  const color = isMissing ? "#64748b" : point.isLocal ? "#facc15" : point.quality > 0.6 ? "#22c55e" : "#38bdf8";

                  return (
                    <div
                      key={`side-map-point-${point.id}`}
                      style={{
                        position: "absolute",
                        left: `${left}%`,
                        top: `${top}%`,
                        width: size,
                        height: size,
                        borderRadius: "50%",
                        transform: "translate(-50%, -50%)",
                        background: color,
                        opacity: isMissing ? 0.45 : 0.9,
                        border: point.isLocal ? "2px solid rgba(254,240,138,0.95)" : "1px solid rgba(255,255,255,0.6)",
                        pointerEvents: "none",
                      }}
                      title={`Landmark ${point.id}: x=${point.x.toFixed(3)}, y=${point.y.toFixed(3)}, z=${point.z.toFixed(3)}, q=${point.quality.toFixed(2)}, ${point.status}`}
                    />
                  );
                })}
                <div
                  style={{
                    position: "absolute",
                    left: 12,
                    bottom: 10,
                    color: "#cbd5e1",
                    fontSize: 12,
                    fontWeight: 700,
                    background: "rgba(2,6,23,0.68)",
                    border: "1px solid rgba(148,163,184,0.24)",
                    borderRadius: 999,
                    padding: "6px 10px",
                  }}
                >
                  Side Z/Y · radius ~{sideMapExtent.toFixed(2)}{sideYExaggeration > 1 ? ` · Y x${sideYExaggeration}` : ""}
                </div>
              </div>
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "repeat(3, minmax(0, 1fr))",
                  gap: 8,
                  marginTop: 10,
                  fontSize: 12,
                }}
              >
                {[
                  ["X range", mapRangeX.toFixed(2)],
                  ["Y range", mapRangeY.toFixed(2)],
                  ["Z range", mapRangeZ.toFixed(2)],
                ].map(([label, value]) => (
                  <div
                    key={String(label)}
                    style={{
                      background: "#f8fafc",
                      border: "1px solid #e2e8f0",
                      borderRadius: 12,
                      padding: "8px 10px",
                    }}
                  >
                    <div style={{ color: "#64748b", fontWeight: 700 }}>{label}</div>
                    <div style={{ color: "#0f172a", fontWeight: 800, marginTop: 2 }}>{String(value)}</div>
                  </div>
                ))}
              </div>
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "repeat(3, minmax(0, 1fr))",
                  gap: 8,
                  marginTop: 10,
                  fontSize: 12,
                }}
              >
                {[
                  ["Visible", cameraPoseData?.visible_landmark_count ?? 0],
                  ["Persistent", cameraPoseData?.persistent_landmark_count ?? 0],
                  ["Missing", cameraPoseData?.missing_landmark_count ?? 0],
                  ["2D stable", cameraPoseData?.stable_2d_landmark_count ?? 0],
                  ["Geom verified", cameraPoseData?.geometry_verified_landmark_count ?? 0],
                  ["Triangulated", cameraPoseData?.triangulated_landmark_count ?? 0],
                  ["Keyframes", cameraPoseData?.keyframes ?? 0],
                  ["Local map", cameraPoseData?.local_landmark_count ?? 0],
                  ["Local visible", cameraPoseData?.local_visible_landmark_count ?? 0],
                  ["Geom inliers", cameraPoseData?.geometric_inlier_count ?? 0],
                  ["Covis edges", cameraPoseData?.covisibility_edges ?? 0],
                  ["Baseline", Number(cameraPoseData?.local_keyframe_baseline ?? 0).toFixed(3)],
                  ["BA refined", cameraPoseData?.ba_lite?.landmarks_refined ?? 0],
                  ["BA low parallax", cameraPoseData?.ba_lite?.last_skipped_low_parallax ?? 0],
                  ["SW BA", cameraPoseData?.sliding_ba?.last_status || "n/a"],
                  ["SW BA obs", cameraPoseData?.sliding_ba?.last_observations ?? 0],
                  ["BA rejected", cameraPoseData?.sliding_ba?.last_rejected ?? 0],
                  ["Tri rej angle", cameraPoseData?.triangulation?.rejected_angle ?? 0],
                  ["Tri rej reproj", cameraPoseData?.triangulation?.rejected_reprojection ?? 0],
                  ["Depth disagree", cameraPoseData?.triangulation?.depth_disagreement ?? 0],
                  ["PnP anchors", cameraPoseData?.pnp_anchor_scope || "n/a"],
                  ["SLAM", cameraPoseData?.slam_backend || "n/a"],
                  ["Re-associated", landmarkLifecycle.descriptor_reassociated ?? 0],
                  ["Pruned", landmarkLifecycle.pruned ?? 0],
                  ["Features", featureBackend.mode || "n/a"],
                  ["XFeat", descriptorBackend.status || "n/a"],
                ].map(([label, value]) => (
                  <div
                    key={String(label)}
                    style={{
                      background: "#f8fafc",
                      border: "1px solid #e2e8f0",
                      borderRadius: 12,
                      padding: "8px 10px",
                    }}
                  >
                    <div style={{ color: "#64748b", fontWeight: 700 }}>{label}</div>
                    <div style={{ color: "#0f172a", fontWeight: 800, marginTop: 2 }}>{String(value)}</div>
                  </div>
                ))}
              </div>
              {mapHealthNotes.length > 0 && (
                <div
                  style={{
                    marginTop: 10,
                    padding: "10px 12px",
                    borderRadius: 14,
                    background: "#fffbeb",
                    border: "1px solid #fde68a",
                    color: "#92400e",
                    fontSize: 12,
                    fontWeight: 700,
                    lineHeight: 1.45,
                  }}
                >
                  {mapHealthNotes.map((note) => (
                    <div key={note}>{note}</div>
                  ))}
                </div>
              )}
            </div>

            <div style={{ ...cardStyle(), padding: 18 }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12, marginBottom: 10 }}>
                <h2 style={{ margin: 0, fontSize: 18 }}>Processed frame</h2>
                <span
                  style={{
                    padding: "5px 9px",
                    borderRadius: 999,
                    background: "#f8fafc",
                    border: "1px solid #e2e8f0",
                    color: "#475569",
                    fontSize: 12,
                    fontWeight: 700,
                  }}
                >
                  {processedFrameTimestamp !== null ? `${Date.now() - processedFrameTimestamp} ms old` : "waiting"}
                </span>
              </div>
              <div
                style={{
                  height: 200,
                  overflow: "hidden",
                  background: "#020617",
                  borderRadius: 14,
                  border: "1px solid #e2e8f0",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                }}
              >
                {embodiedState.processedFrameUrl ? (
                  <div
                    style={{
                      position: "relative",
                      width: "fit-content",
                      height: "100%",
                      maxWidth: "100%",
                      margin: "0 auto",
                      background: "#020617",
                    }}
                  >
                    <img
                      src={embodiedState.processedFrameUrl}
                      alt="Processed frame"
                      style={{ width: "auto", height: "100%", maxWidth: "100%", objectFit: "contain", display: "block" }}
                    />
                    {embodiedState.processedFrameLandmarks.slice(0, 180).map((point: any) => {
                      const imageXY = point?.image_xy;
                      if (!Array.isArray(imageXY) || imageXY.length < 2) return null;

                      const px = Number(imageXY[0]);
                      const py = Number(imageXY[1]);
                      if (!Number.isFinite(px) || !Number.isFinite(py)) return null;

                      const hits = Number(point?.hits || 0);
                      const size = Math.max(4, Math.min(9, 3 + hits * 0.45));

                      return (
                        <div
                          key={`processed-landmark-${point?.id ?? `${px}-${py}`}`}
                          style={{
                            position: "absolute",
                            left: `${(px / embodiedState.processedFrameSize.width) * 100}%`,
                            top: `${(py / embodiedState.processedFrameSize.height) * 100}%`,
                            width: size,
                            height: size,
                            borderRadius: "50%",
                            transform: "translate(-50%, -50%)",
                            background: "rgba(56, 189, 248, 0.95)",
                            border: "1px solid rgba(224, 242, 254, 0.95)",
                            boxShadow: "0 0 0 3px rgba(56, 189, 248, 0.12)",
                            pointerEvents: "none",
                          }}
                          title={`Landmark ${point?.id ?? "?"} • hits ${hits}`}
                        />
                      );
                    })}
                  </div>
                ) : (
                  <span
                    style={{
                      color: "#cbd5e1",
                      fontSize: 12,
                      position: "absolute",
                      left: "50%",
                      top: "50%",
                      transform: "translate(-50%, -50%)",
                    }}
                  >
                    (No processed frame yet)
                  </span>
                )}
              </div>
              <div style={{ marginTop: 14 }}>
                <div style={{ color: "#475569", fontSize: 13, fontWeight: 800, marginBottom: 8 }}>
                  Depth feedback
                </div>
                <div
                  style={{
                    minHeight: 120,
                    overflow: "hidden",
                    background: "#020617",
                    borderRadius: 14,
                    border: "1px solid #e2e8f0",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                  }}
                >
                  {embodiedState.depthDebugUrl ? (
                    <img
                      src={embodiedState.depthDebugUrl}
                      alt="Depth feedback"
                      style={{
                        width: "100%",
                        height: "auto",
                        maxHeight: 180,
                        objectFit: "contain",
                        display: "block",
                      }}
                    />
                  ) : (
                    <span style={{ color: "#cbd5e1", fontSize: 12 }}>(No depth feedback yet)</span>
                  )}
                </div>
              </div>
            </div>

            <div style={{ ...cardStyle(), padding: 18 }}>
              <h2 style={{ marginTop: 0, fontSize: 18 }}>Camera pose</h2>
              <pre
                style={{
                  whiteSpace: "pre-wrap",
                  fontSize: 12,
                  height: 200,
                  overflowY: "auto",
                  background: "#f8fafc",
                  padding: 14,
                  borderRadius: 14,
                  border: "1px solid #e2e8f0",
                  margin: 0,
                }}
              >
                {embodiedState.cameraPoseText || "(No pose yet)"}
              </pre>
            </div>

            <div style={{ ...cardStyle(), padding: 18 }}>
              <h2 style={{ marginTop: 0, fontSize: 18 }}>World debug</h2>
              <pre
                style={{
                  whiteSpace: "pre-wrap",
                  fontSize: 12,
                  height: 200,
                  overflowY: "auto",
                  background: "#f8fafc",
                  padding: 14,
                  borderRadius: 14,
                  border: "1px solid #e2e8f0",
                  margin: 0,
                }}
              >
                {embodiedState.worldDebugText || "(No world debug yet)"}
              </pre>
            </div>

            <div style={{ ...cardStyle(), padding: 18 }}>
              <h2 style={{ marginTop: 0, fontSize: 18 }}>Hands</h2>
              <pre
                style={{
                  whiteSpace: "pre-wrap",
                  fontSize: 12,
                  height: 200,
                  overflowY: "auto",
                  background: "#f8fafc",
                  padding: 14,
                  borderRadius: 14,
                  border: "1px solid #e2e8f0",
                  margin: 0,
                }}
              >
                {embodiedState.handsText || "(No hands yet)"}
              </pre>
            </div>
          </section>
        )}
      </div>
    </main>
  );
}





