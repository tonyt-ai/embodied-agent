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
    const s = Math.max(-1, Math.min(1, float32Array[i]));
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

function base64ByteLength(value: string) {
  const clean = String(value || "").replace(/=+$/, "");
  return Math.floor((clean.length * 3) / 4);
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
type SceneVideoFile = "scene_sophie.mp4";

const SCENE_VIDEO_OPTIONS: Array<{ file: SceneVideoFile; label: string; note: string }> = [
  { file: "scene_sophie.mp4", label: "Sophie demo", note: "bottle/toy mat-tray loop" },
];
const DEFAULT_SCENE_VIDEO: SceneVideoFile = "scene_sophie.mp4";
const SCENE_TIMELINE_CUE_ENABLED = false;
const SOPHIE_SCENE_TIMELINE = [
  { id: "baby_bottle_first_grab", object: "baby bottle", source: "mat", target: "tray", grabStartS: 30.0, releaseS: 41.0 },
  { id: "toy_giraffe_first_grab", object: "toy giraffe", source: "mat", target: "tray", grabStartS: 51.0, releaseS: 60.0 },
  { id: "baby_bottle_tray_to_mat_1", object: "baby bottle", source: "tray", target: "mat", grabStartS: 68.0, releaseS: 74.0 },
  { id: "toy_giraffe_tray_to_mat_1", object: "toy giraffe", source: "tray", target: "mat", grabStartS: 81.0, releaseS: 87.0 },
  { id: "toy_giraffe_mat_to_tray_2", object: "toy giraffe", source: "mat", target: "tray", grabStartS: 92.0, releaseS: 97.0 },
  { id: "baby_bottle_mat_to_tray_2", object: "baby bottle", source: "mat", target: "tray", grabStartS: 104.0, releaseS: 111.0 },
  { id: "baby_bottle_tray_to_mat_2", object: "baby bottle", source: "tray", target: "mat", grabStartS: 121.0, releaseS: 127.0 },
  { id: "toy_giraffe_tray_to_mat_2", object: "toy giraffe", source: "tray", target: "mat", grabStartS: 131.0, releaseS: 138.0 },
  { id: "baby_bottle_mat_to_tray_3", object: "baby bottle", source: "mat", target: "tray", grabStartS: 143.0, releaseS: 148.0 },
  { id: "toy_giraffe_mat_to_tray_3", object: "toy giraffe", source: "mat", target: "tray", grabStartS: 154.0, releaseS: 160.0 },
  { id: "baby_bottle_tray_to_mat_3", object: "baby bottle", source: "tray", target: "mat", grabStartS: 168.0, releaseS: 172.0 },
  { id: "toy_giraffe_tray_to_mat_3", object: "toy giraffe", source: "tray", target: "mat", grabStartS: 178.0, releaseS: 184.0 },
  { id: "toy_giraffe_mat_to_tray_4", object: "toy giraffe", source: "mat", target: "tray", grabStartS: 191.0, releaseS: 196.0 },
  { id: "baby_bottle_mat_to_tray_4", object: "baby bottle", source: "mat", target: "tray", grabStartS: 204.0, releaseS: 209.0 },
] as const;
const SOPHIE_SCENE_REGION_BOXES: Record<string, [number, number, number, number]> = {
  mat: [0.50, 0.24, 0.98, 0.94],
  tray: [0.02, 0.12, 0.47, 0.83],
};
const SCENE_PROFILE = {
  id: "tabletop-transfer",
  targetLabels: ["mat", "tray"],
  movableLabels: ["bottle", "baby bottle", "toy giraffe", "apple", "banana", "orange", "fruit"],
  rawMovableLabels: ["mouse", "donut", "toy"],
  allowedRefinedLabels: ["mat", "tray", "bottle", "baby bottle", "toy giraffe", "object", "unknown"],
  transferTargets: ["mat", "tray"],
};

type ButtonTone = "primary" | "secondary" | "danger";

const HAND_BONES: Array<[number, number]> = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
];

const DEBUG_LABEL_HINTS: Record<string, { label: string; confidence: number }> = {
  apple: { label: "apple", confidence: 0.98 },
  cup: { label: "cup", confidence: 0.97 },
  "dining table": { label: "dining table", confidence: 0.99 },
  person: { label: "hand", confidence: 0.9 },
  unknown_seg: { label: "dish", confidence: 0.95 },
};

function normalizeDisplayLabel(label: string) {
  const text = String(label || "").trim().toLowerCase().replace(/_/g, " ").replace(/\s+/g, " ");
  if (["cake stand", "cake plate", "serving stand", "serving plate", "fruit stand", "fruit plate", "fruit bowl"].includes(text)) {
    return "dish";
  }
  if (["black mat", "table mat", "placemat"].includes(text)) {
    return "mat";
  }
  if (["plastic tray", "white tray", "serving tray", "tray"].includes(text)) {
    return "tray";
  }
  if (["sophie", "sophie giraffe", "sophie the giraffe", "toy giraffe", "giraffe toy", "rubber giraffe", "giraffe"].includes(text)) {
    return "toy giraffe";
  }
  if (["mouse", "donut", "toy"].includes(text)) {
    return "toy giraffe";
  }
  if (["cup", "mug"].includes(text)) {
    return "object";
  }
  if (["teddy bear", "stuffed animal", "plush"].includes(text)) {
    return "toy giraffe";
  }
  return text;
}

const MOVABLE_DISPLAY_LABELS = new Set(SCENE_PROFILE.movableLabels);
const RAW_MOVABLE_DISPLAY_LABELS = new Set(SCENE_PROFILE.rawMovableLabels);
const TARGET_DISPLAY_LABELS = new Set(SCENE_PROFILE.targetLabels);
const GUIDANCE_FRESH_MAX_MS = 1800;
const WORLD_STATE_ACTIONABLE_MAX_MS = 6000;
const WORLD_STATE_RENDERABLE_MAX_MS = 4500;
const GRABBED_SPEECH_CONTACT_MAX_S = 12.0;
const RELEASED_SPEECH_EVENT_MAX_S = 2.0;
const INTENT_HUD_LATCH_MS = 6500;
const RELEASE_UI_ENABLED = true;

function isMovableDisplayLabel(label: string) {
  const normalized = normalizeDisplayLabel(label);
  return MOVABLE_DISPLAY_LABELS.has(normalized) || RAW_MOVABLE_DISPLAY_LABELS.has(normalized);
}

function isTargetDisplayLabel(label: string) {
  return TARGET_DISPLAY_LABELS.has(normalizeDisplayLabel(label));
}

function defaultPlaceTargetForLabel(label: string) {
  const normalized = normalizeDisplayLabel(label);
  if (["apple", "banana", "orange", "fruit"].includes(normalized)) return "dish";
  if (["bottle", "baby bottle", "toy giraffe", "mouse", "donut", "toy"].includes(normalized)) return "tray";
  if (["cup", "mug"].includes(normalized)) return "tray";
  return "target";
}

function targetAliasForPreference(label: string, preferred = "") {
  const normalized = normalizeDisplayLabel(label);
  if (preferred === "mat" && ["dish", "plate", "platter", "coaster"].includes(normalized)) return "mat";
  if (preferred === "tray" && ["dish", "plate", "platter", "coaster"].includes(normalized)) return "tray";
  return normalized;
}

function sanitizeSceneRefinedLabels(labels: Record<string, { label: string; confidence: number }>) {
  const allowed = new Set(SCENE_PROFILE.allowedRefinedLabels);
  const next: Record<string, { label: string; confidence: number }> = {};
  for (const [key, value] of Object.entries(labels || {})) {
    let label = normalizeDisplayLabel(value?.label || "");
    if (["black mat", "table mat", "placemat"].includes(label)) label = "mat";
    if (["white tray", "plastic tray", "serving tray"].includes(label)) label = "tray";
    if (!allowed.has(label)) continue;
    next[key] = { ...value, label };
  }
  return next;
}

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

function colorFromId(id: string) {
  let hash = 0;
  for (let i = 0; i < id.length; i += 1) {
    hash = (hash * 31 + id.charCodeAt(i)) >>> 0;
  }
  const hue = hash % 360;
  return `hsl(${hue}, 78%, 58%)`;
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
  const avatarAudioChunksRef = useRef<string[]>([]);
  const avatarAudioBytesRef = useRef(0);
  const avatarAudioFirstChunkRef = useRef(true);

  const [captureMode, setCaptureMode] = useState<CaptureMode>("embodied");
  const [avatarMode, setAvatarMode] = useState<AvatarMode>("ai");
  const [embodiedVideoSource, setEmbodiedVideoSource] = useState<EmbodiedVideoSource>("scene");
  const [sceneVideoFile] = useState<SceneVideoFile>(DEFAULT_SCENE_VIDEO);
  const [status, setStatus] = useState("idle");
  const [avatarConnected, setAvatarConnected] = useState(false);
  const [bridgeConnected, setBridgeConnected] = useState(false);
  const [worldModelConnected, setWorldModelConnected] = useState(false);
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
  const worldFrameInFlightSinceRef = useRef<number>(0);
  const pendingWorldFrameRef = useRef<{
    image: string;
    timestamp: number;
    width: number;
    height: number;
  } | null>(null);
  const sentWorldFramesRef = useRef<Map<number, {
    image: string;
    timestamp: number;
    width: number;
    height: number;
  }>>(new Map());
  const queuedWorldFrameRef = useRef<{
    image: string;
    timestamp: number;
    width: number;
    height: number;
    captureMs: number;
  } | null>(null);
  const labelReqInFlightRef = useRef(false);
  const labelReqCooldownRef = useRef(false);
  const labelReqCooldownTimerRef = useRef<number | null>(null);
  const labelReqSeqRef = useRef(0);
  const labelCropCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const lastDebugTextUpdateRef = useRef<number>(0);
  const autoModeRef = useRef<boolean>(false);
  const lastGuidanceQueryAtRef = useRef<number>(0);
  const lastGuidanceSignatureRef = useRef<string>("");
  const lastSceneTimelineSpeechRef = useRef<string>("");
  const lastDirectWorldCueRef = useRef<{ signature: string; timeMs: number }>({ signature: "", timeMs: 0 });

  const [worldStateText, setWorldStateText] = useState("");
  const [eventLog, setEventLog] = useState("");
  const [lastQueryResultText, setLastQueryResultText] = useState("");
  const [plannerSummary, setPlannerSummary] = useState("");
  const [plannerSimulations, setPlannerSimulations] = useState<any | null>(null);
  const [bestActionName, setBestActionName] = useState("");
  const [cameraPoseText, setCameraPoseText] = useState("");
  const [cameraPoseDataState, setCameraPoseDataState] = useState<any | null>(null);
  const [objects3dText, setObjects3dText] = useState("");
  const [objects3dDataState, setObjects3dDataState] = useState<any[]>([]);
  const [sparseMapText, setSparseMapText] = useState("");
  const [sparseMapData, setSparseMapData] = useState<any[]>([]);
  const [processedFrameUrl, setProcessedFrameUrl] = useState("");
  const [processedFrameTimestamp, setProcessedFrameTimestamp] = useState<number | null>(null);
  const [processedFrameLandmarks, setProcessedFrameLandmarks] = useState<any[]>([]);
  const [processedFrameSize, setProcessedFrameSize] = useState({ width: 640, height: 360 });
  const [sceneVideoTimeS, setSceneVideoTimeS] = useState(0);
  const [depthDebugSize, setDepthDebugSize] = useState({ width: 160, height: 90 });
  const [handsText, setHandsText] = useState("");
  const [handInteractionsText, setHandInteractionsText] = useState("");
  const [handInteractionsDataState, setHandInteractionsDataState] = useState<any[]>([]);
  const [manipulationEventsText, setManipulationEventsText] = useState("");
  const [manipulationEventsDataState, setManipulationEventsDataState] = useState<any[]>([]);
  const [handsDataState, setHandsDataState] = useState<any[]>([]);
  const [handTrajectoriesDataState, setHandTrajectoriesDataState] = useState<any[]>([]);
  const [cameraTrailState, setCameraTrailState] = useState<number[][]>([]);
  const [worldDebugText, setWorldDebugText] = useState("");
  const [worldDebugDataState, setWorldDebugDataState] = useState<any>({});
  const [refinedLabelsById, setRefinedLabelsById] = useState<Record<string, { label: string; confidence: number }>>({});
  const [refinedLabelsByHint, setRefinedLabelsByHint] = useState<Record<string, { label: string; confidence: number }>>(DEBUG_LABEL_HINTS);
  const [depthDebugUrl, setDepthDebugUrl] = useState("");
  const [cardExpanded, setCardExpanded] = useState<Record<string, boolean>>({
    mapView: true,
    processedFrame: true,
    mapDiagnostics: false,
    object3d: false,
    sparseMap: false,
    cameraPose: false,
    worldDebug: false,
    hands: false,
    handInteractions: false,
    manipulationEvents: false,
  });
  const [diagnosticsHeightPx, setDiagnosticsHeightPx] = useState<number>(280);

  const [autoMode, setAutoMode] = useState(false);
  const [showDebug, setShowDebug] = useState(false);
  const lastSpokenRef = useRef<string>("");
  const lastSpeakTimeRef = useRef<number>(0);
  const avatarSpeechRequestSeqRef = useRef<number>(0);
  const useAvatarSpeechRef = useRef<boolean>(false);
  const geminiReadyRef = useRef<boolean>(false);
  const pendingAvatarCueRef = useRef<string | null>(null);
  const startupCueSpokenRef = useRef<boolean>(false);
  const [useAvatarSpeech, setUseAvatarSpeech] = useState(false);
  const [useWebSpeechDebug, setUseWebSpeechDebug] = useState(true);

  const [frameAgeMs, setFrameAgeMs] = useState<number | null>(null);
  const [captureMs, setCaptureMs] = useState<number | null>(null);
  const [serverDecodeMs, setServerDecodeMs] = useState<number | null>(null);
  const [serverDetectMs, setServerDetectMs] = useState<number | null>(null);
  const [serverDepthMs, setServerDepthMs] = useState<number | null>(null);
  const [serverPoseMs, setServerPoseMs] = useState<number | null>(null);
  const [serverWorldMs, setServerWorldMs] = useState<number | null>(null);
  const [serverTotalMs, setServerTotalMs] = useState<number | null>(null);
  const [pipelineAgeMs, setPipelineAgeMs] = useState<number | null>(null);
  const [uiNowMs, setUiNowMs] = useState<number>(() => Date.now());
  const [worldStateReceivedAtMs, setWorldStateReceivedAtMs] = useState<number>(0);
  const [latchedIntentCue, setLatchedIntentCue] = useState<any | null>(null);
  const [latchedAttentionBlobs, setLatchedAttentionBlobs] = useState<any[]>([]);
  const [latchedAttentionAtMs, setLatchedAttentionAtMs] = useState<number>(0);

  const isEmbodiedMode = captureMode === "embodied";
  const isSocialMode = captureMode === "social";
  const displayedFrameAgeMs = processedFrameTimestamp !== null
    ? Math.max(0, uiNowMs - processedFrameTimestamp)
    : frameAgeMs;
  const useProcessedPlanningFrame = false;
  const worldStateFreshEnough = worldStateReceivedAtMs > 0
    && uiNowMs - worldStateReceivedAtMs <= WORLD_STATE_ACTIONABLE_MAX_MS;
  const planningFreshnessAgeMs = worldStateFreshEnough ? 0 : displayedFrameAgeMs;
  const planningFreshness = planningFreshnessAgeMs == null
    ? 0
    : Math.max(0.08, Math.min(1, 1 - Math.max(0, planningFreshnessAgeMs - 350) / 1800));
  const cameraPoseData = cameraPoseDataState ?? parseJsonSafe<any>(cameraPoseText, null);
  const worldDebugData = worldDebugDataState;
  const persistentMapData = Array.isArray(cameraPoseData?.persistent_map)
    ? cameraPoseData.persistent_map
    : [];
  const localMapData = Array.isArray(cameraPoseData?.local_sparse_map)
    ? cameraPoseData.local_sparse_map
    : [];
  const visibleMapData = Array.isArray(sparseMapData) ? sparseMapData : [];
  const hasFinitePoint = (points: any[]) => points.some((point: any) => {
    const candidates = [
      point?.position_world,
      point?.triangulated_position_world,
      point?.position_world_depth_prior,
    ];
    return candidates.some((candidate: any) => (
      Array.isArray(candidate)
      && candidate.length >= 3
      && Number.isFinite(candidate[0])
      && Number.isFinite(candidate[1])
      && Number.isFinite(candidate[2])
    ));
  });
  const mapSourceData = hasFinitePoint(persistentMapData)
    ? persistentMapData
    : (hasFinitePoint(localMapData) ? localMapData : visibleMapData);
  const cameraPositionWorld = Array.isArray(cameraPoseData?.camera_position_world)
    ? cameraPoseData.camera_position_world
    : [0, 0, 0];
  const handsData = handsDataState;
  const handTrajectoriesData = Array.isArray(handTrajectoriesDataState) && handTrajectoriesDataState.length > 0
    ? handTrajectoriesDataState
    : (Array.isArray(worldDebugData?.hand_trajectories) ? worldDebugData.hand_trajectories : []);
  const handInteractionsData = handInteractionsDataState;
  const manipulationEventsData = manipulationEventsDataState;
  const objects3dData = objects3dDataState;
  const labelKeyForObject = (obj: any) => String(obj?.label || "unknown").trim().toLowerCase();
  const sophieShapeLabelForObject = (obj: any, rawLabel: string) => {
    const visualClass = String(obj?.visual_identity_class || "");
    if (visualClass === "toy_giraffe") return "toy giraffe";
    if (visualClass === "baby_bottle") return "baby bottle";
    const raw = normalizeDisplayLabel(rawLabel);
    if (!["bottle", "baby bottle", "cup", "vase"].includes(raw)) return "";
    const bbox = Array.isArray(obj?.bbox) ? obj.bbox : [];
    if (bbox.length < 4) return "";
    const bw = Math.max(0, Number(bbox[2]) - Number(bbox[0]));
    const bh = Math.max(0, Number(bbox[3]) - Number(bbox[1]));
    const area = bw * bh;
    if (![bw, bh, area].every(Number.isFinite) || bh <= 0) return "";
    const aspect = bw / bh;
    if (area >= 0.006 && area <= 0.16 && aspect >= 0.72) return "toy giraffe";
    if (aspect <= 0.68) return "baby bottle";
    return "";
  };
  const displayLabelForObject = (obj: any) => {
    const id = String(obj?.id || "");
    const key = labelKeyForObject(obj);
    const refinedById = refinedLabelsById[id];
    const refinedByHint = refinedLabelsByHint[key];
    const stableIdLabel = refinedById && refinedById.confidence >= 0.55 ? refinedById.label : "";
    const rawLabel = normalizeDisplayLabel(String(obj?.label || "object"));
    const hintIsSafeToShare = rawLabel.startsWith("unknown") || isTargetDisplayLabel(rawLabel) || ["dining table", "table", "person", "hand"].includes(rawLabel);
    const stableHintLabel = hintIsSafeToShare && refinedByHint && refinedByHint.confidence >= 0.55 ? refinedByHint.label : "";
    const shapeLabel = sophieShapeLabelForObject(obj, rawLabel);
    const raw = normalizeDisplayLabel(String(
      stableIdLabel
      || stableHintLabel
      || shapeLabel
      || obj?.label
      || "object"
    ));
    return raw;
  };
  const contactingObjectIds = new Set(
    handInteractionsData
      .filter((item: any) => Boolean(item?.is_contacting))
      .map((item: any) => String(item?.nearest_object_id || "")),
  );
  const manipulatedObjectIds = new Set(
    manipulationEventsData
      .map((item: any) => String(item?.object_id || ""))
      .filter((v: string) => v.length > 0),
  );
  const highlightedObjectIds = new Set<string>([...contactingObjectIds, ...manipulatedObjectIds]);
  const object3dMarkers = objects3dData
    .map((obj: any) => {
      const p = obj?.position_3d;
      if (!Array.isArray(p) || p.length < 3) return null;
      const x = Number(p[0]); const y = Number(p[1]); const z = Number(p[2]);
      if (![x, y, z].every(Number.isFinite)) return null;
      const id = String(obj?.id || "");
      if (!id) return null;
      return {
        id,
        x,
        y,
        z,
        isHighlighted: highlightedObjectIds.has(id),
        label: displayLabelForObject(obj),
      };
    })
    .filter(Boolean) as Array<{ id: string; x: number; y: number; z: number; isHighlighted: boolean; label: string }>;
  const handPoints3d = handsData
    .map((hand: any) => {
      const center = hand?.center_3d;
      if (!Array.isArray(center) || center.length < 3) return null;
      const [x, y, z] = center.map((value: any) => Number(value));
      if (![x, y, z].every(Number.isFinite)) return null;
      return {
        id: String(hand?.id || hand?.side || "hand"),
        side: String(hand?.side || "unknown"),
        confidence: Number(hand?.confidence ?? 0),
        predicted: Boolean(hand?.predicted),
        missingFrames: Number(hand?.missing_frames ?? 0),
        x,
        y,
        z,
        landmarks3d: Array.isArray(hand?.landmarks_3d) ? hand.landmarks_3d : [],
        volume3d: hand?.volume_3d || null,
      };
    })
    .filter(Boolean) as Array<{
      id: string;
      side: string;
      confidence: number;
      predicted: boolean;
      missingFrames: number;
      x: number;
      y: number;
      z: number;
      landmarks3d: number[][];
      volume3d: any;
      hasDepthPrior: boolean;
      isPriorAnchored: boolean;
    }>;
  const interactionHandPoints3d = handInteractionsData
    .map((it: any) => {
      const c = it?.hand_center_3d;
      if (!Array.isArray(c) || c.length < 3) return null;
      const x = Number(c[0]); const y = Number(c[1]); const z = Number(c[2]);
      if (![x, y, z].every(Number.isFinite)) return null;
      return {
        id: String(it?.hand_id || "hand_interaction"),
        side: String(it?.side || "unknown"),
        confidence: Number(it?.hand_confidence ?? 0),
        predicted: false,
        missingFrames: 0,
        x, y, z,
        landmarks3d: [],
        volume3d: null,
      };
    })
    .filter(Boolean) as Array<{ id: string; side: string; confidence: number; predicted: boolean; missingFrames: number; x: number; y: number; z: number; landmarks3d: number[][]; volume3d: any }>;
  const renderedHandPoints3d = handPoints3d.length > 0 ? handPoints3d : interactionHandPoints3d;
  const renderedHandIds = new Set(renderedHandPoints3d.map((h) => h.id));
  const handCapsules3d = handPoints3d.flatMap((hand) => {
    const caps = Array.isArray(hand?.volume3d?.capsules) ? hand.volume3d.capsules : [];
    return caps
      .map((cap: any, idx: number) => {
        const a = cap?.a;
        const b = cap?.b;
        const r = Number(cap?.r ?? 0.01);
        if (!Array.isArray(a) || !Array.isArray(b) || a.length < 3 || b.length < 3) return null;
        const x1 = Number(a[0]); const y1 = Number(a[1]); const z1 = Number(a[2]);
        const x2 = Number(b[0]); const y2 = Number(b[1]); const z2 = Number(b[2]);
        if (![x1, y1, z1, x2, y2, z2, r].every(Number.isFinite)) return null;
        return { key: `${hand.id}-${idx}`, handId: hand.id, predicted: hand.predicted, x1, y1, z1, x2, y2, z2, r };
      })
      .filter(Boolean) as Array<{ key: string; handId: string; predicted: boolean; x1: number; y1: number; z1: number; x2: number; y2: number; z2: number; r: number }>;
  });
  const handBones3d = handPoints3d.flatMap((hand) => {
    const lm = Array.isArray(hand.landmarks3d) ? hand.landmarks3d : [];
    return HAND_BONES.map(([a, b], idx) => {
      const p1 = lm[a];
      const p2 = lm[b];
      if (!Array.isArray(p1) || !Array.isArray(p2) || p1.length < 3 || p2.length < 3) return null;
      const x1 = Number(p1[0]); const y1 = Number(p1[1]); const z1 = Number(p1[2]);
      const x2 = Number(p2[0]); const y2 = Number(p2[1]); const z2 = Number(p2[2]);
      if (![x1, y1, z1, x2, y2, z2].every(Number.isFinite)) return null;
      return { key: `${hand.id}-${idx}`, x1, y1, z1, x2, y2, z2 };
    }).filter(Boolean) as Array<{ key: string; x1: number; y1: number; z1: number; x2: number; y2: number; z2: number }>;
  });
  const renderedHandCapsules3d = handCapsules3d.filter((cap) => renderedHandIds.has(cap.handId));
  const renderedHandBones3d = handBones3d.filter((bone) => renderedHandIds.has(String(bone.key).split("-").slice(0, -1).join("-")));
  const mapPoints3d = mapSourceData
    .map((point: any) => {
      const candidates = [
        point?.position_world,
        point?.triangulated_position_world,
        point?.position_world_depth_prior,
      ];
      const position = candidates.find((candidate: any) => {
        if (!Array.isArray(candidate) || candidate.length < 3) return false;
        return (
          Number.isFinite(candidate[0]) &&
          Number.isFinite(candidate[1]) &&
          Number.isFinite(candidate[2])
        );
      });
      if (!Array.isArray(position) || position.length < 3) return null;
      const rawX = position[0];
      const rawY = position[1];
      const rawZ = position[2];
      if (!Number.isFinite(rawX) || !Number.isFinite(rawY) || !Number.isFinite(rawZ)) return null;
      const x = Number(rawX);
      const y = Number(rawY);
      const z = Number(rawZ);
      const hasDepthPrior = Array.isArray(point?.position_world_depth_prior)
        && point.position_world_depth_prior.length >= 3
        && Number.isFinite(point.position_world_depth_prior[0])
        && Number.isFinite(point.position_world_depth_prior[1])
        && Number.isFinite(point.position_world_depth_prior[2]);
      const source = String(point?.position_world_source || "");
      return {
        id: point?.id,
        x,
        y,
        z,
        quality: Number(point?.quality ?? 0),
        hits: Number(point?.hits ?? 0),
        status: point?.status || "visible",
        isLocal: Boolean(point?.is_local_map),
        hasDepthPrior,
        isPriorAnchored: hasDepthPrior || source.includes("depth_prior") || source.includes("colmap"),
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
      hasDepthPrior: boolean;
      isPriorAnchored: boolean;
    }>;
  const rankedMapPoints3d = [...mapPoints3d].sort((a, b) => {
    const score = (point: { status: string; isLocal: boolean; quality: number; hits: number }) => (
      (point.status === "visible" ? 1000 : 0)
      + (point.isLocal ? 250 : 0)
      + Math.min(200, point.hits * 8)
      + Math.round(point.quality * 100)
    );
    return score(b) - score(a);
  });
  const mapRenderLimit = Math.max(160, Math.min(420, Math.ceil(rankedMapPoints3d.length * 0.7)));
  const mapPointsForView = rankedMapPoints3d.slice(0, mapRenderLimit);
  const cameraWorldX = Number(cameraPositionWorld[0] || 0);
  const cameraWorldY = Number(cameraPositionWorld[1] || 0);
  const cameraWorldZ = Number(cameraPositionWorld[2] || 0);
  const robustAbsPercentile = (values: number[], p = 0.92) => {
    const finite = values.filter((v) => Number.isFinite(v)).map((v) => Math.abs(v)).sort((a, b) => a - b);
    if (finite.length === 0) return 0.25;
    const idx = Math.max(0, Math.min(finite.length - 1, Math.floor((finite.length - 1) * p)));
    return Math.max(0.25, finite[idx]);
  };
  const landmarkLifecycle = cameraPoseData?.landmark_lifecycle || {};
  const descriptorBackend = cameraPoseData?.descriptor_backend || {};
  const featureBackend = cameraPoseData?.feature_backend || {};
  const centerCandidates = [
    ...mapPoints3d.map((p) => ({ x: p.x, y: p.y, z: p.z })),
    ...object3dMarkers.map((p) => ({ x: p.x, y: p.y, z: p.z })),
    ...renderedHandPoints3d.map((p) => ({ x: p.x, y: p.y, z: p.z })),
    ...renderedHandBones3d.flatMap((b) => [{ x: b.x1, y: b.y1, z: b.z1 }, { x: b.x2, y: b.y2, z: b.z2 }]),
    ...renderedHandCapsules3d.flatMap((c) => [{ x: c.x1, y: c.y1, z: c.z1 }, { x: c.x2, y: c.y2, z: c.z2 }]),
    { x: cameraWorldX, y: cameraWorldY, z: cameraWorldZ },
  ].filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y) && Number.isFinite(p.z));
  const centerMedian = (values: number[], fallback: number) => {
    const finite = values.filter((v) => Number.isFinite(v)).sort((a, b) => a - b);
    if (finite.length === 0) return fallback;
    return finite[Math.floor(finite.length * 0.5)];
  };
  const priorCandidates = mapPoints3d
    .filter((p) => p.isPriorAnchored)
    .map((p) => ({ x: p.x, y: p.y, z: p.z }));
  const priorCenter = priorCandidates.length >= 24
    ? {
        x: centerMedian(priorCandidates.map((p) => p.x), cameraWorldX),
        y: centerMedian(priorCandidates.map((p) => p.y), cameraWorldY),
        z: centerMedian(priorCandidates.map((p) => p.z), cameraWorldZ),
      }
    : null;
  const unlockedCenter = {
    x: centerMedian(centerCandidates.map((p) => p.x), cameraWorldX),
    y: centerMedian(centerCandidates.map((p) => p.y), cameraWorldY),
    z: centerMedian(centerCandidates.map((p) => p.z), cameraWorldZ),
  };
  const lockedCenter = priorCenter;
  const sceneCenterX = lockedCenter?.x ?? unlockedCenter.x;
  const sceneCenterY = lockedCenter?.y ?? unlockedCenter.y;
  const sceneCenterZ = lockedCenter?.z ?? unlockedCenter.z;
  const cameraTrailPoints = cameraTrailState
    .filter((pt) => Array.isArray(pt) && pt.length >= 3)
    .map((pt) => ({ x: Number(pt[0]), y: Number(pt[1]), z: Number(pt[2]) }))
    .filter((pt) => Number.isFinite(pt.x) && Number.isFinite(pt.y) && Number.isFinite(pt.z));
  const mapXZAbs = [...mapPoints3d, ...renderedHandPoints3d, ...object3dMarkers, ...cameraTrailPoints, ...renderedHandBones3d.flatMap((b) => [{ x: b.x1, z: b.z1 }, { x: b.x2, z: b.z2 }]), ...renderedHandCapsules3d.flatMap((c) => [{ x: c.x1, z: c.z1 }, { x: c.x2, z: c.z2 }]), { x: cameraWorldX, z: cameraWorldZ }]
    .flatMap((point: any) => [point.x - sceneCenterX, point.z - sceneCenterZ]);
  const mapExtent = robustAbsPercentile(mapXZAbs, 0.97);
  const mapZoomOutFactor = 3.1;
  const mapScale = 44 / Math.max(0.35, mapExtent * mapZoomOutFactor);
  const mapYZAbs = [...mapPoints3d, ...renderedHandPoints3d, ...object3dMarkers, ...cameraTrailPoints, ...renderedHandBones3d.flatMap((b) => [{ y: b.y1, z: b.z1 }, { y: b.y2, z: b.z2 }]), ...renderedHandCapsules3d.flatMap((c) => [{ y: c.y1, z: c.z1 }, { y: c.y2, z: c.z2 }]), { y: cameraWorldY, z: cameraWorldZ }]
    .flatMap((point: any) => [point.z - sceneCenterZ, point.y - sceneCenterY]);
  const sideMapExtent = robustAbsPercentile(mapYZAbs, 0.97);
  const sideMapScale = 44 / Math.max(0.35, sideMapExtent * mapZoomOutFactor);
  const pointXs = [...mapPoints3d, ...renderedHandPoints3d, ...renderedHandBones3d.flatMap((b) => [{ x: b.x1 }, { x: b.x2 }])].map((point: any) => point.x);
  const pointYs = [...mapPoints3d, ...renderedHandPoints3d, ...renderedHandBones3d.flatMap((b) => [{ y: b.y1 }, { y: b.y2 }])].map((point: any) => point.y);
  const pointZs = [...mapPoints3d, ...renderedHandPoints3d, ...renderedHandBones3d.flatMap((b) => [{ z: b.z1 }, { z: b.z2 }])].map((point: any) => point.z);
  const axisRange = (values: number[]) => (
    values.length > 0 ? Math.max(...values) - Math.min(...values) : 0
  );
  const mapRangeX = axisRange(pointXs);
  const mapRangeY = axisRange(pointYs);
  const mapRangeZ = axisRange(pointZs);
  const sideYExaggeration = (renderedHandPoints3d.length > 0 || renderedHandCapsules3d.length > 0)
    ? 1
    : (mapRangeY > 0 && mapRangeY < mapRangeZ * 0.35 ? 3 : 1);
  const demoGate = worldDebugData?.demo_gate || {};
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
  const debugStats: Array<[string, string | number]> = [
    ["Visible", cameraPoseData?.visible_landmark_count ?? 0],
    ["Persistent", cameraPoseData?.persistent_landmark_count ?? 0],
    ["Missing", cameraPoseData?.missing_landmark_count ?? 0],
    ["Hands 3D", renderedHandPoints3d.length],
    ["Hands predicted", worldDebugData?.hand_tracking?.hands_predicted ?? 0],
    ["Hand trails", handTrajectoriesData.length],
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
    ["Fused map", worldDebugData?.fused_map_points ?? 0],
    ["Fusion voxels", worldDebugData?.depth_fusion?.voxels ?? 0],
    ["PnP anchors", cameraPoseData?.pnp_anchor_scope || "n/a"],
    ["SLAM", cameraPoseData?.slam_backend || "n/a"],
    ["Re-associated", landmarkLifecycle.descriptor_reassociated ?? 0],
    ["Pruned", landmarkLifecycle.pruned ?? 0],
    ["Features", featureBackend.mode || "n/a"],
    ["XFeat", descriptorBackend.status || "n/a"],
  ];

  const staticTargetObjects = Array.isArray(worldDebugData?.static_targets)
    ? worldDebugData.static_targets
    : [];
  const overlayObjects = [...observedObjects, ...staticTargetObjects];
  const objectById = new Map(
    overlayObjects
      .map((obj: any) => {
        const id = String(obj?.id || "");
        const bbox = Array.isArray(obj?.projected_bbox) ? obj.projected_bbox : obj?.bbox;
        if (!id || !Array.isArray(bbox) || bbox.length < 4) return null;
        return [id, { ...obj, bbox }] as const;
      })
      .filter(Boolean) as Array<readonly [string, any]>
  );
  const projectStaticTargetBbox = (obj: any, labelHint = "") => {
    if (Array.isArray(obj?.projected_bbox) && obj.projected_bbox.length >= 4) {
      return obj.projected_bbox;
    }
    const p = obj?.position_3d;
    const cam = cameraPoseData?.camera_position_world;
    const rot =
      Array.isArray(cameraPoseData?.rotation_cw) ? cameraPoseData.rotation_cw :
      (Array.isArray(cameraPoseData?.rotation_wc)
        ? [
            [cameraPoseData.rotation_wc[0]?.[0], cameraPoseData.rotation_wc[1]?.[0], cameraPoseData.rotation_wc[2]?.[0]],
            [cameraPoseData.rotation_wc[0]?.[1], cameraPoseData.rotation_wc[1]?.[1], cameraPoseData.rotation_wc[2]?.[1]],
            [cameraPoseData.rotation_wc[0]?.[2], cameraPoseData.rotation_wc[1]?.[2], cameraPoseData.rotation_wc[2]?.[2]],
          ]
        : null);
    const intr = worldDebugData?.intrinsics;
    if (!Array.isArray(p) || p.length < 3 || !Array.isArray(cam) || cam.length < 3 || !Array.isArray(rot) || rot.length < 3 || !intr) {
      return null;
    }
    const dx = Number(p[0]) - Number(cam[0]);
    const dy = Number(p[1]) - Number(cam[1]);
    const dz = Number(p[2]) - Number(cam[2]);
    const cx3 = Number(rot[0]?.[0]) * dx + Number(rot[0]?.[1]) * dy + Number(rot[0]?.[2]) * dz;
    const cy3 = Number(rot[1]?.[0]) * dx + Number(rot[1]?.[1]) * dy + Number(rot[1]?.[2]) * dz;
    const cz3 = Number(rot[2]?.[0]) * dx + Number(rot[2]?.[1]) * dy + Number(rot[2]?.[2]) * dz;
    if (![cx3, cy3, cz3].every(Number.isFinite) || cz3 <= 0.05) return null;
    const fx = Number(intr.fx);
    const fy = Number(intr.fy);
    const cx = Number(intr.cx);
    const cy = Number(intr.cy);
    if (![fx, fy, cx, cy].every(Number.isFinite)) return null;
    const w = Math.max(1, processedFrameSize.width);
    const h = Math.max(1, processedFrameSize.height);
    const u = (fx * (cx3 / cz3) + cx) / w;
    const v = (fy * (cy3 / cz3) + cy) / h;
    const label = normalizeDisplayLabel(labelHint || displayLabelForObject(obj));
    const diameterM = label === "coaster" ? 0.095 : (label === "dish" ? 0.18 : (label === "mat" ? 0.385 : (label === "tray" ? 0.32 : 0.11)));
    const bw = Math.max(0.035, Math.min(0.22, (fx * diameterM / cz3) / w));
    const bh = Math.max(0.035, Math.min(0.22, (fy * diameterM / cz3) / h));
    if (!Number.isFinite(u) || !Number.isFinite(v) || u < -0.25 || u > 1.25 || v < -0.25 || v > 1.25) return null;
    return [
      Math.max(0, Math.min(1, u - bw * 0.5)),
      Math.max(0, Math.min(1, v - bh * 0.5)),
      Math.max(0, Math.min(1, u + bw * 0.5)),
      Math.max(0, Math.min(1, v + bh * 0.5)),
    ];
  };
  const targetBboxForObject = (obj: any, labelHint = "") => {
    const label = normalizeDisplayLabel(labelHint || displayLabelForObject(obj));
    const projected = projectStaticTargetBbox(obj, label);
    if (projected) return projected;
    const bbox = Array.isArray(obj?.bbox) && obj.bbox.length >= 4 ? obj.bbox : null;
    if (!bbox) return null;
    const area = Math.max(0, Number(bbox[2]) - Number(bbox[0])) * Math.max(0, Number(bbox[3]) - Number(bbox[1]));
    if (isTargetDisplayLabel(label)) {
      const maxArea = label === "mat" ? 0.34 : 0.26;
      if (!Number.isFinite(area) || area > maxArea) return null;
    }
    return bbox;
  };
  const displayLabelForCandidate = (candidate: any, fallback = "object") => {
    const id = String(candidate?.object_id || candidate?.id || "");
    const obj = id ? objectById.get(id) : null;
    return displayLabelForObject(obj || candidate || { label: fallback });
  };
  const heldObjectIdForInteraction = (interaction: any) => String(
    interaction?.held_object_id
    || interaction?.learned_object_id
    || interaction?.nearest_object_id
    || ""
  );
  const heldObjectForInteraction = (interaction: any) => {
    const id = heldObjectIdForInteraction(interaction);
    return id ? objectById.get(id) : null;
  };
  const heldLabelForInteraction = (interaction: any, fallback = "object") => {
    const id = heldObjectIdForInteraction(interaction);
    const obj = id ? objectById.get(id) : null;
    return displayLabelForObject(obj || {
      id,
      label: interaction?.held_object_label
        || interaction?.learned_object_label
        || interaction?.nearest_object_label
        || fallback,
      visual_identity_class: interaction?.visual_identity_class || "",
    });
  };
  const targetFallbackObject = (desiredLabel: string) => {
    const desired = normalizeDisplayLabel(desiredLabel);
    const withBox = overlayObjects.filter((obj: any) => {
      const bbox = targetBboxForObject(obj, desiredLabel);
      return Array.isArray(bbox) && bbox.length >= 4;
    });
    const exact = withBox.find((obj: any) => targetAliasForPreference(displayLabelForObject(obj), desired) === desired);
    if (exact) return exact;
    const unknownTargets = withBox
      .map((obj: any) => {
        const raw = normalizeDisplayLabel(String(obj?.label || ""));
        const bbox = targetBboxForObject(obj, desiredLabel);
        const area = Math.max(0, Number(bbox[2]) - Number(bbox[0])) * Math.max(0, Number(bbox[3]) - Number(bbox[1]));
        return { obj, raw, area };
      })
      .filter((item) => {
        const maxArea = desired === "mat" || desired === "tray" ? 0.82 : 0.18;
        return item.raw.startsWith("unknown") && Number.isFinite(item.area) && item.area > 0.002 && item.area < maxArea;
      });
    if (!unknownTargets.length) return null;
    unknownTargets.sort((a, b) => desired === "coaster" ? a.area - b.area : b.area - a.area);
    return unknownTargets[0]?.obj || null;
  };
  const nearestStaticTargetLabelForObject = (obj: any) => {
    const p = obj?.position_3d;
    if (!Array.isArray(p) || p.length < 3) return "";
    let bestLabel = "";
    let bestDist = Number.POSITIVE_INFINITY;
    for (const target of staticTargetObjects) {
      if (!target?.locked) continue;
      const tp = target?.position_3d;
      if (!Array.isArray(tp) || tp.length < 3) continue;
      const dx = Number(p[0]) - Number(tp[0]);
      const dy = Number(p[1]) - Number(tp[1]);
      const dz = Number(p[2]) - Number(tp[2]);
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (!Number.isFinite(dist) || dist >= bestDist) continue;
      bestDist = dist;
      bestLabel = targetAliasForPreference(displayLabelForObject(target), "mat");
    }
    return bestLabel;
  };
  const alternateTransferTarget = (source: string) => {
    const targets = SCENE_PROFILE.transferTargets.map((item) => normalizeDisplayLabel(item));
    if (targets.length < 2) return "";
    const normalized = normalizeDisplayLabel(source);
    const idx = targets.indexOf(normalized);
    if (idx < 0) return "";
    return targets[(idx + 1) % targets.length];
  };
  const hasStaticTargetLabel = (desiredLabel: string) => {
    const desired = normalizeDisplayLabel(desiredLabel);
    return staticTargetObjects.some((target: any) => (
      Boolean(target?.locked ?? true) &&
      targetAliasForPreference(displayLabelForObject(target), desired) === desired
    ));
  };
  const desiredPlaceTargetForObject = (label: string, obj: any) => {
    const normalized = normalizeDisplayLabel(label);
    if (["bottle", "baby bottle", "toy giraffe", "mouse", "donut", "toy"].includes(normalized)) {
      const source = nearestStaticTargetLabelForObject(obj);
      const transferTarget = alternateTransferTarget(source);
      if (transferTarget) return transferTarget;
      const fallbackTransferTarget = SCENE_PROFILE.transferTargets.find((target) => hasStaticTargetLabel(target));
      if (fallbackTransferTarget) return normalizeDisplayLabel(fallbackTransferTarget);
      if (source === "coaster" || source === "tray") return "mat";
      if (source === "mat" || source === "dish") return hasStaticTargetLabel("tray") ? "tray" : "coaster";
      return hasStaticTargetLabel("tray") ? "tray" : "coaster";
    }
    return defaultPlaceTargetForLabel(label);
  };
  const learnedPlaceTargetForCue = (...items: any[]) => {
    for (const item of items) {
      const label = normalizeDisplayLabel(String(item?.pred_target_label || item?.target_label || ""));
      if (!isTargetDisplayLabel(label)) continue;
      const motionScore = Number(item?.pred_target_motion_score ?? 0);
      if (Number.isFinite(motionScore) && motionScore >= 0.15) {
        return label;
      }
      const prob = Number(item?.pred_target_tray_prob ?? item?.target_tray_prob);
      if (!Number.isFinite(prob) || prob >= 0.56 || prob <= 0.44) {
        return label;
      }
    }
    return "";
  };

  const attentionBlobs: Array<{ id: string; kind: "grab" | "place"; x1: number; y1: number; x2: number; y2: number; score: number; label: string }> = [];
  const attentionLinks: Array<{ id: string; fromX: number; fromY: number; toX: number; toY: number; kind: "grab" | "place"; score: number }> = [];
  for (const interaction of handInteractionsData) {
    const handCenter = handsData
      .find((h: any) => String(h?.id || h?.side || "") === String(interaction?.hand_id || ""))
      ?.pixel_center;
    const hcx = Array.isArray(handCenter) ? Number(handCenter[0]) : Number.NaN;
    const hcy = Array.isArray(handCenter) ? Number(handCenter[1]) : Number.NaN;

    const grab = Array.isArray(interaction?.intent_grab_candidates) ? interaction.intent_grab_candidates : [];
    const place = Array.isArray(interaction?.intent_place_candidates) ? interaction.intent_place_candidates : [];
    const rankedGrab = [...grab].sort((a: any, b: any) => Number(b?.pred_contact_prob ?? 0) - Number(a?.pred_contact_prob ?? 0));
    const rankedPlace = [...place].sort((a: any, b: any) => Number(b?.pred_placement_prob ?? 0) - Number(a?.pred_placement_prob ?? 0));
    const topPredictedGrab = rankedGrab.find((c: any) => {
      const label = displayLabelForCandidate(c);
      return isMovableDisplayLabel(label) && Number(c?.pred_contact_prob ?? 0) >= 0.35;
    }) || null;
    const nearestObjectId = String(interaction?.nearest_object_id || "");
    const nearestObject = nearestObjectId ? objectById.get(nearestObjectId) : null;
    const nearestLabel = displayLabelForObject(nearestObject || { id: nearestObjectId, label: interaction?.nearest_object_label || "object" });
    const heldObjectId = heldObjectIdForInteraction(interaction);
    const heldObject = heldObjectForInteraction(interaction);
    const heldLabel = heldLabelForInteraction(interaction, nearestLabel);
    const plannedObjectId = String(topPredictedGrab?.object_id || nearestObjectId || "");
    const plannedObject = plannedObjectId ? objectById.get(plannedObjectId) : nearestObject;
    const plannedLabel = topPredictedGrab ? displayLabelForCandidate(topPredictedGrab) : nearestLabel;
    const distanceM = Number(interaction?.distance_m ?? 9);
    const predContact = Number(interaction?.pred_contact_prob ?? 0);
    const isContactLike = Boolean(interaction?.learned_is_held || interaction?.is_contacting || interaction?.is_touching_strict);
    const isHoldingMovable = isContactLike && isMovableDisplayLabel(heldLabel);
    const topPredictedGrabScore = Number(topPredictedGrab?.pred_contact_prob ?? 0);
    const isPlanningMovable = isMovableDisplayLabel(plannedLabel)
      && Boolean(isContactLike || interaction?.is_near || distanceM < 0.45 || predContact > 0.16 || topPredictedGrabScore >= 0.35);

    if (isHoldingMovable && heldObject?.bbox && Array.isArray(heldObject.bbox) && heldObject.bbox.length >= 4) {
      const bbox = heldObject.bbox;
      attentionBlobs.push({
        id: `grab-${interaction?.hand_id}-${heldObjectId || "held"}`,
        kind: "grab",
        x1: Number(bbox[0]),
        y1: Number(bbox[1]),
        x2: Number(bbox[2]),
        y2: Number(bbox[3]),
        score: Math.max(0.72, Number(interaction?.pred_contact_prob ?? 0.72) || 0.72),
        label: heldLabel,
      });
    }
    if (!isHoldingMovable && isPlanningMovable && nearestObject?.bbox && Array.isArray(nearestObject.bbox) && nearestObject.bbox.length >= 4) {
      const bbox = nearestObject.bbox;
      const nearScore = Math.max(0.42, Math.min(0.76, 0.78 - distanceM * 0.85));
      attentionBlobs.push({
        id: `grab-near-${interaction?.hand_id}-${nearestObjectId || "candidate"}`,
        kind: "grab",
        x1: Number(bbox[0]),
        y1: Number(bbox[1]),
        x2: Number(bbox[2]),
        y2: Number(bbox[3]),
        score: Math.max(nearScore, Number.isFinite(predContact) ? predContact : 0),
        label: nearestLabel,
      });
    }
    for (const c of (isHoldingMovable ? [] : rankedGrab.slice(0, 1))) {
      const id = String(c?.object_id || "");
      const obj = objectById.get(id);
      const bbox = obj?.bbox;
      if (!Array.isArray(bbox) || bbox.length < 4) continue;
      const label = displayLabelForCandidate(c);
      if (!isMovableDisplayLabel(label)) continue;
      const area = Math.max(0, Number(bbox[2]) - Number(bbox[0])) * Math.max(0, Number(bbox[3]) - Number(bbox[1]));
      if (!Number.isFinite(area) || area > 0.18) continue;
      const rawScore = Number(c?.pred_contact_prob ?? 0);
      const latentConf = Number(c?.pred_future_latent_confidence ?? 0);
      const score = Number.isFinite(latentConf) && latentConf > 0
        ? 0.78 * rawScore + 0.22 * latentConf
        : rawScore;
      if (!Number.isFinite(score) || score < 0.35) continue;
      attentionBlobs.push({ id: `grab-${interaction?.hand_id}-${id}`, kind: "grab", x1: Number(bbox[0]), y1: Number(bbox[1]), x2: Number(bbox[2]), y2: Number(bbox[3]), score, label });
      if (Number.isFinite(hcx) && Number.isFinite(hcy)) {
        const cx = ((Number(bbox[0]) + Number(bbox[2])) * 0.5) * processedFrameSize.width;
        const cy = ((Number(bbox[1]) + Number(bbox[3])) * 0.5) * processedFrameSize.height;
        attentionLinks.push({ id: `grab-link-${interaction?.hand_id}-${id}`, fromX: hcx, fromY: hcy, toX: cx, toY: cy, kind: "grab", score });
      }
    }
    if (!isPlanningMovable && !isHoldingMovable) continue;
    const learnedHeldTarget = normalizeDisplayLabel(String(interaction?.learned_target_label || ""));
    const desiredPlaceLabel = (isTargetDisplayLabel(learnedHeldTarget) ? learnedHeldTarget : "")
      || learnedPlaceTargetForCue(interaction, topPredictedGrab)
      || desiredPlaceTargetForObject(isHoldingMovable ? heldLabel : plannedLabel, isHoldingMovable ? heldObject : plannedObject);
    let addedPlaceBlob = false;
    for (const c of (isHoldingMovable && isTargetDisplayLabel(desiredPlaceLabel) ? [] : rankedPlace.slice(0, 2))) {
      const id = String(c?.object_id || "");
      const obj = objectById.get(id);
      const rawCandidateLabel = normalizeDisplayLabel(String(c?.label || ""));
      const candidateLabel = displayLabelForCandidate(c);
      const candidateTargetLabel = targetAliasForPreference(candidateLabel, desiredPlaceLabel);
      const label = isTargetDisplayLabel(candidateTargetLabel) && !rawCandidateLabel.startsWith("unknown")
        ? candidateTargetLabel
        : desiredPlaceLabel;
      if (!isTargetDisplayLabel(label)) continue;
      if (isTargetDisplayLabel(desiredPlaceLabel) && label !== desiredPlaceLabel) continue;
      const bbox = targetBboxForObject(obj, label);
      if (!Array.isArray(bbox) || bbox.length < 4) continue;
      const area = Math.max(0, Number(bbox[2]) - Number(bbox[0])) * Math.max(0, Number(bbox[3]) - Number(bbox[1]));
      const maxTargetArea = label === "mat" ? 0.82 : 0.24;
      if (!Number.isFinite(area) || area > maxTargetArea) continue;
      const rawScore = Number(c?.pred_placement_prob ?? 0);
      const latentConf = Number(c?.pred_future_latent_confidence ?? 0);
      const score = Number.isFinite(latentConf) && latentConf > 0
        ? 0.78 * rawScore + 0.22 * latentConf
        : rawScore;
      if (!Number.isFinite(score) || score < 0.30) continue;
      attentionBlobs.push({ id: `place-${interaction?.hand_id}-${id}`, kind: "place", x1: Number(bbox[0]), y1: Number(bbox[1]), x2: Number(bbox[2]), y2: Number(bbox[3]), score, label });
      if (Number.isFinite(hcx) && Number.isFinite(hcy)) {
        const cx = ((Number(bbox[0]) + Number(bbox[2])) * 0.5) * processedFrameSize.width;
        const cy = ((Number(bbox[1]) + Number(bbox[3])) * 0.5) * processedFrameSize.height;
        attentionLinks.push({ id: `place-link-${interaction?.hand_id}-${id}`, fromX: hcx, fromY: hcy, toX: cx, toY: cy, kind: "place", score });
      }
      addedPlaceBlob = true;
    }
    if (!addedPlaceBlob && isTargetDisplayLabel(desiredPlaceLabel)) {
      const fallbackTarget = targetFallbackObject(desiredPlaceLabel);
      const bbox = targetBboxForObject(fallbackTarget, desiredPlaceLabel);
      if (Array.isArray(bbox) && bbox.length >= 4) {
        attentionBlobs.push({
          id: `place-fallback-${interaction?.hand_id}-${desiredPlaceLabel}`,
          kind: "place",
          x1: Number(bbox[0]),
          y1: Number(bbox[1]),
          x2: Number(bbox[2]),
          y2: Number(bbox[3]),
          score: isHoldingMovable ? 0.92 : 0.74,
          label: desiredPlaceLabel,
        });
        if (Number.isFinite(hcx) && Number.isFinite(hcy)) {
          const cx = ((Number(bbox[0]) + Number(bbox[2])) * 0.5) * processedFrameSize.width;
          const cy = ((Number(bbox[1]) + Number(bbox[3])) * 0.5) * processedFrameSize.height;
          attentionLinks.push({ id: `place-link-fallback-${interaction?.hand_id}-${desiredPlaceLabel}`, fromX: hcx, fromY: hcy, toX: cx, toY: cy, kind: "place", score: isHoldingMovable ? 0.92 : 0.74 });
        }
      }
    }
  }
  const seenAttentionKeys = new Set<string>();
  const attentionKindCounts: Record<"grab" | "place", number> = { grab: 0, place: 0 };
  const planningAttentionBlobs = [...attentionBlobs]
    .sort((a, b) => b.score - a.score)
    .filter((item) => {
      const key = `${item.kind}:${item.id}`;
      if (seenAttentionKeys.has(key)) return false;
      const limit = item.kind === "grab" ? 1 : 2;
      if (attentionKindCounts[item.kind] >= limit) return false;
      seenAttentionKeys.add(key);
      attentionKindCounts[item.kind] += 1;
      return true;
    });
  const planningFreshEnough = worldStateFreshEnough
    || (displayedFrameAgeMs != null && displayedFrameAgeMs <= WORLD_STATE_ACTIONABLE_MAX_MS);
  const sceneTimelineCue = SCENE_TIMELINE_CUE_ENABLED && isEmbodiedMode && embodiedVideoSource === "scene" && sceneVideoFile === "scene_sophie.mp4"
    ? SOPHIE_SCENE_TIMELINE.find((event) => sceneVideoTimeS >= event.grabStartS - 1.2 && sceneVideoTimeS <= event.releaseS + 0.8)
    : null;
  const sceneTimelineAttentionBlobs = sceneTimelineCue
    ? [
        {
          id: `scene-cue-grab-${sceneTimelineCue.id}`,
          kind: "grab" as const,
          label: sceneTimelineCue.object,
          score: sceneVideoTimeS < sceneTimelineCue.grabStartS ? 0.78 : 0.94,
          x1: SOPHIE_SCENE_REGION_BOXES[sceneTimelineCue.source][0],
          y1: SOPHIE_SCENE_REGION_BOXES[sceneTimelineCue.source][1],
          x2: SOPHIE_SCENE_REGION_BOXES[sceneTimelineCue.source][2],
          y2: SOPHIE_SCENE_REGION_BOXES[sceneTimelineCue.source][3],
        },
        {
          id: `scene-cue-place-${sceneTimelineCue.id}`,
          kind: "place" as const,
          label: sceneTimelineCue.target,
          score: 0.96,
          x1: SOPHIE_SCENE_REGION_BOXES[sceneTimelineCue.target][0],
          y1: SOPHIE_SCENE_REGION_BOXES[sceneTimelineCue.target][1],
          x2: SOPHIE_SCENE_REGION_BOXES[sceneTimelineCue.target][2],
          y2: SOPHIE_SCENE_REGION_BOXES[sceneTimelineCue.target][3],
        },
      ]
    : [];
  const visiblePlanningAttentionBlobs = planningFreshEnough
    ? (planningAttentionBlobs.length > 0 ? planningAttentionBlobs : sceneTimelineAttentionBlobs)
    : sceneTimelineAttentionBlobs;
  const attentionHasReadableTarget = visiblePlanningAttentionBlobs.some((item) => item.kind === "place" && isTargetDisplayLabel(item.label));
  const attentionHasReadableGrab = visiblePlanningAttentionBlobs.some((item) => item.kind === "grab" && isMovableDisplayLabel(item.label));
  useEffect(() => {
    if (!attentionHasReadableTarget || visiblePlanningAttentionBlobs.length === 0) return;
    setLatchedAttentionBlobs(visiblePlanningAttentionBlobs);
    setLatchedAttentionAtMs(uiNowMs);
  }, [
    attentionHasReadableTarget,
    attentionHasReadableGrab,
    uiNowMs,
    visiblePlanningAttentionBlobs.map((item) => `${item.kind}:${item.label}:${Math.round(item.score * 100)}:${item.x1}:${item.y1}:${item.x2}:${item.y2}`).join("|"),
  ]);
  const latchedAttentionFresh = latchedAttentionBlobs.length > 0
    && uiNowMs - latchedAttentionAtMs <= INTENT_HUD_LATCH_MS;
  const displayedPlanningAttentionBlobs = visiblePlanningAttentionBlobs.length > 0
    ? visiblePlanningAttentionBlobs
    : (latchedAttentionFresh ? latchedAttentionBlobs : visiblePlanningAttentionBlobs);
  const primaryAttention = displayedPlanningAttentionBlobs[0] || null;
  const primaryGrab = displayedPlanningAttentionBlobs.find((item) => item.kind === "grab") || null;
  const primaryPlace = displayedPlanningAttentionBlobs.find((item) => item.kind === "place") || null;
  const activeHeldInteraction = handInteractionsData.find((item: any) => {
    if (!Boolean(item?.learned_is_held || item?.is_contacting || item?.is_touching_strict)) return false;
    const label = heldLabelForInteraction(item);
    return isMovableDisplayLabel(label);
  }) || null;
  const bestPredictedGrabCue = handInteractionsData
    .flatMap((item: any) => (
      Array.isArray(item?.intent_grab_candidates)
        ? item.intent_grab_candidates.map((candidate: any) => ({ interaction: item, candidate }))
        : []
    ))
    .map(({ interaction, candidate }: any) => {
      const label = displayLabelForCandidate(candidate);
      const score = Number(candidate?.pred_contact_prob ?? 0);
      const id = String(candidate?.object_id || "");
      const obj = id ? objectById.get(id) : null;
      return {
        interaction,
        candidate,
        id,
        obj,
        label,
        score: Number.isFinite(score) ? score : 0,
      };
    })
    .filter((item: any) => (
      item.score >= 0.35
      && isMovableDisplayLabel(item.label)
    ))
    .sort((a: any, b: any) => b.score - a.score)[0] || null;
  const latestManipulationEvent = manipulationEventsData[manipulationEventsData.length - 1] || null;
  const latestManipulationEventAgeS = latestManipulationEvent?.time
    ? Math.max(0, (uiNowMs / 1000) - Number(latestManipulationEvent.time))
    : null;
  const latestEventLabel = latestManipulationEvent
    ? displayLabelForObject({
        id: latestManipulationEvent.object_id,
        label: latestManipulationEvent.label || latestManipulationEvent.object_label || "object",
      })
    : "object";
  const activeEvent = latestManipulationEventAgeS !== null
    && latestManipulationEventAgeS <= 2.8
    && isMovableDisplayLabel(latestEventLabel)
    && RELEASE_UI_ENABLED
    ? latestManipulationEvent
    : null;
  const hasHandFocus = handsData.length > 0 || handInteractionsData.length > 0;
  const sceneTimelineActive = Boolean(
    sceneTimelineCue &&
    (!planningFreshEnough || (!activeHeldInteraction && !bestPredictedGrabCue))
  );
  const activeHeldIsReleasing = Boolean(activeHeldInteraction?.learned_releasing) || Number(activeHeldInteraction?.pred_release_prob ?? 0) >= 0.55;
  const intentPhase = activeHeldInteraction
    ? (activeHeldIsReleasing ? "RELEASING" : "GRABBED")
    : (activeEvent
      ? "RELEASED"
      : (bestPredictedGrabCue
        ? "PREDICTING"
        : (sceneTimelineActive
          ? (sceneVideoTimeS >= (sceneTimelineCue?.grabStartS ?? 0) ? "GRABBED" : "PREDICTING")
          : (hasHandFocus ? "TRACKING" : "SCANNING"))));
  const intentScore = primaryAttention ? Math.round(Math.max(0, Math.min(1, primaryAttention.score)) * 100) : 0;
  const predictedTargetLabel = bestPredictedGrabCue
    ? (learnedPlaceTargetForCue(bestPredictedGrabCue.candidate, bestPredictedGrabCue.interaction)
      || desiredPlaceTargetForObject(bestPredictedGrabCue.label, bestPredictedGrabCue.obj))
    : "";
  const activeHeldLabel = activeHeldInteraction
    ? heldLabelForInteraction(activeHeldInteraction, primaryGrab?.label || "object")
    : "";
  const activeHeldObject = activeHeldInteraction ? heldObjectForInteraction(activeHeldInteraction) : null;
  const activeLearnedTargetLabel = normalizeDisplayLabel(String(activeHeldInteraction?.learned_target_label || ""));
  const placePredictionLabel = activeHeldInteraction
    ? ((isTargetDisplayLabel(activeLearnedTargetLabel) ? activeLearnedTargetLabel : "") || learnedPlaceTargetForCue(activeHeldInteraction) || primaryPlace?.label || desiredPlaceTargetForObject(activeHeldLabel, activeHeldObject))
    : "none";
  const sceneTimelineCaption = sceneTimelineCue
    ? `${sceneTimelineCue.object} -> ${sceneTimelineCue.target}`
    : "";
  const intentCaption = activeHeldInteraction
    ? `${activeHeldIsReleasing ? "release" : "held"}: ${activeHeldLabel || primaryGrab?.label || "object"} -> ${placePredictionLabel || "target"}`
    : (bestPredictedGrabCue
      ? `likely: ${bestPredictedGrabCue.label} -> ${predictedTargetLabel || "target"}`
      : (sceneTimelineActive && sceneTimelineCaption
        ? sceneTimelineCaption
        : (activeEvent ? `${latestEventLabel}: released` : (hasHandFocus ? "hand" : "static scene"))));
  const activeEventTargetLabel = normalizeDisplayLabel(String(
    activeEvent?.target_label
      || activeEvent?.place_relation?.nearest_object_label
      || activeEvent?.place_relation?.target_label
      || activeEvent?.place_relation?.support_target_label
      || activeEvent?.target
      || ""
  ));

  const liveIntentCue = activeHeldInteraction
    ? {
        phase: intentPhase,
        caption: intentCaption,
        objectLabel: activeHeldLabel || primaryGrab?.label || "object",
        targetLabel: placePredictionLabel || "target",
        objectId: heldObjectIdForInteraction(activeHeldInteraction),
        eventState: activeHeldInteraction.learned_event_state || (activeHeldIsReleasing ? "releasing" : "held"),
        updatedAtMs: uiNowMs,
        score: intentScore,
      }
    : (bestPredictedGrabCue
      ? {
          phase: intentPhase,
          caption: intentCaption,
          objectLabel: bestPredictedGrabCue.label,
          targetLabel: predictedTargetLabel || "target",
          objectId: String(bestPredictedGrabCue.id || ""),
          eventState: "predicting",
          updatedAtMs: uiNowMs,
          score: intentScore,
        }
      : (activeEvent
        ? {
            phase: intentPhase,
            caption: intentCaption,
            objectLabel: latestEventLabel,
            targetLabel: activeEventTargetLabel,
            objectId: String(activeEvent?.object_id || ""),
            eventState: "released",
            updatedAtMs: uiNowMs,
            score: intentScore,
          }
        : null));
  const liveIntentHasObject = Boolean(liveIntentCue && isMovableDisplayLabel(liveIntentCue.objectLabel));
  const liveIntentHasTarget = Boolean(liveIntentCue && (liveIntentCue.eventState === "released" || isTargetDisplayLabel(liveIntentCue.targetLabel)));
  const liveIntentReadable = Boolean(liveIntentCue && liveIntentHasObject && liveIntentHasTarget);

  useEffect(() => {
    if (!liveIntentReadable || !liveIntentCue) return;
    setLatchedIntentCue((prev: any | null) => {
      const key = `${liveIntentCue.eventState}:${liveIntentCue.objectId}:${liveIntentCue.objectLabel}:${liveIntentCue.targetLabel}`;
      const prevKey = prev ? `${prev.eventState}:${prev.objectId}:${prev.objectLabel}:${prev.targetLabel}` : "";
      if (key === prevKey && uiNowMs - Number(prev?.updatedAtMs ?? 0) < 900) return prev;
      return { ...liveIntentCue, updatedAtMs: uiNowMs };
    });
  }, [
    liveIntentReadable,
    liveIntentCue?.phase,
    liveIntentCue?.caption,
    liveIntentCue?.objectId,
    liveIntentCue?.objectLabel,
    liveIntentCue?.targetLabel,
    liveIntentCue?.eventState,
    uiNowMs,
  ]);

  useEffect(() => {
    if (!liveIntentReadable || !liveIntentCue) return;
    if (!useWebSpeechDebug || !isEmbodiedMode) return;
    const state = String(liveIntentCue.eventState || "");
    if (!["predicting", "held", "releasing", "released"].includes(state)) return;
    const objectLabel = normalizeDisplayLabel(String(liveIntentCue.objectLabel || ""));
    const targetLabel = normalizeDisplayLabel(String(liveIntentCue.targetLabel || ""));
    if (!isMovableDisplayLabel(objectLabel) || (state !== "released" && !isTargetDisplayLabel(targetLabel))) return;
    const signature = `${state}:${liveIntentCue.objectId}:${objectLabel}:${targetLabel}`;
    const last = lastDirectWorldCueRef.current;
    const now = Date.now();
    if (signature === last.signature && now - last.timeMs < 6500) return;
    if (now - last.timeMs < 1400) return;
    const phrase = state === "releasing"
      ? `${objectLabel}: releasing near ${targetLabel}.`
      : (state === "released"
        ? (isTargetDisplayLabel(targetLabel) ? `${objectLabel}: released near ${targetLabel}.` : `${objectLabel}: released.`)
        : (state === "predicting"
          ? `Likely ${objectLabel}. Target: ${targetLabel}.`
          : `${objectLabel}: held. Target: ${targetLabel}.`));
    maybeSpeakWorldModelExplanation(phrase);
    lastDirectWorldCueRef.current = { signature, timeMs: now };
  }, [
    liveIntentReadable,
    liveIntentCue?.eventState,
    liveIntentCue?.objectId,
    liveIntentCue?.objectLabel,
    liveIntentCue?.targetLabel,
    useWebSpeechDebug,
    isEmbodiedMode,
  ]);

  const latchedIntentFresh = latchedIntentCue !== null
    && uiNowMs - Number(latchedIntentCue.updatedAtMs ?? 0) <= INTENT_HUD_LATCH_MS;
  const displayedIntentPhase = liveIntentReadable
    ? intentPhase
    : (latchedIntentFresh ? String(latchedIntentCue.phase || "GRABBED") : intentPhase);
  const displayedIntentCaption = liveIntentReadable
    ? intentCaption
    : (latchedIntentFresh ? String(latchedIntentCue.caption || intentCaption) : intentCaption);

  const grabPredictionScore = primaryGrab ? Math.round(Math.max(0, Math.min(1, primaryGrab.score)) * 100) : 0;
  const placePredictionScore = activeHeldInteraction && primaryPlace ? Math.round(Math.max(0, Math.min(1, primaryPlace.score)) * 100) : 0;
  const grabPredictionLabel = primaryGrab?.label || "none";

  const socialState = {
    inputTranscript,
    outputTranscript,
    directAudioMonitor,
    useAvatarSpeech,
    useWebSpeechDebug,
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
    handInteractionsText,
    manipulationEventsText,
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

  const toggleCardExpanded = (key: string) => {
    setCardExpanded((prev) => ({ ...prev, [key]: !Boolean(prev[key]) }));
  };

  const isCardExpanded = (key: string, fallback = true) => {
    if (Object.prototype.hasOwnProperty.call(cardExpanded, key)) return Boolean(cardExpanded[key]);
    return fallback;
  };

  function resetWorldUiState() {
    setWorldStateText("");
    setEventLog("");
    setLastQueryResultText("");
    setPlannerSummary("");
    setPlannerSimulations(null);
    setBestActionName("");
    setCameraPoseText("");
    setCameraPoseDataState(null);
    setObjects3dText("");
    setObjects3dDataState([]);
    setSparseMapText("");
    setSparseMapData([]);
    setProcessedFrameUrl("");
    setProcessedFrameTimestamp(null);
    setProcessedFrameLandmarks([]);
    setProcessedFrameSize({ width: 640, height: 360 });
    setDepthDebugSize({ width: 160, height: 90 });
    setHandsText("");
    setHandInteractionsText("");
    setHandInteractionsDataState([]);
    setManipulationEventsText("");
    setManipulationEventsDataState([]);
    setHandsDataState([]);
    setHandTrajectoriesDataState([]);
    setCameraTrailState([]);
    setWorldDebugText("");
    setWorldDebugDataState({});
    setDepthDebugUrl("");
    setRefinedLabelsById({});
    setRefinedLabelsByHint(DEBUG_LABEL_HINTS);
    try {
      window.localStorage.removeItem("embodied_refined_labels_by_id");
      window.localStorage.removeItem("embodied_refined_labels_by_hint");
    } catch {}
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
    setWorldStateReceivedAtMs(0);
    setLatchedIntentCue(null);
    setLatchedAttentionBlobs([]);
    setLatchedAttentionAtMs(0);
    lastSpokenRef.current = "";
    lastSpeakTimeRef.current = 0;
    lastDirectWorldCueRef.current = { signature: "", timeMs: 0 };
    pendingAvatarCueRef.current = null;
    startupCueSpokenRef.current = false;
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

        if (isEmbodiedMode || avatarMode === "ai" || useAvatarSpeech) {
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
            avatarAudioChunksRef.current = [];
            avatarAudioBytesRef.current = 0;
            avatarAudioFirstChunkRef.current = true;
            setStatus(avatarMode === "ai" ? "Avatar finished speaking" : "Avatar finished direct speech");
          }

          if (msg.type === "agent.speak_interrupted") {
            setIsSpeaking(false);
            avatarTurnStartedRef.current = false;
            currentAvatarEventIdRef.current = null;
            avatarAudioChunksRef.current = [];
            avatarAudioBytesRef.current = 0;
            avatarAudioFirstChunkRef.current = true;
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

      connectGeminiBridge();

      await startLocalCamera();
      startSendingFrames();

      setStatus("World model running (no avatar)");
    } catch (err) {
      console.error("World model startup failed:", err);
      setStatus("World model error");
    }
  }

  function maybeRequestObjectLabels(objects: any[], worldDebug?: any) {
    if (!isEmbodiedMode) return;
    if (!bridgeConnected) return;
    if (!geminiBridgeRef.current || geminiBridgeRef.current.readyState !== WebSocket.OPEN) return;
    if (labelReqInFlightRef.current) return;
    if (labelReqCooldownRef.current) return;

    const video = localCamRef.current;
    if (!video || video.readyState < 2) return;

    const w = video.videoWidth || 0;
    const h = video.videoHeight || 0;
    if (w < 16 || h < 16) return;

    if (!labelCropCanvasRef.current) {
      labelCropCanvasRef.current = document.createElement("canvas");
    }
    const canvas = labelCropCanvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const staticTargets = Array.isArray(worldDebug?.static_targets) ? worldDebug.static_targets : [];
    const labelObjects = [
      ...staticTargets.map((obj: any) => ({ ...obj, label_source: "static_target" })),
      ...(Array.isArray(objects) ? objects.map((obj: any) => ({ ...obj, label_source: "object" })) : []),
    ];
    const candidates = labelObjects
      .map((obj: any) => {
        const id = String(obj?.id || "");
        const bbox = Array.isArray(obj?.projected_bbox) ? obj.projected_bbox : obj?.bbox;
        if (!id || !Array.isArray(bbox) || bbox.length < 4) return null;
        const rawHint = normalizeDisplayLabel(String(obj?.label || "unknown"));
        if (["dining table", "table", "person", "hand"].includes(rawHint)) return null;
        const isStatic = String(obj?.label_source || "") === "static_target" || id.startsWith("target_");
        const hint = isStatic && (rawHint.startsWith("unknown") || rawHint === "object")
          ? "static placement target: mat or tray"
          : rawHint;
        const already = refinedLabelsById[id];
        const needRelabel = !already || already.confidence < 0.86 || rawHint.startsWith("unknown");
        if (!needRelabel) return null;
        const area = Math.max(0, Number(bbox[2]) - Number(bbox[0])) * Math.max(0, Number(bbox[3]) - Number(bbox[1]));
        const priority = (isStatic ? 100 : 0) + (rawHint.startsWith("unknown") ? 40 : 0) + Math.min(20, area * 100);
        return { id, bbox, hint, priority };
      })
      .filter(Boolean)
      .sort((a: any, b: any) => Number(b.priority || 0) - Number(a.priority || 0))
      .slice(0, 8) as Array<{ id: string; bbox: number[]; hint: string; priority: number }>;

    if (candidates.length === 0) return;

    const payloadObjects: Array<{ id: string; label_hint: string; mime_type: string; image_base64: string }> = [];
    for (const c of candidates) {
      const bx1 = Math.max(0, Math.min(1, Number(c.bbox[0])));
      const by1 = Math.max(0, Math.min(1, Number(c.bbox[1])));
      const bx2 = Math.max(0, Math.min(1, Number(c.bbox[2])));
      const by2 = Math.max(0, Math.min(1, Number(c.bbox[3])));
      const bwNorm = Math.max(0, bx2 - bx1);
      const bhNorm = Math.max(0, by2 - by1);
      const pad = Math.max(0.018, Math.min(0.08, Math.max(bwNorm, bhNorm) * 0.18));
      const x1 = Math.max(0, bx1 - pad) * w;
      const y1 = Math.max(0, by1 - pad) * h;
      const x2 = Math.min(1, bx2 + pad) * w;
      const y2 = Math.min(1, by2 + pad) * h;
      const cw = Math.max(8, Math.floor(x2 - x1));
      const ch = Math.max(8, Math.floor(y2 - y1));
      if (cw < 8 || ch < 8) continue;
      canvas.width = cw;
      canvas.height = ch;
      ctx.drawImage(video, Math.floor(x1), Math.floor(y1), cw, ch, 0, 0, cw, ch);
      const dataUrl = canvas.toDataURL("image/jpeg", 0.8);
      const b64 = String(dataUrl.split(",")[1] || "");
      if (!b64) continue;
      payloadObjects.push({ id: c.id, label_hint: c.hint, mime_type: "image/jpeg", image_base64: b64 });
    }

    if (payloadObjects.length === 0) return;

    labelReqSeqRef.current += 1;
    const requestId = `label_${labelReqSeqRef.current}`;
    labelReqInFlightRef.current = true;
    labelReqCooldownRef.current = true;
    if (labelReqCooldownTimerRef.current !== null) {
      window.clearTimeout(labelReqCooldownTimerRef.current);
    }
    labelReqCooldownTimerRef.current = window.setTimeout(() => {
      labelReqCooldownRef.current = false;
      labelReqCooldownTimerRef.current = null;
    }, 2500);
    geminiBridgeRef.current.send(
      JSON.stringify({
        type: "world_label_request",
        request_id: requestId,
        objects: payloadObjects,
      })
    );
  }

  function connectGeminiBridge() {
    if (geminiBridgeRef.current && geminiBridgeRef.current.readyState === WebSocket.OPEN) {
      return;
    }

    geminiBridgeRef.current?.close();
    const ws = new WebSocket("ws://localhost:8081");

    ws.onopen = () => {
      setBridgeConnected(true);
      geminiReadyRef.current = false;
    };

    ws.onmessage = async (event) => {
      const msg = JSON.parse(event.data);

      if (msg.type === "gemini_ready") {
        geminiReadyRef.current = true;
        const pendingCue = pendingAvatarCueRef.current;
        if (pendingCue) {
          pendingAvatarCueRef.current = null;
          maybeSpeakWorldModelExplanation(pendingCue, { forceAvatar: true });
        }
        return;
      }

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
        avatarSpeechRequestSeqRef.current += 1;
        const audio = String(msg.data || "");
        if (!audio) return;
        avatarAudioChunksRef.current.push(audio);
        avatarAudioBytesRef.current += base64ByteLength(audio);
        const thresholdBytes = avatarAudioFirstChunkRef.current
          ? Math.round(0.6 * 24000 * 2)
          : Math.round(1.0 * 24000 * 2);
        if (avatarAudioBytesRef.current >= thresholdBytes) {
          flushAvatarAudioChunk(false);
        }
      }

      if (msg.type === "turn_complete") {
        flushAvatarAudioChunk(true);

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
        avatarAudioChunksRef.current = [];
        avatarAudioBytesRef.current = 0;
        avatarAudioFirstChunkRef.current = true;
      }

      if (msg.type === "world_label_result") {
        labelReqInFlightRef.current = false;
        const labels = (Array.isArray(msg.labels) ? msg.labels : []).filter((item: any) => {
          const label = normalizeDisplayLabel(String(item?.label || ""));
          return Boolean(label) && sanitizeSceneRefinedLabels({ candidate: { label, confidence: Number(item?.confidence ?? 0) } }).candidate;
        });
        if (labels.length > 0) {
          if (worldModelWsRef.current && worldModelWsRef.current.readyState === WebSocket.OPEN) {
            worldModelWsRef.current.send(JSON.stringify({
              type: "query",
              query: "apply_label_refinements",
              labels,
            }));
            const sourceFrameTs = Number(processedFrameTimestamp);
            if (Number.isFinite(sourceFrameTs) && Date.now() - sourceFrameTs <= GUIDANCE_FRESH_MAX_MS) {
              worldModelWsRef.current.send(JSON.stringify({
                type: "query",
                query: "interaction_guidance",
                source: "label_refinement",
                frame_timestamp: sourceFrameTs,
              }));
            }
          }
          setRefinedLabelsById((prev: Record<string, { label: string; confidence: number }>) => {
            const next = { ...prev };
            for (const item of labels) {
              const id = String(item?.id || "");
              const label = normalizeDisplayLabel(String(item?.label || ""));
              const confidence = Number(item?.confidence ?? 0);
              if (!id || !label) continue;
              next[id] = { label, confidence: Number.isFinite(confidence) ? confidence : 0 };
            }
            try {
              window.localStorage.setItem("embodied_refined_labels_by_id", JSON.stringify(next));
            } catch {}
            return next;
          });
          setRefinedLabelsByHint((prev: Record<string, { label: string; confidence: number }>) => {
            const next = { ...prev };
            for (const item of labels) {
              const id = String(item?.id || "");
              const label = normalizeDisplayLabel(String(item?.label || ""));
              const confidence = Number(item?.confidence ?? 0);
              if (!id || !label) continue;
              const obj = observedObjects.find((candidate: any) => String(candidate?.id || "") === id);
              const key = labelKeyForObject(obj || { label: String(item?.label_hint || "") });
              if (!key || key === "unknown") continue;
              const conf = Number.isFinite(confidence) ? confidence : 0;
              if (!next[key] || conf >= next[key].confidence) {
                next[key] = { label, confidence: conf };
              }
            }
            try {
              window.localStorage.setItem("embodied_refined_labels_by_hint", JSON.stringify(next));
            } catch {}
            return next;
          });
        }
      }

      if (msg.type === "world_model_explanation_forwarded" && showDebug) {
        setEventLog((prev) =>
          [
            `[BRIDGE SPEECH FORWARDED]`,
            JSON.stringify({ text: msg.text, prompt: msg.prompt }, null, 2),
            prev,
          ]
            .filter(Boolean)
            .join("\n\n")
            .slice(0, 4000)
        );
      }

      if (msg.type === "world_model_explanation_skipped") {
        const reason = String(msg.reason || "");
        const text = String(msg.text || "");
        if (text && (reason === "gemini-not-ready" || reason === "speech-busy")) {
          setStatus(`Avatar speech skipped (${reason})`);
        }
      }

      if (msg.type === "gemini_error") {
        setStatus(`Gemini error: ${msg.error}`);
      }
    };

    ws.onerror = () => {
      setBridgeConnected(false);
      geminiReadyRef.current = false;
      setStatus("Gemini bridge error");
    };

    ws.onclose = () => {
      setBridgeConnected(false);
      geminiReadyRef.current = false;
    };

    geminiBridgeRef.current = ws;
  }

  function flushAvatarAudioChunk(final = false) {
    if (!avatarWsRef.current || avatarWsRef.current.readyState !== WebSocket.OPEN) {
      avatarAudioChunksRef.current = [];
      avatarAudioBytesRef.current = 0;
      avatarAudioFirstChunkRef.current = true;
      return;
    }
    if (!avatarAudioChunksRef.current.length) {
      if (final && currentAvatarEventIdRef.current) {
        avatarWsRef.current.send(
          JSON.stringify({
            type: "agent.speak_end",
            event_id: currentAvatarEventIdRef.current,
          })
        );
        avatarWsRef.current.send(
          JSON.stringify({
            type: "agent.start_listening",
            event_id: randomEventId(),
          })
        );
        avatarTurnStartedRef.current = false;
        currentAvatarEventIdRef.current = null;
        avatarAudioFirstChunkRef.current = true;
      }
      return;
    }

    const combinedBinary = avatarAudioChunksRef.current.map((chunk) => atob(chunk)).join("");
    const combinedBytes = new Uint8Array(combinedBinary.length);
    for (let i = 0; i < combinedBinary.length; i += 1) {
      combinedBytes[i] = combinedBinary.charCodeAt(i);
    }
    const audio = bytesToBase64(combinedBytes);
    avatarAudioChunksRef.current = [];
    avatarAudioBytesRef.current = 0;

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
        audio,
      })
    );
    avatarAudioFirstChunkRef.current = false;

    if (final && currentAvatarEventIdRef.current) {
      avatarWsRef.current.send(
        JSON.stringify({
          type: "agent.speak_end",
          event_id: currentAvatarEventIdRef.current,
        })
      );
      avatarWsRef.current.send(
        JSON.stringify({
          type: "agent.start_listening",
          event_id: randomEventId(),
        })
      );
      avatarTurnStartedRef.current = false;
      currentAvatarEventIdRef.current = null;
      avatarAudioFirstChunkRef.current = true;
    }
  }

  function connectWorldModel() {
    worldModelWsRef.current?.close();
    worldFrameInFlightRef.current = false;
    worldFrameInFlightSinceRef.current = 0;
    pendingWorldFrameRef.current = null;
    queuedWorldFrameRef.current = null;
    sentWorldFramesRef.current.clear();
    setWorldModelConnected(false);
    const ws = new WebSocket("ws://localhost:8090");

    ws.onopen = () => {
      setWorldModelConnected(true);
      ws.send(JSON.stringify({ type: "query", query: "reset_world_model" }));
      setEventLog((prev) =>
        [`[WORLD MODEL CONNECTED]`, prev].filter(Boolean).join("\n\n").slice(0, 4000)
      );
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);

        if (msg.type === "state_updated") {
          const receivedAtMs = Date.now();
          setWorldStateReceivedAtMs(receivedAtMs);
          worldFrameInFlightRef.current = false;
          worldFrameInFlightSinceRef.current = 0;
          const responseFrameTs = Number(msg.frame_timestamp);
          const responseAgeMs = Number.isFinite(responseFrameTs) ? Math.max(0, receivedAtMs - responseFrameTs) : null;
          const stateRenderable = true;
          const objects = msg.objects || [];
          const cameraPose = msg.camera_pose || null;
          const objects3d = msg.objects_3d || [];
          const sparseMap = msg.sparse_map || [];
          const hands = stateRenderable ? (msg.hands || []) : [];
          const handInteractions = stateRenderable ? (msg.hand_object_interactions || []) : [];
          const manipulationEvents = stateRenderable ? (msg.manipulation_events || []) : [];
          const handTrajectories = msg.hand_trajectories || [];
          const worldDebug = msg.world_debug || {};
          const depthDebug = msg.depth_debug || null;
          const now = performance.now();
          setObservedObjects(objects);
          maybeRequestObjectLabels(objects, worldDebug);
          const shouldUpdateDebugText = now - lastDebugTextUpdateRef.current > 500;
          if (shouldUpdateDebugText) {
            setWorldStateText(JSON.stringify({ objects }, null, 2));
            setCameraPoseText(JSON.stringify(cameraPose, null, 2));
            setObjects3dText(JSON.stringify(objects3d, null, 2));
            setSparseMapText(JSON.stringify(sparseMap, null, 2));
            setWorldDebugText(JSON.stringify(worldDebug, null, 2));
            setHandsText(JSON.stringify(hands, null, 2));
            setHandInteractionsText(JSON.stringify(handInteractions, null, 2));
            setManipulationEventsText(JSON.stringify(manipulationEvents, null, 2));
            lastDebugTextUpdateRef.current = now;
          }
          setHandsDataState(Array.isArray(hands) ? hands : []);
          setObjects3dDataState(Array.isArray(objects3d) ? objects3d : []);
          setHandInteractionsDataState(Array.isArray(handInteractions) ? handInteractions : []);
          setManipulationEventsDataState(Array.isArray(manipulationEvents) ? manipulationEvents : []);
          setHandTrajectoriesDataState(Array.isArray(handTrajectories) ? handTrajectories : []);
          setWorldDebugDataState(worldDebug && typeof worldDebug === "object" ? worldDebug : {});
          setCameraPoseDataState(cameraPose);
          if (Array.isArray(cameraPose?.camera_position_world) && cameraPose.camera_position_world.length >= 3) {
            const cx = Number(cameraPose.camera_position_world[0]);
            const cy = Number(cameraPose.camera_position_world[1]);
            const cz = Number(cameraPose.camera_position_world[2]);
            if ([cx, cy, cz].every(Number.isFinite)) {
              setCameraTrailState((prev) => {
                const next = [...prev, [cx, cy, cz]];
                return next.length > 180 ? next.slice(next.length - 180) : next;
              });
            }
          }
          setSparseMapData(Array.isArray(sparseMap) ? sparseMap : []);
          const sentFrame = Number.isFinite(responseFrameTs)
            ? sentWorldFramesRef.current.get(responseFrameTs)
            : null;
          if (sentFrame) {
            setProcessedFrameUrl(sentFrame.image);
            setProcessedFrameTimestamp(sentFrame.timestamp);
            setProcessedFrameLandmarks(Array.isArray(sparseMap) ? sparseMap : []);
            setProcessedFrameSize({
              width: sentFrame.width,
              height: sentFrame.height,
            });
            sentWorldFramesRef.current.delete(sentFrame.timestamp);
            if (
              pendingWorldFrameRef.current &&
              pendingWorldFrameRef.current.timestamp === sentFrame.timestamp
            ) {
              pendingWorldFrameRef.current = null;
            }
          } else if (Number.isFinite(responseFrameTs)) {
            setProcessedFrameTimestamp(responseFrameTs);
            setProcessedFrameLandmarks(Array.isArray(sparseMap) ? sparseMap : []);
          }
          if (sentWorldFramesRef.current.size > 24) {
            const keys = Array.from(sentWorldFramesRef.current.keys()).sort((a, b) => a - b);
            for (const key of keys.slice(0, Math.max(0, keys.length - 24))) {
              sentWorldFramesRef.current.delete(key);
            }
          }
          if (typeof msg.frame_width === "number" && typeof msg.frame_height === "number") {
            setProcessedFrameSize({
              width: Math.max(1, Math.round(msg.frame_width)),
              height: Math.max(1, Math.round(msg.frame_height)),
            });
          }
          setDepthDebugUrl(depthDebug?.image ? `data:${depthDebug.mime_type};base64,${depthDebug.image}` : "");
          if (depthDebug?.width && depthDebug?.height) {
            setDepthDebugSize({ width: depthDebug.width, height: depthDebug.height });
          }

          if (typeof msg.frame_timestamp === "number") {
            const age = responseAgeMs ?? (receivedAtMs - msg.frame_timestamp);
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

          if ((autoModeRef.current || useAvatarSpeechRef.current || useWebSpeechDebug) && ws.readyState === WebSocket.OPEN) {
            const latestEvent = Array.isArray(manipulationEvents) && manipulationEvents.length > 0
              ? manipulationEvents[manipulationEvents.length - 1]
              : null;
            const interactionSig = Array.isArray(handInteractions)
              ? handInteractions
                  .slice(0, 3)
                  .map((item: any) => [
                    item?.hand_id,
                    item?.held_object_id || item?.learned_object_id || item?.nearest_object_id,
                    normalizeDisplayLabel(String(item?.held_object_label || item?.nearest_object_label || "")),
                    normalizeDisplayLabel(String(item?.pred_target_label || "")),
                    item?.learned_event_state || (item?.is_contacting ? "contact" : (item?.is_near ? "near" : "track")),
                    Math.round(Number(item?.pred_contact_prob ?? 0) * 100),
                    Math.round(Number(item?.pred_release_prob ?? 0) * 100),
                    normalizeDisplayLabel(String(item?.learned_target_label || "")),
                    Math.round(Number(item?.distance_m ?? 9) * 100),
                  ].join(":"))
                  .join("|")
              : "";
            const handSig = Array.isArray(hands) && hands.length > 0
              ? hands
                  .slice(0, 2)
                  .map((hand: any) => [
                    hand?.id || hand?.side || "hand",
                    Math.round(Number(hand?.confidence ?? 0) * 100),
                    hand?.predicted ? "pred" : "seen",
                  ].join(":"))
                  .join("|")
              : "";
            const eventSig = latestEvent
              ? `${latestEvent?.event || ""}:${latestEvent?.object_id || ""}:${latestEvent?.time || ""}`
              : "";
            const signature = `${handSig}::${interactionSig}::${eventSig}`;
            const nowMs = receivedAtMs;
            const sourceFrameTs = Number(msg.frame_timestamp);
            const sourceFrameAgeMs = Number.isFinite(sourceFrameTs) ? Math.max(0, nowMs - sourceFrameTs) : null;
            const hasActionSignal = Boolean(handSig || interactionSig || eventSig);
            const stateActionable = hasActionSignal || (sourceFrameAgeMs !== null && sourceFrameAgeMs <= WORLD_STATE_ACTIONABLE_MAX_MS);
            if (
              hasActionSignal &&
              stateActionable &&
              signature !== lastGuidanceSignatureRef.current &&
              nowMs - lastGuidanceQueryAtRef.current > 450
            ) {
              lastGuidanceSignatureRef.current = signature;
              lastGuidanceQueryAtRef.current = nowMs;
              ws.send(JSON.stringify({
                type: "query",
                query: "interaction_guidance",
                source: "state_update",
                frame_timestamp: msg.frame_timestamp,
              }));
            }
          }
          const queued = queuedWorldFrameRef.current;
          if (queued && ws.readyState === WebSocket.OPEN) {
            if (Date.now() - queued.timestamp > 1200) {
              queuedWorldFrameRef.current = null;
              return;
            }
            queuedWorldFrameRef.current = null;
            worldFrameInFlightRef.current = true;
            worldFrameInFlightSinceRef.current = queued.timestamp;
            pendingWorldFrameRef.current = {
              image: queued.image,
              timestamp: queued.timestamp,
              width: queued.width,
              height: queued.height,
            };
            sentWorldFramesRef.current.set(queued.timestamp, {
              image: queued.image,
              timestamp: queued.timestamp,
              width: queued.width,
              height: queued.height,
            });
            ws.send(
              JSON.stringify({
                type: "frame",
                image: queued.image,
                timestamp: queued.timestamp,
                frame_width: queued.width,
                frame_height: queued.height,
                capture_ms: queued.captureMs,
              })
            );
          }
          return;
        }

        if (msg.type === "query_result") {
          if (msg.result?.mode && typeof msg.result?.message === "string") {
            const sourceFrameTs = Number(msg.result?.source_frame_timestamp);
            const guidanceDelayMs = Number.isFinite(sourceFrameTs) ? Math.max(0, Date.now() - sourceFrameTs) : null;
            setPlannerSummary(
              `${msg.result.message || "(No grounded interaction cue yet)"}${
                guidanceDelayMs !== null ? `\nGuidance delay: ${guidanceDelayMs} ms` : ""
              }`
            );
          }

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

            const now = Date.now();
            const last = lastSpeakTimeRef.current;
            const lastBase = lastSpokenRef.current;
            const base = text;
            const mode = String(msg.result?.mode || "");
            const eventAgeS = Number(msg.result?.event_age_s);
            const contactAgeS = Number(msg.result?.contact_age_s);
            const sourceFrameTs = Number(msg.result?.source_frame_timestamp);
            const guidanceDelayMs = Number.isFinite(sourceFrameTs) ? Math.max(0, Date.now() - sourceFrameTs) : null;
            const isGroundedMode = mode === "grabbed" || mode === "released" || mode === "releasing" || mode === "approach" || mode === "hand_detected";
            const isFreshGuidance = guidanceDelayMs === null
              || guidanceDelayMs <= WORLD_STATE_ACTIONABLE_MAX_MS
              || isGroundedMode;
            const isFreshRelease = mode !== "released" || !Number.isFinite(eventAgeS) || eventAgeS <= RELEASED_SPEECH_EVENT_MAX_S;
            const isFreshGrab = mode !== "grabbed" || !Number.isFinite(contactAgeS) || contactAgeS <= GRABBED_SPEECH_CONTACT_MAX_S;

            const shouldSkipRepeatedStop = base === "stop" && lastBase === "stop";
            const changed = base !== lastBase;
            const enoughTime = now - last > (mode === "hand_detected" ? 3500 : 1400);
            const repeatGroundedCue = mode !== "hand_detected" && now - last > 5200;

            if (isGroundedMode && isFreshGuidance && isFreshRelease && isFreshGrab && !shouldSkipRepeatedStop && enoughTime && (changed || repeatGroundedCue)) {
              maybeSpeakWorldModelExplanation(text);
              if (guidanceDelayMs !== null) {
                setEventLog((prev) =>
                  [`[VOICE CUE ${guidanceDelayMs} ms] ${text}`, prev]
                    .filter(Boolean)
                    .join("\n\n")
                    .slice(0, 4000)
                );
              }
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
      setWorldModelConnected(false);
      worldFrameInFlightRef.current = false;
      worldFrameInFlightSinceRef.current = 0;
      if (showDebug) {
        setEventLog((prev) =>
          [`[WORLD MODEL ERROR]`, prev].filter(Boolean).join("\n\n").slice(0, 4000)
        );
      }
    };

    ws.onclose = () => {
      setWorldModelConnected(false);
      worldFrameInFlightRef.current = false;
      worldFrameInFlightSinceRef.current = 0;
      if (showDebug) {
        setEventLog((prev) =>
          [`[WORLD MODEL CLOSED]`, prev].filter(Boolean).join("\n\n").slice(0, 4000)
        );
      }
    };

    worldModelWsRef.current = ws;
  }

  async function ensureGeminiBridgeOpen(timeoutMs = 2500): Promise<boolean> {
    if (geminiBridgeRef.current && geminiBridgeRef.current.readyState === WebSocket.OPEN) {
      return true;
    }
    connectGeminiBridge();
    const ws = geminiBridgeRef.current;
    if (!ws) return false;
    if (ws.readyState === WebSocket.OPEN) return true;
    return new Promise((resolve) => {
      let settled = false;
      const cleanup = () => {
        ws.removeEventListener("open", handleOpen);
        ws.removeEventListener("close", handleClose);
        ws.removeEventListener("error", handleClose);
        clearTimeout(timer);
      };
      const finish = (value: boolean) => {
        if (settled) return;
        settled = true;
        cleanup();
        resolve(value);
      };
      const handleOpen = () => finish(true);
      const handleClose = () => finish(false);
      const timer = window.setTimeout(() => {
        finish(ws.readyState === WebSocket.OPEN);
      }, timeoutMs);

      ws.addEventListener("open", handleOpen, { once: true });
      ws.addEventListener("close", handleClose, { once: true });
      ws.addEventListener("error", handleClose, { once: true });
    });
  }

  async function startMic() {
    try {
      if (avatarMode === "ai") {
        const ready = await ensureGeminiBridgeOpen(3000);
        if (!ready) {
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

  function speakStartupCue() {
    if (startupCueSpokenRef.current) return;
    if (!isEmbodiedMode || embodiedVideoSource !== "scene") return;
    startupCueSpokenRef.current = true;
    maybeSpeakWorldModelExplanation("Let's start.", { forceAvatar: Boolean(useAvatarSpeechRef.current) });
  }

  async function startSceneVideo(sceneFile: SceneVideoFile = sceneVideoFile) {
    if (localCamStreamRef.current) {
      localCamStreamRef.current.getTracks().forEach((t) => t.stop());
      localCamStreamRef.current = null;
    }

    if (localCamRef.current) {
      localCamRef.current.pause();
      localCamRef.current.srcObject = null;
      localCamRef.current.src = `/${sceneFile}`;
      localCamRef.current.loop = false;
      localCamRef.current.muted = true;
      localCamRef.current.playsInline = true;
      localCamRef.current.playbackRate = 1;
      localCamRef.current.onended = async () => {
        worldModelWsRef.current?.send(JSON.stringify({ type: "query", query: "reset_world_model" }));
        if (localCamRef.current) {
          localCamRef.current.currentTime = 0;
          await localCamRef.current.play();
        }
      };
      localCamRef.current.currentTime = 0;
      await localCamRef.current.play();
      window.setTimeout(() => {
        speakStartupCue();
      }, 350);
    }
  }

  async function startLocalCamera(sourceOverride: EmbodiedVideoSource = embodiedVideoSource) {
    if (isEmbodiedMode && sourceOverride === "scene") {
      await startSceneVideo(sceneVideoFile);
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
    const maxCaptureWidth = 384;

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

      const nowMs = Date.now();

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
      const dataUrl = canvas.toDataURL("image/jpeg", 0.76);
      const t1 = performance.now();
      const timestamp = Date.now();

      setCaptureMs(t1 - t0);

      if (worldFrameInFlightRef.current) {
        queuedWorldFrameRef.current = {
          image: dataUrl,
          timestamp,
          width: frameWidth,
          height: frameHeight,
          captureMs: t1 - t0,
        };
        return;
      }

      worldFrameInFlightRef.current = true;
      worldFrameInFlightSinceRef.current = timestamp;
      pendingWorldFrameRef.current = {
        image: dataUrl,
        timestamp,
        width: frameWidth,
        height: frameHeight,
      };
      sentWorldFramesRef.current.set(timestamp, {
        image: dataUrl,
        timestamp,
        width: frameWidth,
        height: frameHeight,
      });
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
    }, 100);
  }

  function stopLocalCamera() {
    worldFrameInFlightRef.current = false;
    worldFrameInFlightSinceRef.current = 0;
    pendingWorldFrameRef.current = null;
    queuedWorldFrameRef.current = null;
    sentWorldFramesRef.current.clear();

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
    geminiReadyRef.current = false;
    pendingAvatarCueRef.current = null;

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
    setWorldModelConnected(false);

    if (labelReqCooldownTimerRef.current !== null) {
      window.clearTimeout(labelReqCooldownTimerRef.current);
      labelReqCooldownTimerRef.current = null;
    }
    labelReqCooldownRef.current = false;
    labelReqInFlightRef.current = false;

    stopLocalCamera();
    setBridgeConnected(false);
    setAvatarConnected(false);
    setIsSpeaking(false);
  }

  function maybeSpeakWorldModelExplanation(text: string, options: { forceAvatar?: boolean } = {}) {
    const shouldUseAvatar = Boolean(options.forceAvatar || useAvatarSpeechRef.current);
    const speakInBrowser = () => {
      if (!("speechSynthesis" in window)) {
        setStatus("Browser audio is not available in this browser");
        return;
      }
      const utter = new SpeechSynthesisUtterance(text);
      utter.rate = 1.25;
      utter.lang = "en-US";
      const voices = window.speechSynthesis.getVoices();
      const preferredVoice =
        voices.find((voice) => voice.lang === "en-US" && /natural|online|zira|aria|jenny|guy/i.test(voice.name)) ||
        voices.find((voice) => voice.lang === "en-US") ||
        voices.find((voice) => voice.lang.toLowerCase().startsWith("en-"));
      if (preferredVoice) utter.voice = preferredVoice;
      window.speechSynthesis.cancel();
      window.speechSynthesis.speak(utter);
    };

    const sendToBridge = () => {
      const requestSeq = avatarSpeechRequestSeqRef.current;
      geminiBridgeRef.current?.send(
        JSON.stringify({
          type: "world_model_explanation",
          text,
        })
      );
      window.setTimeout(() => {
        if (!shouldUseAvatar) return;
        if (avatarSpeechRequestSeqRef.current !== requestSeq) return;
        setStatus("Avatar audio did not arrive");
      }, 1800);
    };

    if (shouldUseAvatar) {
      if (!avatarWsRef.current || avatarWsRef.current.readyState !== WebSocket.OPEN) {
        setStatus("LiveAvatar is not connected; speech cue skipped");
        return;
      }

      if (geminiBridgeRef.current?.readyState === WebSocket.OPEN) {
        if (geminiReadyRef.current) {
          sendToBridge();
        } else {
          pendingAvatarCueRef.current = text;
          setStatus("Waiting for Gemini speech bridge...");
        }
        return;
      }

      setStatus("Connecting avatar speech bridge...");
      pendingAvatarCueRef.current = text;
      connectGeminiBridge();
      window.setTimeout(() => {
        if (geminiBridgeRef.current?.readyState === WebSocket.OPEN && geminiReadyRef.current) {
          pendingAvatarCueRef.current = null;
          sendToBridge();
        } else {
          setStatus("Waiting for Gemini speech bridge...");
        }
      }, 900);
      return;
    }

    if (useWebSpeechDebug) {
      speakInBrowser();
      return;
    }

    if (text.trim().toLowerCase() === "let's start.") {
      speakInBrowser();
      return;
    }

    setStatus("Browser audio is off; cue skipped");
  }

  useEffect(() => {
    if (!isEmbodiedMode || !autoMode) return;

    const interval = setInterval(() => {
      if (!worldModelWsRef.current || worldModelWsRef.current.readyState !== WebSocket.OPEN) {
        return;
      }
      const sourceFrameTs = Number(processedFrameTimestamp);
      const sourceFrameAgeMs = Number.isFinite(sourceFrameTs) ? Math.max(0, Date.now() - sourceFrameTs) : null;
      if (sourceFrameAgeMs === null || sourceFrameAgeMs > GUIDANCE_FRESH_MAX_MS) {
        return;
      }
      if (Date.now() - lastGuidanceQueryAtRef.current < 1500) {
        return;
      }
      lastGuidanceQueryAtRef.current = Date.now();

      worldModelWsRef.current.send(JSON.stringify({
        type: "query",
        query: "interaction_guidance",
        source: "fresh_periodic",
        frame_timestamp: sourceFrameTs,
      }));
    }, 1000);

    return () => clearInterval(interval);
  }, [autoMode, isEmbodiedMode, processedFrameTimestamp]);

  useEffect(() => {
    if (!isEmbodiedMode || embodiedVideoSource !== "scene") return;
    const interval = window.setInterval(() => {
      const video = localCamRef.current;
      if (!video) return;
      setSceneVideoTimeS(Number.isFinite(video.currentTime) ? video.currentTime : 0);
    }, 100);
    return () => window.clearInterval(interval);
  }, [embodiedVideoSource, isEmbodiedMode]);

  useEffect(() => {
    if (!autoMode || !sceneTimelineCue) return;
    if (!sceneTimelineActive) return;
    if (sceneVideoTimeS < sceneTimelineCue.grabStartS || sceneVideoTimeS > sceneTimelineCue.releaseS) return;
    const key = `${sceneTimelineCue.id}:grab`;
    if (lastSceneTimelineSpeechRef.current === key) return;
    lastSceneTimelineSpeechRef.current = key;
    maybeSpeakWorldModelExplanation(`${sceneTimelineCue.object}. Target: ${sceneTimelineCue.target}.`, {
      forceAvatar: Boolean(useAvatarSpeechRef.current),
    });
  }, [autoMode, sceneTimelineActive, sceneTimelineCue, sceneVideoTimeS]);


  useEffect(() => {
    autoModeRef.current = autoMode;
  }, [autoMode]);

  useEffect(() => {
    useAvatarSpeechRef.current = useAvatarSpeech;
  }, [useAvatarSpeech]);

  useEffect(() => {
    const interval = window.setInterval(() => setUiNowMs(Date.now()), 250);
    return () => window.clearInterval(interval);
  }, []);

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

  useEffect(() => {
    try {
      const raw = window.localStorage.getItem("embodied_ui_card_expanded");
      if (raw) {
        const parsed = JSON.parse(raw);
        if (parsed && typeof parsed === "object") {
          setCardExpanded((prev) => ({ ...prev, ...parsed }));
        }
      }
      const labelByIdRaw = window.localStorage.getItem("embodied_refined_labels_by_id");
      if (labelByIdRaw) {
        const parsed = JSON.parse(labelByIdRaw);
        if (parsed && typeof parsed === "object") setRefinedLabelsById(sanitizeSceneRefinedLabels(parsed));
      }
      const labelByHintRaw = window.localStorage.getItem("embodied_refined_labels_by_hint");
      if (labelByHintRaw) {
        const parsed = JSON.parse(labelByHintRaw);
        if (parsed && typeof parsed === "object") setRefinedLabelsByHint((prev) => sanitizeSceneRefinedLabels({ ...prev, ...parsed }));
      }
      const h = Number(window.localStorage.getItem("embodied_ui_diag_height_px") || "");
      if (Number.isFinite(h) && h >= 160 && h <= 520) {
        setDiagnosticsHeightPx(h);
      }
    } catch {}
  }, []);

  useEffect(() => {
    try {
      window.localStorage.setItem("embodied_ui_card_expanded", JSON.stringify(cardExpanded));
    } catch {}
  }, [cardExpanded]);

  useEffect(() => {
    try {
      window.localStorage.setItem("embodied_ui_diag_height_px", String(diagnosticsHeightPx));
    } catch {}
  }, [diagnosticsHeightPx]);

  const avatarModeLocked = avatarConnected || micOn;
  const captureModeLocked = avatarConnected || micOn || worldModelConnected;

  const statusItems = [
    { id: "status", label: "Status", value: status, active: status !== "idle" },
    { id: "capture", label: "Capture", value: isEmbodiedMode ? "embodied" : "social", active: true },
    { id: "avatar-mode", label: "Avatar", value: avatarMode === "ai" ? "AI mode" : "Direct mode", active: true },
    { id: "avatar-connection", label: "Avatar", value: avatarConnected ? "connected" : "disconnected", active: avatarConnected },
    { id: "gemini-bridge", label: "Gemini bridge", value: bridgeConnected ? "connected" : "disconnected", active: bridgeConnected },
    { id: "mic", label: "Mic", value: micOn ? "on" : "off", active: micOn },
    { id: "avatar-speaking", label: "Avatar speaking", value: isSpeaking ? "yes" : "no", active: isSpeaking },
    {
      id: "guidance-gate",
      label: "Guidance gate",
      value: demoGate?.allow_guidance
        ? `open (${Number(demoGate?.overall_score ?? 0).toFixed(2)})`
        : `gated: ${demoGate?.reason || "n/a"}`,
      active: Boolean(demoGate?.allow_guidance),
    },
    {
      id: "map-conf",
      label: "Map conf",
      value: Number(demoGate?.map_score ?? 0).toFixed(2),
      active: Number(demoGate?.map_score ?? 0) >= 0.5,
    },
    {
      id: "pose-conf",
      label: "Pose conf",
      value: Number(demoGate?.pose_score ?? 0).toFixed(2),
      active: Number(demoGate?.pose_score ?? 0) >= 0.5,
    },
    {
      id: "hand-conf",
      label: "Hand conf",
      value: Number(demoGate?.hand_score ?? 0).toFixed(2),
      active: Number(demoGate?.hand_score ?? 0) >= 0.5,
    },
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
                      onChange: (checked: boolean) => {
                        setUseAvatarSpeech(checked);
                        if (checked) setUseWebSpeechDebug(false);
                      },
                    },
                    {
                      label: "Browser audio",
                      checked: useWebSpeechDebug,
                      onChange: (checked: boolean) => {
                        setUseWebSpeechDebug(checked);
                        if (checked) setUseAvatarSpeech(false);
                      },
                      disabled: !isEmbodiedMode,
                    },
                    {
                      label: "Guidance on",
                      checked: autoMode,
                      onChange: (checked: boolean) => setAutoMode(checked),
                      disabled: !isEmbodiedMode,
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
                    disabled={!avatarConnected || micOn}
                    style={actionButtonStyle("secondary", !avatarConnected || micOn)}
                  >
                    Start mic
                  </button>
                  <button
                    onClick={stopMic}
                    disabled={!micOn}
                    style={actionButtonStyle("secondary", !micOn)}
                  >
                    Stop mic
                  </button>
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
                          Scene video
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
                        <div style={{ display: "flex", flexWrap: "wrap", gap: 8, alignItems: "center" }}>
                          <span style={toggleChipStyle(true)} title={`${SCENE_VIDEO_OPTIONS[0].file}: ${SCENE_VIDEO_OPTIONS[0].note}`}>
                            {SCENE_VIDEO_OPTIONS[0].label}
                          </span>
                          <span style={{ color: "#64748b", fontSize: 13, fontWeight: 700 }}>
                            Locked to /{sceneVideoFile}.
                          </span>
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
{isEmbodiedMode ? "Local camera with grab/place prediction heatmaps." : "Static webcam feed."}
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
                  opacity: 1,
                }}
              />
              {useProcessedPlanningFrame && processedFrameUrl ? (
                <img
                  src={processedFrameUrl}
                  alt="Processed planning frame"
                  style={{
                    position: "absolute",
                    inset: 0,
                    width: "100%",
                    height: "100%",
                    objectFit: "cover",
                    display: "block",
                    zIndex: 1,
                  }}
                />
              ) : null}

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

              {false && isEmbodiedMode ? (
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

              {isEmbodiedMode ? (
                <div
                  style={{
                    position: "absolute",
                    inset: 0,
                    background: "linear-gradient(180deg, rgba(2,6,23,0.10) 0%, rgba(2,6,23,0) 34%, rgba(2,6,23,0.30) 100%)",
                    pointerEvents: "none",
                    zIndex: 7,
                  }}
                />
              ) : null}

              {isEmbodiedMode && planningFreshEnough && attentionLinks.map((ln) => {
                const left = (ln.fromX / processedFrameSize.width) * 100;
                const top = (ln.fromY / processedFrameSize.height) * 100;
                const dx = ln.toX - ln.fromX;
                const dy = ln.toY - ln.fromY;
                const len = Math.sqrt(dx * dx + dy * dy);
                const angle = Math.atan2(dy, dx) * 180 / Math.PI;
                const alpha = Math.max(0.22, Math.min(0.9, ln.score * 0.9));
                const color = ln.kind === "grab" ? `rgba(245,158,11,${alpha})` : `rgba(6,182,212,${alpha})`;
                return (
                  <div
                    key={`planning-${ln.id}`}
                    style={{
                      position: "absolute",
                      left: `${left}%`,
                      top: `${top}%`,
                      width: `${(len / processedFrameSize.width) * 100}%`,
                      height: 3,
                      background: color,
                      boxShadow: `0 0 14px ${color}`,
                      transform: `translateY(-50%) rotate(${angle}deg)`,
                      transformOrigin: "0 50%",
                      pointerEvents: "none",
                      zIndex: 8,
                    }}
                  />
                );
              })}

              {isEmbodiedMode && displayedPlanningAttentionBlobs.map((item) => {
                const left = ((item.x1 + item.x2) * 0.5) * 100;
                const top = ((item.y1 + item.y2) * 0.5) * 100;
                const w = Math.max(item.kind === "place" ? 34 : 28, (item.x2 - item.x1) * (item.kind === "place" ? 285 : 220));
                const h = Math.max(item.kind === "place" ? 34 : 28, (item.y2 - item.y1) * (item.kind === "place" ? 285 : 220));
                const score = Math.max(0, Math.min(1, item.score));
                const isGrab = item.kind === "grab";
                const fill = isGrab
                  ? `radial-gradient(ellipse at center, rgba(255,247,210,${0.20 + 0.24 * score}) 0%, rgba(245,158,11,${0.38 + 0.38 * score}) 24%, rgba(245,158,11,${0.18 + 0.24 * score}) 56%, rgba(245,158,11,0) 84%)`
                  : `radial-gradient(ellipse at center, rgba(207,250,254,${0.20 + 0.22 * score}) 0%, rgba(6,182,212,${0.36 + 0.36 * score}) 26%, rgba(6,182,212,${0.16 + 0.24 * score}) 58%, rgba(6,182,212,0) 86%)`;
                return (
                  <div
                    key={`planning-attention-${item.id}`}
                    style={{
                      position: "absolute",
                      left: `${left}%`,
                      top: `${top}%`,
                      width: `${w}%`,
                      height: `${h}%`,
                      transform: "translate(-50%, -50%)",
                      pointerEvents: "none",
                      opacity: isGrab
                        ? Math.max(0.18, planningFreshness * (item === primaryAttention ? 1 : 0.8))
                        : Math.max(0.32, planningFreshness * 0.95),
                      transition: "left 180ms linear, top 180ms linear, width 180ms linear, height 180ms linear, opacity 220ms ease",
                      zIndex: isGrab ? (item === primaryAttention ? 9 : 8) : 10,
                    }}
                  >
                    <div style={{ position: "absolute", inset: "-14%", borderRadius: 999, background: fill, filter: "blur(10px)" }} />
                    <div
                      style={{
                        position: "absolute",
                        inset: "10%",
                        borderRadius: 999,
                        border: `2px solid ${isGrab ? "rgba(245,158,11,0.78)" : "rgba(6,182,212,0.82)"}`,
                        boxShadow: isGrab
                          ? "0 0 22px rgba(245,158,11,0.55), inset 0 0 18px rgba(245,158,11,0.18)"
                          : "0 0 26px rgba(6,182,212,0.65), inset 0 0 20px rgba(6,182,212,0.22)",
                        background: isGrab ? "rgba(245,158,11,0.08)" : "rgba(6,182,212,0.10)",
                      }}
                    />
                    {false && item === primaryAttention ? (
                      <div
                        style={{
                          position: "absolute",
                          left: "50%",
                          top: "50%",
                          width: 14,
                          height: 14,
                          borderRadius: "50%",
                          transform: "translate(-50%, -50%)",
                          background: "rgba(255,255,255,0.82)",
                          boxShadow: isGrab ? "0 0 22px rgba(245,158,11,0.95)" : "0 0 22px rgba(6,182,212,0.95)",
                        }}
                      />
                    ) : null}
                    <div
                      style={{
                        position: "absolute",
                        display: "flex",
                        alignItems: "center",
                        gap: 6,
                        left: "50%",
                        top: isGrab ? "4%" : "82%",
                        transform: "translateX(-50%)",
                        maxWidth: "92%",
                        padding: "6px 9px",
                        borderRadius: 8,
                        background: isGrab ? "rgba(120,53,15,0.92)" : "rgba(8,47,73,0.92)",
                        color: "#ffffff",
                        border: `1px solid ${isGrab ? "rgba(253,230,138,0.80)" : "rgba(165,243,252,0.80)"}`,
                        fontSize: 12,
                        fontWeight: 900,
                        whiteSpace: "nowrap",
                        textShadow: "0 1px 2px rgba(0,0,0,0.65)",
                        boxShadow: "0 10px 22px rgba(2,6,23,0.36)",
                      }}
                    >
                      <span>{isGrab ? "grab" : "destination"}</span>
                      <span style={{ opacity: 0.84 }}>{item.label}</span>
                      <span style={{ opacity: 0.72 }}>{Math.round(score * 100)}%</span>
                    </div>
                  </div>
                );
              })}

              {isEmbodiedMode ? (
                <div
                  style={{
                    position: "absolute",
                    left: 14,
                    top: 14,
                    padding: "8px 11px",
                    borderRadius: 999,
                    background: "rgba(2,6,23,0.66)",
                    color: "#ffffff",
                    border: "1px solid rgba(255,255,255,0.20)",
                    fontSize: 12,
                    fontWeight: 900,
                    boxShadow: "0 12px 26px rgba(2,6,23,0.24)",
                    pointerEvents: "none",
                    zIndex: 12,
                  }}
                >
                  <span style={{ opacity: 0.78 }}>{displayedIntentPhase}</span>
                  <span style={{ marginLeft: 8 }}>{displayedIntentCaption}</span>
                  {grabPredictionLabel !== "none" ? (
                    <span style={{ marginLeft: 8, opacity: 0.72 }}>
                      grab {grabPredictionLabel} {grabPredictionScore}%
                    </span>
                  ) : null}
                  {placePredictionScore > 0 ? (
                    <span style={{ marginLeft: 8, opacity: 0.72 }}>
                      place {placePredictionScore}%
                    </span>
                  ) : null}
                </div>
              ) : null}

              {isEmbodiedMode ? (
                <div
                  style={{
                    position: "absolute",
                    right: 14,
                    top: 14,
                    display: "flex",
                    alignItems: "center",
                    gap: 10,
                    padding: "7px 10px",
                    borderRadius: 999,
                    background: displayedFrameAgeMs != null && displayedFrameAgeMs > 1600 ? "rgba(127,29,29,0.62)" : "rgba(2,6,23,0.56)",
                    color: "#ffffff",
                    border: "1px solid rgba(255,255,255,0.18)",
                    fontSize: 11,
                    fontWeight: 800,
                    pointerEvents: "none",
                    zIndex: 12,
                    boxShadow: "0 12px 26px rgba(2,6,23,0.20)",
                  }}
                >
                  <span style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
                    <span style={{ width: 9, height: 9, borderRadius: 999, background: "#f59e0b", boxShadow: "0 0 10px rgba(245,158,11,0.8)" }} />
                    grab
                  </span>
                  <span style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
                    <span style={{ width: 9, height: 9, borderRadius: 999, background: "#06b6d4", boxShadow: "0 0 10px rgba(6,182,212,0.8)" }} />
                    place
                  </span>
                  <span style={{ color: "rgba(255,255,255,0.72)" }}>
                    {displayedFrameAgeMs == null ? "syncing" : (displayedFrameAgeMs > 1600 ? `stale ${Math.max(0, displayedFrameAgeMs)} ms` : `${Math.max(0, displayedFrameAgeMs)} ms`)}
                  </span>
                </div>
              ) : null}

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
              alignItems: "start",
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
              <h2 style={{ marginTop: 0, fontSize: 18 }}>Sparse 3D map (JSON debug)</h2>
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
                    left: `${50 + (cameraWorldX - sceneCenterX) * mapScale}%`,
                    top: `${50 - (cameraWorldZ - sceneCenterZ) * mapScale}%`,
                    width: 12,
                    height: 12,
                    borderRadius: "50%",
                    transform: "translate(-50%, -50%)",
                    background: "#f97316",
                    border: "2px solid #ffedd5",
                    boxShadow: "0 0 0 6px rgba(249,115,22,0.18)",
                    zIndex: 3,
                  }}
                  title={`Camera x=${cameraWorldX.toFixed(3)}, z=${cameraWorldZ.toFixed(3)}`}
                />
                {cameraTrailState.map((pt, idx) => {
                  if (!Array.isArray(pt) || pt.length < 3) return null;
                  const left = 50 + (Number(pt[0]) - sceneCenterX) * mapScale;
                  const top = 50 - (Number(pt[2]) - sceneCenterZ) * mapScale;
                  if (left < -25 || left > 125 || top < -25 || top > 125) return null;
                  const alpha = Math.max(0.18, Math.min(0.72, (idx + 1) / Math.max(cameraTrailState.length, 1)));
                  return (
                    <div
                      key={`cam-trail-top-${idx}`}
                      style={{
                        position: "absolute",
                        left: `${left}%`,
                        top: `${top}%`,
                        width: 4,
                        height: 4,
                        borderRadius: "50%",
                        transform: "translate(-50%, -50%)",
                        background: `rgba(251,146,60,${alpha})`,
                        border: "1px solid rgba(255,237,213,0.5)",
                        pointerEvents: "none",
                        zIndex: 2,
                      }}
                    />
                  );
                })}
                {mapPointsForView.map((point) => {
                  const left = 50 + (point.x - sceneCenterX) * mapScale;
                  const top = 50 - (point.z - sceneCenterZ) * mapScale;
                  if (left < -25 || left > 125 || top < -25 || top > 125) return null;

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
                {object3dMarkers.map((obj) => {
                  const left = 50 + (obj.x - sceneCenterX) * mapScale;
                  const top = 50 - (obj.z - sceneCenterZ) * mapScale;
                  if (left < -25 || left > 125 || top < -25 || top > 125) return null;
                  return (
                    <div
                      key={`obj3d-top-${obj.id}`}
                      style={{
                        position: "absolute",
                        left: `${left}%`,
                        top: `${top}%`,
                        width: obj.isHighlighted ? 13 : 9,
                        height: obj.isHighlighted ? 13 : 9,
                        borderRadius: "50%",
                        transform: "translate(-50%, -50%)",
                        background: obj.isHighlighted ? "rgba(239,68,68,0.95)" : "rgba(16,185,129,0.9)",
                        border: obj.isHighlighted ? "2px solid rgba(254,242,242,0.98)" : "2px solid rgba(209,250,229,0.95)",
                        boxShadow: obj.isHighlighted ? "0 0 0 4px rgba(239,68,68,0.22)" : "0 0 0 3px rgba(16,185,129,0.16)",
                        pointerEvents: "none",
                        zIndex: 5,
                      }}
                      title={`${obj.label} (${obj.id})`}
                    />
                  );
                })}
                {handTrajectoriesData.map((track: any) => {
                  const points = Array.isArray(track?.points_3d) ? track.points_3d : [];
                  return points.map((pt: any, idx: number) => {
                    if (!Array.isArray(pt) || pt.length < 3) return null;
                    const x = Number(pt[0]);
                    const z = Number(pt[2]);
                    if (!Number.isFinite(x) || !Number.isFinite(z)) return null;
                    const left = 50 + (x - sceneCenterX) * mapScale;
                    const top = 50 - (z - sceneCenterZ) * mapScale;
                    if (left < -25 || left > 125 || top < -25 || top > 125) return null;
                    const ageAlpha = Math.max(0.2, Math.min(0.9, (idx + 1) / Math.max(points.length, 1)));
                    return (
                      <div
                        key={`hand-traj-top-${track?.hand_id || "h"}-${idx}`}
                        style={{
                          position: "absolute",
                          left: `${left}%`,
                          top: `${top}%`,
                          width: 4,
                          height: 4,
                          borderRadius: "50%",
                          transform: "translate(-50%, -50%)",
                          background: `rgba(244,114,182,${ageAlpha})`,
                          border: "1px solid rgba(255,255,255,0.5)",
                          pointerEvents: "none",
                        }}
                      />
                    );
                  });
                })}
                {renderedHandCapsules3d.map((cap) => {
                  const left1 = 50 + (cap.x1 - sceneCenterX) * mapScale;
                  const top1 = 50 - (cap.z1 - sceneCenterZ) * mapScale;
                  const left2 = 50 + (cap.x2 - sceneCenterX) * mapScale;
                  const top2 = 50 - (cap.z2 - sceneCenterZ) * mapScale;
                  if ([left1, top1, left2, top2].some((v) => v < -10 || v > 110)) return null;
                  const dx = left2 - left1;
                  const dy = top2 - top1;
                  const len = Math.sqrt(dx * dx + dy * dy);
                  const angle = Math.atan2(dy, dx) * 180 / Math.PI;
                  const thickness = Math.max(8, Math.min(24, cap.r * mapScale * 5.6));
                  return (
                    <div
                      key={`hand-capsule-top-${cap.key}`}
                      style={{
                        position: "absolute",
                        left: `${left1}%`,
                        top: `${top1}%`,
                        width: `${len}%`,
                        height: thickness,
                        background: cap.predicted ? "rgba(251,113,133,0.68)" : "rgba(244,114,182,0.72)",
                        border: cap.predicted ? "3px solid rgba(255,228,230,0.96)" : "3px solid rgba(253,242,248,0.96)",
                        boxShadow: cap.predicted
                          ? "0 0 0 5px rgba(251,113,133,0.18), 0 0 22px rgba(251,113,133,0.75)"
                          : "0 0 0 5px rgba(244,114,182,0.18), 0 0 22px rgba(244,114,182,0.75)",
                        borderRadius: 999,
                        transform: `translateY(-50%) rotate(${angle}deg)`,
                        transformOrigin: "0 50%",
                        pointerEvents: "none",
                        zIndex: 18,
                      }}
                    />
                  );
                })}
                {renderedHandBones3d.map((bone) => {
                  const left1 = 50 + (bone.x1 - sceneCenterX) * mapScale;
                  const top1 = 50 - (bone.z1 - sceneCenterZ) * mapScale;
                  const left2 = 50 + (bone.x2 - sceneCenterX) * mapScale;
                  const top2 = 50 - (bone.z2 - sceneCenterZ) * mapScale;
                  if ([left1, top1, left2, top2].some((v) => v < -10 || v > 110)) return null;
                  const dx = left2 - left1;
                  const dy = top2 - top1;
                  const len = Math.sqrt(dx * dx + dy * dy);
                  const angle = Math.atan2(dy, dx) * 180 / Math.PI;
                  return (
                    <div
                      key={`hand-bone-top-${bone.key}`}
                      style={{
                        position: "absolute",
                        left: `${left1}%`,
                        top: `${top1}%`,
                        width: `${len}%`,
                        height: 4,
                        background: "rgba(255,228,230,0.96)",
                        boxShadow: "0 0 12px rgba(244,114,182,0.9)",
                        transform: `translateY(-50%) rotate(${angle}deg)`,
                        transformOrigin: "0 50%",
                        pointerEvents: "none",
                        zIndex: 19,
                      }}
                    />
                  );
                })}
                {renderedHandPoints3d.map((hand) => {
                  const left = 50 + (hand.x - sceneCenterX) * mapScale;
                  const top = 50 - (hand.z - sceneCenterZ) * mapScale;
                  if (left < -25 || left > 125 || top < -25 || top > 125) return null;
                  return (
                    <div
                      key={`hand-top-${hand.id}`}
                      style={{
                        position: "absolute",
                        left: `${left}%`,
                        top: `${top}%`,
                        width: 20,
                        height: 20,
                        borderRadius: "50%",
                        transform: "translate(-50%, -50%)",
                        background: hand.predicted ? "#fb7185" : "#f472b6",
                        border: "3px solid #fdf2f8",
                        boxShadow: "0 0 0 8px rgba(244,114,182,0.22), 0 0 24px rgba(244,114,182,0.9)",
                        zIndex: 20,
                        pointerEvents: "none",
                      }}
                      title={`Hand ${hand.id}: x=${hand.x.toFixed(3)}, y=${hand.y.toFixed(3)}, z=${hand.z.toFixed(3)}`}
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
                    left: `${50 + (cameraWorldZ - sceneCenterZ) * sideMapScale}%`,
                    top: `${50 + (cameraWorldY - sceneCenterY) * sideMapScale * sideYExaggeration}%`,
                    width: 12,
                    height: 12,
                    borderRadius: "50%",
                    transform: "translate(-50%, -50%)",
                    background: "#f97316",
                    border: "2px solid #ffedd5",
                    boxShadow: "0 0 0 6px rgba(249,115,22,0.18)",
                    zIndex: 3,
                  }}
                  title={`Camera y=${cameraWorldY.toFixed(3)}, z=${cameraWorldZ.toFixed(3)}`}
                />
                {cameraTrailState.map((pt, idx) => {
                  if (!Array.isArray(pt) || pt.length < 3) return null;
                  const left = 50 + (Number(pt[2]) - sceneCenterZ) * sideMapScale;
                  const top = 50 + (Number(pt[1]) - sceneCenterY) * sideMapScale * sideYExaggeration;
                  if (left < -25 || left > 125 || top < -25 || top > 125) return null;
                  const alpha = Math.max(0.18, Math.min(0.72, (idx + 1) / Math.max(cameraTrailState.length, 1)));
                  return (
                    <div
                      key={`cam-trail-side-${idx}`}
                      style={{
                        position: "absolute",
                        left: `${left}%`,
                        top: `${top}%`,
                        width: 4,
                        height: 4,
                        borderRadius: "50%",
                        transform: "translate(-50%, -50%)",
                        background: `rgba(251,146,60,${alpha})`,
                        border: "1px solid rgba(255,237,213,0.5)",
                        pointerEvents: "none",
                        zIndex: 2,
                      }}
                    />
                  );
                })}
                {mapPointsForView.map((point) => {
                  const left = 50 + (point.z - sceneCenterZ) * sideMapScale;
                  const top = 50 + (point.y - sceneCenterY) * sideMapScale * sideYExaggeration;
                  if (left < -25 || left > 125 || top < -25 || top > 125) return null;

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
                {object3dMarkers.map((obj) => {
                  const left = 50 + (obj.z - sceneCenterZ) * sideMapScale;
                  const top = 50 + (obj.y - sceneCenterY) * sideMapScale * sideYExaggeration;
                  if (left < -25 || left > 125 || top < -25 || top > 125) return null;
                  return (
                    <div
                      key={`obj3d-side-${obj.id}`}
                      style={{
                        position: "absolute",
                        left: `${left}%`,
                        top: `${top}%`,
                        width: obj.isHighlighted ? 13 : 9,
                        height: obj.isHighlighted ? 13 : 9,
                        borderRadius: "50%",
                        transform: "translate(-50%, -50%)",
                        background: obj.isHighlighted ? "rgba(239,68,68,0.95)" : "rgba(16,185,129,0.9)",
                        border: obj.isHighlighted ? "2px solid rgba(254,242,242,0.98)" : "2px solid rgba(209,250,229,0.95)",
                        boxShadow: obj.isHighlighted ? "0 0 0 4px rgba(239,68,68,0.22)" : "0 0 0 3px rgba(16,185,129,0.16)",
                        pointerEvents: "none",
                        zIndex: 5,
                      }}
                      title={`${obj.label} (${obj.id})`}
                    />
                  );
                })}
                {handTrajectoriesData.map((track: any) => {
                  const points = Array.isArray(track?.points_3d) ? track.points_3d : [];
                  return points.map((pt: any, idx: number) => {
                    if (!Array.isArray(pt) || pt.length < 3) return null;
                    const y = Number(pt[1]);
                    const z = Number(pt[2]);
                    if (!Number.isFinite(y) || !Number.isFinite(z)) return null;
                    const left = 50 + (z - sceneCenterZ) * sideMapScale;
                    const top = 50 + (y - sceneCenterY) * sideMapScale * sideYExaggeration;
                    if (left < -25 || left > 125 || top < -25 || top > 125) return null;
                    const ageAlpha = Math.max(0.2, Math.min(0.9, (idx + 1) / Math.max(points.length, 1)));
                    return (
                      <div
                        key={`hand-traj-side-${track?.hand_id || "h"}-${idx}`}
                        style={{
                          position: "absolute",
                          left: `${left}%`,
                          top: `${top}%`,
                          width: 4,
                          height: 4,
                          borderRadius: "50%",
                          transform: "translate(-50%, -50%)",
                          background: `rgba(244,114,182,${ageAlpha})`,
                          border: "1px solid rgba(255,255,255,0.5)",
                          pointerEvents: "none",
                        }}
                      />
                    );
                  });
                })}
                {renderedHandCapsules3d.map((cap) => {
                  const left1 = 50 + (cap.z1 - sceneCenterZ) * sideMapScale;
                  const top1 = 50 + (cap.y1 - sceneCenterY) * sideMapScale * sideYExaggeration;
                  const left2 = 50 + (cap.z2 - sceneCenterZ) * sideMapScale;
                  const top2 = 50 + (cap.y2 - sceneCenterY) * sideMapScale * sideYExaggeration;
                  if ([left1, top1, left2, top2].some((v) => v < -10 || v > 110)) return null;
                  const dx = left2 - left1;
                  const dy = top2 - top1;
                  const len = Math.sqrt(dx * dx + dy * dy);
                  const angle = Math.atan2(dy, dx) * 180 / Math.PI;
                  const thickness = Math.max(8, Math.min(24, cap.r * sideMapScale * 5.6));
                  return (
                    <div
                      key={`hand-capsule-side-${cap.key}`}
                      style={{
                        position: "absolute",
                        left: `${left1}%`,
                        top: `${top1}%`,
                        width: `${len}%`,
                        height: thickness,
                        background: cap.predicted ? "rgba(251,113,133,0.68)" : "rgba(244,114,182,0.72)",
                        border: cap.predicted ? "3px solid rgba(255,228,230,0.96)" : "3px solid rgba(253,242,248,0.96)",
                        boxShadow: cap.predicted
                          ? "0 0 0 5px rgba(251,113,133,0.18), 0 0 22px rgba(251,113,133,0.75)"
                          : "0 0 0 5px rgba(244,114,182,0.18), 0 0 22px rgba(244,114,182,0.75)",
                        borderRadius: 999,
                        transform: `translateY(-50%) rotate(${angle}deg)`,
                        transformOrigin: "0 50%",
                        pointerEvents: "none",
                        zIndex: 18,
                      }}
                    />
                  );
                })}
                {renderedHandBones3d.map((bone) => {
                  const left1 = 50 + (bone.z1 - sceneCenterZ) * sideMapScale;
                  const top1 = 50 + (bone.y1 - sceneCenterY) * sideMapScale * sideYExaggeration;
                  const left2 = 50 + (bone.z2 - sceneCenterZ) * sideMapScale;
                  const top2 = 50 + (bone.y2 - sceneCenterY) * sideMapScale * sideYExaggeration;
                  if ([left1, top1, left2, top2].some((v) => v < -10 || v > 110)) return null;
                  const dx = left2 - left1;
                  const dy = top2 - top1;
                  const len = Math.sqrt(dx * dx + dy * dy);
                  const angle = Math.atan2(dy, dx) * 180 / Math.PI;
                  return (
                    <div
                      key={`hand-bone-side-${bone.key}`}
                      style={{
                        position: "absolute",
                        left: `${left1}%`,
                        top: `${top1}%`,
                        width: `${len}%`,
                        height: 4,
                        background: "rgba(255,228,230,0.96)",
                        boxShadow: "0 0 12px rgba(244,114,182,0.9)",
                        transform: `translateY(-50%) rotate(${angle}deg)`,
                        transformOrigin: "0 50%",
                        pointerEvents: "none",
                        zIndex: 19,
                      }}
                    />
                  );
                })}
                {renderedHandPoints3d.map((hand) => {
                  const left = 50 + (hand.z - sceneCenterZ) * sideMapScale;
                  const top = 50 + (hand.y - sceneCenterY) * sideMapScale * sideYExaggeration;
                  if (left < -25 || left > 125 || top < -25 || top > 125) return null;
                  return (
                    <div
                      key={`hand-side-${hand.id}`}
                      style={{
                        position: "absolute",
                        left: `${left}%`,
                        top: `${top}%`,
                        width: 20,
                        height: 20,
                        borderRadius: "50%",
                        transform: "translate(-50%, -50%)",
                        background: hand.predicted ? "#fb7185" : "#f472b6",
                        border: "3px solid #fdf2f8",
                        boxShadow: "0 0 0 8px rgba(244,114,182,0.22), 0 0 24px rgba(244,114,182,0.9)",
                        zIndex: 20,
                        pointerEvents: "none",
                      }}
                      title={`Hand ${hand.id}: x=${hand.x.toFixed(3)}, y=${hand.y.toFixed(3)}, z=${hand.z.toFixed(3)}`}
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
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
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
                    {processedFrameTimestamp !== null ? `${uiNowMs - processedFrameTimestamp} ms old` : "waiting"}
                  </span>
                </div>
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
                    {attentionLinks.map((ln) => {
                      const left = (ln.fromX / processedFrameSize.width) * 100;
                      const top = (ln.fromY / processedFrameSize.height) * 100;
                      const dx = ln.toX - ln.fromX;
                      const dy = ln.toY - ln.fromY;
                      const len = Math.sqrt(dx * dx + dy * dy);
                      const angle = Math.atan2(dy, dx) * 180 / Math.PI;
                      const alpha = Math.max(0.18, Math.min(0.85, ln.score * 0.85));
                      const color = ln.kind === "grab" ? `rgba(251,113,133,${alpha})` : `rgba(74,222,128,${alpha})`;
                      return (
                        <div
                          key={ln.id}
                          style={{
                            position: "absolute",
                            left: `${left}%`,
                            top: `${top}%`,
                            width: `${(len / processedFrameSize.width) * 100}%`,
                            height: 2,
                            background: color,
                            boxShadow: `0 0 8px ${color}`,
                            transform: `translateY(-50%) rotate(${angle}deg)`,
                            transformOrigin: "0 50%",
                            pointerEvents: "none",
                            zIndex: 10,
                          }}
                        />
                      );
                    })}
                    {displayedPlanningAttentionBlobs.map((item) => {
                      const left = ((item.x1 + item.x2) * 0.5) * 100;
                      const top = ((item.y1 + item.y2) * 0.5) * 100;
                      const w = Math.max(8, (item.x2 - item.x1) * 130);
                      const h = Math.max(8, (item.y2 - item.y1) * 130);
                      const score = Math.max(0, Math.min(1, item.score));
                      const ring = item.kind === "grab" ? "rgba(251,113,133,0.78)" : "rgba(74,222,128,0.78)";
                      const badge = item.kind === "grab" ? "grab next" : "place target";
                      const glow = item.kind === "grab"
                        ? `radial-gradient(ellipse at center, rgba(251,113,133,${0.32 + 0.36 * score}) 0%, rgba(251,113,133,${0.14 + 0.22 * score}) 42%, rgba(251,113,133,0) 82%)`
                        : `radial-gradient(ellipse at center, rgba(74,222,128,${0.30 + 0.34 * score}) 0%, rgba(74,222,128,${0.14 + 0.20 * score}) 44%, rgba(74,222,128,0) 84%)`;
                      return (
                        <div key={item.id} style={{ position: "absolute", left: `${left}%`, top: `${top}%`, width: `${Math.max(w, 8)}%`, height: `${Math.max(h, 8)}%`, transform: "translate(-50%, -50%)", pointerEvents: "none", zIndex: 11 }}>
                          <div style={{ position: "absolute", inset: "-18%", borderRadius: 999, background: glow, filter: "blur(4px)" }} />
                          <div style={{ position: "absolute", inset: "12%", borderRadius: 999, border: `1px solid ${ring}`, boxShadow: `0 0 18px ${ring}` }} />
                          <div
                            style={{
                              position: "absolute",
                              left: "50%",
                              top: item.kind === "grab" ? "4%" : "78%",
                              transform: "translateX(-50%)",
                              padding: "3px 7px",
                              borderRadius: 999,
                              background: "rgba(15,23,42,0.72)",
                              color: "#ffffff",
                              border: "1px solid rgba(255,255,255,0.22)",
                              fontSize: 10,
                              fontWeight: 800,
                              whiteSpace: "nowrap",
                            }}
                          >
                            {badge} · {Math.round(score * 100)}%
                          </div>
                        </div>
                      );
                    })}

                    {observedObjects.map((obj: any) => {
                      const bbox = obj?.bbox;
                      const objId = String(obj?.id || obj?.label || "obj");
                      if (!Array.isArray(bbox) || bbox.length < 4) return null;
                      const x1 = Number(bbox[0]);
                      const y1 = Number(bbox[1]);
                      const x2 = Number(bbox[2]);
                      const y2 = Number(bbox[3]);
                      if (![x1, y1, x2, y2].every(Number.isFinite)) return null;
                      const left = Math.max(0, Math.min(100, x1 * 100));
                      const top = Math.max(0, Math.min(100, y1 * 100));
                      const width = Math.max(0.5, Math.min(100, (x2 - x1) * 100));
                      const height = Math.max(0.5, Math.min(100, (y2 - y1) * 100));
                      const baseColor = colorFromId(objId);
                      const isContact = contactingObjectIds.has(objId);
                      return (
                        <div key={`obj-box-${objId}`}>
                          <div
                            style={{
                              position: "absolute",
                              left: `${left}%`,
                              top: `${top}%`,
                              width: `${width}%`,
                              height: `${height}%`,
                              border: isContact ? "3px solid rgba(239,68,68,0.95)" : `2px solid ${baseColor}`,
                              borderRadius: 8,
                              boxShadow: isContact ? "0 0 0 4px rgba(239,68,68,0.2)" : "0 0 0 2px rgba(15,23,42,0.12)",
                              pointerEvents: "none",
                              zIndex: 14,
                            }}
                          />
                          <div
                            style={{
                              position: "absolute",
                              left: `${left}%`,
                              top: `${Math.max(0, top - 4)}%`,
                              transform: "translateY(-100%)",
                              background: isContact ? "rgba(127,29,29,0.92)" : "rgba(15,23,42,0.88)",
                              color: "#f8fafc",
                              borderRadius: 6,
                              padding: "2px 6px",
                              fontSize: 10,
                              fontWeight: 700,
                              pointerEvents: "none",
                              whiteSpace: "nowrap",
                              zIndex: 15,
                            }}
                          >
                            {displayLabelForObject(obj)} · {objId}
                          </div>
                        </div>
                      );
                    })}
                    {handsData.map((hand: any) => {
                      const points = Array.isArray(hand?.landmarks_px) ? hand.landmarks_px : [];
                      const handId = String(hand?.id || hand?.side || "hand");
                      const color = "#f472b6";
                      const palmIndices = [0, 5, 9, 13, 17];
                      const palmPoints = palmIndices
                        .map((i) => points[i])
                        .filter((pt: any) => Array.isArray(pt) && pt.length >= 2)
                        .map((pt: any) => [Number(pt[0]), Number(pt[1])]);
                      const palmPolygon = palmPoints.length >= 3
                        ? palmPoints
                            .map(([px, py]: number[]) => `${(px / processedFrameSize.width) * 100}% ${(py / processedFrameSize.height) * 100}%`)
                            .join(", ")
                        : null;
                      const capsuleOverlay = HAND_BONES.map(([a, b], idx) => {
                        const p1 = points[a];
                        const p2 = points[b];
                        if (!Array.isArray(p1) || !Array.isArray(p2) || p1.length < 2 || p2.length < 2) return null;
                        const x1 = Number(p1[0]); const y1 = Number(p1[1]);
                        const x2 = Number(p2[0]); const y2 = Number(p2[1]);
                        if (![x1, y1, x2, y2].every(Number.isFinite)) return null;
                        const dx = x2 - x1;
                        const dy = y2 - y1;
                        const len = Math.sqrt(dx * dx + dy * dy);
                        const angle = Math.atan2(dy, dx) * 180 / Math.PI;
                        const pxThickness = Math.max(4, Math.min(14, len * 0.2));
                        return (
                          <div
                            key={`hand-capsule-overlay-${handId}-${idx}`}
                            style={{
                              position: "absolute",
                              left: `${(x1 / processedFrameSize.width) * 100}%`,
                              top: `${(y1 / processedFrameSize.height) * 100}%`,
                              width: `${(len / processedFrameSize.width) * 100}%`,
                              height: `${(pxThickness / processedFrameSize.height) * 100}%`,
                              borderRadius: 999,
                              background: hand?.predicted ? "rgba(251,113,133,0.46)" : "rgba(244,114,182,0.42)",
                              border: hand?.predicted ? "2px solid rgba(251,113,133,0.88)" : "2px solid rgba(244,114,182,0.88)",
                              boxShadow: hand?.predicted ? "0 0 0 2px rgba(251,113,133,0.2)" : "0 0 0 2px rgba(244,114,182,0.2)",
                              transform: `translateY(-50%) rotate(${angle}deg)`,
                              transformOrigin: "0 50%",
                              pointerEvents: "none",
                              zIndex: 9,
                            }}
                          />
                        );
                      });
                      const segments = HAND_BONES.map(([a, b], idx) => {
                        const p1 = points[a];
                        const p2 = points[b];
                        if (!Array.isArray(p1) || !Array.isArray(p2) || p1.length < 2 || p2.length < 2) return null;
                        const x1 = Number(p1[0]); const y1 = Number(p1[1]);
                        const x2 = Number(p2[0]); const y2 = Number(p2[1]);
                        if (![x1, y1, x2, y2].every(Number.isFinite)) return null;
                        const dx = x2 - x1;
                        const dy = y2 - y1;
                        const len = Math.sqrt(dx * dx + dy * dy);
                        const angle = Math.atan2(dy, dx) * 180 / Math.PI;
                        return (
                          <div
                            key={`hand-seg-${handId}-${idx}`}
                            style={{
                              position: "absolute",
                              left: `${(x1 / processedFrameSize.width) * 100}%`,
                              top: `${(y1 / processedFrameSize.height) * 100}%`,
                              width: `${(len / processedFrameSize.width) * 100}%`,
                              height: 2,
                              background: "rgba(244,114,182,0.95)",
                              transform: `translateY(-50%) rotate(${angle}deg)`,
                              transformOrigin: "0 50%",
                              pointerEvents: "none",
                            }}
                          />
                        );
                      });
                      const joints = points.map((pt: any, idx: number) => {
                        if (!Array.isArray(pt) || pt.length < 2) return null;
                        const px = Number(pt[0]); const py = Number(pt[1]);
                        if (!Number.isFinite(px) || !Number.isFinite(py)) return null;
                        return (
                          <div
                            key={`hand-joint-${handId}-${idx}`}
                            style={{
                              position: "absolute",
                              left: `${(px / processedFrameSize.width) * 100}%`,
                              top: `${(py / processedFrameSize.height) * 100}%`,
                              width: 5,
                              height: 5,
                              borderRadius: "50%",
                              transform: "translate(-50%, -50%)",
                              background: color,
                              border: "1px solid rgba(255,255,255,0.85)",
                              pointerEvents: "none",
                            }}
                          />
                        );
                      });
                      return (
                        <div key={`hand-overlay-${handId}`}>
                          {palmPolygon ? (
                            <div
                              style={{
                                position: "absolute",
                                inset: 0,
                                clipPath: `polygon(${palmPolygon})`,
                                background: hand?.predicted ? "rgba(251,113,133,0.22)" : "rgba(244,114,182,0.2)",
                                border: hand?.predicted ? "2px solid rgba(251,113,133,0.85)" : "2px solid rgba(244,114,182,0.85)",
                                pointerEvents: "none",
                                zIndex: 8,
                              }}
                            />
                          ) : null}
                          {capsuleOverlay}
                          {segments}
                          {joints}
                        </div>
                      );
                    })}
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
                            left: `${(px / processedFrameSize.width) * 100}%`,
                            top: `${(py / processedFrameSize.height) * 100}%`,
                            width: size,
                            height: size,
                            borderRadius: "50%",
                            transform: "translate(-50%, -50%)",
                            background: "rgba(56, 189, 248, 0.95)",
                            border: "1px solid rgba(224, 242, 254, 0.95)",
                            boxShadow: "0 0 0 3px rgba(56, 189, 248, 0.12)",
                            pointerEvents: "none",
                            zIndex: 4,
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
              <div style={{ marginTop: 14 }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                  <div style={{ color: "#475569", fontSize: 13, fontWeight: 800 }}>
                    Map diagnostics
                  </div>
                  <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    <input
                      type="range"
                      min={160}
                      max={520}
                      step={10}
                      value={diagnosticsHeightPx}
                      onChange={(e) => setDiagnosticsHeightPx(Number(e.target.value))}
                    />
                    <span style={{ fontSize: 11, color: "#64748b", fontWeight: 700 }}>{diagnosticsHeightPx}px</span>
                    <button
                      onClick={() => toggleCardExpanded("mapDiagnostics")}
                      style={toggleChipStyle(isCardExpanded("mapDiagnostics", true))}
                    >
                      {isCardExpanded("mapDiagnostics", true) ? "Hide" : "Show"}
                    </button>
                  </div>
                </div>
                {isCardExpanded("mapDiagnostics", true) && (
                <div
                  style={{
                    maxHeight: diagnosticsHeightPx,
                    overflowY: "auto",
                    border: "1px solid #e2e8f0",
                    borderRadius: 12,
                    padding: 8,
                    background: "#f8fafc",
                  }}
                >
                  <div
                    style={{
                      display: "grid",
                      gridTemplateColumns: "repeat(3, minmax(0, 1fr))",
                      gap: 8,
                      fontSize: 12,
                    }}
                  >
                    {debugStats.map(([label, value]) => (
                      <div
                        key={String(label)}
                        style={{
                          background: "#ffffff",
                          border: "1px solid #e2e8f0",
                          borderRadius: 10,
                          padding: "7px 9px",
                        }}
                      >
                        <div style={{ color: "#64748b", fontWeight: 700 }}>{label}</div>
                        <div style={{ color: "#0f172a", fontWeight: 800, marginTop: 2 }}>{String(value)}</div>
                      </div>
                    ))}
                  </div>
                </div>
                )}
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

            <div style={{ ...cardStyle(), padding: 18 }}>
              <h2 style={{ marginTop: 0, fontSize: 18 }}>Hand-object interactions</h2>
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
                {embodiedState.handInteractionsText || "(No interactions yet)"}
              </pre>
            </div>

            <div style={{ ...cardStyle(), padding: 18 }}>
              <h2 style={{ marginTop: 0, fontSize: 18 }}>Manipulation events</h2>
              <pre
                style={{
                  whiteSpace: "pre-wrap",
                  fontSize: 12,
                  height: 180,
                  overflowY: "auto",
                  background: "#f8fafc",
                  padding: 14,
                  borderRadius: 14,
                  border: "1px solid #e2e8f0",
                  margin: 0,
                }}
              >
                {embodiedState.manipulationEventsText || "(No pick/place events yet)"}
              </pre>
            </div>
          </section>
        )}
      </div>
    </main>
  );
}
