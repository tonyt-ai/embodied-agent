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

type Mode = "ai" | "direct";

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

  const [mode, setMode] = useState<Mode>("ai");
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

  const [worldStateText, setWorldStateText] = useState("");
  const [eventLog, setEventLog] = useState("");
  const [lastQueryResultText, setLastQueryResultText] = useState("");
  const [plannerSummary, setPlannerSummary] = useState("");
  const [plannerSimulations, setPlannerSimulations] = useState<any | null>(null);
  const [bestActionName, setBestActionName] = useState("");

  const [autoMode, setAutoMode] = useState(false);
  const [showDebug, setShowDebug] = useState(false);
  const lastSpokenRef = useRef<string>("");
  const lastSpeakTimeRef = useRef<number>(0);
  const [useAvatarSpeech, setUseAvatarSpeech] = useState(false);

  const [frameAgeMs, setFrameAgeMs] = useState<number | null>(null);
  const [captureMs, setCaptureMs] = useState<number | null>(null);
  const [serverDecodeMs, setServerDecodeMs] = useState<number | null>(null);
  const [serverDetectMs, setServerDetectMs] = useState<number | null>(null);
  const [serverTotalMs, setServerTotalMs] = useState<number | null>(null);
  const [pipelineAgeMs, setPipelineAgeMs] = useState<number | null>(null);

  function resetWorldUiState() {
    setWorldStateText("");
    setEventLog("");
    setLastQueryResultText("");
    setPlannerSummary("");
    setPlannerSimulations(null);
    setBestActionName("");
    setInputTranscript("");
    setOutputTranscript("");
    setObservedObjects([]);
    setFrameAgeMs(null);
    setCaptureMs(null);
    setServerDecodeMs(null);
    setServerDetectMs(null);
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
            mode === "ai" || (mode === "direct" && directAudioMonitor);

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
          connectWorldModel();
          await startLocalCamera();
          startSendingFrames();
        } catch (err) {
          console.error("World model camera startup failed:", err);
        }

        if (mode === "ai" || useAvatarSpeech) {
          connectGeminiBridge();
        } else {
          setBridgeConnected(false);
        }

        setStatus(mode === "ai" ? "Avatar ready (AI mode)" : "Avatar ready (Direct mode)");
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
            setStatus(mode === "ai" ? "Avatar finished speaking" : "Avatar finished direct speech");
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

        if (startGuidancePhrases.some((p) => normalized.includes(p))) {
          setAutoMode(true);
          setStatus("Auto guidance enabled by voice");
          return;
        }

        if (stopGuidancePhrases.some((p) => normalized.includes(p))) {
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
      setEventLog((prev) =>
        [`[WORLD MODEL CONNECTED]`, prev].filter(Boolean).join("\n\n").slice(0, 4000)
      );
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);

        if (msg.type === "state_updated") {
          const objects = msg.objects || [];
          setObservedObjects(objects);
          setWorldStateText(JSON.stringify({ objects }, null, 2));

          if (typeof msg.frame_timestamp === "number") {
            const age = Date.now() - msg.frame_timestamp;
            setFrameAgeMs(age);
            setPipelineAgeMs(age);
          }

          if (typeof msg.capture_ms === "number") setCaptureMs(msg.capture_ms);
          if (typeof msg.server_decode_ms === "number") setServerDecodeMs(msg.server_decode_ms);
          if (typeof msg.server_detect_ms === "number") setServerDetectMs(msg.server_detect_ms);
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
      if (showDebug) {
        setEventLog((prev) =>
          [`[WORLD MODEL ERROR]`, prev].filter(Boolean).join("\n\n").slice(0, 4000)
        );
      }
    };

    ws.onclose = () => {
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
      if (mode === "ai") {
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

      const targetSampleRate = mode === "ai" ? 16000 : 24000;
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

        if (mode === "ai") {
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
      setStatus(mode === "ai" ? "Mic on (AI mode)" : "Mic on (Direct mode)");
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

    if (mode === "ai") {
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
    setStatus(mode === "ai" ? "Mic off (AI mode)" : "Mic off (Direct mode)");
  }

  async function loadVideoDevices() {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const cams = devices.filter((d) => d.kind === "videoinput");
    setVideoDevices(cams);

    if (!cams.length) return;

    // If user already selected something, keep it
    if (selectedCameraId && cams.some((c) => c.deviceId === selectedCameraId)) {
      return;
    }

    let preferred;

    if (cams.length === 1) {
      // Only one camera → use it
      preferred = cams[0].deviceId;
    } else {
      // Multiple cameras → assume external is last
      preferred = cams[cams.length - 1].deviceId;
    }

    setSelectedCameraId(preferred);
  }

  async function startLocalCamera() {
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
      localCamRef.current.srcObject = stream;
      await localCamRef.current.play();
    }
  }

  function startSendingFrames() {
    if (!localCamRef.current) return;

    const video = localCamRef.current;

    if (!frameCanvasRef.current) {
      const canvas = document.createElement("canvas");
      canvas.width = 384;
      canvas.height = 216;
      frameCanvasRef.current = canvas;
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
        return;
      }

      if (video.readyState < 2) {
        return;
      }

      const t0 = performance.now();
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const dataUrl = canvas.toDataURL("image/jpeg", 0.3);
      const t1 = performance.now();

      setCaptureMs(t1 - t0);

      worldModelWsRef.current.send(
        JSON.stringify({
          type: "frame",
          image: dataUrl,
          timestamp: Date.now(),
          capture_ms: t1 - t0,
        })
      );
    }, 200);
  }

  function stopLocalCamera() {
    if (frameIntervalRef.current) {
      clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = null;
    }

    localCamStreamRef.current?.getTracks().forEach((t) => t.stop());
    localCamStreamRef.current = null;
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
    if (!autoMode) return;

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
  }, [autoMode]);

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

    if (localCamStreamRef.current) {
      stopLocalCamera();
      startLocalCamera();
    }
  }, [selectedCameraId]);

  useEffect(() => {
    return () => {
      stopAllConnections();
    };
  }, []);

  const modeLocked = avatarConnected || micOn;

  const statusItems = [
    { label: "Status", value: status, active: status !== "idle" },
    { label: "Mode", value: mode === "ai" ? "AI mode" : "Direct mode", active: true },
    { label: "Avatar", value: avatarConnected ? "connected" : "disconnected", active: avatarConnected },
    { label: "Gemini bridge", value: bridgeConnected ? "connected" : "disconnected", active: bridgeConnected },
    { label: "Mic", value: micOn ? "on" : "off", active: micOn },
    { label: "Avatar speaking", value: isSpeaking ? "yes" : "no", active: isSpeaking },
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
                  key={item.label}
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
                Start the avatar, connect the world model, and trigger planning.
              </p>
            </div>

            <div style={{ display: "grid", gap: 14 }}>
              <div>
                <div style={{ fontSize: 13, fontWeight: 700, color: "#64748b", marginBottom: 8 }}>
                  Mode
                </div>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 10 }}>
                  <button
                    onClick={() => setMode("ai")}
                    disabled={modeLocked || mode === "ai"}
                    style={toggleChipStyle(mode === "ai", modeLocked || mode === "ai")}
                  >
                    AI mode
                  </button>
                  <button
                    onClick={() => setMode("direct")}
                    disabled={modeLocked || mode === "direct"}
                    style={toggleChipStyle(mode === "direct", modeLocked || mode === "direct")}
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
                      disabled: modeLocked && mode !== "direct",
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
                  <button onClick={startWorldModelOnly} style={actionButtonStyle("secondary")}>
                    World model only
                  </button>
                  <button onClick={stopAllConnections} style={actionButtonStyle("danger")}>
                    Stop all
                  </button>
                  <button
                    onClick={startMic}
                    disabled={!avatarConnected || micOn || (mode === "ai" && !bridgeConnected)}
                    style={actionButtonStyle("secondary", !avatarConnected || micOn || (mode === "ai" && !bridgeConnected))}
                  >
                    Start mic
                  </button>
                  <button onClick={stopMic} disabled={!micOn} style={actionButtonStyle("secondary", !micOn)}>
                    Stop mic
                  </button>
                  <button
                    onClick={() => setAutoMode((v) => !v)}
                    style={actionButtonStyle(autoMode ? "primary" : "secondary")}
                  >
                    {autoMode ? "Stop guidance" : "Start guidance"}
                  </button>
                  <button onClick={askSimulateActions} style={actionButtonStyle("secondary")}>
                    Simulate futures
                  </button>
                </div>
                <div style={{ marginBottom: 12 }}>
                  <label style={{ marginRight: 8 }}>Camera:</label>
                  <select
                    value={selectedCameraId || ""}
                    onChange={(e) => setSelectedCameraId(e.target.value)}
                  >
                    {videoDevices.map((cam, i) => (
                      <option key={cam.deviceId} value={cam.deviceId}>
                        {cam.label || `Camera ${i + 1}`}
                      </option>
                    ))}
                  </select>
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
                <h2 style={{ margin: 0, fontSize: 18 }}>Camera + planning overlay</h2>
                <p style={{ margin: "6px 0 0 0", color: "#64748b", fontSize: 14 }}>
                  Local camera, goal marker, observed object, and simulated futures.
                </p>
              </div>
              <span
                style={{
                  padding: "6px 10px",
                  borderRadius: 999,
                  background: autoMode ? "#eff6ff" : "#f8fafc",
                  color: autoMode ? "#1d4ed8" : "#475569",
                  border: `1px solid ${autoMode ? "#bfdbfe" : "#e2e8f0"}`,
                  fontWeight: 700,
                  fontSize: 12,
                }}
              >
                {autoMode ? "GUIDANCE ON" : "GUIDANCE OFF"}
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

              {(() => {
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
              })()}

              {plannerSimulations &&
                Object.entries(plannerSimulations).map(([sequenceKey, sim]: any) => {
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
                  const isBest = firstAction === bestActionName;

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
            {bestActionName ? (
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
                Best action: {bestActionName}
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
            {plannerSummary || "(No planner decision yet)"}
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
                <strong>User:</strong> {inputTranscript || "(none)"}
              </p>
              <p style={{ margin: 0 }}>
                <strong>Agent:</strong> {outputTranscript || "(none)"}
              </p>
            </div>
          </section>
        )}

        {showDebug && (
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
                {worldStateText || "(No state yet)"}
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
                {lastQueryResultText || "(No query yet)"}
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
                {eventLog || "(No events yet)"}
              </pre>
            </div>
          </section>
        )}
      </div>
    </main>
  );
}
