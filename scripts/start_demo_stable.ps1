[CmdletBinding(SupportsShouldProcess = $true)]
param(
  [switch]$RequireHands,
  [switch]$HighPointBudget,
  [switch]$HybridQuality,
  [switch]$DenseObjects,
  [switch]$TrainedTemporal,
  [switch]$Biberon,
  [switch]$Sophie
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$serverPath = Join-Path $repoRoot "world_model\server.py"
$geminiBridgePath = Join-Path $repoRoot "server\live-bridge.mjs"
$repoParent = Split-Path -Parent $repoRoot
$xfeatCandidates = @(
  (Join-Path $repoRoot "third_party\xfeat\accelerated_features"),
  (Join-Path $repoRoot "xfeat\accelerated_features"),
  (Join-Path $repoParent "xfeat\accelerated_features"),
  "C:\code\xfeat\accelerated_features"
)
$xfeatRepo = $null
foreach ($candidate in $xfeatCandidates) {
  if (Test-Path -LiteralPath $candidate -PathType Container) {
    $xfeatRepo = (Resolve-Path -LiteralPath $candidate).Path
    break
  }
}

$envPairs = @(
  "DEPTH_LOCAL_ONLY=1",
  "DEPTH_MIN_STABILIZATION_ANCHORS=8",
  "DEPTH_MAX_ALLOWED_RELATIVE_ERROR=0.30",
  "DEPTH_MAX_SHIFT_NORM_FOR_WEAK_ANCHORS=0.06",
  "JEPA_ENABLED=1",
  "JEPA_USE_FOR_CONTACT=1",
  "JEPA_MAX_OBJECTS_PER_FRAME=2",
  "TEMPORAL_HEAD_ENABLED=1",
  "TEMPORAL_HEAD_MODEL_PATH=$repoRoot\\world_model\\models\\temporal_interaction_head_sophie.pt",
  "TEMPORAL_HEAD_CONTACT_THRESHOLD=0.20",
  "TEMPORAL_HEAD_PLACEMENT_THRESHOLD=0.45",
  "TEMPORAL_HEAD_TARGET_TRAY_THRESHOLD=0.45",
  "JEPA_EVENT_CONTACT_THRESHOLD=0.75",
  "JEPA_EVENT_RELEASE_THRESHOLD=0.75",
  "JEPA_EVENT_RELEASE_UI_THRESHOLD=0.35",
  "JEPA_EVENT_MIN_HOLD_S=1.2",
  "JEPA_EVENT_RELEASE_LINGER_S=2.2",
  "GUIDANCE_RELEASE_SPEECH_ENABLED=0",
  "GEMINI_LABEL_PROMPT_MODE=strict",
  "PERCEPTION_DETECTOR_MODE=yolo",
  "YOLO_MODEL_PATH=$repoRoot\\world_model\\models\\yolov8n.pt",
  "YOLO_SEG_MODEL_PATH=$repoRoot\\world_model\\models\\yolov8n-seg.pt",
  "FASTSAM_S_MODEL_PATH=$repoRoot\\world_model\\models\\FastSAM-s.pt",
  "FASTSAM_X_MODEL_PATH=$repoRoot\\world_model\\models\\FastSAM-x.pt",
  "PERCEPTION_SEGMENTATION_BACKEND=yolo-seg",
  "PERCEPTION_VLM_REFINER_ENABLED=0",
  "PERCEPTION_VLM_REFINER_EVERY_N_KEYFRAMES=4",
  "SLAM_SEMANTIC_STABILIZATION=1",
  "COLMAP_DEPTH_PRIOR_ENABLED=1",
  "COLMAP_DEPTH_PRIOR_SPARSE_TXT_DIR=$repoRoot\\world_model\\data\\colmap_scene_sophie_static30\\sparse_txt",
  "COLMAP_DEPTH_PRIOR_FPS=3.0",
  "COLMAP_DEPTH_PRIOR_RUNTIME_FPS=5.0",
  "COLMAP_DEPTH_PRIOR_EXCLUDE_DYNAMIC=1",
  "FEATURE_BACKEND=hybrid",
  "DEMO_MULTI_OBJECT_TRACKING=1",
  "DEMO_MAX_TRACKED_OBJECTS=6",
  "DEMO_TRACKED_OBJECT_MIN_CONFIDENCE=0.20",
  "DEMO_TRACKED_OBJECT_LABEL_DENYLIST=dining table,chair,couch,tv,potted plant",
  "DEMO_TRACKED_OBJECT_LABELS=cup,mug,bowl,vase,bottle,book,cell phone,laptop,mouse,knife,scissors,banana,apple,orange",
  "DEMO_MAX_OBJECT_DINO_EMBEDS_PER_UPDATE=1",
  "PERCEPTION_ADD_UNMATCHED_SEG_OBJECTS=1",
  "PERCEPTION_UNMATCHED_SEG_MAX=6",
  "PERCEPTION_UNMATCHED_SEG_MIN_CONF=0.15",
  "PERCEPTION_UNMATCHED_SEG_MIN_AREA=0.004",
  "PERCEPTION_UNMATCHED_SEG_MAX_AREA=0.25",
  "MANIP_MIN_MOVE_DISTANCE_M=0.05",
  "MANIP_REL_NEAR_XY_M=0.12",
  "MANIP_REL_BEHIND_DZ_M=0.05",
  "MANIP_REL_NEAR_3D_M=0.18",
  "HAND_TRACKING_ENABLED=1",
  "HAND_MAX_HANDS=2",
  "HAND_MIN_DET_CONF=0.35",
  "HAND_MIN_TRACK_CONF=0.35",
  "HAND_EMA_ALPHA=0.55",
  "HAND_FORCE_SIDE=right",
  "HAND_INTERACTION_SIDES=right",
  "HAND_METRIC_PRIOR_PALM_WIDTH_M=0.085",
  "HAND_FINGER_RADIUS_M=0.009",
  "HAND_THUMB_RADIUS_M=0.010",
  "HAND_PALM_CAPSULE_RADIUS_M=0.018",
  "SLAM_HAND_DYNAMIC_MASK_ENABLED=1",
  "SLAM_HAND_DYNAMIC_MASK_RADIUS_NORM=0.04",
  "HAND_CONTACT_DISTANCE_M=0.12",
  "HAND_NEAR_DISTANCE_M=0.16",
  "HAND_CONTACT_2D_DISTANCE_NORM=0.08",
  "HAND_CONTACT_2D_OVERLAP_MIN=0.025",
  "HAND_CONTACT_2D_EFFECTIVE_DISTANCE_M=0.075",
  "HAND_CONTACT_ENTER_FRAMES=2",
  "HAND_CONTACT_EXIT_FRAMES=3",
  "HAND_LABEL_CONTACT_ENTER_FRAMES=apple:1,banana:1,orange:1",
  "HAND_TOUCH_DISTANCE_M=0.055",
  "HAND_TOUCH_START_DISTANCE_M=0.065",
  "HAND_TOUCH_END_DISTANCE_M=0.095",
  "HAND_LABEL_TOUCH_DISTANCE_M=apple:0.08,banana:0.08,orange:0.08",
  "HAND_LABEL_TOUCH_START_DISTANCE_M=apple:0.09,banana:0.09,orange:0.09",
  "HAND_LABEL_TOUCH_END_DISTANCE_M=apple:0.12,banana:0.12,orange:0.12",
  "DEMO_MIN_POSE_SCORE=0.40",
  "DEMO_MIN_MAP_SCORE=0.40",
  "DEMO_MIN_HAND_SCORE=0.35",
  "DEMO_REQUIRE_TARGET_HAND_ENGAGEMENT=0",
  "DEMO_MIN_INTERACTION_SCORE=0.30",
  "DEMO_MIN_OVERALL_SCORE=0.45",
  "WORLD_MATCH_MAX_DIST=0.34",
  "WORLD_MATCH_CONFIDENCE_WEIGHT=0.05",
  "WORLD_MATCH_EMBEDDING_WEIGHT=0.55",
  "WORLD_MATCH_3D_WEIGHT=0.30",
  "WORLD_MATCH_MAX_3D_DIST=0.45",
  "WORLD_OBJECT_MEMORY_SECONDS=45",
  "SLAM_KEYFRAME_MIN_INTERVAL=7",
  "SLAM_KEYFRAME_MIN_TRANSLATION=0.016",
  "SLAM_KEYFRAME_MIN_VISIBLE=28",
  "SLAM_MAX_KEYFRAMES=32",
  "SLAM_MAX_OBSERVATIONS_PER_LANDMARK=12",
  "SLAM_MAX_LANDMARKS=2800",
  "SLAM_VISIBLE_MAP_EXPORT_LIMIT=760",
  "SLAM_LOCAL_MAP_MAX_KEYFRAMES=6",
  "SLAM_LOCAL_MAP_MAX_LANDMARKS=320",
  "SLAM_SLIDING_BA_MAX_KEYFRAMES=5",
  "SLAM_SLIDING_BA_MAX_LANDMARKS=90",
  "SLAM_SLIDING_BA_MAX_RESIDUAL_OBS=320",
  "SLAM_BA_LITE_MAX_UPDATES_PER_KEYFRAME=120",
  "SLAM_COVISIBILITY_MIN_SHARED=6",
  "SLAM_MAX_TRACK_POINTS=320",
  "SLAM_MIN_TRACK_POINTS=100",
  "SLAM_FEATURE_QUALITY_LEVEL=0.004",
  "SLAM_FEATURE_MIN_DISTANCE=6",
  "SLAM_PERSISTENT_MAP_EXPORT_LIMIT=800",
  "SLAM_PNP_MAX_ANCHORS=420",
  "SLAM_PNP_MIN_INLIERS=6",
  "SLAM_PNP_LOCK_MIN_INLIERS=12",
  "SLAM_PNP_LOCK_MAX_REPROJECTION_ERROR=6.0",
  "SLAM_TRIANGULATED_MIN_BASELINE=0.02",
  "SLAM_TRIANGULATION_POSITION_BLEND=0.40",
  "SLAM_ESSENTIAL_FALLBACK_SCALE_DAMPING=0.55",
  "SLAM_ESSENTIAL_FALLBACK_MAX_TRANSLATION=0.03",
  "SLAM_ESSENTIAL_ROTATION_ONLY_AFTER_MISSED_PNP=10",
  "SLAM_GEOMETRY_REOBSERVATION_SIMILARITY=0.72",
  "SLAM_GEOMETRY_REOBSERVATION_DISTANCE_PX=140",
  "SLAM_PROTECTED_GEOMETRY_MIN_HITS=6",
  "SLAM_PROTECTED_GEOMETRY_MISSING_MULTIPLIER=2.0",
  "DEMO_REQUIRE_HANDS_FOR_GUIDANCE=0"
)

if ($Biberon) {
  $biberonPrior = Join-Path $repoRoot "world_model\data\colmap_scene_biberon_static40_abs2\sparse_txt"
  $biberonTemporal = Join-Path $repoRoot "world_model\models\temporal_interaction_head_biberon.pt"
  $envPairs = $envPairs | ForEach-Object {
    if ($_ -like "COLMAP_DEPTH_PRIOR_SPARSE_TXT_DIR=*") { "COLMAP_DEPTH_PRIOR_SPARSE_TXT_DIR=$biberonPrior" }
    elseif ($_ -like "TEMPORAL_HEAD_MODEL_PATH=*") { "TEMPORAL_HEAD_MODEL_PATH=$biberonTemporal" }
    elseif ($_ -like "DEMO_TRACKED_OBJECT_LABELS=*") { "DEMO_TRACKED_OBJECT_LABELS=cup,mug,bottle,book,cell phone,mouse,bowl,vase" }
    elseif ($_ -like "MANIP_MIN_MOVE_DISTANCE_M=*") { "MANIP_MIN_MOVE_DISTANCE_M=0.035" }
    elseif ($_ -like "HAND_LABEL_CONTACT_ENTER_FRAMES=*") { "HAND_LABEL_CONTACT_ENTER_FRAMES=apple:1,banana:1,orange:1,bottle:1" }
    elseif ($_ -like "HAND_LABEL_TOUCH_DISTANCE_M=*") { "HAND_LABEL_TOUCH_DISTANCE_M=apple:0.08,banana:0.08,orange:0.08,bottle:0.09" }
    elseif ($_ -like "HAND_LABEL_TOUCH_START_DISTANCE_M=*") { "HAND_LABEL_TOUCH_START_DISTANCE_M=apple:0.09,banana:0.09,orange:0.09,bottle:0.10" }
    elseif ($_ -like "HAND_LABEL_TOUCH_END_DISTANCE_M=*") { "HAND_LABEL_TOUCH_END_DISTANCE_M=apple:0.12,banana:0.12,orange:0.12,bottle:0.13" }
    else { $_ }
  }
  $envPairs += "NEXT_PUBLIC_DEFAULT_SCENE_VIDEO=scene_biberon.mp4"
}

if ($Sophie) {
  $sophiePrior = Join-Path $repoRoot "world_model\data\colmap_scene_sophie_static30\sparse_txt"
  $sophieTemporal = Join-Path $repoRoot "world_model\models\temporal_interaction_head_sophie.pt"
  $envPairs = $envPairs | ForEach-Object {
    if ($_ -like "COLMAP_DEPTH_PRIOR_SPARSE_TXT_DIR=*") { "COLMAP_DEPTH_PRIOR_SPARSE_TXT_DIR=$sophiePrior" }
    elseif ($_ -like "TEMPORAL_HEAD_MODEL_PATH=*") { "TEMPORAL_HEAD_MODEL_PATH=$sophieTemporal" }
    elseif ($_ -like "DEMO_TRACKED_OBJECT_LABELS=*") { "DEMO_TRACKED_OBJECT_LABELS=bottle,mouse,donut,toy,cup" }
    elseif ($_ -like "MANIP_MIN_MOVE_DISTANCE_M=*") { "MANIP_MIN_MOVE_DISTANCE_M=0.035" }
    elseif ($_ -like "HAND_LABEL_CONTACT_ENTER_FRAMES=*") { "HAND_LABEL_CONTACT_ENTER_FRAMES=bottle:1,donut:1,mouse:1,toy:1" }
    elseif ($_ -like "HAND_LABEL_TOUCH_DISTANCE_M=*") { "HAND_LABEL_TOUCH_DISTANCE_M=bottle:0.12,donut:0.12,mouse:0.11,toy:0.11" }
    elseif ($_ -like "HAND_LABEL_TOUCH_START_DISTANCE_M=*") { "HAND_LABEL_TOUCH_START_DISTANCE_M=bottle:0.13,donut:0.13,mouse:0.12,toy:0.12" }
    elseif ($_ -like "HAND_LABEL_TOUCH_END_DISTANCE_M=*") { "HAND_LABEL_TOUCH_END_DISTANCE_M=bottle:0.16,donut:0.16,mouse:0.15,toy:0.15" }
    else { $_ }
  }
  $envPairs += "DEMO_SCENE_PROFILE=sophie"
  $envPairs += "NEXT_PUBLIC_DEFAULT_SCENE_VIDEO=scene_sophie.mp4"
  $envPairs += "SCENE_TARGET_LABELS=mat,tray"
  $envPairs += "SCENE_MOVABLE_LABELS=bottle,baby bottle,toy giraffe"
  $envPairs += "SCENE_FORBIDDEN_LABELS=coaster,dish,plate,cup,mug,mouse,donut"
  $envPairs += "DEMO_TRANSFER_TARGETS=mat,tray"
  $envPairs += "STATIC_TARGET_LABELS=tray,mat,black mat,table mat,placemat,dish,plate,unknown_seg"
  $envPairs += "STATIC_TARGET_INFER_LARGE_LABEL=tray"
  $envPairs += "STATIC_TARGET_INFER_SMALL_LABEL="
  $envPairs += "STATIC_TARGET_INFER_DARK_LABEL=mat"
  $envPairs += "STATIC_TARGET_DARK_LUMA_MAX=85"
  $envPairs += "STATIC_TARGET_LOCK_UNKNOWN=0"
  $envPairs += "STATIC_TARGET_HITS_MIN=2"
  $envPairs += "OBJECT_SURFACE_STATIC_SECONDS=30"
}

if ($xfeatRepo) {
  $envPairs += "XFEAT_REPO=$xfeatRepo"
}

if ($HighPointBudget) {
  $envPairs = $envPairs | ForEach-Object {
    if ($_ -like "SLAM_MAX_LANDMARKS=*") { "SLAM_MAX_LANDMARKS=2600" }
    elseif ($_ -like "SLAM_VISIBLE_MAP_EXPORT_LIMIT=*") { "SLAM_VISIBLE_MAP_EXPORT_LIMIT=720" }
    elseif ($_ -like "SLAM_LOCAL_MAP_MAX_LANDMARKS=*") { "SLAM_LOCAL_MAP_MAX_LANDMARKS=520" }
    elseif ($_ -like "SLAM_SLIDING_BA_MAX_LANDMARKS=*") { "SLAM_SLIDING_BA_MAX_LANDMARKS=140" }
    elseif ($_ -like "SLAM_MAX_TRACK_POINTS=*") { "SLAM_MAX_TRACK_POINTS=520" }
    elseif ($_ -like "SLAM_MIN_TRACK_POINTS=*") { "SLAM_MIN_TRACK_POINTS=140" }
    elseif ($_ -like "SLAM_PERSISTENT_MAP_EXPORT_LIMIT=*") { "SLAM_PERSISTENT_MAP_EXPORT_LIMIT=1600" }
    elseif ($_ -like "SLAM_PNP_MAX_ANCHORS=*") { "SLAM_PNP_MAX_ANCHORS=380" }
    elseif ($_ -like "SLAM_BA_LITE_MAX_UPDATES_PER_KEYFRAME=*") { "SLAM_BA_LITE_MAX_UPDATES_PER_KEYFRAME=180" }
    else { $_ }
  }
}

if ($HybridQuality) {
  $envPairs = $envPairs | ForEach-Object {
    if ($_ -like "SLAM_MAX_LANDMARKS=*") { "SLAM_MAX_LANDMARKS=2400" }
    elseif ($_ -like "SLAM_VISIBLE_MAP_EXPORT_LIMIT=*") { "SLAM_VISIBLE_MAP_EXPORT_LIMIT=640" }
    elseif ($_ -like "SLAM_LOCAL_MAP_MAX_LANDMARKS=*") { "SLAM_LOCAL_MAP_MAX_LANDMARKS=480" }
    elseif ($_ -like "SLAM_SLIDING_BA_MAX_LANDMARKS=*") { "SLAM_SLIDING_BA_MAX_LANDMARKS=120" }
    elseif ($_ -like "SLAM_MAX_TRACK_POINTS=*") { "SLAM_MAX_TRACK_POINTS=460" }
    elseif ($_ -like "SLAM_MIN_TRACK_POINTS=*") { "SLAM_MIN_TRACK_POINTS=130" }
    elseif ($_ -like "SLAM_PERSISTENT_MAP_EXPORT_LIMIT=*") { "SLAM_PERSISTENT_MAP_EXPORT_LIMIT=1400" }
    elseif ($_ -like "SLAM_PNP_MAX_ANCHORS=*") { "SLAM_PNP_MAX_ANCHORS=340" }
    elseif ($_ -like "SLAM_BA_LITE_MAX_UPDATES_PER_KEYFRAME=*") { "SLAM_BA_LITE_MAX_UPDATES_PER_KEYFRAME=170" }
    elseif ($_ -like "SLAM_KEYFRAME_MIN_INTERVAL=*") { "SLAM_KEYFRAME_MIN_INTERVAL=9" }
    elseif ($_ -like "SLAM_KEYFRAME_MIN_TRANSLATION=*") { "SLAM_KEYFRAME_MIN_TRANSLATION=0.022" }
    else { $_ }
  }
}

if ($DenseObjects) {
  $envPairs = $envPairs | ForEach-Object {
    if ($_ -like "PERCEPTION_SEGMENTATION_BACKEND=*") { "PERCEPTION_SEGMENTATION_BACKEND=yolo-seg" }
    elseif ($_ -like "DEMO_MAX_TRACKED_OBJECTS=*") { "DEMO_MAX_TRACKED_OBJECTS=14" }
    elseif ($_ -like "DEMO_TRACKED_OBJECT_MIN_CONFIDENCE=*") { "DEMO_TRACKED_OBJECT_MIN_CONFIDENCE=0.15" }
    elseif ($_ -like "DEMO_TRACKED_OBJECT_LABELS=*") { "DEMO_TRACKED_OBJECT_LABELS=" }
    elseif ($_ -like "PERCEPTION_UNMATCHED_SEG_MAX=*") { "PERCEPTION_UNMATCHED_SEG_MAX=6" }
    elseif ($_ -like "PERCEPTION_UNMATCHED_SEG_MIN_CONF=*") { "PERCEPTION_UNMATCHED_SEG_MIN_CONF=0.12" }
    elseif ($_ -like "PERCEPTION_UNMATCHED_SEG_MIN_AREA=*") { "PERCEPTION_UNMATCHED_SEG_MIN_AREA=0.003" }
    elseif ($_ -like "PERCEPTION_UNMATCHED_SEG_MAX_AREA=*") { "PERCEPTION_UNMATCHED_SEG_MAX_AREA=0.30" }
    else { $_ }
  }
}

$useTrainedTemporal = $TrainedTemporal.IsPresent

if ($useTrainedTemporal) {
  $trainedModel = Join-Path $repoRoot "world_model\models\temporal_interaction_head_sophie.pt"
  if ($Biberon) {
    $trainedModel = Join-Path $repoRoot "world_model\models\temporal_interaction_head_biberon.pt"
  }
  if ($Sophie) {
    $trainedModel = Join-Path $repoRoot "world_model\models\temporal_interaction_head_sophie.pt"
  }
  if (Test-Path -LiteralPath $trainedModel -PathType Leaf) {
    $envPairs = $envPairs | ForEach-Object {
      if ($_ -like "TEMPORAL_HEAD_MODEL_PATH=*") { "TEMPORAL_HEAD_MODEL_PATH=$trainedModel" } else { $_ }
    }
  } else {
    Write-Host "Trained temporal checkpoint not found: $trainedModel" -ForegroundColor Yellow
  }
}

if ($RequireHands) {
  $envPairs = $envPairs | ForEach-Object {
    if ($_ -like "DEMO_REQUIRE_HANDS_FOR_GUIDANCE=*") { "DEMO_REQUIRE_HANDS_FOR_GUIDANCE=1" } else { $_ }
  }
}

$envScript = ($envPairs | ForEach-Object {
  $parts = $_.Split("=", 2)
  "`$env:$($parts[0])='$($parts[1])'"
}) -join "; "

Write-Host "Starting stable demo profile..." -ForegroundColor Cyan
Write-Host "Repo: $repoRoot"
Write-Host "Require hands for guidance: $($RequireHands.IsPresent)"
Write-Host "High point budget: $($HighPointBudget.IsPresent)"
Write-Host "Hybrid quality profile: $($HybridQuality.IsPresent)"
Write-Host "Dense objects profile: $($DenseObjects.IsPresent)"
Write-Host "Trained temporal profile: $useTrainedTemporal"
Write-Host "Biberon profile: $($Biberon.IsPresent)"
Write-Host "Sophie profile: $($Sophie.IsPresent)"

function Stop-PortOwner {
  param(
    [Parameter(Mandatory = $true)][int]$Port,
    [Parameter(Mandatory = $true)][string]$Name
  )
  $connections = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
  if (-not $connections) { return }

  $owners = $connections | Select-Object -ExpandProperty OwningProcess -Unique
  foreach ($owner in $owners) {
    if ($owner -eq $PID) { continue }
    $process = Get-Process -Id $owner -ErrorAction SilentlyContinue
    if (-not $process) { continue }
    Write-Host "Stopping stale $Name on port $Port (PID $owner, $($process.ProcessName))..." -ForegroundColor Yellow
    Stop-Process -Id $owner -Force -ErrorAction SilentlyContinue
  }
}

Stop-PortOwner -Port 8090 -Name "world-model server"
Stop-PortOwner -Port 8081 -Name "Gemini bridge"
Stop-PortOwner -Port 3000 -Name "Next.js UI"

$nextLock = Join-Path $repoRoot ".next\dev\lock"
if (Test-Path -LiteralPath $nextLock -PathType Leaf) {
  Remove-Item -LiteralPath $nextLock -Force -ErrorAction SilentlyContinue
}

$serverCmd = "$envScript; Set-Location '$repoRoot'; python '$serverPath'"
$uiCmd = "$envScript; Set-Location '$repoRoot'; npm run dev"

$geminiApiKey = $env:GEMINI_API_KEY
if (-not $geminiApiKey) {
  $envFile = Join-Path $repoRoot ".env.local"
  if (Test-Path -LiteralPath $envFile -PathType Leaf) {
    $geminiLine = Get-Content -LiteralPath $envFile | Where-Object { $_ -match '^\s*GEMINI_API_KEY\s*=' } | Select-Object -First 1
    if ($geminiLine) {
      $geminiApiKey = ($geminiLine -split "=", 2)[1].Trim()
      if (($geminiApiKey.StartsWith("'") -and $geminiApiKey.EndsWith("'")) -or ($geminiApiKey.StartsWith('"') -and $geminiApiKey.EndsWith('"'))) {
        $geminiApiKey = $geminiApiKey.Substring(1, $geminiApiKey.Length - 2)
      }
    }
  }
}
if ($geminiApiKey) {
  $bridgeCmd = "$envScript; `$env:GEMINI_API_KEY='$geminiApiKey'; Set-Location '$repoRoot'; node '$geminiBridgePath'"
} else {
  $bridgeCmd = $null
}

if ($PSCmdlet.ShouldProcess("World model server + Next.js UI", "Launch stable demo profile")) {
  Start-Process powershell.exe -ArgumentList "-NoExit", "-Command", $serverCmd | Out-Null
  Start-Sleep -Seconds 2
  if ($bridgeCmd) {
    Start-Process powershell.exe -ArgumentList "-NoExit", "-Command", $bridgeCmd | Out-Null
    Start-Sleep -Seconds 1
  } else {
    Write-Host "GEMINI_API_KEY not found in env or .env.local; Gemini bridge was not started." -ForegroundColor Yellow
  }
  Start-Process powershell.exe -ArgumentList "-NoExit", "-Command", $uiCmd | Out-Null
  Write-Host "Launched world model server, Gemini bridge (if configured), and Next.js UI in new PowerShell windows." -ForegroundColor Green
  Write-Host "Use Ctrl+C in each window to stop."
}
else {
  Write-Host "Dry run only (-WhatIf): no process was launched." -ForegroundColor Yellow
}
