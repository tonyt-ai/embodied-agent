[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

$env:PERCEPTION_SEGMENTATION_BACKEND = "fastsam-s"
$env:DEMO_SCENE_PROFILE = "sophie"
$env:DEMO_TRANSFER_TARGETS = "mat,tray"
$env:STATIC_TARGET_LABELS = "tray,mat,black mat,table mat,placemat,dish,plate,unknown_seg"
$env:STATIC_TARGET_INFER_LARGE_LABEL = "tray"
$env:STATIC_TARGET_INFER_SMALL_LABEL = ""
$env:STATIC_TARGET_INFER_DARK_LABEL = "mat"
$env:STATIC_TARGET_DARK_LUMA_MAX = "85"
$env:STATIC_TARGET_LOCK_UNKNOWN = "0"
$env:HAND_LABEL_CONTACT_ENTER_FRAMES = "bottle:1,donut:1,mouse:1,toy:1"
$env:HAND_LABEL_TOUCH_DISTANCE_M = "bottle:0.12,donut:0.12,mouse:0.11,toy:0.11"
$env:HAND_LABEL_TOUCH_START_DISTANCE_M = "bottle:0.13,donut:0.13,mouse:0.12,toy:0.12"
$env:HAND_LABEL_TOUCH_END_DISTANCE_M = "bottle:0.16,donut:0.16,mouse:0.15,toy:0.15"

$testRows = Join-Path $repoRoot "world_model\data\temporal_head_test_rows_sophie.json"
$modelPath = Join-Path $repoRoot "world_model\models\temporal_interaction_head_sophie.pt"
$reportPath = Join-Path $repoRoot "world_model\data\temporal_head_eval_sophie_test_latest.json"
$collectLog = Join-Path $repoRoot "world_model\data\jepa_sophie_test_collect_stderr.log"

$prevErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
python (Join-Path $repoRoot "world_model\train_temporal_heads.py") `
  --video (Join-Path $repoRoot "public\scene_sophie_test.mp4") `
  --out-model $modelPath `
  --out-json $testRows `
  --horizon 10 `
  --no-train `
  2> $collectLog
if ($LASTEXITCODE -ne 0) {
  $ErrorActionPreference = $prevErrorActionPreference
  throw "Test row collection failed with exit code $LASTEXITCODE. See $collectLog"
}

python (Join-Path $repoRoot "world_model\eval_temporal_head.py") `
  --rows $testRows `
  --model $modelPath `
  --output $reportPath `
  --fit-epochs 120
if ($LASTEXITCODE -ne 0) {
  $ErrorActionPreference = $prevErrorActionPreference
  throw "Temporal head test eval failed with exit code $LASTEXITCODE."
}
$ErrorActionPreference = $prevErrorActionPreference
