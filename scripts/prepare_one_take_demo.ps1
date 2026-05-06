param(
  [string]$Source = "public\scene_sophie.mp4",
  [double]$StaticSeconds = 30.0,
  [double]$IdentitySeconds = 0.0,
  [double]$InteractionStartSeconds = 0.0,
  [int]$StaticFrameStride = 4
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

python scripts\prepare_demo_segments.py `
  --source $Source `
  --static-out public\scene_sophie_static_30s.mp4 `
  --identity-out public\scene_sophie_identity_30s.mp4 `
  --interaction-out public\scene_sophie_interactions.mp4 `
  --static-seconds $StaticSeconds `
  --identity-seconds $IdentitySeconds `
  --interaction-start-seconds $InteractionStartSeconds `
  --manifest world_model\data\demo_segments_manifest_sophie.json

python world_model\validate_static_targets.py `
  --video public\scene_sophie_static_30s.mp4 `
  --seconds $StaticSeconds `
  --frame-stride $StaticFrameStride `
  --out-json world_model\data\static_targets_validation_static_segment_dense.json

python world_model\eval_tracking_embeddings_ab.py `
  --video public\scene_sophie_identity_30s.mp4 `
  --max-frames 100 `
  --frame-stride 5 `
  --enable-segmentation `
  --add-unmatched-seg-objects `
  --output world_model\data\tracking_embeddings_ab_identity_latest.json

Write-Host "Prepared one-take demo assets."
Write-Host "Manifest: world_model\data\demo_segments_manifest.json"
Write-Host "Static validation: world_model\data\static_targets_validation_static_segment_dense.json"
Write-Host "DINO/HSV A/B: world_model\data\tracking_embeddings_ab_identity_latest.json"
