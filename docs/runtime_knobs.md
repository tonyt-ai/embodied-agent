# Runtime Knobs

This page is the tuning drawer. The recommended demo path should stay simple:

```bash
npm run demo:sophie
npm run jepa:train
npm run jepa:test
```

Most defaults are set by `scripts/start_demo_stable.ps1`, `package.json`, and the `world_model/` modules. Only tune these when debugging a specific failure mode.

## Launch Profiles

```bash
npm run demo
npm run demo:sophie
```

For lower-level profile experiments, call `scripts/start_demo_stable.ps1` directly
with flags such as `-DenseObjects`, `-TrainedTemporal`, `-Sophie`,
`-RequireHands`, `-HighPointBudget`, or `-HybridQuality`.

## Sophie Training Profile

`npm run jepa:train` currently delegates to `npm run jepa:sophie:train`.
`npm run jepa:test` builds held-out rows from `public/scene_sophie_test.mp4`
without updating the model, then evaluates the saved Sophie temporal head.
The eval JSON includes a `row_fit_upper_bound` section. That model is fit on
the same rows being measured, so use it only to check whether the rows are
learnable; use `saved_model` for the real checkpoint score.

The live demo uses the temporal head as an anticipatory signal, not as the
authoritative contact detector. Current Sophie defaults are:

```text
TEMPORAL_HEAD_CONTACT_THRESHOLD=0.20
TEMPORAL_HEAD_PLACEMENT_THRESHOLD=0.45
```

Geometry/contact thresholds still decide actual grasp and release events.

The Sophie profile intentionally keeps contact thresholds on raw detector labels:

```text
bottle
donut
mouse
toy
```

Semantic names such as `baby bottle` and `Sophie the giraffe` should come from Gemini/refined labels, not from detector-label remapping in the launch command.

## Perception

Common controls:

```text
PERCEPTION_SEGMENTATION_BACKEND
PERCEPTION_DETECTOR_MODE
PERCEPTION_ADD_UNMATCHED_SEG_OBJECTS
PERCEPTION_UNMATCHED_SEG_MAX
PERCEPTION_UNMATCHED_SEG_MIN_CONF
PERCEPTION_UNMATCHED_SEG_MIN_AREA
PERCEPTION_UNMATCHED_SEG_MAX_AREA
YOLO_MODEL_PATH
YOLO_SEG_MODEL_PATH
FASTSAM_S_MODEL_PATH
FASTSAM_X_MODEL_PATH
```

The Sophie training command uses `fastsam-s` for segmentation.

## V-JEPA2.1 / Appearance Features

The encoder auto-detects common local V-JEPA2.1 checkouts such as
`C:\code\vjepa2-main`. If V-JEPA2.1 cannot be loaded, it falls back to DINOv2
crop embeddings.

Useful controls:

```text
JEPA_ENABLED
VJEPA2_REPO
VJEPA2_MODEL
VJEPA2_CHECKPOINT
JEPA_MODEL_ID
JEPA_OUT_DIM
```

## Gemini Label Refinement

Useful controls:

```text
GEMINI_API_KEY
GEMINI_LABEL_PROMPT_MODE
PERCEPTION_VLM_REFINER_ENABLED
PERCEPTION_VLM_REFINER_EVERY_N_KEYFRAMES
```

Use strict/open prompting for non-cheating label refinement. Guided prompting is useful for debugging but should not be used as evidence that perception discovered the object identity.

## Hand Tracking And Contact

Common controls:

```text
HAND_TRACKING_ENABLED
HAND_MAX_HANDS
HAND_MIN_DET_CONF
HAND_MIN_TRACK_CONF
HAND_EMA_ALPHA
HAND_FORCE_SIDE
HAND_LABEL_CONTACT_ENTER_FRAMES
HAND_LABEL_TOUCH_DISTANCE_M
HAND_LABEL_TOUCH_START_DISTANCE_M
HAND_LABEL_TOUCH_END_DISTANCE_M
```

These are physical/contact calibration knobs, not semantic-label shortcuts. For Sophie, they should stay on raw detector families unless you are running an explicit calibration experiment.

## SLAM And Persistent Geometry

The Sophie demo uses an offline COLMAP sparse SfM prior together with online SLAM/depth.
The prior is a practical scaffold for commodity webcam input; better depth or
stereo sensing could relax this requirement in future work.

Common controls:

```text
SLAM_KEYFRAME_MIN_INTERVAL
SLAM_KEYFRAME_MIN_TRANSLATION
SLAM_KEYFRAME_MIN_VISIBLE
SLAM_MAX_KEYFRAMES
SLAM_MAX_LANDMARKS
SLAM_VISIBLE_MAP_EXPORT_LIMIT
SLAM_LOCAL_MAP_MAX_KEYFRAMES
SLAM_LOCAL_MAP_MAX_LANDMARKS
SLAM_PNP_MIN_INLIERS
SLAM_PNP_LOCK_MIN_INLIERS
SLAM_PNP_LOCK_MAX_REPROJECTION_ERROR
SLAM_HAND_DYNAMIC_MASK_ENABLED
SLAM_HAND_DYNAMIC_MASK_RADIUS_NORM
```

Depth scaling and the COLMAP prior are used to keep sparse landmarks, surface estimates, static targets, and online camera motion in a consistent frame.

## Demo Gating

Common controls:

```text
DEMO_REQUIRE_HANDS_FOR_GUIDANCE
DEMO_REQUIRE_TARGET_HAND_ENGAGEMENT
DEMO_MIN_INTERACTION_SCORE
DEMO_MIN_POSE_SCORE
DEMO_MIN_MAP_SCORE
DEMO_MIN_HAND_SCORE
DEMO_MIN_OVERALL_SCORE
```

Use these only to diagnose readiness and timing. The wow demo should rely on the stable Sophie profile rather than manual knob combinations.

## Logging

MediaPipe native logs are quieted by default in `world_model/hands.py`.

To re-enable verbose MediaPipe logs:

```text
MEDIAPIPE_LOG_LEVEL=0
```

DINOv2 xFormers warnings are optional acceleration warnings. Install xFormers only if DINO inference is a measured bottleneck and your PyTorch/CUDA versions are compatible.
