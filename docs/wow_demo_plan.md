# Sophie World-Model Demo Plan

The demo is about grounded manipulation understanding, not 2D object steering.

## Scenario

Use:

- `public/scene_sophie.mp4` for training.
- `public/scene_sophie_test.mp4` for held-out evaluation.

The scene starts with a static bootstrap period, then a right hand repeatedly moves a baby bottle and Sophie the giraffe between a table mat and a tray.

The expected public story:

```text
I build a physical 3D world from SfM/COLMAP plus online SLAM/depth, keep targets
persistent, track the hand and objects, infer contact, predict what will be
moved and where, ask Gemini to name weakly detected objects, then speak and
highlight the prediction in real time.
```

## Under-The-Hood Models

- YOLOv8: light detection and initial detector labels.
- FastSAM-s: segmentation regions for the Sophie profile.
- V-JEPA2.1: primary visual embedding path for object tracking, re-ID, temporal memory, and prediction features.
- DINOv2: fallback/baseline crop embeddings for tracking and re-ID.
- MediaPipe Hands: hand landmarks before 3D lifting.
- Depth Anything: monocular surface estimate via Hugging Face Transformers when available, scaled to sparse anchors; heuristic fallback if unavailable.
- COLMAP sparse SfM: static/offline scene prior from commodity webcam video.
- Online sparse SLAM/PnP: camera motion and persistent 3D alignment while interaction is running.
- PyTorch interaction-conditioned temporal head: JEPA-like dynamics/prediction over grounded hand-object-target tokens.
- Gemini + LiveAvatar: label refinement, speech, and avatar output.

## What To Show

- Static 3D prior points and current SLAM/depth surface.
- Persistent mat/tray targets reprojected into the live video.
- Right hand in the 3D/debug maps.
- Object tracks that persist across detector flicker and weak labels.
- Grab attention heatmap over the likely object.
- Destination heatmap over mat or tray.
- Voice feedback from first hand detection, approach, grasp, predicted destination, and release.
- Gemini-refined labels such as `baby bottle` and `Sophie the giraffe`.

## LLM Role

Gemini has two demo-facing jobs:

- refine weak detector labels from crops/context,
- speak as the assistant through the live avatar.

It should not be presented as the source of 3D truth. Geometry, SLAM, depth,
tracking, and contact logic provide grounded state and teacher signals; the
learned temporal world model predicts contact, placement, motion, and future
latents over that state. Gemini turns grounded state and predictions into
human-readable labels and timely language.

## Current JEPA Interpretation

Geometry is the teacher and the authority for contact/release. The interaction-conditioned temporal head predicts:

- future contact probability,
- placement/destination likelihood,
- object motion delta,
- future visual latent state.

This is the part closest to JEPA-AC, but it is not a full action-conditioned
JEPA-AC planner yet. The current head is conditioned on observed interaction
state: hand/object embeddings, 3D distance, motion, and contact history. A full
action-conditioned version would add explicit action/control tokens and roll the
state forward under candidate actions.

This is a hybrid explicit/latent world model. Persistence is explicit: objects,
hands, mat, and tray maintain IDs and 3D anchors across time. V-JEPA2.1
embeddings provide appearance memory and prediction features over those grounded
tokens, with DINOv2 available as a fallback/baseline. That makes the result
inspectable and physically grounded, while still using latent prediction for
futures.

For live feedback, use the temporal head as an early cue. Current Sophie thresholds:

```text
TEMPORAL_HEAD_CONTACT_THRESHOLD=0.20
TEMPORAL_HEAD_PLACEMENT_THRESHOLD=0.45
```

Held-out test performance is modest but useful as an anticipatory signal. More varied training captures should improve generalization more than tiny architecture changes.

## Evaluation Loop

```powershell
npm run jepa:train
npm run jepa:eval
npm run jepa:test
python world_model/validate_interactions.py --video public/scene_sophie.mp4 --profile sophie
python world_model/validate_interactions.py --video public/scene_sophie_test.mp4 --profile sophie
```

Read `saved_model` as real checkpoint performance. Read `row_fit_upper_bound` only as a sanity check that the rows contain learnable signal.

## Improvement Plan

1. Capture more Sophie-scene training videos.
   Vary lighting, viewpoint, hand speed, object rotation, grasp style, and mat/tray starting side.

2. Train on multiple captures.
   Keep one or more held-out test captures untouched for reporting.

3. Sweep the cheap things first.
   Thresholds, horizon, learning rate, epochs, and contact class weighting.

4. Keep A/B tests practical.
   Compare YOLO-seg vs FastSAM, DINO on/off, geometry-only vs geometry+JEPA, and different temporal-head thresholds.

5. Improve future decoding.
   Decode future latent state into 3D object/destination distributions rather than only displaying scalar probabilities.

6. Push toward multi-step futures.
   Roll the learned state forward over several hand/object/target tokens, while keeping geometry as the consistency check.

7. Add explicit action conditioning.
   Add action/control tokens so the temporal head becomes closer to a JEPA-AC predictive world model.

8. Preserve demo honesty.
   Gemini provides semantic names. Geometry provides physical state. JEPA predicts futures.

## Sensor Assumption

The COLMAP sparse SfM prior is a practical bridge for commodity webcam input. It gives
the live system a better metric scene scaffold than monocular video alone. With
RGB-D, calibrated stereo, IMU-assisted capture, or stronger online dense SLAM,
the offline prior could be relaxed. Today, COLMAP reconstruction is an offline
preparation step; online SLAM/local mapping refinement is what runs during live
interaction.

## Robotics Extension

The same representation can become a robot-task dataset:

- observation tokens: hand/object/static-target state, DINO/JEPA memory, 3D geometry,
- action labels: hand/object motion or robot end-effector command,
- teacher labels: contact, release, carried object, source target, destination target,
- prediction targets: next contact object, release location, object motion, future latent state.

That makes the demo a small observable manipulation task, not just a UI overlay.
