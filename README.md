# Embodied Agent

A real-time, 3D grounded world model demo for egocentric assisted-living tasks with an embodied agent.

The current repository showcases hand-object interactions in a real-world environment, observed by an embodied AI agent.
The system watches the scene from an egocentric video, builds a persistent 3D scene prior, tracks hands and objects online, predicts likely interaction futures and targets, asks an LLM (Gemini) to refine weak object labels, and speaks timely guidance through a live avatar.

![Sophie demo prediction overlay: hand/object tracking with a predicted tray destination.](public/sophie_demo_prediction1s.gif)

Current demo shows JEPA prediction cues: the hand is holding a baby bottle, the destination heatmap is on the tray, and the UI shows the predicted target.

## What The System Contains

The current design focuses on grounded interaction u  nderstanding and consists of several modules:

- COLMAP sparse SfM for the offline/static scene prior.
- Online sparse SLAM/PnP for camera motion and persistent map alignment during interaction.
- YOLOv8 for light object detection and coarse detector labels.
- FastSAM-s for higher-recall segmentation regions in the Sophie demo profile.
- V-JEPA2.1 embeddings for object/hand appearance memory, object tracking, re-identification, and temporal prediction features.
- DINOv2 crop embeddings as a portability fallback and supporting baseline for tracking/re-ID.
- MediaPipe Hands for 2D hand landmarks, lifted into 3D with depth and camera geometry.
- Depth Anything monocular depth via Hugging Face Transformers when available, with a heuristic fallback if the model cannot load.
- Gemini label refinement for objects that detectors label weakly or generically.
- Geometry-teacher contact and release detection from hand/object distance in 3D.
- An interaction-conditioned temporal head that predicts next contact, destination likelihood, object motion, and future latent state from recent hand-object-target state.
- UI attention heatmaps over likely grabbed objects and predicted destinations.
- Voice/avatar feedback aligned to hand detection, object grasp, and predicted placement.

```text
static SfM/COLMAP prior -> online SLAM/depth -> persistent objects/targets
              -> hand-object contact -> JEPA future prediction
              -> Gemini labels/speech -> UI heatmaps/avatar feedback
```

## LLM And Avatar Bridge

Gemini is used in two places:

- Label refinement: the perception stack sends object/target crops and lightweight context to Gemini when detector labels are weak or generic. Gemini can refine labels such as `baby bottle`, `Sophie the giraffe`, `mat`, or `tray`, but those names do not create physical state by themselves.
- Assistant speech: the UI and Node bridge use Gemini/LiveAvatar to speak concise, spatially grounded feedback such as first hand detection, likely grabbed object, and predicted destination.

The LLM is deliberately downstream of the world model. It translates and refines what the grounded system sees. Geometry, SLAM, depth, tracking, and contact logic provide the metric state and teacher signals; the learned temporal world model predicts how that grounded state is likely to evolve.

## Explicit And Latent State

This demo deliberately uses both explicit and latent structure:

- Explicit state: persistent 3D targets, object tracks, hand tracks, camera pose, contact/release events, and source/destination relations.
- Latent state: V-JEPA2.1 visual embeddings, temporal embedding memory, and the learned temporal head's future latent prediction. DINOv2 embeddings remain available as a fallback/baseline.

Persistence is meaningful here because targets and objects keep stable IDs and 3D anchors across frames, even when detector labels flicker or objects are temporarily weakly observed. That persistence is currently maintained by explicit geometry, tracking, and appearance memory; the learned temporal head predicts over those grounded tokens rather than replacing them.

This is a pragmatic world model rather than a pure latent-only model. The explicit 3D state makes the demo inspectable and physically grounded with commodity webcam input. The latent side is used where it helps most now: re-ID, temporal memory, contact/placement prediction, motion prediction, and future visual-state prediction.

The COLMAP sparse SfM prior exists because the sensing hardware is limited: a normal webcam does not provide calibrated metric depth or dense, reliable 3D on its own. A better sensor stack, such as RGB-D, stereo, or event/IMU-assisted capture, could relax the need for an offline prior. Today, COLMAP reconstruction is an offline preparation step; online SLAM and local mapping/refinement keep the live interaction running.

## Run

Install dependencies:

```bash
npm install
pip install -r requirements.txt
```

Set API keys if you want Gemini/voice/avatar support:

```bash
GEMINI_API_KEY=your_key
LIVEAVATAR_API_KEY=your_key
```

Run the Sophie demo:

```bash
npm run demo:sophie
```

Open:

```text
http://localhost:3000
```

The demo expects these local videos:

- `public/scene_sophie.mp4` for training.
- `public/scene_sophie_test.mp4` for held-out evaluation/playback.

## Train And Evaluate JEPA

Train the Sophie interaction-conditioned temporal head:

```bash
npm run jepa:sophie:train
```

Evaluate on training rows:

```bash
npm run jepa:sophie:eval
```

Evaluate on the held-out test video:

```bash
npm run jepa:sophie:test
```

Valid shorter aliases:

```bash
npm run jepa:train
npm run jepa:eval
npm run jepa:test
```

Those currently delegate to the Sophie commands above. `jepa:sophie:test` first builds rows from `public/scene_sophie_test.mp4` with `--no-train`, then evaluates the saved checkpoint. The eval report includes:

- `saved_model`: the real checkpoint score.
- `row_fit_upper_bound`: a sanity check trained and evaluated on the same rows. This only tells us whether the rows are learnable; it is not held-out performance.
- `contact_threshold_sweep` and `placement_threshold_sweep`: useful for choosing demo signal thresholds.

Current Sophie demo defaults use the temporal head as an early learned attention/prediction cue:

```text
TEMPORAL_HEAD_CONTACT_THRESHOLD=0.20
TEMPORAL_HEAD_PLACEMENT_THRESHOLD=0.45
TEMPORAL_HEAD_TARGET_TRAY_THRESHOLD=0.45
JEPA_EVENT_RELEASE_UI_THRESHOLD=0.35
```

Training is self-supervised from the grounded scene state. The geometry-based hand/object contact and release signals are accurate enough to act as a teacher, so the JEPA temporal head learns interaction dynamics from 3D contact/release episodes rather than from hand-written event labels. Geometry still provides the conservative physical state used for validation and safety, while JEPA learns to predict upcoming contact, release, destination, motion, and future latent state from those grounded episodes.

## Architecture

```text
Browser UI
  -> Node/Gemini bridge
  -> Python world model server
  -> detection, segmentation, depth, SLAM, 3D state
  -> hand/object contact, JEPA prediction, guidance
  -> overlays, heatmaps, speech, avatar
```

Core files:

- `app/page.tsx`: UI modes, overlays, heatmaps, timing, speech feedback.
- `server/live-bridge.mjs`: Gemini label/speech/avatar bridge.
- `world_model/server.py`: realtime perception and WebSocket backend.
- `world_model/world_state.py`: persistent 3D state, targets, hands, contact, prediction candidates.
- `world_model/hands.py`: MediaPipe hand tracking and 3D hand localization.
- `world_model/perception_candidates.py`: detector/segmenter candidate construction.
- `world_model/jepa_encoder.py`: V-JEPA2.1 visual features, with DINOv2 fallback for portability.
- `world_model/temporal_interaction_head.py`: interaction-conditioned contact/motion/placement/future-latent predictor.
- `world_model/train_temporal_heads.py`: train/test row generation and temporal-head training.
- `world_model/eval_temporal_head.py`: saved-model eval, row-fit upper bound, threshold sweeps.
- `world_model/validate_interactions.py`: offline interaction/contact validation.

## Useful Commands

```bash
npm run demo:sophie
npm run jepa:sophie:train
npm run jepa:sophie:eval
npm run jepa:sophie:test
python world_model/validate_interactions.py --video public/scene_sophie_test.mp4 --profile sophie
```

Detailed runtime switches live in [docs/runtime_knobs.md](docs/runtime_knobs.md). The normal demo path should stay simple.

## License

MIT
