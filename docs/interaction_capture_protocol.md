# Interaction Capture Protocol

Goal: collect self-supervised action-conditioned dynamics data without pressing buttons during the manipulation sequence.

## Run Mode

Start the world model in capture mode:

```powershell
python world_model/server.py --capture
```

This enables:

- `world_model/data/transitions.jsonl`: compact object-motion transitions.
- `world_model/data/interaction_capture.jsonl`: rich geometry-teacher interaction rows.

The rich capture log is written automatically when:

- the hand is touching/contacting an object,
- a contact/release/pick-place event is emitted,
- an interaction is visible on the periodic sample interval.

Useful environment knobs:

```powershell
$env:AUTO_INTERACTION_CAPTURE="1"
$env:INTERACTION_CAPTURE_EVERY_FRAMES="10"
$env:HAND_INTERACTION_SIDES="right"
$env:HAND_FORCE_SIDE="right"
$env:USE_DINO_EMBEDDING="1"
$env:JEPA_ENABLED="1"
```

## Physical Scenario

Use a single continuous take. Do not stop between actions.

1. Static calibration phase, 20-30 seconds:
   - Move the camera slowly around the table.
   - Keep the right hand out of frame.
   - Show the black mat, coasters, dish, cups, apple, banana from several angles.
   - This builds COLMAP/depth anchors, static targets, DINO identities, and object labels.

2. Identity phase, 5-10 seconds:
   - Still no manipulation.
   - Look at each movable object and each target from 2-3 angles.
   - This gives DINO/LLM clean object views before occlusion.

3. Neutral hover phase, 5 seconds:
   - Bring the right hand into view.
   - Hover near each object without touching.
   - This gives negative examples: hand near object, no contact.

4. Pick/place phase, 2-3 repeats per object:
   - Pick mug/cup A and place it on coaster 1.
   - Pick mug/cup B and place it on coaster 2.
   - Pick apple and place it on the dish.
   - Pick banana and place it on the dish.
   - Reverse some actions: pick from target and put the object near its original position.
   - Include slow approaches, firm grasps, short carries, and clear release.

5. Hard negatives:
   - Reach over an object without grabbing.
   - Touch the mat or coaster without moving an object.
   - Move the hand behind/near an object where 2D overlap is high but 3D contact is false.

6. Recovery cases:
   - Briefly occlude the hand behind an object.
   - Let MediaPipe lose the hand for a few frames.
   - Continue the same action so depth-supported prediction can be evaluated.

## Labels Produced Automatically

The geometry teacher emits:

- `is_touching_strict`
- `is_contacting`
- `contact_start`
- `contact_end`
- `pick_place`
- moved object id/label
- start/end 3D positions
- nearest static target at release
- DINO/JEPA track embeddings
- hand depth evidence during short missing-hand spans

## Training Use

The next learned model should train from `interaction_capture.jsonl` as a sequence dataset:

- input: recent hand/object/static-target tokens over N frames,
- action: hand motion/contact state/object carried,
- targets: future contact object, future release target, object motion delta, contact/release timing.

Geometry remains the teacher, but the model learns temporal prediction instead of memorizing frame-local thresholds.

Build horizon-labeled rows after a recording:

```powershell
python world_model/build_interaction_dataset.py `
  --input world_model/data/interaction_capture.jsonl `
  --output world_model/data/interaction_dynamics_rows.json `
  --horizon 8
```
