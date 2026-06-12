# Logical Vehicle Consistency Module

This project isolates the logical vehicle consistency layer for the
`MVI_0866_520_560.mp4` real-video clip.

The module starts from YOLO/ByteTrack detection evidence and reviewed raw target
tracks, then builds stable `logical_vehicle_id` trajectories. It is target
identity logic only. It does not infer lanes, OD, turning movements, SUMO
routes, or demand.

## Project Scope

Current v3 processing keeps only the active logical-consistency toolchain in the
top-level `tools/` directory:

- `tools/logical_vehicle_consistency.py`
  - duplicate suppression
  - tracklet construction and association
  - same-raw continuity merging
  - fragment-to-mature path absorption
  - target validity, identity purity, and final gate reports
- `tools/build_logical_vehicle_consistency_v1.py`
  - command-line entrypoint for rebuilding the logical consistency outputs
  - writes CSV audit outputs and renders videos
- `tools/build_logical_vehicle_id_video.py`
  - final/debug/review video renderer for logical vehicle IDs

Earlier target extraction, track-stitching mock, direction, SUMO, and review
helpers are archived under `tools/old/` for traceability. They are not part of
the active v3 project surface.

## Key Inputs

The v3 rebuild uses:

```text
outputs/old/yolo26x_trial/target_tracks/detections_tracked.csv
outputs/old/yolo26x_manual_filter_v1/target_tracks_final.csv
inputs/video_clips/MVI_0866_520_560.mp4
```

The detection CSV is treated as evidence. The reviewed final target table
defines the allowed raw-track set.

## Key Outputs

Current v3 outputs are under:

```text
outputs/logical_vehicle_consistency/v3/
```

Important files:

- `logical_vehicle_tracks.csv`
- `logical_vehicle_summary.csv`
- `raw_track_to_logical_vehicle.csv`
- `duplicate_groups.csv`
- `tracklets.csv`
- `tracklet_link_candidates.csv`
- `tracklet_links_accepted.csv`
- `ambiguous_link_review.csv`
- `raw_track_split_review.csv`
- `risky_accepted_link_review.csv`
- `fragment_path_absorption_review.csv`
- `target_validity_report.csv`
- `identity_purity_report.csv`
- `final_target_gate.csv`
- `logical_vehicle_id_final.mp4`
- `logical_vehicle_id_debug.mp4`
- `logical_vehicle_id_review.mp4`

## Rebuild

```bash
python3 tools/build_logical_vehicle_consistency_v1.py \
  --detections outputs/old/yolo26x_trial/target_tracks/detections_tracked.csv \
  --final-targets outputs/old/yolo26x_manual_filter_v1/target_tracks_final.csv \
  --clip inputs/video_clips/MVI_0866_520_560.mp4 \
  --output-dir outputs/logical_vehicle_consistency/v3 \
  --fps 50.0 \
  --max-gap-frames 10 \
  --max-link-distance-px 80.0 \
  --max-iou 0.85 \
  --window-slices
```

## Test

```bash
python3 -m unittest discover tests
```

The default test suite covers only the active v3 logical-consistency project.
Archived tests for old tools live under `tools/old/tests/`.

## Current Bug-Fix Behavior

The v3 path-level fragment absorption pass handles short, low-confidence
fragment paths that appear near a mature path. Overlapping fragment rows become
`fragment_suppressed`; fragment rows that fill a mature-path missing frame are
kept as `accepted`.

For the `mot_0226` case, the current output records:

```text
lv_0017 -> lv_0009 AUTO_ABSORB_FRAGMENT_PATH
lv_0018 -> lv_0009 AUTO_ABSORB_FRAGMENT_PATH
```

This prevents short duplicate fragments from becoming independent final logical
vehicles.
