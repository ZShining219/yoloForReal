# YOLO26X Trial Summary

## Scope

This is a detector/tracker trial only. It does not replace the current reviewed yolo11m target layer and does not produce a final target table for downstream lane-level event extraction.

## Command Environment

- Python: `/opt/miniconda3/envs/python/bin/python`
- Added packages in that environment: `ultralytics`, `opencv-python`, `lap`
- Model weight: `inputs/model_weights/yolo26x.pt`
- Tracker: `bytetrack.yaml`
- Confidence threshold: `0.05`
- Image size: `1280`
- Device: CPU in this local run. MPS was not available in this environment.

## Outputs

Trial output directory:

```text
outputs/yolo26x_trial/
```

Key files:

- `outputs/yolo26x_trial/target_tracks/detections_tracked.csv`
- `outputs/yolo26x_trial/target_tracks/target_tracks_summary.csv`
- `outputs/yolo26x_trial/target_tracks/target_recall_gate.csv`
- `outputs/yolo26x_trial/target_tracks/target_track_overlay_frames/`
- `outputs/yolo26x_trial/audit_board/TARGET_EXTRACTION_REVIEW.html`
- `outputs/yolo26x_trial/audit_video/target_overlay_all_tracks.mp4`

## First-Pass Quantitative Comparison

| Metric | yolo11m reviewed branch | yolo26x trial |
| --- | ---: | ---: |
| tracked detection rows | 22,878 | 23,363 |
| total track summaries | 74 | 133 |
| target review candidates | 45 | 52 |
| excluded by automatic summary gate | 29 | 81 |
| short tracks, duration <= 1s | 33 | 90 |
| low-motion tracks, displacement <= 5px | 29 | 68 |
| low-confidence tracks, mean confidence < 0.3 | 21 | 7 |
| tracks starting after 30s | 17 | 39 |

## Interpretation

The yolo26x trial is more sensitive but not automatically better for this task. It increases tracked detections slightly and target-review candidates modestly, but it also creates many more short or low-motion tracks. That suggests a higher fragmentation burden and a likely increase in manual review cost.

This trial should therefore be reviewed using the yolo26x all-track overlay video before deciding whether to use it as the next target source.

## Current Status

```text
PENDING_TARGET_RECALL_REVIEW
```

Do not use `outputs/yolo26x_trial/target_tracks/target_recall_gate.csv` or `target_tracks_summary.csv` as downstream SUMO input until full target recall/precision review is completed.

