# File Inventory

## Provenance

Original source bundle:

```text
论文稿件/框架设计文稿/外部真实到SUMO闭环证据子模块_20260601/
```

Original target-layer output source:

```text
outputs/real_video_to_sumo_export_20260601_yolo11m_mot/target_tracks/
```

No original files were moved or deleted. This module is a duplicated, bounded package for the target extraction and target review layer.

## Inputs

| Path | Role |
| --- | --- |
| `inputs/video_clips/MVI_0866_520_560.mp4` | Real video window used for YOLO/ByteTrack target extraction. |
| `inputs/model_weights/yolo11m.pt` | YOLO11m model weight used by the extraction pass. |

## Implementation

| Path | Role |
| --- | --- |
| `tools/mot_target_extraction.py` | Performs YOLO tracking, target-track summarization, recall gate generation, overlay rendering, and extraction review page generation. |
| `tools/build_target_false_positive_review.py` | Builds false-positive review crops and HTML from target summaries and tracked detections. |
| `tools/apply_target_false_positive_gate.py` | Applies human KEEP/EXCLUDE gate to produce final target tables. Added in this module because the source project contained the final CSV but no isolated filter script for this step. |
| `tools/build_target_overlay_video.py` | Builds continuous MP4 audit videos by drawing target boxes and track ids onto the original video. |

## Tests

| Path | Role |
| --- | --- |
| `tests/test_mot_target_extraction.py` | Tests target summary and recall gate behavior. |
| `tests/test_target_false_positive_review.py` | Tests false-positive review frame/crop selection behavior. |
| `tests/test_apply_target_false_positive_gate.py` | Tests that final target rows are produced only from review-accepted target tracks and never from manual additions. |
| `tests/test_build_target_overlay_video.py` | Tests target status classification and all/final video row filtering. |

## Target-Layer Outputs

| Path | Current Count | Role |
| --- | ---: | --- |
| `outputs/target_tracks/detections_tracked.csv` | 22,878 data rows | Frame-level tracked YOLO detections with bbox, class, confidence, frame, and time. |
| `outputs/target_tracks/target_tracks_summary.csv` | 74 data rows | Track-level summaries before human false-positive filtering. |
| `outputs/target_tracks/target_recall_gate.csv` | 45 candidate rows | Coarse recall review gate for target-review candidates. |
| `outputs/target_tracks/target_false_positive_review_gate.csv` | 14 candidate rows | Human false-positive review gate: 7 `EXCLUDE`, 7 `KEEP`. |
| `outputs/target_tracks/target_tracks_after_fp_filter.csv` | 38 data rows | Target tracks retained after false-positive exclusion. |
| `outputs/target_tracks/target_tracks_final.csv` | 38 data rows | Final accepted target tracks for downstream use. |
| `outputs/target_tracks/target_overall_review_gate.csv` | 1 data row | Overall target extraction review status. |
| `outputs/target_tracks/target_track_contact_sheet.jpg` | 1 image | Contact sheet for target-track overlay frames. |
| `outputs/target_tracks/target_track_overlay_frames/` | 80 images | Full-frame overlay evidence. |
| `outputs/target_tracks/false_positive_review_crops/` | 33 images | Crops used by false-positive review. |

## Audit Videos

| Path | Current Video Properties | Role |
| --- | --- | --- |
| `outputs/audit_video/target_overlay_all_tracks.mp4` | 1920x1080, 50fps, 40s, 2000 frames | Continuous review video showing all tracked detections with status colors. Best for checking whether visually obvious vehicles were captured. |
| `outputs/audit_video/target_overlay_final_tracks.mp4` | 1920x1080, 50fps, 40s, 2000 frames | Continuous review video showing only final retained tracks. Best for checking whether the filtered target layer remains plausible. |

## Detector Trials

| Path | Status | Role |
| --- | --- | --- |
| `inputs/model_weights/yolo26x.pt` | downloaded and packaged | YOLO26X weight used for detector/tracker trial. |
| `outputs/yolo26x_trial/` | `PENDING_TARGET_RECALL_REVIEW` | Isolated yolo26x trial output. This does not replace the reviewed yolo11m target layer. |
| `outputs/yolo26x_trial/audit_video/target_overlay_all_tracks.mp4` | 1920x1080, 50fps, 40s, 2000 frames | Continuous yolo26x all-track review video. |
| `docs/YOLO26X_TRIAL_SUMMARY.md` | active note | First-pass yolo11m vs yolo26x comparison and status. |

## Audit And Notes

| Path | Role |
| --- | --- |
| `outputs/audit_board/TARGET_EXTRACTION_REVIEW.html` | Human-facing coarse extraction review page. |
| `outputs/audit_board/TARGET_FALSE_POSITIVE_REVIEW.html` | Human-facing false-positive review page. |
| `outputs/notes/target_extraction_summary.md` | Current review status, counts, excluded IDs, and downstream handoff note. |

## Explicitly Excluded From This Module

The following layers are intentionally not packaged here:

- `mot_vehicle_events_candidate_draft/`
- `mot_direction_reconstruction/`
- `mot_edge_route_reconstruction/`
- `lane_anchor_mapping`
- `lane_temporal_target_panels`
- `semantic_sumo_packages`
- SUMO route, demand, and simulation outputs

Those layers are downstream of target extraction and must be debugged separately.
