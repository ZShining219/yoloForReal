# Reproduction Commands

Run commands from the repository workspace root:

```text
/Users/zfh/Desktop/论文撰写汇报
```

Use a temporary output directory first when checking reproducibility. Do not overwrite the packaged outputs unless intentionally regenerating this module.

## 1. YOLO/ByteTrack Target Extraction

```bash
python3 论文稿件/框架设计文稿/外部真实到SUMO闭环证据子模块_20260601/modules/yolo_target_extraction_review/tools/mot_target_extraction.py \
  --clip 论文稿件/框架设计文稿/外部真实到SUMO闭环证据子模块_20260601/modules/yolo_target_extraction_review/inputs/video_clips/MVI_0866_520_560.mp4 \
  --model 论文稿件/框架设计文稿/外部真实到SUMO闭环证据子模块_20260601/modules/yolo_target_extraction_review/inputs/model_weights/yolo11m.pt \
  --output-dir /tmp/yolo_target_extraction_review_regen \
  --tracker bytetrack.yaml \
  --conf 0.05 \
  --imgsz 1280 \
  --fps 50.0 \
  --overlay-step 25
```

Expected target-layer files under `/tmp/yolo_target_extraction_review_regen/`:

- `target_tracks/detections_tracked.csv`
- `target_tracks/target_tracks_summary.csv`
- `target_tracks/target_recall_gate.csv`
- `target_tracks/target_track_overlay_frames/`
- `target_tracks/target_track_contact_sheet.jpg`
- `audit_board/TARGET_EXTRACTION_REVIEW.html`
- `notes/target_extraction_summary.md`

## 2. False-Positive Review Page

After the false-positive candidate gate exists, generate the focused review page and crops:

```bash
python3 论文稿件/框架设计文稿/外部真实到SUMO闭环证据子模块_20260601/modules/yolo_target_extraction_review/tools/build_target_false_positive_review.py \
  --review-gate 论文稿件/框架设计文稿/外部真实到SUMO闭环证据子模块_20260601/modules/yolo_target_extraction_review/outputs/target_tracks/target_false_positive_review_gate.csv \
  --detections 论文稿件/框架设计文稿/外部真实到SUMO闭环证据子模块_20260601/modules/yolo_target_extraction_review/outputs/target_tracks/detections_tracked.csv \
  --overlay-dir 论文稿件/框架设计文稿/外部真实到SUMO闭环证据子模块_20260601/modules/yolo_target_extraction_review/outputs/target_tracks/target_track_overlay_frames \
  --crop-dir /tmp/yolo_target_extraction_review_regen/target_tracks/false_positive_review_crops \
  --output-html /tmp/yolo_target_extraction_review_regen/audit_board/TARGET_FALSE_POSITIVE_REVIEW.html
```

The gate used in the current packaged result has 14 candidates:

- 7 `EXCLUDE`
- 7 `KEEP`

## 3. Apply Human False-Positive Gate

This reproduces the current packaged final target tables from `target_tracks_summary.csv` and `target_false_positive_review_gate.csv`:

```bash
python3 论文稿件/框架设计文稿/外部真实到SUMO闭环证据子模块_20260601/modules/yolo_target_extraction_review/tools/apply_target_false_positive_gate.py \
  --target-summary 论文稿件/框架设计文稿/外部真实到SUMO闭环证据子模块_20260601/modules/yolo_target_extraction_review/outputs/target_tracks/target_tracks_summary.csv \
  --false-positive-gate 论文稿件/框架设计文稿/外部真实到SUMO闭环证据子模块_20260601/modules/yolo_target_extraction_review/outputs/target_tracks/target_false_positive_review_gate.csv \
  --output-after-fp-filter /tmp/yolo_target_extraction_review_verify/target_tracks_after_fp_filter.csv \
  --output-final /tmp/yolo_target_extraction_review_verify/target_tracks_final.csv \
  --output-overall-gate /tmp/yolo_target_extraction_review_verify/target_overall_review_gate.csv
```

Expected stdout:

```text
target_tracks_after_fp_filter=38
target_tracks_final=38
```

The regenerated files should match the packaged files:

```bash
cmp -s /tmp/yolo_target_extraction_review_verify/target_tracks_after_fp_filter.csv 论文稿件/框架设计文稿/外部真实到SUMO闭环证据子模块_20260601/modules/yolo_target_extraction_review/outputs/target_tracks/target_tracks_after_fp_filter.csv
cmp -s /tmp/yolo_target_extraction_review_verify/target_tracks_final.csv 论文稿件/框架设计文稿/外部真实到SUMO闭环证据子模块_20260601/modules/yolo_target_extraction_review/outputs/target_tracks/target_tracks_final.csv
cmp -s /tmp/yolo_target_extraction_review_verify/target_overall_review_gate.csv 论文稿件/框架设计文稿/外部真实到SUMO闭环证据子模块_20260601/modules/yolo_target_extraction_review/outputs/target_tracks/target_overall_review_gate.csv
```

Exit code `0` means byte-level match.

## 4. Overlay Audit Videos

Generate the all-track audit video:

```bash
python3 -B 论文稿件/框架设计文稿/外部真实到SUMO闭环证据子模块_20260601/modules/yolo_target_extraction_review/tools/build_target_overlay_video.py \
  --clip 论文稿件/框架设计文稿/外部真实到SUMO闭环证据子模块_20260601/modules/yolo_target_extraction_review/inputs/video_clips/MVI_0866_520_560.mp4 \
  --detections 论文稿件/框架设计文稿/外部真实到SUMO闭环证据子模块_20260601/modules/yolo_target_extraction_review/outputs/target_tracks/detections_tracked.csv \
  --target-summary 论文稿件/框架设计文稿/外部真实到SUMO闭环证据子模块_20260601/modules/yolo_target_extraction_review/outputs/target_tracks/target_tracks_summary.csv \
  --false-positive-gate 论文稿件/框架设计文稿/外部真实到SUMO闭环证据子模块_20260601/modules/yolo_target_extraction_review/outputs/target_tracks/target_false_positive_review_gate.csv \
  --final-targets 论文稿件/框架设计文稿/外部真实到SUMO闭环证据子模块_20260601/modules/yolo_target_extraction_review/outputs/target_tracks/target_tracks_final.csv \
  --output 论文稿件/框架设计文稿/外部真实到SUMO闭环证据子模块_20260601/modules/yolo_target_extraction_review/outputs/audit_video/target_overlay_all_tracks.mp4 \
  --mode all \
  --fps 50.0
```

Generate the final-track audit video:

```bash
python3 -B 论文稿件/框架设计文稿/外部真实到SUMO闭环证据子模块_20260601/modules/yolo_target_extraction_review/tools/build_target_overlay_video.py \
  --clip 论文稿件/框架设计文稿/外部真实到SUMO闭环证据子模块_20260601/modules/yolo_target_extraction_review/inputs/video_clips/MVI_0866_520_560.mp4 \
  --detections 论文稿件/框架设计文稿/外部真实到SUMO闭环证据子模块_20260601/modules/yolo_target_extraction_review/outputs/target_tracks/detections_tracked.csv \
  --target-summary 论文稿件/框架设计文稿/外部真实到SUMO闭环证据子模块_20260601/modules/yolo_target_extraction_review/outputs/target_tracks/target_tracks_summary.csv \
  --false-positive-gate 论文稿件/框架设计文稿/外部真实到SUMO闭环证据子模块_20260601/modules/yolo_target_extraction_review/outputs/target_tracks/target_false_positive_review_gate.csv \
  --final-targets 论文稿件/框架设计文稿/外部真实到SUMO闭环证据子模块_20260601/modules/yolo_target_extraction_review/outputs/target_tracks/target_tracks_final.csv \
  --output 论文稿件/框架设计文稿/外部真实到SUMO闭环证据子模块_20260601/modules/yolo_target_extraction_review/outputs/audit_video/target_overlay_final_tracks.mp4 \
  --mode final \
  --fps 50.0
```

Expected stdout for each command:

```text
frames_written=2000
```

The script requires `ffmpeg` and `ffprobe` on `PATH`. It does not require `cv2`.

## 5. Tests

```bash
python3 -m unittest discover -s 论文稿件/框架设计文稿/外部真实到SUMO闭环证据子模块_20260601/modules/yolo_target_extraction_review/tests
```

Expected result:

```text
Ran 11 tests
OK
```
