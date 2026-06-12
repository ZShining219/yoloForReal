#!/usr/bin/env python3
"""Mature MOT-based vehicle target extraction for MVI_0866.

This script stops at the target-track layer. It does not infer lanes,
movements, SUMO routes, or vehicle events.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


MAIN_ROUTE_CLASSES = {"car", "truck", "bus"}
MIN_REVIEW_FRAMES = 2
MIN_REVIEW_DURATION_SEC = 0.5


def as_float(row: dict, key: str) -> float:
    return float(row[key])


def center(row: dict) -> tuple[float, float]:
    return ((as_float(row, "x1") + as_float(row, "x2")) / 2.0, (as_float(row, "y1") + as_float(row, "y2")) / 2.0)


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def normalized_track_id(track_id: str) -> str:
    return f"mot_{int(float(track_id)):04d}"


def build_track_summary_rows(detection_rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in detection_rows:
        if row.get("track_id", ""):
            grouped[row["track_id"]].append(row)

    summaries: list[dict] = []
    for raw_track_id, rows in sorted(grouped.items(), key=lambda item: int(float(item[0]))):
        ordered = sorted(rows, key=lambda row: as_float(row, "time_sec"))
        points = [center(row) for row in ordered]
        first = points[0]
        max_displacement = max(distance(first, point) for point in points)
        times = [as_float(row, "time_sec") for row in ordered]
        class_votes = Counter(row["class_name"] for row in ordered)
        mean_conf = sum(as_float(row, "confidence") for row in ordered) / len(ordered)
        summary = {
            "track_id": normalized_track_id(raw_track_id),
            "raw_track_id": str(raw_track_id),
            "class_name": class_votes.most_common(1)[0][0],
            "class_votes": json.dumps(dict(class_votes), ensure_ascii=False, sort_keys=True),
            "start_time": f"{min(times):.2f}",
            "end_time": f"{max(times):.2f}",
            "duration_sec": f"{max(times) - min(times):.2f}",
            "frame_count": str(len({row["frame_id"] for row in ordered})),
            "detection_count": str(len(ordered)),
            "mean_confidence": f"{mean_conf:.4f}",
            "max_displacement_px": f"{max_displacement:.2f}",
        }
        summary.update(keep_track_for_review(summary))
        summaries.append(summary)
    return summaries


def keep_track_for_review(summary: dict) -> dict:
    frame_count = int(float(summary["frame_count"]))
    duration = as_float(summary, "duration_sec")
    class_name = summary.get("class_name", "")
    if class_name not in MAIN_ROUTE_CLASSES:
        return {"track_status": "excluded", "exclude_reason": "non_main_motor_vehicle_class"}
    if frame_count <= 1:
        return {"track_status": "excluded", "exclude_reason": "single_sample_track"}
    if frame_count < MIN_REVIEW_FRAMES or duration < MIN_REVIEW_DURATION_SEC:
        return {"track_status": "excluded", "exclude_reason": "too_short_for_target_review"}
    return {"track_status": "target_review_candidate", "exclude_reason": ""}


def build_review_gate_rows(summaries: list[dict]) -> list[dict]:
    rows = []
    for summary in summaries:
        if summary["track_status"] != "target_review_candidate":
            continue
        rows.append(
            {
                "case_id": "mvi0866",
                "review_unit": "target_track_overlay",
                "track_id": summary["track_id"],
                "status": "PENDING_RECALL_REVIEW",
                "visible_route_bearing_vehicle_missed": "",
                "false_or_duplicate_track": "",
                "next_action": "pass/warning/fail; do not add vehicles manually",
                "note": "",
            }
        )
    return rows


def crop_name_for_track(track_id: str, label: str, time_sec: float) -> str:
    return f"{track_id}_{label}_t{time_sec:.1f}.jpg".replace(".", "p").replace("pjpg", ".jpg")


def selected_track_rows(rows: list[dict]) -> list[dict]:
    ordered = sorted(rows, key=lambda row: as_float(row, "time_sec"))
    if len(ordered) <= 2:
        return ordered
    start = as_float(ordered[0], "time_sec")
    end = as_float(ordered[-1], "time_sec")
    mid_time = (start + end) / 2.0
    mid = min(ordered, key=lambda row: abs(as_float(row, "time_sec") - mid_time))
    return [ordered[0], mid, ordered[-1]]


def build_overlay_gallery(overlay_paths: list[Path], rel_dir: str) -> str:
    figures = []
    for path in sorted(overlay_paths):
        figures.append(
            f"""
            <figure>
              <img src="../{html.escape(rel_dir)}/{html.escape(path.name)}" alt="{html.escape(path.stem)}">
              <figcaption>{html.escape(path.name)}</figcaption>
            </figure>
            """
        )
    return "\n".join(figures)


def write_contact_sheet(image_paths: list[Path], output_path: Path, cols: int = 5) -> None:
    from PIL import Image, ImageDraw

    if not image_paths:
        return
    thumbs = []
    for path in sorted(image_paths):
        image = Image.open(path).convert("RGB")
        image.thumbnail((360, 203))
        canvas = Image.new("RGB", (360, 230), (245, 247, 249))
        canvas.paste(image, (0, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 208), path.name, fill=(20, 30, 40))
        thumbs.append(canvas)
    rows = math.ceil(len(thumbs) / cols)
    sheet = Image.new("RGB", (cols * 360, rows * 230), (235, 238, 242))
    for idx, thumb in enumerate(thumbs):
        row, col = divmod(idx, cols)
        sheet.paste(thumb, (col * 360, row * 230))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def build_html(summaries: list[dict], overlay_paths: list[Path], overlay_rel_dir: str, contact_sheet_rel: str) -> str:
    counts = Counter(row["track_status"] for row in summaries)
    class_counts = Counter(row["class_name"] for row in summaries)
    cards = []
    for row in summaries:
        if row["track_status"] != "target_review_candidate":
            continue
        cards.append(
            f"""
            <article class="track-card">
              <div class="track-head">
                <h3>{html.escape(row["track_id"])} | {html.escape(row["class_name"])}</h3>
                <span>{html.escape(row["start_time"])}-{html.escape(row["end_time"])}s / frames={html.escape(row["frame_count"])}</span>
              </div>
              <p>只判断这个 track 是否覆盖一辆真实机动车，以及是否存在明显漏车/错 track。不要判断车道、转向或 SUMO 事件。</p>
              <div class="metrics">
                <span>mean_conf={html.escape(row["mean_confidence"])}</span>
                <span>disp={html.escape(row["max_displacement_px"])}px</span>
                <span>raw_id={html.escape(row["raw_track_id"])}</span>
              </div>
            </article>
            """
        )
    overlay_gallery = build_overlay_gallery(overlay_paths, overlay_rel_dir)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MVI_0866 车辆目标提取审计</title>
  <style>
    :root {{
      --ink: #202832;
      --muted: #62707f;
      --line: #d7dde4;
      --paper: #f4f1ea;
      --panel: #fff;
      --blue: #17364c;
      --green: #236b4f;
      --amber: #986613;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--paper);
      color: var(--ink);
      font-family: "Songti SC", "Noto Serif CJK SC", "Times New Roman", serif;
    }}
    header {{
      background: var(--blue);
      color: white;
      padding: 22px 28px;
      border-bottom: 4px solid #c59c55;
    }}
    header h1 {{ margin: 0 0 8px; font-size: 28px; }}
    header p {{ margin: 0; color: #dce7ef; line-height: 1.6; }}
    main {{ max-width: 1500px; margin: 0 auto; padding: 18px; }}
    section, .track-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 16px;
      margin-bottom: 16px;
      box-shadow: 0 10px 24px rgba(22, 32, 42, 0.10);
    }}
    .guide {{ border-left: 6px solid var(--green); background: #eef8f2; }}
    .stats {{ display: grid; grid-template-columns: repeat(4, minmax(130px, 1fr)); gap: 10px; }}
    .stat {{ border: 1px solid var(--line); border-radius: 6px; padding: 12px; background: #f8fafc; }}
    .stat b {{ display: block; font: 700 25px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
    .track-head {{ display: flex; justify-content: space-between; gap: 12px; border-bottom: 1px solid var(--line); padding-bottom: 10px; }}
    .track-head span, .metrics span {{ color: var(--muted); font: 13px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
    .metrics {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }}
    .metrics span {{ border: 1px solid var(--line); border-radius: 999px; padding: 5px 9px; background: #f6f8fa; }}
    .overlay-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
    }}
    figure {{
      margin: 0;
      border: 1px solid var(--line);
      border-radius: 6px;
      overflow: hidden;
      background: #101820;
    }}
    figure img {{
      display: block;
      width: 100%;
      aspect-ratio: 16 / 9;
      object-fit: cover;
    }}
    figcaption {{
      background: #edf2f6;
      padding: 6px 8px;
      font: 12px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      overflow-wrap: anywhere;
    }}
    @media (max-width: 1000px) {{
      .stats, .overlay-grid {{ grid-template-columns: 1fr; }}
      .track-head {{ display: block; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>MVI_0866 车辆目标提取审计</h1>
    <p>本页只审车辆目标提取和 track recall，不生成车道、转向或 SUMO route。</p>
  </header>
  <main>
    <section class="guide">
      <h2>怎么审</h2>
      <ol>
        <li>看 overlay/contact sheet 中主车流机动车是否都有稳定 track id。</li>
        <li>如果明显有车连续多帧没有框或没有 track，记 FAIL。</li>
        <li>如果只是单帧丢框或局部抖动，可以记 WARNING 或继续。</li>
        <li>不要补车、不要画框、不要改 track id、不要判断转向。</li>
      </ol>
      <div class="stats">
        <div class="stat"><b>{len(summaries)}</b><span>total tracks</span></div>
        <div class="stat"><b>{counts.get("target_review_candidate", 0)}</b><span>review candidates</span></div>
        <div class="stat"><b>{counts.get("excluded", 0)}</b><span>excluded tracks</span></div>
        <div class="stat"><b>{html.escape(str(dict(class_counts)))}</b><span>class counts</span></div>
      </div>
      <p>Contact sheet: <code>{html.escape(contact_sheet_rel)}</code></p>
    </section>
    <section>
      <h2>连续跟踪 overlay</h2>
      <p>这里按 0.5s 间隔展示完整 40s clip 的目标跟踪结果。人工只看主车流机动车是否有稳定框和 track id，是否存在连续多帧漏车。</p>
      <div class="overlay-grid">
        {overlay_gallery}
      </div>
    </section>
    <section>
      <h2>目标 track 列表</h2>
      {"".join(cards)}
    </section>
  </main>
</body>
</html>
"""


def run_ultralytics_track(args) -> list[dict]:
    import cv2
    from ultralytics import YOLO

    model = YOLO(args.model)
    rows: list[dict] = []
    output_frames = Path(args.output_dir) / "target_tracks" / "target_track_overlay_frames"
    output_frames.mkdir(parents=True, exist_ok=True)

    results = model.track(
        source=args.clip,
        conf=args.conf,
        imgsz=args.imgsz,
        tracker=args.tracker,
        stream=True,
        persist=True,
        classes=[2, 5, 7],
        verbose=False,
    )
    for frame_index, result in enumerate(results):
        frame = result.orig_img.copy()
        time_sec = frame_index / args.fps
        if result.boxes is not None:
            for box in result.boxes:
                if box.id is None:
                    continue
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                cls_id = int(box.cls[0])
                class_name = result.names.get(cls_id, str(cls_id))
                track_id = int(box.id[0])
                conf = float(box.conf[0])
                rows.append(
                    {
                        "frame_id": str(frame_index),
                        "time_sec": f"{time_sec:.2f}",
                        "track_id": str(track_id),
                        "class_name": class_name,
                        "confidence": f"{conf:.6f}",
                        "x1": f"{x1:.2f}",
                        "y1": f"{y1:.2f}",
                        "x2": f"{x2:.2f}",
                        "y2": f"{y2:.2f}",
                    }
                )
                color = (0, 255, 0)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(
                    frame,
                    f"mot_{track_id:04d} {class_name} {conf:.2f}",
                    (int(x1), max(20, int(y1) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                )
        if frame_index % max(1, args.overlay_step) == 0:
            cv2.imwrite(str(output_frames / f"target_track_f{frame_index:06d}.jpg"), frame)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--overlay-step", type=int, default=25)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    target_dir = output_dir / "target_tracks"
    audit_dir = output_dir / "audit_board"
    notes_dir = output_dir / "notes"
    target_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    notes_dir.mkdir(parents=True, exist_ok=True)

    detection_rows = run_ultralytics_track(args)
    overlay_dir = target_dir / "target_track_overlay_frames"
    overlay_paths = sorted(overlay_dir.glob("*.jpg"))
    contact_sheet = target_dir / "target_track_contact_sheet.jpg"
    write_contact_sheet(overlay_paths, contact_sheet)
    detection_fields = ["frame_id", "time_sec", "track_id", "class_name", "confidence", "x1", "y1", "x2", "y2"]
    write_csv(target_dir / "detections_tracked.csv", detection_rows, detection_fields)

    summaries = build_track_summary_rows(detection_rows)
    summary_fields = [
        "track_id",
        "raw_track_id",
        "class_name",
        "class_votes",
        "start_time",
        "end_time",
        "duration_sec",
        "frame_count",
        "detection_count",
        "mean_confidence",
        "max_displacement_px",
        "track_status",
        "exclude_reason",
    ]
    write_csv(target_dir / "target_tracks_summary.csv", summaries, summary_fields)
    write_csv(
        target_dir / "target_recall_gate.csv",
        build_review_gate_rows(summaries),
        [
            "case_id",
            "review_unit",
            "track_id",
            "status",
            "visible_route_bearing_vehicle_missed",
            "false_or_duplicate_track",
            "next_action",
            "note",
        ],
    )
    (audit_dir / "TARGET_EXTRACTION_REVIEW.html").write_text(
        build_html(
            summaries,
            overlay_paths,
            "target_tracks/target_track_overlay_frames",
            "target_tracks/target_track_contact_sheet.jpg",
        ),
        encoding="utf-8",
    )
    counts = Counter(row["track_status"] for row in summaries)
    notes = [
        "# Target Extraction Summary",
        "",
        "- status: `PENDING_TARGET_RECALL_REVIEW`",
        f"- model: `{args.model}`",
        f"- tracker: `{args.tracker}`",
        f"- detections_tracked: `{len(detection_rows)}`",
        f"- target_tracks: `{len(summaries)}`",
        f"- status_counts: `{dict(counts)}`",
        "",
        "This branch stops at target extraction. It does not infer lanes, movements, or SUMO routes.",
    ]
    (notes_dir / "target_extraction_summary.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
    print(f"detections_tracked={len(detection_rows)}")
    print(f"target_tracks={len(summaries)}")
    print(f"review_candidates={counts.get('target_review_candidate', 0)}")
    print(f"html={audit_dir / 'TARGET_EXTRACTION_REVIEW.html'}")


if __name__ == "__main__":
    main()
