#!/usr/bin/env python3
"""Build a direction-ready YOLO26 post-processing review layer.

This layer does not modify YOLO detections. It creates auditable quality gates,
logical vehicle IDs, and approach-level OD candidates before downstream use.
"""

from __future__ import annotations

import argparse
import csv
import html
import math
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from build_center_follow_mock import build_center_follow_mock
from build_single_track_direction_evidence import (
    CROSSING_FIELDS,
    DIRECTION_FIELDS,
    TrackPoint,
    build_direction_result,
    detect_gate_crossings,
    load_direction_semantics,
    load_gates,
    normalized_track_id,
)


QUALITY_FIELDS = [
    "track_id",
    "class_name",
    "duration_sec",
    "max_displacement_px",
    "crossing_count",
    "od_review_status",
    "result_direction",
    "static_or_parked_flag",
    "short_track_flag",
    "window_partial_status",
    "motor_vehicle_review_status",
    "keep_for_direction_od",
    "exclude_reason",
    "review_note",
]

LINK_FIELDS = [
    "logical_vehicle_id",
    "from_raw_track_id",
    "to_raw_track_id",
    "from_frame",
    "to_frame",
    "gap_frames",
    "predicted_distance_px",
    "size_ratio",
    "link_action",
    "review_decision",
]

LOGICAL_TARGET_FIELDS = [
    "logical_vehicle_id",
    "raw_track_ids",
    "raw_track_id_count",
    "detected_frame_count",
    "start_frame",
    "end_frame",
    "duration_sec",
    "max_displacement_px",
    "component_quality_status",
    "keep_for_direction_od",
    "review_note",
]


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def as_float(row: dict, key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    return float(value)


def frame_value(row: dict, key: str, fps: float, default: int) -> int:
    if row.get(key, "") != "":
        return int(float(row[key]))
    time_key = "start_time" if key == "start_frame" else "end_time"
    if row.get(time_key, "") != "":
        return int(round(float(row[time_key]) * fps))
    return default


def partial_window_status(
    final_row: dict,
    od_row: dict,
    video_last_frame: int,
    fps: float = 50.0,
    boundary_tolerance_frames: int = 5,
) -> str:
    crossing_count = int(float(od_row.get("crossing_count") or 0))
    has_entering = bool(od_row.get("first_crossing_gate", ""))
    has_exiting = bool(od_row.get("last_crossing_gate", ""))
    start_frame = frame_value(final_row, "start_frame", fps, default=video_last_frame)
    end_frame = frame_value(final_row, "end_frame", fps, default=0)

    starts_at_window = start_frame <= boundary_tolerance_frames
    ends_at_window = end_frame >= video_last_frame - boundary_tolerance_frames
    if starts_at_window and crossing_count == 1 and has_exiting and not has_entering:
        return "WINDOW_START_PARTIAL_EXIT_ONLY"
    if ends_at_window and crossing_count == 1 and has_entering and not has_exiting:
        return "WINDOW_END_PARTIAL_ENTRY_ONLY"
    if starts_at_window and ends_at_window and crossing_count == 0:
        return "WINDOW_SPANNING_NO_GATE_CROSSING"
    return ""


def should_keep_for_direction_od(quality_keep: str, motor_vehicle_review_status: str) -> bool:
    return quality_keep == "yes" and motor_vehicle_review_status != "NON_MOTOR_REVIEW_EXCLUDE"


def build_quality_review_row(
    final_row: dict,
    od_row: dict,
    video_last_frame: int = 1999,
    fps: float = 50.0,
    static_displacement_px: float = 10.0,
    min_static_duration_sec: float = 2.0,
    short_duration_sec: float = 2.0,
) -> dict:
    duration = as_float(final_row, "duration_sec")
    displacement = as_float(final_row, "max_displacement_px")
    crossing_count = int(float(od_row.get("crossing_count") or 0))
    static_flag = duration >= min_static_duration_sec and displacement <= static_displacement_px and crossing_count == 0
    short_flag = duration <= short_duration_sec
    partial_status = partial_window_status(final_row, od_row, video_last_frame, fps=fps)
    motor_status = "PENDING_MOTOR_REVIEW"

    quality_keep = "no"
    exclude_reason = ""
    review_note = ""
    if static_flag:
        exclude_reason = "static_or_parked_without_crossing"
        review_note = "Low displacement and no accepted crossing; review as parked/static before OD."
    elif od_row.get("review_status") == "ACCEPTED":
        quality_keep = "yes"
        review_note = "Complete entering and exiting crossings are available."
    elif partial_status:
        exclude_reason = "window_boundary_partial"
        review_note = "Track is partial because the video window starts or ends mid-trajectory."
    elif short_flag:
        exclude_reason = "short_track_fragment"
        review_note = "Short fragment; do not force direction OD."
    else:
        exclude_reason = "insufficient_crossing_for_direction_od"
        review_note = "No complete entering/exiting pair; keep for review but not final OD."

    keep = "yes" if should_keep_for_direction_od(quality_keep, motor_status) else "no"
    return {
        "track_id": final_row["track_id"],
        "class_name": final_row.get("class_name", ""),
        "duration_sec": final_row.get("duration_sec", ""),
        "max_displacement_px": final_row.get("max_displacement_px", ""),
        "crossing_count": str(crossing_count),
        "od_review_status": od_row.get("review_status", ""),
        "result_direction": od_row.get("result_direction", ""),
        "static_or_parked_flag": "yes" if static_flag else "no",
        "short_track_flag": "yes" if short_flag else "no",
        "window_partial_status": partial_status,
        "motor_vehicle_review_status": motor_status,
        "keep_for_direction_od": keep,
        "exclude_reason": exclude_reason,
        "review_note": review_note,
    }


def link_decision(
    link: dict,
    max_gap_frames: int = 30,
    max_prediction_distance_px: float = 80.0,
    max_size_ratio: float = 2.5,
) -> str:
    gap = int(float(link.get("gap_frames") or 0))
    predicted_distance = float(link.get("predicted_distance_px") or 0.0)
    ratio = float(link.get("size_ratio") or 0.0)
    from_id = link.get("from_raw_track_id", "")
    to_id = link.get("to_raw_track_id", "")
    if gap <= max_gap_frames and predicted_distance <= max_prediction_distance_px and ratio <= max_size_ratio:
        if from_id != to_id:
            return "ACCEPT_LOW_RISK_CONTINUITY"
        return "ACCEPT_SAME_RAW_GAP"
    return "REVIEW_LINK_CANDIDATE"


def selected_track_rows(rows: list[dict]) -> list[dict]:
    ordered = sorted(rows, key=lambda row: int(float(row["frame_id"])))
    if len(ordered) <= 3:
        return ordered
    return [ordered[0], ordered[len(ordered) // 2], ordered[-1]]


def review_case_groups(quality_rows: list[dict], link_rows: list[dict]) -> dict[str, list[dict]]:
    return {
        "link_candidates": [
            row
            for row in link_rows
            if row.get("review_decision") == "ACCEPT_LOW_RISK_CONTINUITY"
        ],
        "motor_vehicle": [
            row
            for row in quality_rows
            if row.get("keep_for_direction_od") == "yes"
            and row.get("motor_vehicle_review_status") == "PENDING_MOTOR_REVIEW"
        ],
        "static_or_parked": [
            row
            for row in quality_rows
            if row.get("static_or_parked_flag") == "yes"
        ],
        "window_partial": [
            row
            for row in quality_rows
            if row.get("window_partial_status", "")
        ],
    }


def rows_by_normalized_track(detection_rows: list[dict], allowed_ids: set[str]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in detection_rows:
        track_id = normalized_track_id(row["track_id"])
        if track_id not in allowed_ids:
            continue
        copy = dict(row)
        copy["track_id"] = track_id
        grouped[track_id].append(copy)
    return {
        track_id: sorted(rows, key=lambda row: int(float(row["frame_id"])))
        for track_id, rows in grouped.items()
    }


def nearest_detection_row(rows: list[dict], frame_id: int) -> dict | None:
    if not rows:
        return None
    return min(rows, key=lambda row: abs(int(float(row["frame_id"])) - frame_id))


def frame_image_path(clip_path: Path, frame_id: int, frame_dir: Path) -> Path:
    frame_dir.mkdir(parents=True, exist_ok=True)
    output_path = frame_dir / f"frame_{frame_id:06d}.jpg"
    if output_path.exists():
        return output_path
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(clip_path),
            "-vf",
            f"select=eq(n\\,{frame_id})",
            "-frames:v",
            "1",
            str(output_path),
        ],
        check=True,
    )
    return output_path


def load_font(size: int):
    from PIL import ImageFont

    for path in [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_detection_crop(
    clip_path: Path,
    row: dict,
    output_path: Path,
    label: str,
    frame_cache_dir: Path,
    pad: int = 80,
) -> str:
    from PIL import Image, ImageDraw

    frame_id = int(float(row["frame_id"]))
    frame_path = frame_image_path(clip_path, frame_id, frame_cache_dir)
    image = Image.open(frame_path).convert("RGB")
    x1 = float(row["x1"])
    y1 = float(row["y1"])
    x2 = float(row["x2"])
    y2 = float(row["y2"])
    crop_box = (
        max(0, int(x1 - pad)),
        max(0, int(y1 - pad)),
        min(image.width, int(x2 + pad)),
        min(image.height, int(y2 + pad)),
    )
    crop = image.crop(crop_box)
    draw = ImageDraw.Draw(crop, "RGBA")
    font = load_font(18)
    small_font = load_font(14)
    local_box = (x1 - crop_box[0], y1 - crop_box[1], x2 - crop_box[0], y2 - crop_box[1])
    draw.rectangle(local_box, outline=(235, 36, 42, 255), width=4)
    draw.ellipse(
        (
            (local_box[0] + local_box[2]) / 2 - 4,
            local_box[3] - 4,
            (local_box[0] + local_box[2]) / 2 + 4,
            local_box[3] + 4,
        ),
        fill=(255, 255, 255, 230),
    )
    draw.rectangle((0, 0, crop.width, 48), fill=(16, 25, 34, 210))
    draw.text((10, 8), label[:80], fill=(255, 255, 255), font=font)
    metric = f"frame {frame_id}  t={float(row.get('time_sec') or 0):.2f}s  {row.get('track_id', '')}"
    draw.text((10, 30), metric, fill=(210, 224, 235), font=small_font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(output_path, quality=92)
    return output_path.name


def visual_image_grid(image_names: list[str], rel_dir: str) -> str:
    figures = []
    for image_name in image_names:
        figures.append(f'<img src="{html.escape(rel_dir)}/{html.escape(image_name)}" alt="{html.escape(image_name)}">')
    return '<div class="image-strip">' + "".join(figures) + "</div>"


def render_track_triplet_crops(
    clip_path: Path,
    track_id: str,
    rows_by_track: dict[str, list[dict]],
    output_dir: Path,
    frame_cache_dir: Path,
    prefix: str,
) -> list[str]:
    images = []
    for index, row in enumerate(selected_track_rows(rows_by_track.get(track_id, []))):
        label = f"{track_id} {'first/mid/last'.split('/')[min(index, 2)]}"
        image_name = f"{prefix}_{track_id}_{index + 1}_f{int(float(row['frame_id'])):06d}.jpg"
        images.append(render_detection_crop(clip_path, row, output_dir / image_name, label, frame_cache_dir))
    return images


def render_link_crops(
    clip_path: Path,
    link: dict,
    rows_by_track: dict[str, list[dict]],
    output_dir: Path,
    frame_cache_dir: Path,
    index: int,
) -> list[str]:
    crops = []
    from_id = link["from_raw_track_id"]
    to_id = link["to_raw_track_id"]
    pairs = [
        ("from", from_id, int(float(link["from_frame"]))),
        ("to", to_id, int(float(link["to_frame"]))),
    ]
    for side, track_id, frame_id in pairs:
        row = nearest_detection_row(rows_by_track.get(track_id, []), frame_id)
        if row is None:
            continue
        label = f"{side.upper()} {track_id} -> {to_id if side == 'from' else from_id}"
        image_name = f"link_{index:03d}_{side}_{track_id}_f{int(float(row['frame_id'])):06d}.jpg"
        crops.append(render_detection_crop(clip_path, row, output_dir / image_name, label, frame_cache_dir))
    return crops


def write_visual_audit_html(
    path: Path,
    clip_path: Path,
    quality_rows: list[dict],
    link_rows: list[dict],
    detection_rows: list[dict],
) -> None:
    final_ids = {row["track_id"] for row in quality_rows}
    rows_by_track = rows_by_normalized_track(detection_rows, final_ids)
    groups = review_case_groups(quality_rows, link_rows)
    crop_root = path.parent / "review_crops"
    frame_cache = crop_root / "_frames"
    rel_crop = "review_crops"

    link_cards = []
    for index, link in enumerate(groups["link_candidates"], start=1):
        images = render_link_crops(clip_path, link, rows_by_track, crop_root / "link_candidates", frame_cache, index)
        link_cards.append(
            visual_card(
                title=f"{link['from_raw_track_id']} -> {link['to_raw_track_id']}",
                badge=link["review_decision"],
                meta=[
                    f"gap={link['gap_frames']} frames",
                    f"pred_dist={link['predicted_distance_px']}px",
                    f"size_ratio={link['size_ratio']}",
                ],
                images=images,
                rel_dir=f"{rel_crop}/link_candidates",
                actions=["ACCEPT_LINK", "REJECT_LINK", "UNCERTAIN"],
            )
        )

    motor_cards = []
    for row in groups["motor_vehicle"]:
        images = render_track_triplet_crops(clip_path, row["track_id"], rows_by_track, crop_root / "motor_vehicle_review", frame_cache, "motor")
        motor_cards.append(
            visual_card(
                title=f"{row['track_id']}  {row['result_direction']}",
                badge="PENDING_MOTOR_REVIEW",
                meta=[
                    f"duration={row['duration_sec']}s",
                    f"disp={row['max_displacement_px']}px",
                    f"crossings={row['crossing_count']}",
                ],
                images=images,
                rel_dir=f"{rel_crop}/motor_vehicle_review",
                actions=["MOTOR_VEHICLE", "NON_MOTOR", "UNCERTAIN"],
            )
        )

    static_cards = []
    for row in groups["static_or_parked"]:
        images = render_track_triplet_crops(clip_path, row["track_id"], rows_by_track, crop_root / "static_or_parked_review", frame_cache, "static")
        static_cards.append(
            visual_card(
                title=f"{row['track_id']} static/parked candidate",
                badge=row["exclude_reason"],
                meta=[
                    f"duration={row['duration_sec']}s",
                    f"disp={row['max_displacement_px']}px",
                    f"crossings={row['crossing_count']}",
                ],
                images=images,
                rel_dir=f"{rel_crop}/static_or_parked_review",
                actions=["PARKED_EXCLUDE", "KEEP_WAITING_VEHICLE", "UNCERTAIN"],
            )
        )

    partial_cards = []
    for row in groups["window_partial"]:
        images = render_track_triplet_crops(clip_path, row["track_id"], rows_by_track, crop_root / "window_partial_review", frame_cache, "partial")
        partial_cards.append(
            visual_card(
                title=f"{row['track_id']} window partial",
                badge=row["window_partial_status"],
                meta=[
                    f"duration={row['duration_sec']}s",
                    f"disp={row['max_displacement_px']}px",
                    f"crossings={row['crossing_count']}",
                ],
                images=images,
                rel_dir=f"{rel_crop}/window_partial_review",
                actions=["ACCEPT_PARTIAL", "REVIEW_PARTIAL", "EXCLUDE"],
            )
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    content = visual_audit_document(
        link_cards=link_cards,
        motor_cards=motor_cards,
        static_cards=static_cards,
        partial_cards=partial_cards,
    )
    path.write_text(content, encoding="utf-8")


def visual_card(title: str, badge: str, meta: list[str], images: list[str], rel_dir: str, actions: list[str]) -> str:
    meta_html = "".join(f"<span>{html.escape(item)}</span>" for item in meta)
    action_html = "".join(f"<button type=\"button\">{html.escape(action)}</button>" for action in actions)
    return f"""
    <article class="case-card">
      <div class="case-head">
        <h3>{html.escape(title)}</h3>
        <strong>{html.escape(badge)}</strong>
      </div>
      {visual_image_grid(images, rel_dir)}
      <div class="metric-row">{meta_html}</div>
      <div class="action-row">{action_html}</div>
    </article>
    """


def visual_audit_document(link_cards: list[str], motor_cards: list[str], static_cards: list[str], partial_cards: list[str]) -> str:
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>YOLO26 Direction Ready Visual Review</title>
  <style>
    :root {{
      --paper: #ece7de;
      --ink: #1a232c;
      --muted: #5d6974;
      --line: #c8c0b3;
      --panel: #fbfaf7;
      --navy: #192d3c;
      --green: #247052;
      --red: #b83238;
      --amber: #b2741f;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--paper);
      color: var(--ink);
      font-family: "Avenir Next", "Helvetica Neue", Arial, sans-serif;
    }}
    header {{
      padding: 22px 30px;
      background: var(--navy);
      color: #f8f3ea;
      border-bottom: 5px solid var(--amber);
    }}
    header h1 {{ margin: 0 0 6px; font-size: 28px; letter-spacing: 0; }}
    header p {{ margin: 0; max-width: 980px; color: #dbe4ea; line-height: 1.55; }}
    main {{ max-width: 1720px; margin: 0 auto; padding: 18px; }}
    section {{ margin-bottom: 24px; }}
    .section-head {{
      display: flex;
      align-items: end;
      justify-content: space-between;
      border-bottom: 2px solid var(--line);
      padding: 8px 2px 10px;
      margin-bottom: 12px;
    }}
    .section-head h2 {{ margin: 0; font-size: 22px; }}
    .section-head span {{ color: var(--muted); font-size: 13px; }}
    .case-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(520px, 1fr));
      gap: 14px;
    }}
    .case-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 6px;
      overflow: hidden;
      box-shadow: 0 8px 18px rgba(35, 31, 24, 0.09);
    }}
    .case-head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      padding: 11px 12px;
      background: #f5f0e7;
      border-bottom: 1px solid var(--line);
    }}
    .case-head h3 {{ margin: 0; font-size: 17px; }}
    .case-head strong {{
      color: #fff;
      background: var(--navy);
      border-radius: 4px;
      padding: 5px 7px;
      font-size: 12px;
      white-space: nowrap;
    }}
    .image-strip {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 1px;
      background: #2b333b;
    }}
    .image-strip img {{
      width: 100%;
      aspect-ratio: 4 / 3;
      object-fit: cover;
      display: block;
      background: #111820;
    }}
    .metric-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      padding: 10px 12px;
      border-top: 1px solid var(--line);
    }}
    .metric-row span {{
      border: 1px solid #d7d0c5;
      border-radius: 4px;
      padding: 5px 7px;
      font-size: 12px;
      color: var(--muted);
      background: #fffdf8;
    }}
    .action-row {{
      display: flex;
      gap: 8px;
      padding: 0 12px 12px;
    }}
    button {{
      border: 1px solid var(--navy);
      background: var(--navy);
      color: #fff;
      border-radius: 4px;
      padding: 8px 10px;
      font-weight: 700;
      font-size: 12px;
      cursor: default;
    }}
    button:nth-child(2) {{ background: #fff; color: var(--red); border-color: var(--red); }}
    button:nth-child(3) {{ background: #fff; color: var(--amber); border-color: var(--amber); }}
    @media (max-width: 760px) {{
      .case-grid {{ grid-template-columns: 1fr; }}
      .image-strip {{ grid-template-columns: 1fr; }}
      .section-head {{ display: block; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>YOLO26 Direction Ready Visual Review</h1>
    <p>用原视频帧、检测框和关键指标审核 ID 拼接、机动车属性、静止/路边目标、窗口截断 partial。按钮是审核动作标签，当前版本用于人工记录决策。</p>
  </header>
  <main>
    <section>
      <div class="section-head"><h2>ID 拼接审核</h2><span>{len(link_cards)} cases</span></div>
      <div class="case-grid">{''.join(link_cards)}</div>
    </section>
    <section>
      <div class="section-head"><h2>机动车属性审核</h2><span>{len(motor_cards)} cases</span></div>
      <div class="case-grid">{''.join(motor_cards)}</div>
    </section>
    <section>
      <div class="section-head"><h2>静止/路边目标审核</h2><span>{len(static_cards)} cases</span></div>
      <div class="case-grid">{''.join(static_cards)}</div>
    </section>
    <section>
      <div class="section-head"><h2>窗口截断 Partial 审核</h2><span>{len(partial_cards)} cases</span></div>
      <div class="case-grid">{''.join(partial_cards)}</div>
    </section>
  </main>
</body>
</html>
"""


def final_track_ids(final_rows: list[dict]) -> set[str]:
    return {row["track_id"] for row in final_rows}


def points_by_track_id(rows: list[dict], id_field: str) -> dict[str, list[TrackPoint]]:
    grouped: dict[str, list[TrackPoint]] = defaultdict(list)
    for row in rows:
        track_id = row[id_field]
        x1 = float(row["x1"])
        y1 = float(row["y1"])
        x2 = float(row["x2"])
        y2 = float(row["y2"])
        grouped[track_id].append(
            TrackPoint(
                track_id=track_id,
                frame_id=int(float(row["frame_id"])),
                time_sec=float(row["time_sec"]),
                bottom_center_x=(x1 + x2) / 2.0,
                bottom_center_y=y2,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
            )
        )
    return {key: sorted(points, key=lambda point: point.frame_id) for key, points in grouped.items()}


def build_od_rows(points_by_id: dict[str, list[TrackPoint]], gate_rows: list[dict], semantic_rows: list[dict], stable_frames: int) -> tuple[list[dict], list[dict]]:
    gates = load_gates(gate_rows)
    semantics = load_direction_semantics(semantic_rows)
    crossing_rows: list[dict] = []
    direction_rows: list[dict] = []
    for track_id in sorted(points_by_id):
        crossings = [crossing.to_row() for crossing in detect_gate_crossings(points_by_id[track_id], gates, stable_frames=stable_frames)]
        crossing_rows.extend(crossings)
        direction_rows.append(build_direction_result(track_id, crossings, semantics))
    return direction_rows, crossing_rows


def enrich_final_rows_with_frames(final_rows: list[dict], detection_rows: list[dict], fps: float) -> list[dict]:
    frames_by_track: dict[str, list[int]] = defaultdict(list)
    for row in detection_rows:
        track_id = normalized_track_id(row["track_id"])
        frames_by_track[track_id].append(int(float(row["frame_id"])))
    enriched = []
    for row in final_rows:
        copy = dict(row)
        frames = frames_by_track.get(row["track_id"], [])
        if frames:
            copy["start_frame"] = str(min(frames))
            copy["end_frame"] = str(max(frames))
        else:
            copy["start_frame"] = str(frame_value(row, "start_frame", fps, 0))
            copy["end_frame"] = str(frame_value(row, "end_frame", fps, 0))
        enriched.append(copy)
    return enriched


def center_distance(first: TrackPoint, point: TrackPoint) -> float:
    cx0 = (first.x1 + first.x2) / 2.0
    cy0 = (first.y1 + first.y2) / 2.0
    cx = (point.x1 + point.x2) / 2.0
    cy = (point.y1 + point.y2) / 2.0
    return math.hypot(cx - cx0, cy - cy0)


def logical_target_rows(summary_rows: list[dict], trajectory_rows: list[dict], raw_quality_by_id: dict[str, dict], logical_od_by_id: dict[str, dict]) -> list[dict]:
    points = points_by_track_id(trajectory_rows, "logical_vehicle_id")
    rows = []
    for summary in summary_rows:
        logical_id = summary["logical_vehicle_id"]
        if logical_id == "__overall__":
            continue
        group = points.get(logical_id, [])
        if not group:
            continue
        frames = [point.frame_id for point in group]
        duration = (max(frames) - min(frames)) / 50.0
        max_displacement = max(center_distance(group[0], point) for point in group)
        raw_ids = summary.get("raw_track_ids", "")
        component_statuses = [
            raw_quality_by_id.get(raw_id, {}).get("keep_for_direction_od", "missing")
            for raw_id in raw_ids.split("|")
            if raw_id
        ]
        od_row = logical_od_by_id.get(logical_id, {})
        keep = "yes" if od_row.get("review_status") == "ACCEPTED" and "yes" in component_statuses else "no"
        rows.append(
            {
                "logical_vehicle_id": logical_id,
                "raw_track_ids": raw_ids,
                "raw_track_id_count": summary.get("raw_track_id_count", ""),
                "detected_frame_count": summary.get("detected_frame_count", ""),
                "start_frame": str(min(frames)),
                "end_frame": str(max(frames)),
                "duration_sec": f"{duration:.2f}",
                "max_displacement_px": f"{max_displacement:.2f}",
                "component_quality_status": "|".join(component_statuses),
                "keep_for_direction_od": keep,
                "review_note": "Logical vehicle has complete OD and at least one accepted raw component." if keep == "yes" else "Review before direction OD use.",
            }
        )
    return rows


def write_audit_html(path: Path, quality_rows: list[dict], link_rows: list[dict], logical_rows: list[dict], direction_rows: list[dict]) -> None:
    quality_counts = Counter(row["exclude_reason"] or "direction_ready" for row in quality_rows)
    od_counts = Counter(row["review_status"] for row in direction_rows)
    link_counts = Counter(row["review_decision"] for row in link_rows)

    def table(rows: list[dict], fields: list[str], limit: int = 80) -> str:
        head = "".join(f"<th>{html.escape(field)}</th>" for field in fields)
        body = []
        for row in rows[:limit]:
            body.append("<tr>" + "".join(f"<td>{html.escape(str(row.get(field, '')))}</td>" for field in fields) + "</tr>")
        return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"

    content = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>YOLO26 Direction Ready V1 Review</title>
  <style>
    body {{ margin: 0; font-family: Arial, "PingFang SC", sans-serif; background: #f4f5f7; color: #18212b; }}
    header {{ padding: 20px 28px; background: #1f3446; color: white; }}
    main {{ padding: 18px; max-width: 1600px; margin: 0 auto; }}
    section {{ background: white; border: 1px solid #d7dde4; border-radius: 6px; padding: 14px; margin-bottom: 14px; }}
    .stats {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }}
    pre {{ background: #111820; color: #e8eef5; padding: 10px; overflow: auto; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
    th, td {{ border: 1px solid #d9e0e7; padding: 5px 7px; text-align: left; vertical-align: top; }}
    th {{ background: #eef2f6; position: sticky; top: 0; }}
  </style>
</head>
<body>
  <header>
    <h1>YOLO26 Direction Ready V1 Review</h1>
    <p>Quality gate, logical ID continuity, and approach-level direction OD candidates. This layer is not lane assignment and not SUMO route generation.</p>
  </header>
  <main>
    <section>
      <h2>Summary</h2>
      <div class="stats">
        <pre>quality_counts={html.escape(str(dict(quality_counts)))}</pre>
        <pre>link_counts={html.escape(str(dict(link_counts)))}</pre>
        <pre>logical_od_counts={html.escape(str(dict(od_counts)))}</pre>
      </div>
    </section>
    <section>
      <h2>Raw Track Quality Gate</h2>
      {table(quality_rows, QUALITY_FIELDS)}
    </section>
    <section>
      <h2>Fragment Link Review</h2>
      {table(link_rows, LINK_FIELDS)}
    </section>
    <section>
      <h2>Logical Vehicle Targets</h2>
      {table(logical_rows, LOGICAL_TARGET_FIELDS)}
    </section>
    <section>
      <h2>Logical Direction OD</h2>
      {table(direction_rows, DIRECTION_FIELDS)}
    </section>
  </main>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_note(path: Path, quality_rows: list[dict], link_rows: list[dict], logical_rows: list[dict], direction_rows: list[dict]) -> None:
    lines = [
        "# YOLO26 Direction Ready V1",
        "",
        "This output is a reviewable post-processing layer for direction-level OD.",
        "It does not modify YOLO detections, assign lanes, or generate SUMO routes.",
        "",
        f"- raw_track_quality_rows: `{len(quality_rows)}`",
        f"- raw_direction_ready_tracks: `{sum(row['keep_for_direction_od'] == 'yes' for row in quality_rows)}`",
        f"- accepted_cross_raw_links: `{sum(row['review_decision'] == 'ACCEPT_LOW_RISK_CONTINUITY' for row in link_rows)}`",
        f"- logical_vehicle_targets: `{len(logical_rows)}`",
        f"- logical_direction_accepted: `{sum(row['review_status'] == 'ACCEPTED' for row in direction_rows)}`",
        "",
        "Window-boundary partial tracks are separated from true failures. A track that starts at frame 0 and only exits, or ends at the final frame and only enters, should not be forced into a complete OD label.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_direction_ready_v1(
    final_rows: list[dict],
    detection_rows: list[dict],
    gate_rows: list[dict],
    semantic_rows: list[dict],
    output_dir: Path,
    clip_path: Path | None = None,
    fps: float = 50.0,
    stable_frames: int = 5,
    video_last_frame: int = 1999,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    final_rows = enrich_final_rows_with_frames(final_rows, detection_rows, fps)
    final_ids = final_track_ids(final_rows)
    raw_detection_rows = [
        {**row, "track_id": normalized_track_id(row["track_id"])}
        for row in detection_rows
        if normalized_track_id(row["track_id"]) in final_ids
    ]
    raw_points = points_by_track_id(raw_detection_rows, "track_id")
    raw_direction_rows, raw_crossing_rows = build_od_rows(raw_points, gate_rows, semantic_rows, stable_frames)
    raw_od_by_id = {row["track_id"]: row for row in raw_direction_rows}
    quality_rows = [
        build_quality_review_row(row, raw_od_by_id.get(row["track_id"], {"crossing_count": "0", "review_status": "UNKNOWN", "result_direction": "unknown"}), video_last_frame=video_last_frame, fps=fps)
        for row in final_rows
    ]
    raw_quality_by_id = {row["track_id"]: row for row in quality_rows}

    center_outputs = build_center_follow_mock(
        detection_rows=detection_rows,
        allowed_track_ids=final_ids,
        fps=fps,
        max_gap_frames=20,
        max_prediction_distance_px=120.0,
        max_size_ratio=3.0,
    )
    link_rows = []
    for row in center_outputs.links:
        link_rows.append({**row, "review_decision": link_decision(row)})

    logical_points = points_by_track_id(center_outputs.trajectory_rows, "logical_vehicle_id")
    logical_direction_rows, logical_crossing_rows = build_od_rows(logical_points, gate_rows, semantic_rows, stable_frames)
    logical_od_by_id = {row["track_id"]: row for row in logical_direction_rows}
    logical_rows = logical_target_rows(center_outputs.summary_rows, center_outputs.trajectory_rows, raw_quality_by_id, logical_od_by_id)

    write_csv(output_dir / "raw_track_quality_review.csv", quality_rows, QUALITY_FIELDS)
    write_csv(output_dir / "raw_track_direction_od.csv", raw_direction_rows, DIRECTION_FIELDS)
    write_csv(output_dir / "raw_track_approach_crossings.csv", raw_crossing_rows, CROSSING_FIELDS)
    write_csv(output_dir / "track_fragment_link_candidates.csv", link_rows, LINK_FIELDS)
    write_csv(output_dir / "track_fragment_links_accepted.csv", [row for row in link_rows if row["review_decision"].startswith("ACCEPT")], LINK_FIELDS)
    write_csv(output_dir / "logical_vehicle_tracks.csv", center_outputs.trajectory_rows, list(center_outputs.trajectory_rows[0].keys()) if center_outputs.trajectory_rows else [])
    write_csv(output_dir / "logical_vehicle_targets.csv", logical_rows, LOGICAL_TARGET_FIELDS)
    write_csv(output_dir / "logical_direction_od.csv", logical_direction_rows, DIRECTION_FIELDS)
    write_csv(output_dir / "logical_approach_crossings.csv", logical_crossing_rows, CROSSING_FIELDS)
    write_audit_html(output_dir / "audit_board" / "YOLO26_DIRECTION_READY_REVIEW.html", quality_rows, link_rows, logical_rows, logical_direction_rows)
    if clip_path is not None:
        write_visual_audit_html(
            output_dir / "audit_board" / "YOLO26_DIRECTION_READY_VISUAL_REVIEW.html",
            clip_path,
            quality_rows,
            link_rows,
            detection_rows,
        )
    write_note(output_dir / "DIRECTION_READY_V1_NOTE.md", quality_rows, link_rows, logical_rows, logical_direction_rows)
    return {
        "quality_rows": quality_rows,
        "link_rows": link_rows,
        "logical_rows": logical_rows,
        "logical_direction_rows": logical_direction_rows,
        "raw_direction_rows": raw_direction_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-targets", required=True)
    parser.add_argument("--detections", required=True)
    parser.add_argument("--approach-gates", required=True)
    parser.add_argument("--direction-semantics", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--clip", default="")
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--stable-frames", type=int, default=5)
    parser.add_argument("--video-last-frame", type=int, default=1999)
    args = parser.parse_args()
    outputs = build_direction_ready_v1(
        final_rows=read_csv(Path(args.final_targets)),
        detection_rows=read_csv(Path(args.detections)),
        gate_rows=read_csv(Path(args.approach_gates)),
        semantic_rows=read_csv(Path(args.direction_semantics)),
        output_dir=Path(args.output_dir),
        clip_path=Path(args.clip) if args.clip else None,
        fps=args.fps,
        stable_frames=args.stable_frames,
        video_last_frame=args.video_last_frame,
    )
    print(f"raw_tracks={len(outputs['quality_rows'])}")
    print(f"raw_direction_ready={sum(row['keep_for_direction_od'] == 'yes' for row in outputs['quality_rows'])}")
    print(f"accepted_cross_raw_links={sum(row['review_decision'] == 'ACCEPT_LOW_RISK_CONTINUITY' for row in outputs['link_rows'])}")
    print(f"logical_vehicle_targets={len(outputs['logical_rows'])}")
    print(f"logical_direction_accepted={sum(row['review_status'] == 'ACCEPTED' for row in outputs['logical_direction_rows'])}")
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
