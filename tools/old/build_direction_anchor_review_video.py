#!/usr/bin/env python3
"""Render dynamic direction-anchor review video for the 0-30s window."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import defaultdict
from pathlib import Path

from build_logical_consistency_review_video import (
    draw_label,
    draw_trails,
    color_for_logical_id,
    points_by_id,
    probe_video,
    read_csv,
    rows_by_frame,
    text_size,
)


DIRECTION_CN = {
    "E": "东",
    "W": "西",
    "N": "北",
    "S": "南",
}


def load_review_font(size: int):
    from PIL import ImageFont

    for path in [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def direction_label(direction: str) -> str:
    return DIRECTION_CN.get(direction, direction or "?")


def anchor_summary_label(row: dict) -> str:
    entry_method = row.get("entry_estimation_method", "")
    method_label = "manual" if entry_method == "manual_entry_time_override" else "inferred"
    return (
        f"{row.get('track_id', '')} {row.get('result_direction', 'unknown')} "
        f"入{direction_label(row.get('estimated_entry_direction', ''))}@{row.get('estimated_entry_time_sec', '')}s "
        f"出{direction_label(row.get('observed_exit_direction', ''))}@{row.get('observed_exit_time_sec', '')}s "
        f"{method_label} {row.get('confidence_level', '')}"
    )


def anchor_method_label(anchor_row: dict) -> str:
    if anchor_row.get("entry_estimation_method") == "manual_review_supplement":
        return "review"
    if anchor_row.get("entry_estimation_method") == "observed_gate_crossing":
        return "observed"
    return "manual" if anchor_row.get("entry_estimation_method") == "manual_entry_time_override" else "inferred"


def anchor_box_label(anchor_row: dict) -> str:
    return (
        f"{anchor_row.get('result_direction', 'unknown')} | "
        f"入{direction_label(anchor_row.get('estimated_entry_direction', ''))} {anchor_row.get('estimated_entry_time_sec', '')}s | "
        f"出{direction_label(anchor_row.get('observed_exit_direction', ''))} {anchor_row.get('observed_exit_time_sec', '')}s | "
        f"{anchor_method_label(anchor_row)}"
    )


def anchor_event_labels(anchor_row: dict, frame_id: int, hold_frames: int, start_frame_offset: int = 0) -> list[str]:
    labels = []
    track_id = anchor_row.get("track_id", "")
    entry_frame_text = anchor_row.get("estimated_entry_frame", "")
    if entry_frame_text:
        entry_frame = int(round(float(entry_frame_text)))
        entry_method = anchor_row.get("entry_estimation_method", "")
        if entry_frame < 0:
            is_active = start_frame_offset <= frame_id <= start_frame_offset + hold_frames
            source = "manual warmup" if entry_method == "manual_entry_time_override" else "backprojected warmup"
        else:
            is_active = entry_frame <= frame_id <= entry_frame + hold_frames
            source = "observed"
        if is_active:
            labels.append(
                f"{track_id} 入{direction_label(anchor_row.get('estimated_entry_direction', ''))} "
                f"@ {anchor_row.get('estimated_entry_time_sec', '')}s {source}"
            )

    exit_frame_text = anchor_row.get("observed_exit_frame", "")
    if exit_frame_text:
        exit_frame = int(round(float(exit_frame_text)))
        if exit_frame <= frame_id <= exit_frame + hold_frames:
            source = "window-start" if anchor_row.get("exit_time_source") == "already_outside_at_window_start" else "observed"
            labels.append(
                f"{track_id} 出{direction_label(anchor_row.get('observed_exit_direction', ''))} "
                f"@ {anchor_row.get('observed_exit_time_sec', '')}s {source}"
            )
    return labels


def anchor_event_highlight(anchor_row: dict, frame_id: int, fps: float, hold_frames: int) -> dict:
    current_time = frame_id / fps
    entry_frame_text = anchor_row.get("estimated_entry_frame", "")
    if entry_frame_text:
        entry_frame = int(round(float(entry_frame_text)))
        if 0 <= entry_frame <= frame_id <= entry_frame + hold_frames:
            source = "review" if anchor_row.get("entry_estimation_method") == "manual_review_supplement" else "observed"
            return {
                "active": True,
                "event_kind": "entry",
                "label": (
                    f"当前{current_time:.2f}s | 入{direction_label(anchor_row.get('estimated_entry_direction', ''))} "
                    f"@ {anchor_row.get('estimated_entry_time_sec', '')}s {source}"
                ),
            }
    exit_frame_text = anchor_row.get("observed_exit_frame", "")
    if exit_frame_text:
        exit_frame = int(round(float(exit_frame_text)))
        if exit_frame <= frame_id <= exit_frame + hold_frames:
            source = "window-start" if anchor_row.get("exit_time_source") == "already_outside_at_window_start" else "observed"
            return {
                "active": True,
                "event_kind": "exit",
                "label": (
                    f"当前{current_time:.2f}s | 出{direction_label(anchor_row.get('observed_exit_direction', ''))} "
                    f"@ {anchor_row.get('observed_exit_time_sec', '')}s {source}"
                ),
            }
    return {"active": False, "event_kind": "", "label": ""}


def build_anchor_statuses(anchor_rows: list[dict]) -> dict[str, dict]:
    statuses = {}
    for row in anchor_rows:
        confidence = row.get("confidence_level", "low").upper()
        statuses[row["track_id"]] = {
            "status": f"ANCHOR_{confidence}",
            "result_direction": row.get("result_direction", "unknown"),
            "raw_track_ids": "",
        }
    return statuses


def anchors_by_track(anchor_rows: list[dict]) -> dict[str, dict]:
    return {row["track_id"]: row for row in anchor_rows}


def build_observed_od_anchor_row(od_row: dict, fps: float) -> dict:
    first_frame = int(float(od_row["first_crossing_frame"]))
    last_frame = int(float(od_row["last_crossing_frame"]))
    origin = od_row.get("origin_direction") or od_row.get("first_crossing_gate", "")
    destination = od_row.get("destination_direction") or od_row.get("last_crossing_gate", "")
    return {
        "track_id": od_row["track_id"],
        "manual_origin_direction": origin,
        "manual_destination_direction": destination,
        "result_direction": od_row.get("result_direction", f"{origin}_to_{destination}"),
        "route_source": "observed_complete_od",
        "frame0_frame": "",
        "frame0_time_sec": "",
        "frame0_bottom_center_x": "",
        "frame0_bottom_center_y": "",
        "frame0_speed_px_per_sec": "",
        "frame0_heading_dx": "",
        "frame0_heading_dy": "",
        "estimated_entry_direction": origin,
        "estimated_entry_frame": f"{first_frame:.1f}",
        "estimated_entry_time_sec": f"{first_frame / fps:.2f}",
        "entry_estimation_method": "observed_gate_crossing",
        "entry_projection_distance_px": "",
        "observed_exit_direction": destination,
        "observed_exit_frame": str(last_frame),
        "observed_exit_time_sec": f"{last_frame / fps:.2f}",
        "exit_time_source": "observed_gate_crossing",
        "warmup_status": "OBSERVED_COMPLETE_OD",
        "confidence_level": od_row.get("confidence_level", "high"),
        "evidence_note": "Complete in-window OD from accepted direction review.",
    }


def build_manual_supplement_anchor_row(supplement: dict, fps: float) -> dict:
    entry_frame = int(float(supplement["entry_frame"]))
    exit_frame = int(float(supplement["exit_frame"]))
    origin = supplement["origin_direction"]
    destination = supplement["destination_direction"]
    return {
        "track_id": supplement["track_id"],
        "manual_origin_direction": origin,
        "manual_destination_direction": destination,
        "result_direction": supplement.get("result_direction", f"{origin}_to_{destination}"),
        "route_source": "manual_review_supplement",
        "frame0_frame": "",
        "frame0_time_sec": "",
        "frame0_bottom_center_x": "",
        "frame0_bottom_center_y": "",
        "frame0_speed_px_per_sec": "",
        "frame0_heading_dx": "",
        "frame0_heading_dy": "",
        "estimated_entry_direction": origin,
        "estimated_entry_frame": f"{entry_frame:.1f}",
        "estimated_entry_time_sec": f"{entry_frame / fps:.2f}",
        "entry_estimation_method": "manual_review_supplement",
        "entry_projection_distance_px": "",
        "observed_exit_direction": destination,
        "observed_exit_frame": str(exit_frame),
        "observed_exit_time_sec": f"{exit_frame / fps:.2f}",
        "exit_time_source": supplement.get("exit_time_source", "outside_current_window_manual_review"),
        "warmup_status": "MANUAL_REVIEW_SUPPLEMENT",
        "confidence_level": supplement.get("confidence_level", "review"),
        "evidence_note": supplement.get("manual_note", "Manual review supplement."),
    }


def is_accepted_complete_od(od_row: dict) -> bool:
    return (
        od_row.get("review_status") == "ACCEPTED"
        and od_row.get("result_direction", "") not in {"", "unknown"}
        and bool(od_row.get("first_crossing_frame"))
        and bool(od_row.get("last_crossing_frame"))
    )


def build_combined_anchor_rows(
    warmup_rows: list[dict],
    od_rows: list[dict],
    track_ids: set[str],
    fps: float,
    manual_supplement_rows: list[dict] | None = None,
) -> list[dict]:
    combined = list(warmup_rows)
    existing_ids = {row["track_id"] for row in combined}
    for row in manual_supplement_rows or []:
        track_id = row.get("track_id", "")
        if track_id in existing_ids or track_id not in track_ids:
            continue
        combined.append(build_manual_supplement_anchor_row(row, fps=fps))
        existing_ids.add(track_id)
    for row in od_rows:
        track_id = row.get("track_id", "")
        if track_id in existing_ids or track_id not in track_ids or not is_accepted_complete_od(row):
            continue
        combined.append(build_observed_od_anchor_row(row, fps=fps))
        existing_ids.add(track_id)
    return combined


def draw_anchor_legend(draw, width: int, frame_id: int, fps: float, font, small_font) -> None:
    left, top, right, bottom = 10, 10, 980, 88
    draw.rectangle((left, top, right, bottom), fill=(241, 244, 246), outline=(35, 43, 50), width=1)
    draw.text(
        (left + 10, top + 8),
        f"direction anchor review | frame={frame_id:06d} time={frame_id / fps:.2f}s | label: 逻辑ID/原始ID 锚定状态 方向",
        fill=(20, 26, 32),
        font=font,
    )
    draw.text(
        (left + 10, top + 46),
        "口门线: W/N/S/E | 每车框旁显示方向锚定 | 黄色粗框表示当前附近帧存在入/出事件",
        fill=(20, 26, 32),
        font=small_font,
    )
    right_text = "source: warmup anchors + accepted OD + manual review tracks"
    text_w, text_h = text_size(draw, right_text, small_font)
    draw.rectangle((width - text_w - 28, 14, width - 12, 14 + text_h + 14), fill=(241, 244, 246), outline=(35, 43, 50))
    draw.text((width - text_w - 20, 21), right_text, fill=(20, 26, 32), font=small_font)


def draw_gates(draw, gate_rows: list[dict], font) -> None:
    gate_color = (0, 225, 235)
    for row in gate_rows:
        x1 = int(round(float(row["line_x1"])))
        y1 = int(round(float(row["line_y1"])))
        x2 = int(round(float(row["line_x2"])))
        y2 = int(round(float(row["line_y2"])))
        draw.line((x1, y1, x2, y2), fill=gate_color, width=5)
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        draw_label(draw, row.get("approach_id", ""), cx, cy, gate_color, font)


def draw_text_box(draw, lines: list[str], x: int, y: int, fill: tuple[int, int, int], font) -> None:
    widths = []
    heights = []
    for line in lines:
        width, height = text_size(draw, line, font)
        widths.append(width)
        heights.append(height)
    box_width = max(widths) + 12
    line_height = max(heights) + 5
    box_height = line_height * len(lines) + 8
    draw.rectangle((x, y, x + box_width, y + box_height), fill=fill, outline=(25, 30, 35), width=1)
    text_y = y + 5
    for line in lines:
        draw.text((x + 6, text_y), line, fill=(18, 24, 30), font=font)
        text_y += line_height


def draw_anchor_detection(
    draw,
    row: dict,
    statuses: dict[str, dict],
    anchor_rows_by_id: dict[str, dict],
    frame_id: int,
    fps: float,
    event_hold_frames: int,
    frame_width: int,
    font,
) -> None:
    logical_id = row.get("logical_vehicle_id", "")
    status = statuses.get(logical_id, {"status": "REVIEW", "result_direction": "unknown"})
    anchor = anchor_rows_by_id.get(logical_id)
    color = color_for_logical_id(logical_id, status["status"])
    highlight = anchor_event_highlight(anchor, frame_id, fps, event_hold_frames) if anchor else {"active": False}
    if highlight.get("active"):
        color = (255, 220, 35)
    x1, y1, x2, y2 = [int(round(float(row[key]))) for key in ("x1", "y1", "x2", "y2")]
    width = 6 if highlight.get("active") else 3
    for offset in range(width):
        draw.rectangle((x1 - offset, y1 - offset, x2 + offset, y2 + offset), outline=color)

    confidence = float(row.get("confidence") or 0.0)
    lines = [
        f"{logical_id}/{row.get('track_id', '')} {status.get('status', 'REVIEW')} {row.get('class_name', '')} {confidence:.2f}",
    ]
    if anchor:
        lines.append(anchor_box_label(anchor))
    if highlight.get("active"):
        lines.append(highlight["label"])
    label_x = max(0, min(x1, frame_width - 780))
    label_y = max(92, y1 - 12)
    draw_text_box(draw, lines, label_x, label_y, color, font)


def draw_anchor_callouts(
    draw,
    row: dict,
    anchor_rows_by_id: dict[str, dict],
    anchor_start_offsets: dict[str, int],
    frame_id: int,
    hold_frames: int,
    font,
) -> None:
    logical_id = row.get("logical_vehicle_id", "")
    anchor = anchor_rows_by_id.get(logical_id)
    if not anchor:
        return
    labels = anchor_event_labels(
        anchor,
        frame_id=frame_id,
        hold_frames=hold_frames,
        start_frame_offset=anchor_start_offsets.get(logical_id, 0),
    )
    if not labels:
        return
    x1 = int(round(float(row["x1"])))
    y2 = int(round(float(row["y2"])))
    for index, label in enumerate(labels):
        draw_label(draw, label, x1, y2 + 24 + index * 26, (255, 216, 80), font)


def active_anchor_event_labels(
    anchor_rows: list[dict],
    anchor_start_offsets: dict[str, int],
    frame_id: int,
    hold_frames: int,
) -> list[str]:
    labels = []
    for row in anchor_rows:
        labels.extend(
            anchor_event_labels(
                row,
                frame_id=frame_id,
                hold_frames=hold_frames,
                start_frame_offset=anchor_start_offsets.get(row.get("track_id", ""), 0),
            )
        )
    return labels


def draw_active_event_panel(draw, labels: list[str], font, small_font) -> None:
    left, top = 10, 100
    width = 590
    row_height = 28
    visible_labels = labels[:5]
    height = 42 + max(1, len(visible_labels)) * row_height
    draw.rectangle((left, top, left + width, top + height), fill=(255, 249, 219), outline=(65, 55, 30), width=1)
    draw.text((left + 10, top + 8), "active anchor events", fill=(30, 26, 18), font=font)
    if not visible_labels:
        draw.text((left + 10, top + 38), "no active entry/exit event at this frame", fill=(30, 26, 18), font=small_font)
        return
    y = top + 38
    for label in visible_labels:
        draw.text((left + 10, y), label, fill=(30, 26, 18), font=small_font)
        y += row_height


def draw_anchor_summary_panel(draw, anchor_rows: list[dict], width: int, height: int, font, small_font) -> None:
    panel_width = 620
    left = width - panel_width - 12
    top = height - 250
    right = width - 12
    bottom = height - 12
    draw.rectangle((left, top, right, bottom), fill=(245, 246, 247), outline=(35, 43, 50), width=1)
    draw.text((left + 10, top + 8), "0-30s direction anchors", fill=(20, 26, 32), font=font)
    y = top + 40
    for row in anchor_rows[:7]:
        draw.text((left + 10, y), anchor_summary_label(row), fill=(20, 26, 32), font=small_font)
        y += 28


def render_video(
    clip_path: Path,
    logical_rows: list[dict],
    anchor_rows: list[dict],
    gate_rows: list[dict],
    output_path: Path,
    fps: float,
    history_frames: int,
    event_hold_frames: int,
) -> int:
    from PIL import Image, ImageDraw

    grouped = rows_by_frame(logical_rows)
    trails = points_by_id(logical_rows)
    statuses = build_anchor_statuses(anchor_rows)
    anchor_rows_by_id = anchors_by_track(anchor_rows)
    anchor_start_offsets = {row["track_id"]: index * 45 for index, row in enumerate(anchor_rows)}
    width, height = probe_video(clip_path)
    frame_bytes = width * height * 3
    output_path.parent.mkdir(parents=True, exist_ok=True)
    decode = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(clip_path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE,
    )
    encode = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            f"{fps}",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ],
        stdin=subprocess.PIPE,
    )
    if decode.stdout is None or encode.stdin is None:
        raise RuntimeError("Failed to open ffmpeg pipes")
    label_font = load_review_font(16)
    legend_font = load_review_font(18)
    small_font = load_review_font(14)
    frame_id = 0
    written = 0
    while True:
        raw = decode.stdout.read(frame_bytes)
        if not raw:
            break
        if len(raw) != frame_bytes:
            raise RuntimeError(f"Incomplete raw frame at {frame_id}")
        image = Image.frombytes("RGB", (width, height), raw)
        draw = ImageDraw.Draw(image)
        draw_gates(draw, gate_rows, small_font)
        draw_trails(draw, frame_id, trails, statuses, history_frames)
        for row in grouped.get(frame_id, []):
            draw_anchor_detection(draw, row, statuses, anchor_rows_by_id, frame_id, fps, event_hold_frames, width, label_font)
        draw_anchor_legend(draw, width, frame_id, fps, legend_font, small_font)
        encode.stdin.write(image.tobytes())
        frame_id += 1
        written += 1
    decode.stdout.close()
    encode.stdin.close()
    decode_code = decode.wait()
    encode_code = encode.wait()
    if decode_code != 0:
        raise RuntimeError(f"decode failed: {decode_code}")
    if encode_code != 0:
        raise RuntimeError(f"encode failed: {encode_code}")
    return written


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", required=True)
    parser.add_argument("--logical-tracks", required=True)
    parser.add_argument("--anchors", required=True)
    parser.add_argument("--logical-od")
    parser.add_argument("--manual-supplements")
    parser.add_argument("--gates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--combined-anchors-output")
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--history-frames", type=int, default=180)
    parser.add_argument("--event-hold-frames", type=int, default=140)
    args = parser.parse_args()

    logical_rows = read_csv(Path(args.logical_tracks))
    track_ids = {row["logical_vehicle_id"] for row in logical_rows}
    anchor_rows = read_csv(Path(args.anchors))
    if args.logical_od:
        anchor_rows = build_combined_anchor_rows(
            warmup_rows=anchor_rows,
            od_rows=read_csv(Path(args.logical_od)),
            track_ids=track_ids,
            fps=args.fps,
            manual_supplement_rows=read_csv(Path(args.manual_supplements)) if args.manual_supplements else None,
        )
    elif args.manual_supplements:
        anchor_rows = build_combined_anchor_rows(
            warmup_rows=anchor_rows,
            od_rows=[],
            track_ids=track_ids,
            fps=args.fps,
            manual_supplement_rows=read_csv(Path(args.manual_supplements)),
        )
    if args.combined_anchors_output:
        write_csv(Path(args.combined_anchors_output), anchor_rows, list(anchor_rows[0].keys()) if anchor_rows else [])
    written = render_video(
        clip_path=Path(args.clip),
        logical_rows=logical_rows,
        anchor_rows=anchor_rows,
        gate_rows=read_csv(Path(args.gates)),
        output_path=Path(args.output),
        fps=args.fps,
        history_frames=args.history_frames,
        event_hold_frames=args.event_hold_frames,
    )
    status_counts = defaultdict(int)
    for status in build_anchor_statuses(anchor_rows).values():
        status_counts[status["status"]] += 1
    print(f"anchor_tracks={len(anchor_rows)}")
    print(f"status_counts={dict(status_counts)}")
    print(f"frames_written={written}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
