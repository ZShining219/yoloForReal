#!/usr/bin/env python3
"""Render dynamic logical-vehicle consistency review video.

The video shows original frames with bbox overlays, logical IDs, raw YOLO IDs,
direction readiness, and approach-level OD status.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import defaultdict
from pathlib import Path


STATUS_STYLE = {
    "READY": {"color": (40, 210, 90), "description": "complete OD"},
    "LINK": {"color": (60, 170, 240), "description": "cross raw-id continuity"},
    "STATIC": {"color": (230, 65, 65), "description": "static/parked candidate"},
    "PARTIAL": {"color": (245, 176, 45), "description": "window-boundary partial"},
    "DUPLICATE": {"color": (190, 110, 230), "description": "duplicate representative"},
    "REVIEW": {"color": (170, 170, 170), "description": "review only"},
}


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def rows_by_frame(rows: list[dict]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[int(float(row["frame_id"]))].append(row)
    return grouped


def build_logical_statuses(logical_targets: list[dict], logical_od: list[dict], raw_quality: list[dict]) -> dict[str, dict]:
    od_by_id = {row["track_id"]: row for row in logical_od}
    raw_quality_by_id = {row["track_id"]: row for row in raw_quality}
    statuses = {}
    for row in logical_targets:
        logical_id = row["logical_vehicle_id"]
        od = od_by_id.get(logical_id, {})
        raw_ids = [item for item in row.get("raw_track_ids", "").split("|") if item]
        raw_rows = [raw_quality_by_id.get(raw_id, {}) for raw_id in raw_ids]
        result_direction = od.get("result_direction", "unknown")
        if row.get("keep_for_direction_od") == "yes" and od.get("review_status") == "ACCEPTED":
            status = "READY"
        elif od.get("review_status") == "DUPLICATE_REVIEW":
            status = "DUPLICATE"
        elif any(raw.get("static_or_parked_flag") == "yes" for raw in raw_rows):
            status = "STATIC"
        elif any(raw.get("window_partial_status") for raw in raw_rows):
            status = "PARTIAL"
        elif int(float(row.get("raw_track_id_count") or 1)) > 1:
            status = "LINK"
        else:
            status = "REVIEW"
        statuses[logical_id] = {
            "status": status,
            "result_direction": result_direction,
            "raw_track_ids": row.get("raw_track_ids", ""),
        }
    return statuses


def overlay_label(row: dict, status: dict) -> str:
    logical_id = row.get("logical_vehicle_id", "")
    raw_id = row.get("track_id", "")
    class_name = row.get("class_name", "")
    confidence = float(row.get("confidence") or 0.0)
    result = status.get("result_direction", "unknown")
    direction = "" if result == "unknown" else f" {result}"
    return f"{logical_id}/{raw_id} {status.get('status', 'REVIEW')}{direction} {class_name} {confidence:.2f}"


def color_for_logical_id(logical_id: str, status: str) -> tuple[int, int, int]:
    if status in {"STATIC", "PARTIAL"}:
        return STATUS_STYLE[status]["color"]
    value = sum((index + 1) * ord(char) for index, char in enumerate(logical_id))
    return (
        55 + (value * 37) % 180,
        55 + (value * 73) % 180,
        55 + (value * 109) % 180,
    )


def probe_video(clip_path: Path) -> tuple[int, int]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            str(clip_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    return int(stream["width"]), int(stream["height"])


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


def text_size(draw, text: str, font) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def draw_label(draw, text: str, x: int, y: int, color: tuple[int, int, int], font) -> None:
    width, height = text_size(draw, text, font)
    top = max(0, y - height - 8)
    draw.rectangle((x, top, x + width + 10, top + height + 8), fill=color)
    draw.text((x + 5, top + 4), text, fill=(18, 24, 30), font=font)


def compact_legend_box() -> tuple[int, int, int, int]:
    return 10, 10, 880, 74


def draw_legend(draw, width: int, frame_id: int, fps: float, font, small_font) -> None:
    left, top, right, bottom = compact_legend_box()
    draw.rectangle((left, top, right, bottom), fill=(244, 241, 235), outline=(45, 52, 58), width=1)
    draw.text(
        (left + 10, top + 8),
        f"logical vehicle consistency review | frame={frame_id:06d} time={frame_id / fps:.2f}s",
        fill=(20, 26, 32),
        font=font,
    )
    x = left + 10
    y = top + 41
    compact_entries = [
        ("READY", "OD"),
        ("LINK", "ID link"),
        ("STATIC", "parked"),
        ("PARTIAL", "edge"),
        ("DUPLICATE", "rep"),
        ("REVIEW", "only"),
    ]
    for status, label in compact_entries:
        style = STATUS_STYLE[status]
        draw.rectangle((x, y, x + 14, y + 14), fill=style["color"])
        text = f"{status} {label}"
        draw.text((x + 20, y - 2), text, fill=(20, 26, 32), font=small_font)
        text_w, _ = text_size(draw, text, small_font)
        x += text_w + 46
    right_text = "label = logical_id/raw_yolo_id status direction class confidence"
    text_w, text_h = text_size(draw, right_text, small_font)
    draw.rectangle((width - text_w - 30, 14, width - 12, 14 + text_h + 14), fill=(244, 241, 235), outline=(45, 52, 58), width=1)
    draw.text((width - text_w - 22, 21), right_text, fill=(20, 26, 32), font=small_font)


def points_by_id(rows: list[dict]) -> dict[str, list[tuple[int, float, float]]]:
    grouped: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
    for row in rows:
        x1 = float(row["x1"])
        y1 = float(row["y1"])
        x2 = float(row["x2"])
        y2 = float(row["y2"])
        grouped[row["logical_vehicle_id"]].append((int(float(row["frame_id"])), (x1 + x2) / 2.0, y2))
    return {key: sorted(points) for key, points in grouped.items()}


def draw_trails(draw, frame_id: int, trails: dict[str, list[tuple[int, float, float]]], statuses: dict[str, dict], history_frames: int) -> None:
    min_frame = max(0, frame_id - history_frames)
    for logical_id, points in trails.items():
        recent = [(x, y) for point_frame, x, y in points if min_frame <= point_frame <= frame_id]
        if len(recent) < 2:
            continue
        status = statuses.get(logical_id, {}).get("status", "REVIEW")
        color = color_for_logical_id(logical_id, status)
        draw.line(recent, fill=color, width=3)
        x, y = recent[-1]
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)


def draw_detection(draw, row: dict, statuses: dict[str, dict], font) -> None:
    logical_id = row.get("logical_vehicle_id", "")
    status = statuses.get(logical_id, {"status": "REVIEW", "result_direction": "unknown"})
    color = color_for_logical_id(logical_id, status["status"])
    x1, y1, x2, y2 = [int(round(float(row[key]))) for key in ("x1", "y1", "x2", "y2")]
    for offset in range(3):
        draw.rectangle((x1 - offset, y1 - offset, x2 + offset, y2 + offset), outline=color)
    draw_label(draw, overlay_label(row, status), x1, max(18, y1 - 4), color, font)


def render_video(
    clip_path: Path,
    logical_rows: list[dict],
    statuses: dict[str, dict],
    output_path: Path,
    fps: float,
    history_frames: int,
) -> int:
    from PIL import Image, ImageDraw

    grouped = rows_by_frame(logical_rows)
    trails = points_by_id(logical_rows)
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
    label_font = load_font(16)
    legend_font = load_font(18)
    small_font = load_font(14)
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
        draw_trails(draw, frame_id, trails, statuses, history_frames)
        for row in grouped.get(frame_id, []):
            draw_detection(draw, row, statuses, label_font)
        draw_legend(draw, width, frame_id, fps, legend_font, small_font)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", required=True)
    parser.add_argument("--logical-tracks", required=True)
    parser.add_argument("--logical-targets", required=True)
    parser.add_argument("--logical-od", required=True)
    parser.add_argument("--raw-quality", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--history-frames", type=int, default=180)
    args = parser.parse_args()

    logical_rows = read_csv(Path(args.logical_tracks))
    statuses = build_logical_statuses(
        logical_targets=read_csv(Path(args.logical_targets)),
        logical_od=read_csv(Path(args.logical_od)),
        raw_quality=read_csv(Path(args.raw_quality)),
    )
    written = render_video(
        clip_path=Path(args.clip),
        logical_rows=logical_rows,
        statuses=statuses,
        output_path=Path(args.output),
        fps=args.fps,
        history_frames=args.history_frames,
    )
    status_counts = defaultdict(int)
    for status in statuses.values():
        status_counts[status["status"]] += 1
    print(f"logical_tracks={len(statuses)}")
    print(f"status_counts={dict(status_counts)}")
    print(f"frames_written={written}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
