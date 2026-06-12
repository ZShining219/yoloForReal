#!/usr/bin/env python3
"""Render target extraction audit videos on top of the original video clip.

This tool is for human target-extraction review only. It does not infer lanes,
turns, intersection entry/exit times, SUMO routes, or events.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


STATUS_STYLE = {
    "final_target": {"color": (40, 210, 80), "name": "FINAL"},
    "kept_by_user_gate": {"color": (255, 165, 0), "name": "KEEP"},
    "excluded_by_user": {"color": (230, 40, 40), "name": "EXCLUDE"},
    "summary_excluded": {"color": (160, 160, 160), "name": "RAW-EXCLUDED"},
    "raw_unreviewed": {"color": (190, 190, 190), "name": "RAW"},
}


@dataclass
class DisplaySuppression:
    segment_rows: list[dict]
    suppressed_frame_keys: set[tuple[str, int]]


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def normalized_track_id(track_id: str) -> str:
    if track_id.startswith("mot_"):
        return track_id
    return f"mot_{int(float(track_id)):04d}"


def rows_by_frame(detection_rows: list[dict]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in detection_rows:
        grouped[int(float(row["frame_id"]))].append(row)
    return grouped


def trajectory_id(row: dict) -> str:
    logical_vehicle_id = row.get("logical_vehicle_id", "").strip()
    return logical_vehicle_id if logical_vehicle_id else normalized_track_id(row["track_id"])


def trajectory_points_by_id(detection_rows: list[dict]) -> dict[str, list[tuple[int, float, float]]]:
    grouped: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
    for row in detection_rows:
        frame_id = int(float(row["frame_id"]))
        x1, y1, x2, y2 = [float(row[key]) for key in ("x1", "y1", "x2", "y2")]
        grouped[trajectory_id(row)].append((frame_id, (x1 + x2) / 2.0, (y1 + y2) / 2.0))
    return {key: sorted(points) for key, points in grouped.items()}


def rows_by_track(detection_rows: list[dict], final_ids: set[str]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in detection_rows:
        track_id = normalized_track_id(row["track_id"])
        if track_id in final_ids:
            grouped[track_id].append(row)
    return grouped


def consecutive_segments(rows: list[dict]) -> list[list[dict]]:
    ordered = sorted(rows, key=lambda row: int(float(row["frame_id"])))
    if not ordered:
        return []
    segments: list[list[dict]] = []
    current = [ordered[0]]
    previous_frame = int(float(ordered[0]["frame_id"]))
    for row in ordered[1:]:
        frame_id = int(float(row["frame_id"]))
        if frame_id == previous_frame + 1:
            current.append(row)
        else:
            segments.append(current)
            current = [row]
        previous_frame = frame_id
    segments.append(current)
    return segments


def segment_summary_row(
    track_id: str,
    segment: list[dict],
    previous_gap_frames: int | None,
    next_gap_frames: int | None,
    fps: float,
) -> dict:
    frames = [int(float(row["frame_id"])) for row in segment]
    centers_x = []
    centers_y = []
    confidences = []
    for row in segment:
        x1, y1, x2, y2 = [float(row[key]) for key in ("x1", "y1", "x2", "y2")]
        centers_x.append((x1 + x2) / 2.0)
        centers_y.append((y1 + y2) / 2.0)
        confidences.append(float(row.get("confidence") or 0.0))
    start_frame = min(frames)
    end_frame = max(frames)
    return {
        "track_id": track_id,
        "class_name": segment[0].get("class_name", ""),
        "start_frame": str(start_frame),
        "end_frame": str(end_frame),
        "start_time": f"{start_frame / fps:.2f}",
        "end_time": f"{end_frame / fps:.2f}",
        "frame_count": str(len(segment)),
        "previous_gap_frames": "" if previous_gap_frames is None else str(previous_gap_frames),
        "next_gap_frames": "" if next_gap_frames is None else str(next_gap_frames),
        "mean_center_x": f"{sum(centers_x) / len(centers_x):.2f}",
        "mean_center_y": f"{sum(centers_y) / len(centers_y):.2f}",
        "mean_confidence": f"{sum(confidences) / len(confidences):.4f}",
        "display_filter_action": "SUPPRESS_OVERLAY_ONLY",
    }


def build_isolated_display_suppression(
    detection_rows: list[dict],
    final_ids: set[str],
    max_segment_frames: int,
    min_gap_frames: int,
    fps: float,
) -> DisplaySuppression:
    segment_rows: list[dict] = []
    suppressed_frame_keys: set[tuple[str, int]] = set()
    for track_id, track_rows in rows_by_track(detection_rows, final_ids).items():
        segments = consecutive_segments(track_rows)
        for index, segment in enumerate(segments):
            previous_gap = None
            next_gap = None
            start_frame = int(float(segment[0]["frame_id"]))
            end_frame = int(float(segment[-1]["frame_id"]))
            if index > 0:
                previous_end = int(float(segments[index - 1][-1]["frame_id"]))
                previous_gap = start_frame - previous_end - 1
            if index < len(segments) - 1:
                next_start = int(float(segments[index + 1][0]["frame_id"]))
                next_gap = next_start - end_frame - 1
            previous_isolated = previous_gap is None or previous_gap >= min_gap_frames
            next_isolated = next_gap is None or next_gap >= min_gap_frames
            if len(segment) <= max_segment_frames and previous_isolated and next_isolated:
                segment_rows.append(segment_summary_row(track_id, segment, previous_gap, next_gap, fps))
                for row in segment:
                    suppressed_frame_keys.add((track_id, int(float(row["frame_id"]))))
    segment_rows.sort(key=lambda row: (int(row["start_frame"]), row["track_id"]))
    return DisplaySuppression(segment_rows=segment_rows, suppressed_frame_keys=suppressed_frame_keys)


def suppress_rows_for_display(rows: list[dict], suppressed_frame_keys: set[tuple[str, int]]) -> list[dict]:
    return [
        row
        for row in rows
        if (normalized_track_id(row["track_id"]), int(float(row["frame_id"]))) not in suppressed_frame_keys
    ]


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_track_statuses(
    summary_rows: list[dict],
    false_positive_rows: list[dict],
    final_rows: list[dict],
) -> dict[str, str]:
    statuses = {}
    for row in summary_rows:
        track_id = row["track_id"]
        if row.get("track_status") == "target_review_candidate":
            statuses[track_id] = "raw_unreviewed"
        else:
            statuses[track_id] = "summary_excluded"

    for row in final_rows:
        statuses[row["track_id"]] = "final_target"

    for row in false_positive_rows:
        track_id = row["track_id"]
        review_status = row.get("review_status", "").strip().upper()
        if review_status == "EXCLUDE":
            statuses[track_id] = "excluded_by_user"
        elif review_status == "KEEP":
            statuses[track_id] = "kept_by_user_gate"
    return statuses


def final_track_ids(final_rows: list[dict]) -> set[str]:
    return {row["track_id"] for row in final_rows}


def filter_rows_for_mode(rows: list[dict], mode: str, final_ids: set[str]) -> list[dict]:
    if mode == "all":
        return rows
    if mode == "final":
        return [row for row in rows if normalized_track_id(row["track_id"]) in final_ids]
    raise ValueError(f"Unknown overlay mode: {mode}")


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
    text_w, text_h = text_size(draw, text, font)
    top = max(0, y - text_h - 8)
    left = max(0, x)
    draw.rectangle((left, top, left + text_w + 10, top + text_h + 8), fill=color)
    draw.text((left + 5, top + 4), text, fill=(20, 20, 20), font=font)


def draw_legend(draw, mode: str, font, small_font) -> None:
    x = 14
    y = 48
    draw.rectangle((8, 8, 520, 142), fill=(245, 245, 245), outline=(60, 60, 60), width=1)
    draw.text((14, 16), f"target overlay review | mode={mode}", fill=(20, 20, 20), font=font)
    for status in ["final_target", "kept_by_user_gate", "excluded_by_user", "summary_excluded"]:
        style = STATUS_STYLE[status]
        draw.rectangle((x, y - 13, x + 20, y + 7), fill=style["color"])
        draw.text((x + 28, y - 13), style["name"], fill=(20, 20, 20), font=small_font)
        y += 26


def draw_frame_header(draw, width: int, frame_id: int, fps: float, font) -> None:
    text = f"frame={frame_id:06d}  time={frame_id / fps:.2f}s"
    text_w, text_h = text_size(draw, text, font)
    x = max(12, width - text_w - 26)
    y = 15
    draw.rectangle((x - 8, y - 6, width - 12, y + text_h + 10), fill=(245, 245, 245), outline=(60, 60, 60), width=1)
    draw.text((x, y), text, fill=(20, 20, 20), font=font)


def draw_detection(draw, row: dict, status: str, font) -> None:
    style = STATUS_STYLE.get(status, STATUS_STYLE["raw_unreviewed"])
    color = style["color"]
    x1, y1, x2, y2 = [int(round(float(row[key]))) for key in ("x1", "y1", "x2", "y2")]
    for offset in range(2):
        draw.rectangle((x1 - offset, y1 - offset, x2 + offset, y2 + offset), outline=color)
    label = detection_label(row, style["name"])
    draw_label(draw, label, x1, max(18, y1 - 4), color, font)


def color_for_id(identifier: str) -> tuple[int, int, int]:
    value = sum((idx + 1) * ord(char) for idx, char in enumerate(identifier))
    red = 60 + (value * 47) % 170
    green = 60 + (value * 83) % 170
    blue = 60 + (value * 131) % 170
    return (red, green, blue)


def draw_trajectory_trails(
    draw,
    frame_id: int,
    points_by_id: dict[str, list[tuple[int, float, float]]],
    history_frames: int,
) -> None:
    min_frame = max(0, frame_id - history_frames)
    for identifier, points in points_by_id.items():
        recent = [(x, y) for point_frame, x, y in points if min_frame <= point_frame <= frame_id]
        if len(recent) < 2:
            continue
        color = color_for_id(identifier)
        draw.line(recent, fill=color, width=3)
        end_x, end_y = recent[-1]
        draw.ellipse((end_x - 4, end_y - 4, end_x + 4, end_y + 4), fill=color)


def detection_label(row: dict, status_name: str) -> str:
    track_id = normalized_track_id(row["track_id"])
    logical_vehicle_id = row.get("logical_vehicle_id", "").strip()
    display_id = f"{logical_vehicle_id}/{track_id}" if logical_vehicle_id else track_id
    class_name = row.get("class_name", "")
    confidence = float(row.get("confidence", 0.0))
    return f"{display_id} {status_name} {class_name} {confidence:.2f}"


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
    data = json.loads(result.stdout)
    stream = data["streams"][0]
    return int(stream["width"]), int(stream["height"])


def render_overlay_video(
    clip_path: Path,
    detections: list[dict],
    statuses: dict[str, str],
    final_ids: set[str],
    output_path: Path,
    mode: str,
    fps: float,
    suppressed_frame_keys: set[tuple[str, int]] | None = None,
    draw_trajectories: bool = False,
    trajectory_history_frames: int = 250,
) -> int:
    from PIL import Image, ImageDraw

    grouped = rows_by_frame(detections)
    points_by_id = trajectory_points_by_id(detections) if draw_trajectories else {}
    suppressed_frame_keys = suppressed_frame_keys or set()
    width, height = probe_video(clip_path)
    output_fps = fps or 50.0
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
            f"{output_fps}",
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
        raise RuntimeError("Failed to open ffmpeg raw-video pipes")

    frame_id = 0
    written = 0
    label_font = load_font(16)
    legend_font = load_font(18)
    small_font = load_font(15)
    while True:
        raw_frame = decode.stdout.read(frame_bytes)
        if not raw_frame:
            break
        if len(raw_frame) != frame_bytes:
            raise RuntimeError(f"Incomplete raw frame at frame_id={frame_id}")
        image = Image.frombytes("RGB", (width, height), raw_frame)
        draw = ImageDraw.Draw(image)
        if draw_trajectories:
            draw_trajectory_trails(draw, frame_id, points_by_id, trajectory_history_frames)
        draw_legend(draw, mode, legend_font, small_font)
        draw_frame_header(draw, width, frame_id, output_fps, legend_font)
        selected_rows = filter_rows_for_mode(grouped.get(frame_id, []), mode=mode, final_ids=final_ids)
        selected_rows = suppress_rows_for_display(selected_rows, suppressed_frame_keys)
        for row in selected_rows:
            track_id = normalized_track_id(row["track_id"])
            draw_detection(draw, row, statuses.get(track_id, "raw_unreviewed"), label_font)
        encode.stdin.write(image.tobytes())
        frame_id += 1
        written += 1

    decode.stdout.close()
    encode.stdin.close()
    decode_code = decode.wait()
    encode_code = encode.wait()
    if decode_code != 0:
        raise RuntimeError(f"ffmpeg decode failed with exit code {decode_code}")
    if encode_code != 0:
        raise RuntimeError(f"ffmpeg encode failed with exit code {encode_code}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", required=True)
    parser.add_argument("--detections", required=True)
    parser.add_argument("--target-summary", required=True)
    parser.add_argument("--false-positive-gate", required=True)
    parser.add_argument("--final-targets", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["all", "final"], required=True)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--suppress-isolated-display", action="store_true")
    parser.add_argument("--max-isolated-segment-frames", type=int, default=5)
    parser.add_argument("--min-isolated-gap-frames", type=int, default=6)
    parser.add_argument("--suppressed-segments-output")
    parser.add_argument("--draw-trajectories", action="store_true")
    parser.add_argument("--trajectory-history-frames", type=int, default=250)
    args = parser.parse_args()

    detection_rows = read_csv(Path(args.detections))
    summary_rows = read_csv(Path(args.target_summary))
    false_positive_rows = read_csv(Path(args.false_positive_gate))
    final_rows = read_csv(Path(args.final_targets))
    statuses = build_track_statuses(summary_rows, false_positive_rows, final_rows)
    final_ids = final_track_ids(final_rows)
    suppression = DisplaySuppression(segment_rows=[], suppressed_frame_keys=set())
    if args.suppress_isolated_display:
        suppression = build_isolated_display_suppression(
            detection_rows=detection_rows,
            final_ids=final_ids,
            max_segment_frames=args.max_isolated_segment_frames,
            min_gap_frames=args.min_isolated_gap_frames,
            fps=args.fps,
        )
        if args.suppressed_segments_output:
            write_csv(
                Path(args.suppressed_segments_output),
                suppression.segment_rows,
                [
                    "track_id",
                    "class_name",
                    "start_frame",
                    "end_frame",
                    "start_time",
                    "end_time",
                    "frame_count",
                    "previous_gap_frames",
                    "next_gap_frames",
                    "mean_center_x",
                    "mean_center_y",
                    "mean_confidence",
                    "display_filter_action",
                ],
            )
    written = render_overlay_video(
        clip_path=Path(args.clip),
        detections=detection_rows,
        statuses=statuses,
        final_ids=final_ids,
        output_path=Path(args.output),
        mode=args.mode,
        fps=args.fps,
        suppressed_frame_keys=suppression.suppressed_frame_keys,
        draw_trajectories=args.draw_trajectories,
        trajectory_history_frames=args.trajectory_history_frames,
    )
    print(f"mode={args.mode}")
    print(f"display_suppressed_segments={len(suppression.segment_rows)}")
    print(f"display_suppressed_frames={len(suppression.suppressed_frame_keys)}")
    print(f"frames_written={written}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
