#!/usr/bin/env python3
"""Render logical vehicle ID videos.

This renderer is target-consistency only. It does not infer lanes, OD, turns,
SUMO routes, or demand.
"""

from __future__ import annotations

import argparse
import colorsys
import csv
import json
import subprocess
from collections import defaultdict
from pathlib import Path


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def final_detection_label(row: dict) -> str:
    logical_id = row.get("logical_vehicle_id", "").strip()
    if not logical_id:
        raise ValueError("Final render row is missing logical_vehicle_id")
    return logical_id


def rows_by_frame(rows: list[dict]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[int(float(row["frame_id"]))].append(row)
    return grouped


def final_color_map(rows: list[dict]) -> dict[str, tuple[int, int, int]]:
    identifiers = sorted({final_detection_label(row) for row in rows})
    colors: dict[str, tuple[int, int, int]] = {}
    for index, identifier in enumerate(identifiers):
        hue = ((index * 137.508) % 360.0) / 360.0
        red, green, blue = colorsys.hsv_to_rgb(hue, 0.82, 0.95)
        colors[identifier] = (int(red * 255), int(green * 255), int(blue * 255))
    return colors


def style_for_row(row: dict, colors: dict[str, tuple[int, int, int]]) -> tuple[int, int, int]:
    return colors[final_detection_label(row)]


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
    top = max(0, y - text_h - 12)
    left = max(0, x)
    draw.rectangle((left, top, left + text_w + 14, top + text_h + 10), fill=(15, 20, 24))
    draw.rectangle((left, top, left + text_w + 14, top + text_h + 10), outline=color, width=2)
    draw.text((left + 7, top + 5), text, fill=(255, 255, 255), font=font)


def draw_detection(draw, row: dict, colors: dict[str, tuple[int, int, int]], font) -> None:
    color = style_for_row(row, colors)
    x1, y1, x2, y2 = [int(round(float(row[key]))) for key in ("x1", "y1", "x2", "y2")]
    width = 5
    for offset in range(width):
        draw.rectangle((x1 - offset, y1 - offset, x2 + offset, y2 + offset), outline=color)
    draw_label(draw, final_detection_label(row), x1, max(24, y1 - 5), color, font)


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


def probe_video_frame_count(clip_path: Path) -> int:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames,nb_frames",
            "-of",
            "json",
            str(clip_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    count = stream.get("nb_read_frames") or stream.get("nb_frames")
    if count in {None, "N/A"}:
        raise RuntimeError(f"Could not determine frame count for {clip_path}")
    return int(count)


def render_final_logical_vehicle_video(
    clip_path: Path,
    logical_rows: list[dict],
    output_path: Path,
    fps: float = 50.0,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> int:
    from PIL import Image, ImageDraw

    grouped = rows_by_frame(logical_rows)
    colors = final_color_map(logical_rows)
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
        raise RuntimeError("Failed to open ffmpeg raw-video pipes")
    label_font = load_font(24)
    frame_id = 0
    written = 0
    while True:
        raw_frame = decode.stdout.read(frame_bytes)
        if not raw_frame:
            break
        if len(raw_frame) != frame_bytes:
            raise RuntimeError(f"Incomplete raw frame at frame_id={frame_id}")
        if frame_id < start_frame:
            frame_id += 1
            continue
        if end_frame is not None and frame_id >= end_frame:
            frame_id += 1
            continue
        image = Image.frombytes("RGB", (width, height), raw_frame)
        draw = ImageDraw.Draw(image)
        for row in grouped.get(frame_id, []):
            draw_detection(draw, row, colors, label_font)
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
    parser.add_argument("--final-tracks", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int)
    args = parser.parse_args()

    rows = read_csv(Path(args.final_tracks))
    written = render_final_logical_vehicle_video(
        clip_path=Path(args.clip),
        logical_rows=rows,
        output_path=Path(args.output),
        fps=args.fps,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    print("render=final")
    print(f"frames_written={written}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
