#!/usr/bin/env python3
"""Render a SUMO replay trajectory CSV into an audit MP4."""

from __future__ import annotations

import argparse
import csv
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


COLORS = {
    "spatial_init_at_video_zero": (67, 160, 71),
    "hold_until_window_end": (251, 140, 0),
    "corrected_route_replay": (25, 118, 210),
}


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_lane_shapes(net_file: Path) -> list[list[tuple[float, float]]]:
    root = ET.parse(net_file).getroot()
    shapes = []
    for lane in root.findall(".//lane"):
        lane_id = lane.attrib.get("id", "")
        shape = lane.attrib.get("shape", "")
        if lane_id.startswith(":") or not shape:
            continue
        points = []
        for item in shape.split():
            x, y = item.split(",")[:2]
            points.append((float(x), float(y)))
        if len(points) >= 2:
            shapes.append(points)
    return shapes


def bounds_for(shapes: list[list[tuple[float, float]]], trajectories: list[dict]) -> tuple[float, float, float, float]:
    xs = [x for shape in shapes for x, _ in shape]
    ys = [y for shape in shapes for _, y in shape]
    xs.extend(float(row["x"]) for row in trajectories)
    ys.extend(float(row["y"]) for row in trajectories)
    return min(xs), min(ys), max(xs), max(ys)


def make_transform(bounds: tuple[float, float, float, float], width: int, height: int, margin: int = 48):
    min_x, min_y, max_x, max_y = bounds
    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)
    scale = min((width - margin * 2) / span_x, (height - margin * 2) / span_y)
    offset_x = (width - span_x * scale) / 2.0
    offset_y = (height - span_y * scale) / 2.0

    def transform(x: float, y: float) -> tuple[int, int]:
        px = offset_x + (x - min_x) * scale
        py = height - (offset_y + (y - min_y) * scale)
        return int(round(px)), int(round(py))

    return transform


def group_trajectories(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["sumo_vehicle_id"]].append(row)
    return {
        vehicle_id: sorted(vehicle_rows, key=lambda row: float(row["sim_time_sec"]))
        for vehicle_id, vehicle_rows in grouped.items()
    }


def row_at_or_before(rows: list[dict], sim_time: float, tolerance: float) -> dict | None:
    selected = None
    for row in rows:
        row_time = float(row["sim_time_sec"])
        if row_time <= sim_time + 1e-9:
            selected = row
        else:
            break
    if selected is None:
        return None
    if abs(float(selected["sim_time_sec"]) - sim_time) > tolerance:
        return None
    return selected


def load_traceability(path: Path) -> dict[str, dict]:
    return {row["sumo_vehicle_id"]: row for row in read_csv(path)}


def draw_frame(
    shapes: list[list[tuple[float, float]]],
    grouped: dict[str, list[dict]],
    traceability: dict[str, dict],
    transform,
    sim_time: float,
    width: int,
    height: int,
    tolerance: float,
) -> Image.Image:
    image = Image.new("RGB", (width, height), (248, 249, 250))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for shape in shapes:
        points = [transform(x, y) for x, y in shape]
        draw.line(points, fill=(170, 174, 180), width=4)
    draw.rectangle((0, 0, width, 42), fill=(30, 34, 40))
    draw.text((16, 12), f"SUMO replay v2 | sim {sim_time:05.1f}s | video {sim_time - 6.0:05.1f}s", fill=(255, 255, 255), font=font)
    for vehicle_id, rows in grouped.items():
        row = row_at_or_before(rows, sim_time, tolerance)
        if row is None:
            continue
        meta = traceability.get(vehicle_id, {})
        color = COLORS.get(meta.get("control_mode", ""), (97, 97, 97))
        x, y = transform(float(row["x"]), float(row["y"]))
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=color, outline=(0, 0, 0), width=1)
        label = meta.get("logical_vehicle_id") or row.get("track_id") or vehicle_id
        draw.text((x + 9, y - 7), label, fill=(20, 20, 20), font=font)
    legend = [
        ("spatial init", COLORS["spatial_init_at_video_zero"]),
        ("waiting hold", COLORS["hold_until_window_end"]),
        ("gate aligned", COLORS["corrected_route_replay"]),
    ]
    lx = 16
    ly = height - 30
    for text, color in legend:
        draw.rectangle((lx, ly, lx + 12, ly + 12), fill=color)
        draw.text((lx + 18, ly), text, fill=(20, 20, 20), font=font)
        lx += 150
    return image


def render_video(
    net_file: Path,
    trajectory_path: Path,
    traceability_path: Path,
    output_path: Path,
    fps: int,
    width: int,
    height: int,
    start_sim_time: float,
    end_sim_time: float,
) -> None:
    shapes = parse_lane_shapes(net_file)
    trajectories = read_csv(trajectory_path)
    traceability = load_traceability(traceability_path)
    grouped = group_trajectories(trajectories)
    transform = make_transform(bounds_for(shapes, trajectories), width, height)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tolerance = max(0.25, 1.0 / fps + 0.21)
    with tempfile.TemporaryDirectory(prefix="sumo_replay_frames_") as tmpdir:
        frame_dir = Path(tmpdir)
        frame_count = int(round((end_sim_time - start_sim_time) * fps)) + 1
        for index in range(frame_count):
            sim_time = start_sim_time + index / fps
            image = draw_frame(shapes, grouped, traceability, transform, sim_time, width, height, tolerance)
            image.save(frame_dir / f"frame_{index:06d}.png")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-framerate",
                str(fps),
                "-i",
                str(frame_dir / "frame_%06d.png"),
                "-pix_fmt",
                "yuv420p",
                "-c:v",
                "libx264",
                str(output_path),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--net-file", required=True)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--traceability", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--start-sim-time", type=float, default=0.0)
    parser.add_argument("--end-sim-time", type=float, default=46.0)
    args = parser.parse_args()
    render_video(
        net_file=Path(args.net_file),
        trajectory_path=Path(args.trajectory),
        traceability_path=Path(args.traceability),
        output_path=Path(args.output),
        fps=args.fps,
        width=args.width,
        height=args.height,
        start_sim_time=args.start_sim_time,
        end_sim_time=args.end_sim_time,
    )
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
