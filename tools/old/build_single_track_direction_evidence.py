#!/usr/bin/env python3
"""Build a single-track approach-direction evidence image.

This tool reads approach gate geometry from the mapping module. It does not
hard-code north/south/east/west rules, assign lanes, or generate SUMO events.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


DIRECTION_FIELDS = [
    "track_id",
    "origin_direction",
    "destination_direction",
    "result_direction",
    "first_crossing_gate",
    "first_crossing_frame",
    "last_crossing_gate",
    "last_crossing_frame",
    "crossing_count",
    "confidence_level",
    "review_status",
    "evidence_note",
]

CROSSING_FIELDS = [
    "track_id",
    "gate_id",
    "crossing_type",
    "crossing_frame",
    "crossing_time_sec",
    "from_side",
    "to_side",
    "stable_before_frames",
    "stable_after_frames",
    "bottom_center_x",
    "bottom_center_y",
    "accepted",
    "reject_reason",
]


@dataclass(frozen=True)
class Gate:
    approach_id: str
    approach_name: str
    x1: float
    y1: float
    x2: float
    y2: float
    intersection_side: str
    approach_outside_side: str
    entering_transition: str
    exiting_transition: str


@dataclass(frozen=True)
class TrackPoint:
    track_id: str
    frame_id: int
    time_sec: float
    bottom_center_x: float
    bottom_center_y: float
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class Crossing:
    track_id: str
    gate_id: str
    crossing_type: str
    crossing_frame: int
    crossing_time_sec: float
    from_side: str
    to_side: str
    stable_before_frames: int
    stable_after_frames: int
    bottom_center_x: float
    bottom_center_y: float

    def to_row(self) -> dict:
        return {
            "track_id": self.track_id,
            "gate_id": self.gate_id,
            "crossing_type": self.crossing_type,
            "crossing_frame": str(self.crossing_frame),
            "crossing_time_sec": f"{self.crossing_time_sec:.2f}",
            "from_side": self.from_side,
            "to_side": self.to_side,
            "stable_before_frames": str(self.stable_before_frames),
            "stable_after_frames": str(self.stable_after_frames),
            "bottom_center_x": f"{self.bottom_center_x:.2f}",
            "bottom_center_y": f"{self.bottom_center_y:.2f}",
            "accepted": "yes",
            "reject_reason": "",
        }


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


def normalized_track_id(track_id: str) -> str:
    value = str(track_id).strip()
    if value.startswith("mot_"):
        return value
    return f"mot_{int(float(value)):04d}"


def load_gates(rows: list[dict]) -> list[Gate]:
    gates = []
    for row in rows:
        gates.append(
            Gate(
                approach_id=row["approach_id"],
                approach_name=row.get("approach_name", row["approach_id"]),
                x1=float(row["line_x1"]),
                y1=float(row["line_y1"]),
                x2=float(row["line_x2"]),
                y2=float(row["line_y2"]),
                intersection_side=row["intersection_side"],
                approach_outside_side=row["approach_outside_side"],
                entering_transition=row["entering_transition"],
                exiting_transition=row["exiting_transition"],
            )
        )
    return gates


def load_direction_semantics(rows: list[dict]) -> dict[tuple[str, str], dict]:
    return {
        (row["first_crossing_approach_id"], row["last_crossing_approach_id"]): row
        for row in rows
    }


def side_label(gate: Gate, px: float, py: float, tolerance: float = 1e-6) -> str:
    signed = (gate.x2 - gate.x1) * (py - gate.y1) - (gate.y2 - gate.y1) * (px - gate.x1)
    if abs(signed) <= tolerance:
        return "on"
    if signed > 0:
        return "left"
    return "right"


def crossing_type_for_transition(gate: Gate, from_side: str, to_side: str) -> str:
    transition = f"{from_side}_to_{to_side}"
    if transition == gate.entering_transition:
        return "entering"
    if transition == gate.exiting_transition:
        return "exiting"
    return ""


def detect_gate_crossings(points: list[TrackPoint], gates: list[Gate], stable_frames: int = 3) -> list[Crossing]:
    if len(points) < stable_frames * 2:
        return []
    ordered = sorted(points, key=lambda point: point.frame_id)
    crossings: list[Crossing] = []

    for gate in gates:
        sides = [side_label(gate, point.bottom_center_x, point.bottom_center_y) for point in ordered]
        for index in range(stable_frames, len(ordered) - stable_frames + 1):
            from_side = sides[index - 1]
            to_side = sides[index]
            if from_side == to_side or from_side == "on" or to_side == "on":
                continue
            before = sides[index - stable_frames:index]
            after = sides[index:index + stable_frames]
            if any(side != from_side for side in before):
                continue
            if any(side != to_side for side in after):
                continue
            crossing_type = crossing_type_for_transition(gate, from_side, to_side)
            if not crossing_type:
                continue
            point = ordered[index]
            if crossings and crossings[-1].gate_id == gate.approach_id:
                if point.frame_id - crossings[-1].crossing_frame <= stable_frames:
                    continue
            crossings.append(
                Crossing(
                    track_id=point.track_id,
                    gate_id=gate.approach_id,
                    crossing_type=crossing_type,
                    crossing_frame=point.frame_id,
                    crossing_time_sec=point.time_sec,
                    from_side=from_side,
                    to_side=to_side,
                    stable_before_frames=stable_frames,
                    stable_after_frames=stable_frames,
                    bottom_center_x=point.bottom_center_x,
                    bottom_center_y=point.bottom_center_y,
                )
            )
    return sorted(crossings, key=lambda crossing: (crossing.crossing_frame, crossing.gate_id))


def build_direction_result(track_id: str, crossings: list[dict], semantics: dict[tuple[str, str], dict]) -> dict:
    entering = [row for row in crossings if row.get("crossing_type") == "entering"]
    exiting = [row for row in crossings if row.get("crossing_type") == "exiting"]
    base = {
        "track_id": track_id,
        "origin_direction": "unknown",
        "destination_direction": "unknown",
        "result_direction": "unknown",
        "first_crossing_gate": entering[0]["gate_id"] if entering else "",
        "first_crossing_frame": entering[0]["crossing_frame"] if entering else "",
        "last_crossing_gate": exiting[-1]["gate_id"] if exiting else "",
        "last_crossing_frame": exiting[-1]["crossing_frame"] if exiting else "",
        "crossing_count": str(len(crossings)),
        "confidence_level": "low",
        "review_status": "UNKNOWN",
        "evidence_note": "",
    }
    if not entering or not exiting:
        base["evidence_note"] = "Insufficient accepted entering/exiting crossings."
        return base

    first_entering = entering[0]
    last_exiting = exiting[-1]
    if int(float(last_exiting["crossing_frame"])) <= int(float(first_entering["crossing_frame"])):
        base["review_status"] = "REVIEW"
        base["evidence_note"] = "Accepted exiting crossing is not after the first entering crossing."
        return base

    semantic = semantics.get((first_entering["gate_id"], last_exiting["gate_id"]))
    if not semantic:
        base["review_status"] = "REVIEW"
        base["evidence_note"] = "No direction semantic row for first/last gate pair."
        return base

    base.update(
        {
            "origin_direction": semantic["origin_direction"],
            "destination_direction": semantic["destination_direction"],
            "result_direction": semantic["result_direction"],
            "first_crossing_gate": first_entering["gate_id"],
            "first_crossing_frame": first_entering["crossing_frame"],
            "last_crossing_gate": last_exiting["gate_id"],
            "last_crossing_frame": last_exiting["crossing_frame"],
            "confidence_level": "high" if len(crossings) == 2 else "medium",
            "review_status": "ACCEPTED",
            "evidence_note": "First accepted entering crossing and last accepted exiting crossing mapped by direction_semantics.csv.",
        }
    )
    return base


def load_track_points(detection_rows: list[dict], final_ids: set[str]) -> dict[str, list[TrackPoint]]:
    grouped: dict[str, list[TrackPoint]] = {}
    for row in detection_rows:
        track_id = normalized_track_id(row["track_id"])
        if track_id not in final_ids:
            continue
        x1 = float(row["x1"])
        y1 = float(row["y1"])
        x2 = float(row["x2"])
        y2 = float(row["y2"])
        grouped.setdefault(track_id, []).append(
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
    return {track_id: sorted(points, key=lambda point: point.frame_id) for track_id, points in grouped.items()}


def choose_sample_result(results: list[dict], preferred_track_id: str = "") -> dict:
    if preferred_track_id:
        for row in results:
            if row["track_id"] == preferred_track_id:
                return row
        raise ValueError(f"Preferred track id not found in direction results: {preferred_track_id}")
    accepted = [row for row in results if row["review_status"] == "ACCEPTED"]
    if accepted:
        return sorted(accepted, key=lambda row: (row["confidence_level"] != "high", row["track_id"]))[0]
    return sorted(results, key=lambda row: row["track_id"])[0]


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


def draw_label(draw, xy: tuple[int, int], text: str, fill: tuple[int, int, int], font) -> None:
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    pad = 5
    draw.rectangle(
        (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad),
        fill=fill,
    )
    draw.text((x, y), text, fill=(20, 20, 20), font=font)


def sampled_points(points: list[TrackPoint], crossing_frames: set[int], max_boxes: int = 18) -> list[TrackPoint]:
    if len(points) <= max_boxes:
        selected = list(points)
    else:
        step = max(1, len(points) // max_boxes)
        selected = points[::step]
    by_frame = {point.frame_id: point for point in selected}
    for point in points:
        if point.frame_id in crossing_frames:
            by_frame[point.frame_id] = point
    return sorted(by_frame.values(), key=lambda point: point.frame_id)


def render_evidence_image(
    base_image: Path,
    output_image: Path,
    gates: list[Gate],
    points: list[TrackPoint],
    result: dict,
    crossings: list[dict],
) -> None:
    from PIL import Image, ImageDraw

    image = Image.open(base_image).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    font = load_font(18)
    small_font = load_font(14)
    title_font = load_font(26)
    gate_colors = {
        "W": (255, 175, 25, 255),
        "N": (45, 190, 255, 255),
        "S": (70, 230, 105, 255),
        "E": (255, 80, 80, 255),
    }

    for gate in gates:
        color = gate_colors.get(gate.approach_id, (255, 255, 255, 255))
        draw.line((gate.x1, gate.y1, gate.x2, gate.y2), fill=color, width=7)
        draw.ellipse((gate.x1 - 7, gate.y1 - 7, gate.x1 + 7, gate.y1 + 7), fill=color)
        draw.ellipse((gate.x2 - 7, gate.y2 - 7, gate.x2 + 7, gate.y2 + 7), fill=color)
        label_x = int((gate.x1 + gate.x2) / 2)
        label_y = int((gate.y1 + gate.y2) / 2)
        draw_label(draw, (label_x, label_y), f"{gate.approach_id} {gate.approach_name}", color, small_font)

    centers = [(point.bottom_center_x, point.bottom_center_y) for point in points]
    if len(centers) >= 2:
        draw.line(centers, fill=(30, 70, 255, 210), width=4)

    crossing_frames = {int(float(row["crossing_frame"])) for row in crossings}
    selected = sampled_points(points, crossing_frames)
    total = max(1, len(selected) - 1)
    for index, point in enumerate(selected):
        red = int(40 + 200 * index / total)
        blue = int(230 - 150 * index / total)
        color = (red, 80, blue, 120)
        outline = (red, 60, blue, 255)
        draw.rectangle((point.x1, point.y1, point.x2, point.y2), outline=outline, width=3, fill=color)
        draw.ellipse(
            (
                point.bottom_center_x - 4,
                point.bottom_center_y - 4,
                point.bottom_center_x + 4,
                point.bottom_center_y + 4,
            ),
            fill=(255, 255, 255, 230),
        )
        if point.frame_id in crossing_frames:
            draw_label(
                draw,
                (int(point.x1), int(max(0, point.y1 - 24))),
                f"frame {point.frame_id} / {point.time_sec:.2f}s",
                (255, 255, 255, 230),
                small_font,
            )

    by_frame = {point.frame_id: point for point in points}
    for row in crossings:
        frame = int(float(row["crossing_frame"]))
        point = by_frame.get(frame)
        if not point:
            continue
        label = f"{row['crossing_type'].upper()} {row['gate_id']} f{frame} {row['crossing_time_sec']}s"
        fill = (80, 235, 115, 245) if row["crossing_type"] == "entering" else (255, 210, 55, 245)
        draw.ellipse(
            (
                point.bottom_center_x - 12,
                point.bottom_center_y - 12,
                point.bottom_center_x + 12,
                point.bottom_center_y + 12,
            ),
            fill=fill,
        )
        draw_label(draw, (int(point.bottom_center_x + 14), int(point.bottom_center_y - 10)), label, fill, font)

    title = (
        f"{result['track_id']}  {result['result_direction']}  "
        f"{result['origin_direction']} -> {result['destination_direction']}  "
        f"{result['confidence_level']} / {result['review_status']}"
    )
    draw.rectangle((18, 18, 1180, 70), fill=(0, 0, 0, 180))
    draw.text((32, 30), title, fill=(255, 255, 255), font=title_font)
    output_image.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_image)


def write_summary(path: Path, result: dict, crossings: list[dict], image_path: Path) -> None:
    entering = [row for row in crossings if row["crossing_type"] == "entering"]
    exiting = [row for row in crossings if row["crossing_type"] == "exiting"]
    lines = [
        f"# Single Track Direction Evidence: {result['track_id']}",
        "",
        f"- result_direction: `{result['result_direction']}`",
        f"- origin_direction: `{result['origin_direction']}`",
        f"- destination_direction: `{result['destination_direction']}`",
        f"- confidence_level: `{result['confidence_level']}`",
        f"- review_status: `{result['review_status']}`",
        f"- evidence_image: `{image_path.name}`",
        "",
        "## Entering",
        "",
    ]
    if entering:
        row = entering[0]
        lines.extend(
            [
                f"- time_sec: `{row['crossing_time_sec']}`",
                f"- frame: `{row['crossing_frame']}`",
                f"- gate: `{row['gate_id']}`",
            ]
        )
    else:
        lines.append("- unknown")
    lines.extend(["", "## Exiting", ""])
    if exiting:
        row = exiting[-1]
        lines.extend(
            [
                f"- time_sec: `{row['crossing_time_sec']}`",
                f"- frame: `{row['crossing_frame']}`",
                f"- gate: `{row['gate_id']}`",
            ]
        )
    else:
        lines.append("- unknown")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_outputs(
    final_rows: list[dict],
    detection_rows: list[dict],
    gate_rows: list[dict],
    semantic_rows: list[dict],
    base_image: Path,
    output_dir: Path,
    stable_frames: int,
    preferred_track_id: str = "",
) -> dict:
    final_ids = {row["track_id"] for row in final_rows}
    gates = load_gates(gate_rows)
    semantics = load_direction_semantics(semantic_rows)
    points_by_track = load_track_points(detection_rows, final_ids)

    all_crossing_rows: list[dict] = []
    direction_rows: list[dict] = []
    crossings_by_track: dict[str, list[dict]] = {}
    for track_id in sorted(final_ids):
        points = points_by_track.get(track_id, [])
        crossing_objects = detect_gate_crossings(points, gates, stable_frames=stable_frames)
        crossing_rows = [crossing.to_row() for crossing in crossing_objects]
        crossings_by_track[track_id] = crossing_rows
        all_crossing_rows.extend(crossing_rows)
        direction_rows.append(build_direction_result(track_id, crossing_rows, semantics))

    sample = choose_sample_result(direction_rows, preferred_track_id=preferred_track_id)
    sample_track_id = sample["track_id"]
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "track_approach_crossings.csv", all_crossing_rows, CROSSING_FIELDS)
    write_csv(output_dir / "track_direction_od.csv", direction_rows, DIRECTION_FIELDS)

    image_path = output_dir / f"{sample_track_id}_direction_evidence.png"
    render_evidence_image(
        base_image=base_image,
        output_image=image_path,
        gates=gates,
        points=points_by_track[sample_track_id],
        result=sample,
        crossings=crossings_by_track[sample_track_id],
    )
    summary_path = output_dir / f"{sample_track_id}_direction_summary.md"
    write_summary(summary_path, sample, crossings_by_track[sample_track_id], image_path)
    return {
        "sample_track_id": sample_track_id,
        "sample_result": sample,
        "accepted_count": str(sum(row["review_status"] == "ACCEPTED" for row in direction_rows)),
        "direction_rows": direction_rows,
        "crossing_rows": all_crossing_rows,
        "image_path": str(image_path),
        "summary_path": str(summary_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-targets", required=True)
    parser.add_argument("--detections", required=True)
    parser.add_argument("--approach-gates", required=True)
    parser.add_argument("--direction-semantics", required=True)
    parser.add_argument("--base-image", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stable-frames", type=int, default=3)
    parser.add_argument("--track-id", default="")
    args = parser.parse_args()

    outputs = build_outputs(
        final_rows=read_csv(Path(args.final_targets)),
        detection_rows=read_csv(Path(args.detections)),
        gate_rows=read_csv(Path(args.approach_gates)),
        semantic_rows=read_csv(Path(args.direction_semantics)),
        base_image=Path(args.base_image),
        output_dir=Path(args.output_dir),
        stable_frames=args.stable_frames,
        preferred_track_id=args.track_id,
    )
    result = outputs["sample_result"]
    print(f"sample_track_id={outputs['sample_track_id']}")
    print(f"result_direction={result['result_direction']}")
    print(f"origin_direction={result['origin_direction']}")
    print(f"destination_direction={result['destination_direction']}")
    print(f"first_crossing_frame={result['first_crossing_frame']}")
    print(f"last_crossing_frame={result['last_crossing_frame']}")
    print(f"confidence_level={result['confidence_level']}")
    print(f"review_status={result['review_status']}")
    print(f"accepted_direction_tracks={outputs['accepted_count']}")
    print(f"evidence_image={outputs['image_path']}")
    print(f"summary={outputs['summary_path']}")


if __name__ == "__main__":
    main()
