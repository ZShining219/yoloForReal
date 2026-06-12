#!/usr/bin/env python3
"""Build a geometry-only center-follow trajectory layer.

This mock layer ignores YOLO track IDs during association. Raw IDs are retained
only as evidence labels after each detection has been assigned to a logical
center-follow trajectory.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ActiveTrajectory:
    logical_vehicle_id: str
    last_frame: int
    last_center: tuple[float, float]
    velocity: tuple[float, float]
    last_size: tuple[float, float]
    class_name: str
    last_raw_track_id: str


@dataclass
class CenterFollowOutputs:
    overlay_rows: list[dict]
    trajectory_rows: list[dict]
    links: list[dict]
    summary_rows: list[dict]


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
    if track_id.startswith("mot_"):
        return track_id
    return f"mot_{int(float(track_id)):04d}"


def as_float(row: dict, key: str) -> float:
    return float(row[key])


def center(row: dict) -> tuple[float, float]:
    return ((as_float(row, "x1") + as_float(row, "x2")) / 2.0, (as_float(row, "y1") + as_float(row, "y2")) / 2.0)


def size(row: dict) -> tuple[float, float]:
    return (as_float(row, "x2") - as_float(row, "x1"), as_float(row, "y2") - as_float(row, "y1"))


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def add(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return (a[0] + b[0], a[1] + b[1])


def sub(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return (a[0] - b[0], a[1] - b[1])


def mul(a: tuple[float, float], value: float) -> tuple[float, float]:
    return (a[0] * value, a[1] * value)


def size_ratio(a: tuple[float, float], b: tuple[float, float]) -> float:
    width_ratio = max(a[0], b[0]) / max(min(a[0], b[0]), 0.01)
    height_ratio = max(a[1], b[1]) / max(min(a[1], b[1]), 0.01)
    return max(width_ratio, height_ratio)


def class_compatible(a: str, b: str) -> bool:
    motor_classes = {"car", "truck", "bus"}
    return a == b or (a in motor_classes and b in motor_classes)


def final_track_ids(final_rows: list[dict]) -> set[str]:
    return {row["track_id"] for row in final_rows}


def candidate_cost(
    trajectory: ActiveTrajectory,
    row: dict,
    max_gap_frames: int,
    max_prediction_distance_px: float,
    max_size_ratio: float,
) -> tuple[float, dict] | None:
    frame_id = int(float(row["frame_id"]))
    frame_delta = frame_id - trajectory.last_frame
    if frame_delta <= 0 or frame_delta > max_gap_frames + 1:
        return None
    if not class_compatible(trajectory.class_name, row.get("class_name", "")):
        return None
    current_center = center(row)
    predicted_center = add(trajectory.last_center, mul(trajectory.velocity, frame_delta))
    predicted_distance = distance(predicted_center, current_center)
    current_size = size(row)
    ratio = size_ratio(trajectory.last_size, current_size)
    if predicted_distance > max_prediction_distance_px or ratio > max_size_ratio:
        return None
    gap_frames = frame_delta - 1
    raw_track_id = normalized_track_id(row["track_id"])
    raw_id_change_penalty = 3.0 if raw_track_id != trajectory.last_raw_track_id else 0.0
    cost = predicted_distance + gap_frames * 2.0 + raw_id_change_penalty
    return cost, {
        "frame_delta": frame_delta,
        "gap_frames": gap_frames,
        "predicted_distance_px": predicted_distance,
        "size_ratio": ratio,
        "raw_track_id": raw_track_id,
    }


def update_trajectory(trajectory: ActiveTrajectory, row: dict, metrics: dict) -> None:
    frame_id = int(float(row["frame_id"]))
    current_center = center(row)
    frame_delta = max(1, frame_id - trajectory.last_frame)
    observed_velocity = mul(sub(current_center, trajectory.last_center), 1.0 / frame_delta)
    trajectory.velocity = observed_velocity
    trajectory.last_frame = frame_id
    trajectory.last_center = current_center
    trajectory.last_size = size(row)
    trajectory.class_name = row.get("class_name", trajectory.class_name)
    trajectory.last_raw_track_id = metrics["raw_track_id"]


def overlay_row(logical_vehicle_id: str, row: dict, fps: float) -> dict:
    raw_track_id = normalized_track_id(row["track_id"])
    frame_id = int(float(row["frame_id"]))
    cx, cy = center(row)
    return {
        "frame_id": str(frame_id),
        "time_sec": f"{frame_id / fps:.2f}",
        "track_id": raw_track_id,
        "logical_vehicle_id": logical_vehicle_id,
        "class_name": row.get("class_name", ""),
        "confidence": row.get("confidence", ""),
        "x1": row.get("x1", ""),
        "y1": row.get("y1", ""),
        "x2": row.get("x2", ""),
        "y2": row.get("y2", ""),
        "center_x": f"{cx:.2f}",
        "center_y": f"{cy:.2f}",
    }


def build_center_follow_mock(
    detection_rows: list[dict],
    allowed_track_ids: set[str],
    fps: float,
    max_gap_frames: int = 20,
    max_prediction_distance_px: float = 120.0,
    max_size_ratio: float = 3.0,
) -> CenterFollowOutputs:
    filtered = [
        row
        for row in detection_rows
        if normalized_track_id(row["track_id"]) in allowed_track_ids
    ]
    filtered.sort(key=lambda row: (int(float(row["frame_id"])), float(row["x1"]), float(row["y1"])))

    active: list[ActiveTrajectory] = []
    logical_index = 1
    overlay_rows: list[dict] = []
    links: list[dict] = []

    by_frame: dict[int, list[dict]] = defaultdict(list)
    for row in filtered:
        by_frame[int(float(row["frame_id"]))].append(row)

    for frame_id in sorted(by_frame):
        rows = by_frame[frame_id]
        candidates = []
        for row_index, row in enumerate(rows):
            for trajectory_index, trajectory in enumerate(active):
                result = candidate_cost(trajectory, row, max_gap_frames, max_prediction_distance_px, max_size_ratio)
                if result is None:
                    continue
                cost, metrics = result
                candidates.append((cost, trajectory_index, row_index, metrics))

        assigned_trajectories: set[int] = set()
        assigned_rows: set[int] = set()
        assignments: dict[int, tuple[int, dict]] = {}
        for _, trajectory_index, row_index, metrics in sorted(candidates):
            if trajectory_index in assigned_trajectories or row_index in assigned_rows:
                continue
            assigned_trajectories.add(trajectory_index)
            assigned_rows.add(row_index)
            assignments[row_index] = (trajectory_index, metrics)

        for row_index, row in enumerate(rows):
            raw_track_id = normalized_track_id(row["track_id"])
            if row_index in assignments:
                trajectory_index, metrics = assignments[row_index]
                trajectory = active[trajectory_index]
                if metrics["gap_frames"] > 0 or raw_track_id != trajectory.last_raw_track_id:
                    links.append(
                        {
                            "logical_vehicle_id": trajectory.logical_vehicle_id,
                            "from_raw_track_id": trajectory.last_raw_track_id,
                            "to_raw_track_id": raw_track_id,
                            "from_frame": str(trajectory.last_frame),
                            "to_frame": str(frame_id),
                            "gap_frames": str(metrics["gap_frames"]),
                            "predicted_distance_px": f"{metrics['predicted_distance_px']:.2f}",
                            "size_ratio": f"{metrics['size_ratio']:.2f}",
                            "link_action": "CENTER_FOLLOW_CONTINUITY",
                        }
                    )
                logical_vehicle_id = trajectory.logical_vehicle_id
                update_trajectory(trajectory, row, metrics)
            else:
                logical_vehicle_id = f"cf_{logical_index:04d}"
                logical_index += 1
                current_center = center(row)
                active.append(
                    ActiveTrajectory(
                        logical_vehicle_id=logical_vehicle_id,
                        last_frame=frame_id,
                        last_center=current_center,
                        velocity=(0.0, 0.0),
                        last_size=size(row),
                        class_name=row.get("class_name", ""),
                        last_raw_track_id=raw_track_id,
                    )
                )
            overlay_rows.append(overlay_row(logical_vehicle_id, row, fps))

    trajectory_rows = sorted(overlay_rows, key=lambda row: (row["logical_vehicle_id"], int(row["frame_id"])))
    summary_rows = build_summary_rows(trajectory_rows, links)
    return CenterFollowOutputs(
        overlay_rows=sorted(overlay_rows, key=lambda row: (int(row["frame_id"]), row["logical_vehicle_id"])),
        trajectory_rows=trajectory_rows,
        links=links,
        summary_rows=summary_rows,
    )


def build_summary_rows(trajectory_rows: list[dict], links: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in trajectory_rows:
        grouped[row["logical_vehicle_id"]].append(row)
    rows = []
    link_counts = Counter(row["logical_vehicle_id"] for row in links)
    for logical_vehicle_id, group in sorted(grouped.items()):
        frames = [int(row["frame_id"]) for row in group]
        raw_ids = sorted({row["track_id"] for row in group})
        rows.append(
            {
                "logical_vehicle_id": logical_vehicle_id,
                "raw_track_ids": "|".join(raw_ids),
                "raw_track_id_count": str(len(raw_ids)),
                "detected_frame_count": str(len(group)),
                "start_frame": str(min(frames)),
                "end_frame": str(max(frames)),
                "link_count": str(link_counts[logical_vehicle_id]),
            }
        )
    rows.append(
        {
            "logical_vehicle_id": "__overall__",
            "raw_track_ids": "",
            "raw_track_id_count": "",
            "detected_frame_count": str(len(trajectory_rows)),
            "start_frame": "",
            "end_frame": "",
            "link_count": str(len(links)),
            "logical_trajectory_count": str(len(grouped)),
        }
    )
    return rows


OVERLAY_FIELDS = [
    "frame_id",
    "time_sec",
    "track_id",
    "logical_vehicle_id",
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "center_x",
    "center_y",
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
]

SUMMARY_FIELDS = [
    "logical_vehicle_id",
    "raw_track_ids",
    "raw_track_id_count",
    "detected_frame_count",
    "start_frame",
    "end_frame",
    "link_count",
    "logical_trajectory_count",
]


def write_outputs(output_dir: Path, outputs: CenterFollowOutputs) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "center_follow_overlay_detections.csv", outputs.overlay_rows, OVERLAY_FIELDS)
    write_csv(output_dir / "center_follow_trajectory_points.csv", outputs.trajectory_rows, OVERLAY_FIELDS)
    write_csv(output_dir / "center_follow_links.csv", outputs.links, LINK_FIELDS)
    write_csv(output_dir / "center_follow_summary.csv", outputs.summary_rows, SUMMARY_FIELDS)
    note = (
        "# yolo26x manual_filter_v1 center-follow mock\n\n"
        "This output ignores YOLO track IDs during association. Raw IDs are retained only as evidence labels.\n\n"
        "The overlay video should be used to review whether each center-follow trajectory line follows one real vehicle.\n"
    )
    (output_dir / "CENTER_FOLLOW_MOCK_NOTE.md").write_text(note, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections", required=True)
    parser.add_argument("--final-targets", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--max-gap-frames", type=int, default=20)
    parser.add_argument("--max-prediction-distance-px", type=float, default=120.0)
    parser.add_argument("--max-size-ratio", type=float, default=3.0)
    args = parser.parse_args()

    outputs = build_center_follow_mock(
        detection_rows=read_csv(Path(args.detections)),
        allowed_track_ids=final_track_ids(read_csv(Path(args.final_targets))),
        fps=args.fps,
        max_gap_frames=args.max_gap_frames,
        max_prediction_distance_px=args.max_prediction_distance_px,
        max_size_ratio=args.max_size_ratio,
    )
    write_outputs(Path(args.output_dir), outputs)
    overall = outputs.summary_rows[-1]
    print(f"logical_trajectory_count={overall['logical_trajectory_count']}")
    print(f"detected_rows={overall['detected_frame_count']}")
    print(f"center_follow_links={overall['link_count']}")
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
