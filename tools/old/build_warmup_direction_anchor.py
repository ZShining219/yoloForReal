#!/usr/bin/env python3
"""Build warm-up direction anchors for vehicles already inside the window."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from build_single_track_direction_evidence import (
    Gate,
    TrackPoint,
    build_direction_result,
    detect_gate_crossings,
    load_direction_semantics,
    load_gates,
    side_label,
)


WARMUP_FIELDS = [
    "track_id",
    "manual_origin_direction",
    "manual_destination_direction",
    "result_direction",
    "route_source",
    "frame0_frame",
    "frame0_time_sec",
    "frame0_bottom_center_x",
    "frame0_bottom_center_y",
    "frame0_speed_px_per_sec",
    "frame0_heading_dx",
    "frame0_heading_dy",
    "estimated_entry_direction",
    "estimated_entry_frame",
    "estimated_entry_time_sec",
    "entry_estimation_method",
    "entry_projection_distance_px",
    "observed_exit_direction",
    "observed_exit_frame",
    "observed_exit_time_sec",
    "exit_time_source",
    "warmup_status",
    "confidence_level",
    "evidence_note",
]


@dataclass(frozen=True)
class RayHit:
    distance_px: float
    segment_position: float
    x: float
    y: float


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


def points_by_track(track_rows: list[dict]) -> dict[str, list[TrackPoint]]:
    grouped: dict[str, list[TrackPoint]] = defaultdict(list)
    for row in track_rows:
        x1 = float(row["x1"])
        y1 = float(row["y1"])
        x2 = float(row["x2"])
        y2 = float(row["y2"])
        track_id = row["logical_vehicle_id"]
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
    return {track_id: sorted(points, key=lambda point: point.frame_id) for track_id, points in grouped.items()}


def ray_segment_intersection(
    origin: tuple[float, float],
    direction: tuple[float, float],
    segment_start: tuple[float, float],
    segment_end: tuple[float, float],
) -> RayHit | None:
    px, py = origin
    dx, dy = direction
    ax, ay = segment_start
    bx, by = segment_end
    sx, sy = bx - ax, by - ay
    det = -dx * sy + dy * sx
    if abs(det) < 1e-9:
        return None
    qx, qy = ax - px, ay - py
    distance = (-sy * qx + sx * qy) / det
    segment_position = (-dy * qx + dx * qy) / det
    if distance < 0 or segment_position < 0 or segment_position > 1:
        return None
    return RayHit(distance, segment_position, px + distance * dx, py + distance * dy)


def early_motion(points: list[TrackPoint], horizon_frames: int = 30) -> tuple[TrackPoint, TrackPoint, float, float]:
    ordered = sorted(points, key=lambda point: point.frame_id)
    first = ordered[0]
    target = ordered[-1]
    for point in ordered:
        if point.frame_id >= first.frame_id + horizon_frames:
            target = point
            break
    if target.frame_id == first.frame_id and len(ordered) > 1:
        target = ordered[1]
    frame_delta = max(1, target.frame_id - first.frame_id)
    vx = (target.bottom_center_x - first.bottom_center_x) / frame_delta
    vy = (target.bottom_center_y - first.bottom_center_y) / frame_delta
    return first, target, vx, vy


def estimate_gate_frame(
    frame0: TrackPoint,
    vx: float,
    vy: float,
    gate: Gate,
    sign: int,
) -> tuple[float, float, RayHit] | None:
    speed = math.hypot(vx, vy)
    if speed <= 1e-9:
        return None
    direction = (sign * vx / speed, sign * vy / speed)
    hit = ray_segment_intersection(
        (frame0.bottom_center_x, frame0.bottom_center_y),
        direction,
        (gate.x1, gate.y1),
        (gate.x2, gate.y2),
    )
    if hit is None:
        return None
    frame = frame0.frame_id + sign * hit.distance_px / speed
    return frame, hit.distance_px, hit


def movement_duration_priors(od_rows: list[dict], fps: float) -> dict[str, float]:
    durations: dict[str, list[float]] = defaultdict(list)
    for row in od_rows:
        if row.get("review_status") != "ACCEPTED":
            continue
        result = row.get("result_direction", "")
        if not result or result == "unknown":
            continue
        if not row.get("first_crossing_frame") or not row.get("last_crossing_frame"):
            continue
        duration = (int(float(row["last_crossing_frame"])) - int(float(row["first_crossing_frame"]))) / fps
        if duration > 0:
            durations[result].append(duration)
    priors = {}
    for result, values in durations.items():
        ordered = sorted(values)
        middle = len(ordered) // 2
        if len(ordered) % 2:
            priors[result] = ordered[middle]
        else:
            priors[result] = (ordered[middle - 1] + ordered[middle]) / 2.0
    return priors


def selected_exit_crossing(crossings: list[dict], destination: str) -> dict | None:
    exits = [
        row
        for row in crossings
        if row.get("crossing_type") == "exiting" and row.get("gate_id") == destination
    ]
    if not exits:
        return None
    return sorted(exits, key=lambda row: int(float(row["crossing_frame"])))[-1]


def manual_entry_time_override(route: dict) -> float | None:
    value = route.get("manual_estimated_entry_time_sec", "").strip()
    if not value:
        return None
    return float(value)


def build_warmup_anchor_row(
    route: dict,
    points: list[TrackPoint],
    gates: dict[str, Gate],
    crossings: list[dict],
    duration_priors: dict[str, float],
    fps: float,
    horizon_frames: int = 30,
) -> dict:
    track_id = route["track_id"]
    origin = route["origin_direction"]
    destination = route["destination_direction"]
    result_direction = f"{origin}_to_{destination}"
    frame0, target, vx, vy = early_motion(points, horizon_frames=horizon_frames)
    speed_px_per_sec = math.hypot(vx, vy) * fps
    heading_norm = math.hypot(vx, vy)
    heading_dx = vx / heading_norm if heading_norm > 0 else 0.0
    heading_dy = vy / heading_norm if heading_norm > 0 else 0.0

    exit_crossing = selected_exit_crossing(crossings, destination)
    observed_exit_direction = ""
    observed_exit_frame = ""
    observed_exit_time_sec = ""
    exit_time_source = "not_observed"
    if exit_crossing:
        observed_exit_direction = destination
        observed_exit_frame = exit_crossing["crossing_frame"]
        observed_exit_time_sec = exit_crossing["crossing_time_sec"]
        exit_time_source = "observed_gate_crossing"
    elif side_label(gates[destination], frame0.bottom_center_x, frame0.bottom_center_y) == gates[destination].approach_outside_side:
        observed_exit_direction = destination
        observed_exit_frame = str(frame0.frame_id)
        observed_exit_time_sec = f"{frame0.time_sec:.2f}"
        exit_time_source = "already_outside_at_window_start"

    estimated_entry_frame = ""
    estimated_entry_time_sec = ""
    entry_method = ""
    projection_distance = ""
    confidence = "low"
    status = "WARMUP_REVIEW_REQUIRED"
    note_parts = []

    manual_entry_time = manual_entry_time_override(route)
    if manual_entry_time is not None:
        estimated_entry_frame = f"{manual_entry_time * fps:.1f}"
        estimated_entry_time_sec = f"{manual_entry_time:.2f}"
        entry_method = "manual_entry_time_override"
        confidence = "high"
        status = "WARMUP_MANUAL_ENTRY_OVERRIDE"
        note_parts.append("Manual entry time override applied from route review constraint.")
    else:
        entry_hit = estimate_gate_frame(frame0, vx, vy, gates[origin], sign=-1)
        if entry_hit is not None:
            frame, distance, _ = entry_hit
            estimated_entry_frame = f"{frame:.1f}"
            estimated_entry_time_sec = f"{frame / fps:.2f}"
            entry_method = "backward_ray_to_origin_gate"
            projection_distance = f"{distance:.1f}"
            confidence = "high" if frame >= -8 * fps else "medium"
            status = "WARMUP_BACK_PROJECTED"
            note_parts.append("Entry estimated by backward ray from frame0 motion to manual origin gate.")
        elif exit_crossing and result_direction in duration_priors:
            exit_time = float(exit_crossing["crossing_time_sec"])
            estimated_time = exit_time - duration_priors[result_direction]
            estimated_entry_frame = f"{estimated_time * fps:.1f}"
            estimated_entry_time_sec = f"{estimated_time:.2f}"
            entry_method = "same_movement_duration_prior"
            confidence = "medium"
            status = "WARMUP_BACK_PROJECTED_WITH_ROUTE_PRIOR"
            note_parts.append("Backward ray missed origin gate; entry estimated from same-movement complete OD duration prior.")
        else:
            note_parts.append("Manual route is available but entry time could not be estimated from mapping geometry or route prior.")

    if exit_crossing:
        note_parts.append(f"Destination anchored by observed exiting crossing on {destination}.")
    elif exit_time_source == "already_outside_at_window_start":
        note_parts.append(f"Frame0 is already on the outside side of destination gate {destination}; exit occurred before or at window start.")
    else:
        note_parts.append("Destination crossing was not stably observed in the 0-30s window.")

    return {
        "track_id": track_id,
        "manual_origin_direction": origin,
        "manual_destination_direction": destination,
        "result_direction": result_direction,
        "route_source": route.get("route_source", "manual"),
        "frame0_frame": str(frame0.frame_id),
        "frame0_time_sec": f"{frame0.time_sec:.2f}",
        "frame0_bottom_center_x": f"{frame0.bottom_center_x:.2f}",
        "frame0_bottom_center_y": f"{frame0.bottom_center_y:.2f}",
        "frame0_speed_px_per_sec": f"{speed_px_per_sec:.2f}",
        "frame0_heading_dx": f"{heading_dx:.4f}",
        "frame0_heading_dy": f"{heading_dy:.4f}",
        "estimated_entry_direction": origin if estimated_entry_time_sec else "",
        "estimated_entry_frame": estimated_entry_frame,
        "estimated_entry_time_sec": estimated_entry_time_sec,
        "entry_estimation_method": entry_method,
        "entry_projection_distance_px": projection_distance,
        "observed_exit_direction": observed_exit_direction,
        "observed_exit_frame": observed_exit_frame,
        "observed_exit_time_sec": observed_exit_time_sec,
        "exit_time_source": exit_time_source,
        "warmup_status": status,
        "confidence_level": confidence,
        "evidence_note": " ".join(note_parts),
    }


def build_warmup_rows(
    route_rows: list[dict],
    track_rows: list[dict],
    gate_rows: list[dict],
    semantic_rows: list[dict],
    fps: float,
    duration_prior_od_rows: list[dict],
) -> tuple[list[dict], list[dict]]:
    gates = {gate.approach_id: gate for gate in load_gates(gate_rows)}
    semantics = load_direction_semantics(semantic_rows)
    points = points_by_track(track_rows)
    duration_priors = movement_duration_priors(duration_prior_od_rows, fps=fps)
    warmup_rows = []
    crossing_rows = []
    for route in route_rows:
        track_id = route["track_id"]
        track_points = points.get(track_id, [])
        if not track_points:
            row = {field: "" for field in WARMUP_FIELDS}
            row.update(
                {
                    "track_id": track_id,
                    "manual_origin_direction": route.get("origin_direction", ""),
                    "manual_destination_direction": route.get("destination_direction", ""),
                    "result_direction": f"{route.get('origin_direction', '')}_to_{route.get('destination_direction', '')}",
                    "route_source": route.get("route_source", "manual"),
                    "warmup_status": "TRACK_NOT_FOUND",
                    "confidence_level": "low",
                    "evidence_note": "No track rows found for manual warmup route.",
                }
            )
            warmup_rows.append(row)
            continue
        crossings = [crossing.to_row() for crossing in detect_gate_crossings(track_points, list(gates.values()), stable_frames=5)]
        crossing_rows.extend(crossings)
        # Build once to keep the semantic mapping exercised; manual route remains the strong constraint.
        build_direction_result(track_id, crossings, semantics)
        warmup_rows.append(build_warmup_anchor_row(route, track_points, gates, crossings, duration_priors, fps=fps))
    return warmup_rows, crossing_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual-routes", required=True)
    parser.add_argument("--logical-tracks", required=True)
    parser.add_argument("--approach-gates", required=True)
    parser.add_argument("--direction-semantics", required=True)
    parser.add_argument("--duration-prior-od", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fps", type=float, default=50.0)
    args = parser.parse_args()

    warmup_rows, crossing_rows = build_warmup_rows(
        route_rows=read_csv(Path(args.manual_routes)),
        track_rows=read_csv(Path(args.logical_tracks)),
        gate_rows=read_csv(Path(args.approach_gates)),
        semantic_rows=read_csv(Path(args.direction_semantics)),
        fps=args.fps,
        duration_prior_od_rows=read_csv(Path(args.duration_prior_od)),
    )
    output_dir = Path(args.output_dir)
    write_csv(output_dir / "warmup_direction_anchors.csv", warmup_rows, WARMUP_FIELDS)
    crossing_fields = list(crossing_rows[0].keys()) if crossing_rows else [
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
    write_csv(output_dir / "warmup_observed_crossings.csv", crossing_rows, crossing_fields)
    print(f"warmup_rows={len(warmup_rows)}")
    print(f"observed_crossings={len(crossing_rows)}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
