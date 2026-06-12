#!/usr/bin/env python3
"""Build a display/mock logical vehicle track layer from YOLO detections.

This tool does not overwrite YOLO track IDs. It creates a separate
`logical_vehicle_id` layer for reviewing whether short tracker breaks can be
stitched by center-point continuity.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrackSegment:
    segment_id: str
    raw_track_id: str
    rows: list[dict]
    start_frame: int
    end_frame: int
    start_center: tuple[float, float]
    end_center: tuple[float, float]
    start_size: tuple[float, float]
    end_size: tuple[float, float]
    class_name: str


@dataclass
class StitchOutputs:
    logical_tracks: list[dict]
    links: list[dict]
    rejections: list[dict]
    interpolated_points: list[dict]
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


def sub(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return (a[0] - b[0], a[1] - b[1])


def add(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return (a[0] + b[0], a[1] + b[1])


def mul(a: tuple[float, float], value: float) -> tuple[float, float]:
    return (a[0] * value, a[1] * value)


def segment_velocity(rows: list[dict], at_end: bool) -> tuple[float, float]:
    ordered = sorted(rows, key=lambda row: int(float(row["frame_id"])))
    if len(ordered) < 2:
        return (0.0, 0.0)
    window = ordered[-3:] if at_end else ordered[:3]
    first = window[0]
    last = window[-1]
    frame_delta = int(float(last["frame_id"])) - int(float(first["frame_id"]))
    if frame_delta <= 0:
        return (0.0, 0.0)
    return mul(sub(center(last), center(first)), 1.0 / frame_delta)


def build_segments(detection_rows: list[dict], final_ids: set[str]) -> list[TrackSegment]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in detection_rows:
        raw_track_id = normalized_track_id(row["track_id"])
        if raw_track_id in final_ids:
            copied = dict(row)
            copied["raw_track_id"] = raw_track_id
            grouped[raw_track_id].append(copied)

    segments: list[TrackSegment] = []
    for raw_track_id, rows in grouped.items():
        ordered = sorted(rows, key=lambda row: int(float(row["frame_id"])))
        current = [ordered[0]]
        previous_frame = int(float(ordered[0]["frame_id"]))
        segment_index = 1
        for row in ordered[1:]:
            frame_id = int(float(row["frame_id"]))
            if frame_id == previous_frame + 1:
                current.append(row)
            else:
                segments.append(make_segment(raw_track_id, segment_index, current))
                segment_index += 1
                current = [row]
            previous_frame = frame_id
        segments.append(make_segment(raw_track_id, segment_index, current))
    return sorted(segments, key=lambda segment: (segment.start_frame, segment.raw_track_id, segment.segment_id))


def make_segment(raw_track_id: str, segment_index: int, rows: list[dict]) -> TrackSegment:
    ordered = sorted(rows, key=lambda row: int(float(row["frame_id"])))
    segment_id = f"{raw_track_id}_seg{segment_index:02d}"
    return TrackSegment(
        segment_id=segment_id,
        raw_track_id=raw_track_id,
        rows=ordered,
        start_frame=int(float(ordered[0]["frame_id"])),
        end_frame=int(float(ordered[-1]["frame_id"])),
        start_center=center(ordered[0]),
        end_center=center(ordered[-1]),
        start_size=size(ordered[0]),
        end_size=size(ordered[-1]),
        class_name=ordered[0].get("class_name", ""),
    )


def class_compatible(a: str, b: str) -> bool:
    motor_classes = {"car", "truck", "bus"}
    return a == b or (a in motor_classes and b in motor_classes)


def link_candidate_metrics(from_segment: TrackSegment, to_segment: TrackSegment) -> dict:
    gap_frames = to_segment.start_frame - from_segment.end_frame - 1
    frame_delta = to_segment.start_frame - from_segment.end_frame
    from_velocity = segment_velocity(from_segment.rows, at_end=True)
    observed_velocity = mul(sub(to_segment.start_center, from_segment.end_center), 1.0 / frame_delta)
    predicted_center = add(from_segment.end_center, mul(from_velocity, frame_delta))
    predicted_distance = distance(predicted_center, to_segment.start_center)
    from_speed = max(distance((0.0, 0.0), from_velocity), 0.01)
    observed_speed = distance((0.0, 0.0), observed_velocity)
    speed_change_ratio = max(from_speed, observed_speed) / max(min(from_speed, observed_speed), 0.01)
    size_ratio_w = max(from_segment.end_size[0], to_segment.start_size[0]) / max(min(from_segment.end_size[0], to_segment.start_size[0]), 0.01)
    size_ratio_h = max(from_segment.end_size[1], to_segment.start_size[1]) / max(min(from_segment.end_size[1], to_segment.start_size[1]), 0.01)
    return {
        "gap_frames": gap_frames,
        "frame_delta": frame_delta,
        "predicted_distance_px": predicted_distance,
        "speed_change_ratio": speed_change_ratio,
        "size_ratio": max(size_ratio_w, size_ratio_h),
    }


def reject_reason(
    from_segment: TrackSegment,
    to_segment: TrackSegment,
    metrics: dict,
    max_gap_frames: int,
    max_link_distance_px: float,
    max_speed_change_ratio: float,
    max_size_ratio: float,
) -> str:
    if to_segment.start_frame <= from_segment.end_frame:
        return "temporal_overlap_or_reverse"
    if metrics["gap_frames"] > max_gap_frames:
        return "gap_too_large"
    if not class_compatible(from_segment.class_name, to_segment.class_name):
        return "class_incompatible"
    if metrics["predicted_distance_px"] > max_link_distance_px:
        return "predicted_distance_too_large"
    if metrics["speed_change_ratio"] > max_speed_change_ratio:
        return "speed_change_too_large"
    if metrics["size_ratio"] > max_size_ratio:
        return "bbox_size_change_too_large"
    return ""


def has_ambiguous_cross_id_swap(from_segment: TrackSegment, to_segment: TrackSegment, segments: list[TrackSegment], max_gap_frames: int) -> bool:
    if from_segment.raw_track_id == to_segment.raw_track_id:
        return False
    target_previous = [
        segment
        for segment in segments
        if segment.raw_track_id == to_segment.raw_track_id
        and segment.segment_id != to_segment.segment_id
        and segment.end_frame <= from_segment.end_frame
        and from_segment.end_frame - segment.end_frame <= max_gap_frames
    ]
    source_next = [
        segment
        for segment in segments
        if segment.raw_track_id == from_segment.raw_track_id
        and segment.segment_id != from_segment.segment_id
        and segment.start_frame >= to_segment.start_frame
        and segment.start_frame - to_segment.start_frame <= max_gap_frames
    ]
    return bool(target_previous and source_next)


def choose_links(
    segments: list[TrackSegment],
    max_gap_frames: int,
    max_link_distance_px: float,
    max_speed_change_ratio: float,
    max_size_ratio: float,
) -> tuple[dict[str, str], list[dict], list[dict]]:
    next_by_segment: dict[str, str] = {}
    used_to_segments: set[str] = set()
    links: list[dict] = []
    rejections: list[dict] = []
    by_id = {segment.segment_id: segment for segment in segments}

    for from_segment in sorted(segments, key=lambda segment: (segment.end_frame, segment.segment_id)):
        candidates = [
            to_segment
            for to_segment in segments
            if to_segment.segment_id != from_segment.segment_id
            and to_segment.segment_id not in used_to_segments
            and to_segment.start_frame > from_segment.end_frame
            and to_segment.start_frame - from_segment.end_frame - 1 <= max_gap_frames
        ]
        ranked = []
        for to_segment in candidates:
            metrics = link_candidate_metrics(from_segment, to_segment)
            reason = reject_reason(
                from_segment,
                to_segment,
                metrics,
                max_gap_frames,
                max_link_distance_px,
                max_speed_change_ratio,
                max_size_ratio,
            )
            if reason:
                rejections.append(rejection_row(from_segment, to_segment, metrics, reason))
                continue
            if has_ambiguous_cross_id_swap(from_segment, to_segment, segments, max_gap_frames):
                rejections.append(rejection_row(from_segment, to_segment, metrics, "ambiguous_cross_id_swap"))
                continue
            ranked.append((metrics["predicted_distance_px"], metrics["gap_frames"], to_segment.segment_id, metrics))
        if ranked:
            _, _, to_segment_id, metrics = sorted(ranked)[0]
            to_segment = by_id[to_segment_id]
            next_by_segment[from_segment.segment_id] = to_segment.segment_id
            used_to_segments.add(to_segment.segment_id)
            links.append(link_row(from_segment, to_segment, metrics))
    links.sort(key=lambda row: (int(row["from_end_frame"]), row["from_segment_id"]))
    rejections.sort(key=lambda row: (int(row["from_end_frame"]), row["from_segment_id"], row["to_segment_id"]))
    return next_by_segment, links, rejections


def link_row(from_segment: TrackSegment, to_segment: TrackSegment, metrics: dict) -> dict:
    return {
        "from_segment_id": from_segment.segment_id,
        "to_segment_id": to_segment.segment_id,
        "from_track_id": from_segment.raw_track_id,
        "to_track_id": to_segment.raw_track_id,
        "from_end_frame": str(from_segment.end_frame),
        "to_start_frame": str(to_segment.start_frame),
        "gap_frames": str(metrics["gap_frames"]),
        "predicted_distance_px": f"{metrics['predicted_distance_px']:.2f}",
        "speed_change_ratio": f"{metrics['speed_change_ratio']:.2f}",
        "size_ratio": f"{metrics['size_ratio']:.2f}",
        "link_action": "STITCH_TO_SAME_LOGICAL_VEHICLE",
    }


def rejection_row(from_segment: TrackSegment, to_segment: TrackSegment, metrics: dict, reason: str) -> dict:
    return {
        "from_segment_id": from_segment.segment_id,
        "to_segment_id": to_segment.segment_id,
        "from_track_id": from_segment.raw_track_id,
        "to_track_id": to_segment.raw_track_id,
        "from_end_frame": str(from_segment.end_frame),
        "to_start_frame": str(to_segment.start_frame),
        "gap_frames": str(metrics["gap_frames"]),
        "predicted_distance_px": f"{metrics['predicted_distance_px']:.2f}",
        "speed_change_ratio": f"{metrics['speed_change_ratio']:.2f}",
        "size_ratio": f"{metrics['size_ratio']:.2f}",
        "reject_reason": reason,
    }


def assign_logical_ids(segments: list[TrackSegment], next_by_segment: dict[str, str]) -> dict[str, str]:
    previous_by_segment = {to_segment: from_segment for from_segment, to_segment in next_by_segment.items()}
    logical_ids: dict[str, str] = {}
    logical_index = 1
    for segment in sorted(segments, key=lambda item: (item.start_frame, item.segment_id)):
        if segment.segment_id in previous_by_segment:
            continue
        logical_id = f"lv_{logical_index:04d}"
        logical_index += 1
        current = segment.segment_id
        while current:
            logical_ids[current] = logical_id
            current = next_by_segment.get(current, "")
    for segment in segments:
        if segment.segment_id not in logical_ids:
            logical_ids[segment.segment_id] = f"lv_{logical_index:04d}"
            logical_index += 1
    return logical_ids


def build_detected_logical_rows(segments: list[TrackSegment], logical_ids: dict[str, str], fps: float) -> list[dict]:
    rows = []
    for segment in segments:
        logical_id = logical_ids[segment.segment_id]
        for row in segment.rows:
            cx, cy = center(row)
            rows.append(
                {
                    "logical_vehicle_id": logical_id,
                    "raw_track_id": segment.raw_track_id,
                    "segment_id": segment.segment_id,
                    "frame_id": str(int(float(row["frame_id"]))),
                    "time_sec": f"{int(float(row['frame_id'])) / fps:.2f}",
                    "source": "detected",
                    "class_name": row.get("class_name", ""),
                    "confidence": row.get("confidence", ""),
                    "x1": row.get("x1", ""),
                    "y1": row.get("y1", ""),
                    "x2": row.get("x2", ""),
                    "y2": row.get("y2", ""),
                    "center_x": f"{cx:.2f}",
                    "center_y": f"{cy:.2f}",
                    "stitch_note": "original_yolo_detection",
                }
            )
    return sorted(rows, key=lambda row: (int(row["frame_id"]), row["logical_vehicle_id"], row["raw_track_id"]))


def build_interpolated_rows(segments: list[TrackSegment], logical_ids: dict[str, str], links: list[dict], fps: float) -> list[dict]:
    by_segment = {segment.segment_id: segment for segment in segments}
    rows = []
    for link in links:
        from_segment = by_segment[link["from_segment_id"]]
        to_segment = by_segment[link["to_segment_id"]]
        gap = int(link["gap_frames"])
        if gap <= 0:
            continue
        logical_id = logical_ids[from_segment.segment_id]
        for step in range(1, gap + 1):
            ratio = step / (gap + 1)
            frame_id = from_segment.end_frame + step
            cx = from_segment.end_center[0] + (to_segment.start_center[0] - from_segment.end_center[0]) * ratio
            cy = from_segment.end_center[1] + (to_segment.start_center[1] - from_segment.end_center[1]) * ratio
            rows.append(
                {
                    "logical_vehicle_id": logical_id,
                    "raw_track_id": f"{from_segment.raw_track_id}|{to_segment.raw_track_id}",
                    "segment_id": f"{from_segment.segment_id}->{to_segment.segment_id}",
                    "frame_id": str(frame_id),
                    "time_sec": f"{frame_id / fps:.2f}",
                    "source": "interpolated",
                    "class_name": from_segment.class_name,
                    "confidence": "",
                    "x1": "",
                    "y1": "",
                    "x2": "",
                    "y2": "",
                    "center_x": f"{cx:.2f}",
                    "center_y": f"{cy:.2f}",
                    "stitch_note": "linear_center_interpolation_between_stitched_segments",
                }
            )
    return sorted(rows, key=lambda row: (int(row["frame_id"]), row["logical_vehicle_id"]))


def build_summary_rows(segments: list[TrackSegment], logical_ids: dict[str, str], links: list[dict], rejections: list[dict], interpolated_rows: list[dict]) -> list[dict]:
    logical_to_segments: dict[str, list[TrackSegment]] = defaultdict(list)
    for segment in segments:
        logical_to_segments[logical_ids[segment.segment_id]].append(segment)
    rows = []
    for logical_id, logical_segments in sorted(logical_to_segments.items()):
        frames = []
        raw_track_ids = []
        for segment in logical_segments:
            raw_track_ids.append(segment.raw_track_id)
            frames.extend(int(float(row["frame_id"])) for row in segment.rows)
        rows.append(
            {
                "logical_vehicle_id": logical_id,
                "raw_track_ids": "|".join(sorted(set(raw_track_ids))),
                "segment_count": str(len(logical_segments)),
                "detected_frame_count": str(len(frames)),
                "start_frame": str(min(frames)),
                "end_frame": str(max(frames)),
                "linked_segment_count": str(max(0, len(logical_segments) - 1)),
            }
        )
    rows.append(
        {
            "logical_vehicle_id": "__overall__",
            "raw_track_ids": "",
            "segment_count": str(len(segments)),
            "detected_frame_count": str(sum(len(segment.rows) for segment in segments)),
            "start_frame": "",
            "end_frame": "",
            "linked_segment_count": str(len(links)),
            "rejected_candidate_links": str(len(rejections)),
            "interpolated_points": str(len(interpolated_rows)),
        }
    )
    return rows


def build_track_stitch_mock(
    detection_rows: list[dict],
    final_ids: set[str],
    fps: float,
    max_gap_frames: int = 10,
    max_link_distance_px: float = 80.0,
    max_speed_change_ratio: float = 4.0,
    max_size_ratio: float = 2.5,
) -> StitchOutputs:
    segments = build_segments(detection_rows, final_ids)
    next_by_segment, links, rejections = choose_links(
        segments,
        max_gap_frames=max_gap_frames,
        max_link_distance_px=max_link_distance_px,
        max_speed_change_ratio=max_speed_change_ratio,
        max_size_ratio=max_size_ratio,
    )
    logical_ids = assign_logical_ids(segments, next_by_segment)
    detected_rows = build_detected_logical_rows(segments, logical_ids, fps)
    interpolated_rows = build_interpolated_rows(segments, logical_ids, links, fps)
    logical_tracks = sorted(detected_rows + interpolated_rows, key=lambda row: (int(row["frame_id"]), row["logical_vehicle_id"], row["source"]))
    summary_rows = build_summary_rows(segments, logical_ids, links, rejections, interpolated_rows)
    return StitchOutputs(
        logical_tracks=logical_tracks,
        links=links,
        rejections=rejections,
        interpolated_points=interpolated_rows,
        summary_rows=summary_rows,
    )


LOGICAL_TRACK_FIELDS = [
    "logical_vehicle_id",
    "raw_track_id",
    "segment_id",
    "frame_id",
    "time_sec",
    "source",
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "center_x",
    "center_y",
    "stitch_note",
]

LINK_FIELDS = [
    "from_segment_id",
    "to_segment_id",
    "from_track_id",
    "to_track_id",
    "from_end_frame",
    "to_start_frame",
    "gap_frames",
    "predicted_distance_px",
    "speed_change_ratio",
    "size_ratio",
    "link_action",
]

REJECTION_FIELDS = [
    "from_segment_id",
    "to_segment_id",
    "from_track_id",
    "to_track_id",
    "from_end_frame",
    "to_start_frame",
    "gap_frames",
    "predicted_distance_px",
    "speed_change_ratio",
    "size_ratio",
    "reject_reason",
]

SUMMARY_FIELDS = [
    "logical_vehicle_id",
    "raw_track_ids",
    "segment_count",
    "detected_frame_count",
    "start_frame",
    "end_frame",
    "linked_segment_count",
    "rejected_candidate_links",
    "interpolated_points",
]

LOGICAL_OVERLAY_FIELDS = [
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
]


def final_track_ids(final_rows: list[dict]) -> set[str]:
    return {row["track_id"] for row in final_rows}


def write_outputs(output_dir: Path, outputs: StitchOutputs) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "logical_vehicle_tracks.csv", outputs.logical_tracks, LOGICAL_TRACK_FIELDS)
    overlay_rows = [
        {
            "frame_id": row["frame_id"],
            "time_sec": row["time_sec"],
            "track_id": row["raw_track_id"],
            "logical_vehicle_id": row["logical_vehicle_id"],
            "class_name": row["class_name"],
            "confidence": row["confidence"],
            "x1": row["x1"],
            "y1": row["y1"],
            "x2": row["x2"],
            "y2": row["y2"],
        }
        for row in outputs.logical_tracks
        if row["source"] == "detected"
    ]
    write_csv(output_dir / "logical_overlay_detections.csv", overlay_rows, LOGICAL_OVERLAY_FIELDS)
    write_csv(output_dir / "track_stitching_links.csv", outputs.links, LINK_FIELDS)
    write_csv(output_dir / "track_stitching_rejections.csv", outputs.rejections, REJECTION_FIELDS)
    write_csv(output_dir / "interpolated_points.csv", outputs.interpolated_points, LOGICAL_TRACK_FIELDS)
    write_csv(output_dir / "track_stitching_summary.csv", outputs.summary_rows, SUMMARY_FIELDS)
    note = (
        "# yolo26x manual_filter_v1 track stitching mock\n\n"
        "This directory contains an offline logical-vehicle-ID layer. It does not overwrite YOLO track IDs or manual-filter outputs.\n\n"
        "Detected rows retain their original `raw_track_id`. Interpolated rows are marked with `source=interpolated`.\n"
    )
    (output_dir / "TRACK_STITCHING_MOCK_NOTE.md").write_text(note, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections", required=True)
    parser.add_argument("--final-targets", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--max-gap-frames", type=int, default=10)
    parser.add_argument("--max-link-distance-px", type=float, default=80.0)
    parser.add_argument("--max-speed-change-ratio", type=float, default=4.0)
    parser.add_argument("--max-size-ratio", type=float, default=2.5)
    args = parser.parse_args()

    detection_rows = read_csv(Path(args.detections))
    final_rows = read_csv(Path(args.final_targets))
    outputs = build_track_stitch_mock(
        detection_rows=detection_rows,
        final_ids=final_track_ids(final_rows),
        fps=args.fps,
        max_gap_frames=args.max_gap_frames,
        max_link_distance_px=args.max_link_distance_px,
        max_speed_change_ratio=args.max_speed_change_ratio,
        max_size_ratio=args.max_size_ratio,
    )
    write_outputs(Path(args.output_dir), outputs)
    overall = outputs.summary_rows[-1]
    print(f"logical_vehicle_rows={len(outputs.logical_tracks)}")
    print(f"stitched_links={len(outputs.links)}")
    print(f"rejected_candidate_links={overall['rejected_candidate_links']}")
    print(f"interpolated_points={len(outputs.interpolated_points)}")
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
