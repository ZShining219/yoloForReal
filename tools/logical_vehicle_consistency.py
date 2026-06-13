#!/usr/bin/env python3
"""Logical vehicle consistency helpers.

This module is intentionally target-layer only. It does not infer lanes,
turns, OD, SUMO routes, or simulation demand.
"""

from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


VEHICLE_CLASSES = {"car", "truck", "bus"}

LOGICAL_TRACK_FIELDS = [
    "frame_id",
    "time_sec",
    "logical_vehicle_id",
    "raw_track_id",
    "tracklet_id",
    "source",
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "center_x",
    "center_y",
    "association_status",
    "vehicle_validity_status",
    "purity_status",
    "final_gate_status",
]

DUPLICATE_GROUP_FIELDS = [
    "duplicate_group_id",
    "frame_id",
    "representative_raw_track_id",
    "suppressed_raw_track_ids",
    "member_raw_track_ids",
    "member_count",
    "status",
]

TRACKLET_FIELDS = [
    "tracklet_id",
    "raw_track_id",
    "class_name",
    "start_frame",
    "end_frame",
    "frame_count",
    "mean_confidence",
    "start_center_x",
    "start_center_y",
    "end_center_x",
    "end_center_y",
]

LINK_FIELDS = [
    "from_tracklet_id",
    "to_tracklet_id",
    "from_raw_track_id",
    "to_raw_track_id",
    "from_end_frame",
    "to_start_frame",
    "gap_frames",
    "predicted_distance_px",
    "size_ratio",
    "link_cost",
    "link_status",
    "review_status",
]

SUMMARY_FIELDS = [
    "logical_vehicle_id",
    "raw_track_ids",
    "tracklet_ids",
    "detected_frame_count",
    "start_frame",
    "end_frame",
    "association_status",
]

RAW_MAPPING_FIELDS = [
    "raw_track_id",
    "logical_vehicle_id",
    "tracklet_ids",
    "detected_frame_count",
    "association_status",
]

VALIDATION_FIELDS = [
    "check_name",
    "logical_vehicle_id",
    "frame_id",
    "status",
    "message",
]

TARGET_VALIDITY_FIELDS = [
    "logical_vehicle_id",
    "class_votes",
    "detected_frame_count",
    "median_bbox_area",
    "vehicle_validity_status",
    "exclude_reason",
    "review_reason",
]

IDENTITY_PURITY_FIELDS = [
    "logical_vehicle_id",
    "purity_status",
    "review_reason",
    "max_speed_px_per_frame",
    "max_bbox_area_ratio",
    "raw_switch_count",
]

FINAL_GATE_FIELDS = [
    "logical_vehicle_id",
    "vehicle_validity_status",
    "purity_status",
    "final_gate_status",
    "exclude_reason",
]

RAW_SPLIT_REVIEW_FIELDS = [
    "raw_track_id",
    "from_logical_vehicle_id",
    "to_logical_vehicle_id",
    "gap_frames",
    "center_distance_px",
    "center_distance_per_frame",
    "bbox_size_ratio",
    "class_compatible",
    "suggested_action",
    "merge_reason",
]

RISKY_LINK_REVIEW_FIELDS = [
    "from_tracklet_id",
    "to_tracklet_id",
    "from_raw_track_id",
    "to_raw_track_id",
    "gap_frames",
    "predicted_distance_px",
    "size_ratio",
    "link_cost",
    "risk_reason",
]

FRAGMENT_PATH_ABSORPTION_FIELDS = [
    "fragment_logical_vehicle_id",
    "mature_logical_vehicle_id",
    "fragment_raw_track_ids",
    "mature_raw_track_ids",
    "fragment_frame_count",
    "mature_frame_count",
    "overlap_frame_count",
    "gap_fill_frame_count",
    "overlap_ratio",
    "median_iou",
    "median_center_distance_px",
    "median_size_ratio",
    "action",
    "reason",
]

TARGET_QUALITY_FIELDS = [
    "logical_vehicle_id",
    "raw_track_ids",
    "detected_frame_count",
    "start_frame",
    "end_frame",
    "duration_frames",
    "coverage_ratio",
    "tracklet_count",
    "gap_count",
    "low_confidence_count",
    "low_confidence_ratio",
    "median_bbox_area",
    "total_displacement_px",
    "border_touch_ratio",
    "quality_status",
    "risk_reasons",
]

CROSS_RAW_RECOVERY_FIELDS = [
    "from_logical_vehicle_id",
    "to_logical_vehicle_id",
    "from_raw_track_ids",
    "to_raw_track_ids",
    "from_end_frame",
    "to_start_frame",
    "gap_frames",
    "center_distance_px",
    "center_distance_per_frame",
    "bbox_size_ratio",
    "class_compatible",
    "candidate_rank",
    "review_status",
    "reason",
]


@dataclass
class DuplicateGroupingResult:
    representative_rows: list[dict]
    suppressed_rows: list[dict]
    groups: list[dict]


@dataclass
class Tracklet:
    tracklet_id: str
    raw_track_id: str
    rows: list[dict]
    class_name: str
    start_frame: int
    end_frame: int
    start_center: tuple[float, float]
    end_center: tuple[float, float]
    start_size: tuple[float, float]
    end_size: tuple[float, float]
    velocity: tuple[float, float]
    mean_confidence: float

    def to_row(self) -> dict:
        return {
            "tracklet_id": self.tracklet_id,
            "raw_track_id": self.raw_track_id,
            "class_name": self.class_name,
            "start_frame": str(self.start_frame),
            "end_frame": str(self.end_frame),
            "frame_count": str(len(self.rows)),
            "mean_confidence": f"{self.mean_confidence:.4f}",
            "start_center_x": f"{self.start_center[0]:.2f}",
            "start_center_y": f"{self.start_center[1]:.2f}",
            "end_center_x": f"{self.end_center[0]:.2f}",
            "end_center_y": f"{self.end_center[1]:.2f}",
        }


@dataclass
class AssociationResult:
    logical_tracks: list[dict]
    logical_vehicle_summary: list[dict]
    raw_track_mapping: list[dict]
    link_candidates: list[dict]
    accepted_links: list[dict]
    ambiguous_links: list[dict]


@dataclass
class ConsistencyOutputs:
    logical_tracks: list[dict]
    logical_vehicle_summary: list[dict]
    raw_track_to_logical_vehicle: list[dict]
    duplicate_groups: list[dict]
    tracklets: list[dict]
    tracklet_link_candidates: list[dict]
    tracklet_links_accepted: list[dict]
    ambiguous_link_review: list[dict]
    consistency_validation_report: list[dict]
    target_validity_report: list[dict]
    identity_purity_report: list[dict]
    final_target_gate: list[dict]
    raw_track_split_review: list[dict]
    risky_accepted_link_review: list[dict]
    fragment_path_absorption_review: list[dict]
    target_quality_report: list[dict]
    cross_raw_recovery_review: list[dict]

    def as_dict(self) -> dict[str, list[dict]]:
        return {
            "logical_vehicle_tracks": self.logical_tracks,
            "logical_vehicle_summary": self.logical_vehicle_summary,
            "raw_track_to_logical_vehicle": self.raw_track_to_logical_vehicle,
            "duplicate_groups": self.duplicate_groups,
            "tracklets": self.tracklets,
            "tracklet_link_candidates": self.tracklet_link_candidates,
            "tracklet_links_accepted": self.tracklet_links_accepted,
            "ambiguous_link_review": self.ambiguous_link_review,
            "consistency_validation_report": self.consistency_validation_report,
            "target_validity_report": self.target_validity_report,
            "identity_purity_report": self.identity_purity_report,
            "final_target_gate": self.final_target_gate,
            "raw_track_split_review": self.raw_track_split_review,
            "risky_accepted_link_review": self.risky_accepted_link_review,
            "fragment_path_absorption_review": self.fragment_path_absorption_review,
            "target_quality_report": self.target_quality_report,
            "cross_raw_recovery_review": self.cross_raw_recovery_review,
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
    if track_id.startswith("mot_"):
        return track_id
    return f"mot_{int(float(track_id)):04d}"


def final_track_ids(final_targets: list[dict]) -> set[str]:
    return {normalized_track_id(row["track_id"]) for row in final_targets}


def as_float(row: dict, key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    return default if value == "" else float(value)


def frame_id(row: dict) -> int:
    return int(float(row["frame_id"]))


def center(row: dict) -> tuple[float, float]:
    return ((as_float(row, "x1") + as_float(row, "x2")) / 2.0, (as_float(row, "y1") + as_float(row, "y2")) / 2.0)


def size(row: dict) -> tuple[float, float]:
    return (max(0.01, as_float(row, "x2") - as_float(row, "x1")), max(0.01, as_float(row, "y2") - as_float(row, "y1")))


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def bbox(row: dict) -> tuple[float, float, float, float]:
    return tuple(as_float(row, key) for key in ("x1", "y1", "x2", "y2"))


def area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def bbox_iou(row_a: dict, row_b: dict) -> float:
    ax1, ay1, ax2, ay2 = bbox(row_a)
    bx1, by1, bx2, by2 = bbox(row_b)
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    union = area((ax1, ay1, ax2, ay2)) + area((bx1, by1, bx2, by2)) - inter
    return inter / union if union > 0 else 0.0


def class_compatible(a: str, b: str) -> bool:
    return a == b or (a in VEHICLE_CLASSES and b in VEHICLE_CLASSES)


def normalized_detection(row: dict) -> dict:
    copied = dict(row)
    raw_track_id = row.get("raw_track_id") or row.get("track_id", "")
    copied["raw_track_id"] = normalized_track_id(raw_track_id)
    copied["track_id"] = normalized_track_id(raw_track_id)
    cx, cy = center(copied)
    copied["center_x"] = f"{cx:.2f}"
    copied["center_y"] = f"{cy:.2f}"
    return copied


def group_same_frame_duplicates(
    rows: list[dict],
    iou_threshold: float = 0.85,
) -> DuplicateGroupingResult:
    normalized_rows = [normalized_detection(row) for row in rows]
    raw_support = Counter(row["raw_track_id"] for row in normalized_rows)
    by_frame: dict[int, list[dict]] = defaultdict(list)
    for row in normalized_rows:
        by_frame[frame_id(row)].append(row)

    representative_rows: list[dict] = []
    suppressed_rows: list[dict] = []
    groups: list[dict] = []
    group_index = 1

    for current_frame in sorted(by_frame):
        frame_rows = by_frame[current_frame]
        parent = list(range(len(frame_rows)))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(a: int, b: int) -> None:
            root_a = find(a)
            root_b = find(b)
            if root_a != root_b:
                parent[root_b] = root_a

        for index, row_a in enumerate(frame_rows):
            for other_index, row_b in enumerate(frame_rows[index + 1 :], start=index + 1):
                if not class_compatible(row_a.get("class_name", ""), row_b.get("class_name", "")):
                    continue
                if bbox_iou(row_a, row_b) >= iou_threshold:
                    union(index, other_index)

        components: dict[int, list[dict]] = defaultdict(list)
        for index, row in enumerate(frame_rows):
            components[find(index)].append(row)

        for component in sorted(components.values(), key=lambda item: min(row["raw_track_id"] for row in item)):
            if len(component) == 1:
                representative_rows.append(component[0])
                continue
            representative = max(
                component,
                key=lambda row: (
                    raw_support[row["raw_track_id"]],
                    as_float(row, "confidence"),
                    row["raw_track_id"],
                ),
            )
            suppressed = [row for row in component if row is not representative]
            representative_rows.append(representative)
            suppressed_rows.extend(suppressed)
            groups.append(
                {
                    "duplicate_group_id": f"dup_{group_index:04d}",
                    "frame_id": str(current_frame),
                    "representative_raw_track_id": representative["raw_track_id"],
                    "suppressed_raw_track_ids": "|".join(sorted(row["raw_track_id"] for row in suppressed)),
                    "member_raw_track_ids": "|".join(sorted(row["raw_track_id"] for row in component)),
                    "member_count": str(len(component)),
                    "status": "duplicate_suppressed",
                }
            )
            group_index += 1

    representative_rows.sort(key=lambda row: (frame_id(row), row["raw_track_id"]))
    suppressed_rows.sort(key=lambda row: (frame_id(row), row["raw_track_id"]))
    return DuplicateGroupingResult(representative_rows=representative_rows, suppressed_rows=suppressed_rows, groups=groups)


def tracklet_velocity(rows: list[dict]) -> tuple[float, float]:
    if len(rows) < 2:
        return (0.0, 0.0)
    first = rows[0]
    last = rows[-1]
    delta = max(1, frame_id(last) - frame_id(first))
    first_center = center(first)
    last_center = center(last)
    return ((last_center[0] - first_center[0]) / delta, (last_center[1] - first_center[1]) / delta)


def make_tracklet(raw_track_id: str, segment_index: int, rows: list[dict]) -> Tracklet:
    ordered = sorted([normalized_detection(row) for row in rows], key=frame_id)
    class_votes = Counter(row.get("class_name", "") for row in ordered)
    confidences = [as_float(row, "confidence") for row in ordered]
    return Tracklet(
        tracklet_id=f"{raw_track_id}_tl{segment_index:02d}",
        raw_track_id=raw_track_id,
        rows=ordered,
        class_name=class_votes.most_common(1)[0][0],
        start_frame=frame_id(ordered[0]),
        end_frame=frame_id(ordered[-1]),
        start_center=center(ordered[0]),
        end_center=center(ordered[-1]),
        start_size=size(ordered[0]),
        end_size=size(ordered[-1]),
        velocity=tracklet_velocity(ordered),
        mean_confidence=sum(confidences) / len(confidences),
    )


def build_tracklets(
    rows: list[dict],
    allowed_track_ids: set[str] | None = None,
    max_gap_frames: int = 1,
) -> list[Tracklet]:
    allowed = {normalized_track_id(track_id) for track_id in allowed_track_ids} if allowed_track_ids else None
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        normalized = normalized_detection(row)
        raw_track_id = normalized["raw_track_id"]
        if allowed is not None and raw_track_id not in allowed:
            continue
        grouped[raw_track_id].append(normalized)

    tracklets: list[Tracklet] = []
    for raw_track_id, group in sorted(grouped.items()):
        ordered = sorted(group, key=frame_id)
        current = [ordered[0]]
        previous_frame = frame_id(ordered[0])
        segment_index = 1
        for row in ordered[1:]:
            current_frame = frame_id(row)
            if current_frame - previous_frame <= max_gap_frames:
                current.append(row)
            else:
                tracklets.append(make_tracklet(raw_track_id, segment_index, current))
                segment_index += 1
                current = [row]
            previous_frame = current_frame
        tracklets.append(make_tracklet(raw_track_id, segment_index, current))
    return sorted(tracklets, key=lambda item: (item.start_frame, item.raw_track_id, item.tracklet_id))


def size_ratio(a: tuple[float, float], b: tuple[float, float]) -> float:
    return max(max(a[0], b[0]) / max(min(a[0], b[0]), 0.01), max(a[1], b[1]) / max(min(a[1], b[1]), 0.01))


def candidate_link_row(
    from_tracklet: Tracklet,
    to_tracklet: Tracklet,
    max_link_distance_px: float,
    max_size_ratio: float,
) -> dict:
    gap_frames = to_tracklet.start_frame - from_tracklet.end_frame - 1
    frame_delta = to_tracklet.start_frame - from_tracklet.end_frame
    predicted_center = (
        from_tracklet.end_center[0] + from_tracklet.velocity[0] * frame_delta,
        from_tracklet.end_center[1] + from_tracklet.velocity[1] * frame_delta,
    )
    predicted_distance = distance(predicted_center, to_tracklet.start_center)
    ratio = size_ratio(from_tracklet.end_size, to_tracklet.start_size)
    raw_penalty = 0.0 if from_tracklet.raw_track_id == to_tracklet.raw_track_id else 3.0
    cost = predicted_distance + max(0, gap_frames) * 2.0 + raw_penalty
    if not class_compatible(from_tracklet.class_name, to_tracklet.class_name):
        status = "REJECT_CLASS"
    elif predicted_distance > max_link_distance_px:
        status = "REJECT_DISTANCE"
    elif ratio > max_size_ratio:
        status = "REJECT_SIZE"
    else:
        status = "ACCEPT_CANDIDATE"
    return {
        "from_tracklet_id": from_tracklet.tracklet_id,
        "to_tracklet_id": to_tracklet.tracklet_id,
        "from_raw_track_id": from_tracklet.raw_track_id,
        "to_raw_track_id": to_tracklet.raw_track_id,
        "from_end_frame": str(from_tracklet.end_frame),
        "to_start_frame": str(to_tracklet.start_frame),
        "gap_frames": str(gap_frames),
        "predicted_distance_px": f"{predicted_distance:.2f}",
        "size_ratio": f"{ratio:.2f}",
        "link_cost": f"{cost:.2f}",
        "link_status": status,
        "review_status": "AUTO_ACCEPT" if status == "ACCEPT_CANDIDATE" else "PENDING_REVIEW",
    }


def associate_tracklets(
    tracklets: list[Tracklet],
    max_gap_frames: int = 10,
    max_link_distance_px: float = 80.0,
    max_size_ratio: float = 2.5,
    ambiguous_cost_margin: float = 5.0,
) -> AssociationResult:
    by_id = {tracklet.tracklet_id: tracklet for tracklet in tracklets}
    candidates: list[dict] = []
    for from_tracklet in tracklets:
        for to_tracklet in tracklets:
            if to_tracklet.start_frame <= from_tracklet.end_frame:
                continue
            gap = to_tracklet.start_frame - from_tracklet.end_frame - 1
            if gap > max_gap_frames:
                continue
            candidates.append(candidate_link_row(from_tracklet, to_tracklet, max_link_distance_px, max_size_ratio))

    accepted_candidates = [row for row in candidates if row["link_status"] == "ACCEPT_CANDIDATE"]
    by_from: dict[str, list[dict]] = defaultdict(list)
    for row in accepted_candidates:
        by_from[row["from_tracklet_id"]].append(row)

    next_by_tracklet: dict[str, str] = {}
    used_to: set[str] = set()
    accepted_links: list[dict] = []
    ambiguous_links: list[dict] = []
    for from_id in sorted(by_from, key=lambda item: by_id[item].end_frame):
        ranked = sorted(by_from[from_id], key=lambda row: (float(row["link_cost"]), int(row["gap_frames"]), row["to_tracklet_id"]))
        best = ranked[0]
        same_raw_low_risk = (
            best["from_raw_track_id"] == best["to_raw_track_id"]
            and int(float(best["gap_frames"])) <= 3
            and float(best["predicted_distance_px"]) <= 2.0
            and float(best["size_ratio"]) <= 1.2
        )
        if (
            not same_raw_low_risk
            and len(ranked) > 1
            and float(ranked[1]["link_cost"]) - float(best["link_cost"]) <= ambiguous_cost_margin
        ):
            ambiguous = dict(ranked[0])
            ambiguous["review_status"] = "PENDING_REVIEW"
            ambiguous_links.append(ambiguous)
            continue
        if best["to_tracklet_id"] in used_to:
            continue
        next_by_tracklet[from_id] = best["to_tracklet_id"]
        used_to.add(best["to_tracklet_id"])
        accepted = dict(best)
        accepted["review_status"] = "AUTO_ACCEPT"
        accepted_links.append(accepted)

    logical_ids = assign_logical_ids(tracklets, next_by_tracklet)
    logical_tracks = build_logical_track_rows(tracklets, logical_ids)
    return AssociationResult(
        logical_tracks=logical_tracks,
        logical_vehicle_summary=build_logical_summary_rows(tracklets, logical_ids),
        raw_track_mapping=build_raw_mapping_rows(tracklets, logical_ids),
        link_candidates=sorted(candidates, key=lambda row: (int(row["from_end_frame"]), row["from_tracklet_id"], row["to_tracklet_id"])),
        accepted_links=sorted(accepted_links, key=lambda row: (int(row["from_end_frame"]), row["from_tracklet_id"])),
        ambiguous_links=sorted(ambiguous_links, key=lambda row: (int(row["from_end_frame"]), row["from_tracklet_id"])),
    )


def assign_logical_ids(tracklets: list[Tracklet], next_by_tracklet: dict[str, str]) -> dict[str, str]:
    previous_by_tracklet = {to_id: from_id for from_id, to_id in next_by_tracklet.items()}
    logical_ids: dict[str, str] = {}
    logical_index = 1
    for tracklet in sorted(tracklets, key=lambda item: (item.start_frame, item.tracklet_id)):
        if tracklet.tracklet_id in previous_by_tracklet:
            continue
        logical_id = f"lv_{logical_index:04d}"
        logical_index += 1
        current = tracklet.tracklet_id
        while current:
            logical_ids[current] = logical_id
            current = next_by_tracklet.get(current, "")
    for tracklet in tracklets:
        if tracklet.tracklet_id not in logical_ids:
            logical_ids[tracklet.tracklet_id] = f"lv_{logical_index:04d}"
            logical_index += 1
    return logical_ids


def logical_row_from_detection(row: dict, logical_vehicle_id: str, tracklet_id: str, status: str = "accepted") -> dict:
    cx, cy = center(row)
    return {
        "frame_id": str(frame_id(row)),
        "time_sec": row.get("time_sec", f"{frame_id(row) / 50.0:.2f}"),
        "logical_vehicle_id": logical_vehicle_id,
        "raw_track_id": normalized_track_id(row.get("raw_track_id") or row["track_id"]),
        "tracklet_id": tracklet_id,
        "source": "detected",
        "class_name": row.get("class_name", ""),
        "confidence": row.get("confidence", ""),
        "x1": row.get("x1", ""),
        "y1": row.get("y1", ""),
        "x2": row.get("x2", ""),
        "y2": row.get("y2", ""),
        "center_x": f"{cx:.2f}",
        "center_y": f"{cy:.2f}",
        "association_status": status,
    }


def build_logical_track_rows(tracklets: list[Tracklet], logical_ids: dict[str, str]) -> list[dict]:
    rows = []
    for tracklet in tracklets:
        logical_id = logical_ids[tracklet.tracklet_id]
        for row in tracklet.rows:
            rows.append(logical_row_from_detection(row, logical_id, tracklet.tracklet_id))
    return sorted(rows, key=lambda row: (int(row["frame_id"]), row["logical_vehicle_id"], row["raw_track_id"]))


def build_logical_summary_rows(tracklets: list[Tracklet], logical_ids: dict[str, str]) -> list[dict]:
    grouped: dict[str, list[Tracklet]] = defaultdict(list)
    for tracklet in tracklets:
        grouped[logical_ids[tracklet.tracklet_id]].append(tracklet)
    rows = []
    for logical_id, items in sorted(grouped.items()):
        frames = []
        raw_ids = set()
        tracklet_ids = []
        for tracklet in items:
            frames.extend(frame_id(row) for row in tracklet.rows)
            raw_ids.add(tracklet.raw_track_id)
            tracklet_ids.append(tracklet.tracklet_id)
        rows.append(
            {
                "logical_vehicle_id": logical_id,
                "raw_track_ids": "|".join(sorted(raw_ids)),
                "tracklet_ids": "|".join(sorted(tracklet_ids)),
                "detected_frame_count": str(len(frames)),
                "start_frame": str(min(frames)),
                "end_frame": str(max(frames)),
                "association_status": "accepted",
            }
        )
    return rows


def build_raw_mapping_rows(tracklets: list[Tracklet], logical_ids: dict[str, str]) -> list[dict]:
    grouped: dict[tuple[str, str], list[Tracklet]] = defaultdict(list)
    for tracklet in tracklets:
        grouped[(tracklet.raw_track_id, logical_ids[tracklet.tracklet_id])].append(tracklet)
    rows = []
    for (raw_track_id, logical_id), items in sorted(grouped.items()):
        rows.append(
            {
                "raw_track_id": raw_track_id,
                "logical_vehicle_id": logical_id,
                "tracklet_ids": "|".join(tracklet.tracklet_id for tracklet in sorted(items, key=lambda item: item.tracklet_id)),
                "detected_frame_count": str(sum(len(tracklet.rows) for tracklet in items)),
                "association_status": "accepted",
            }
        )
    return rows


def append_suppressed_duplicate_rows(
    logical_tracks: list[dict],
    suppressed_rows: list[dict],
    duplicate_groups: list[dict],
) -> list[dict]:
    representative_lookup = {
        (row["frame_id"], row["representative_raw_track_id"]): row
        for row in duplicate_groups
    }
    accepted_by_frame_raw = {
        (row["frame_id"], row["raw_track_id"]): row
        for row in logical_tracks
        if row["association_status"] == "accepted"
    }
    output = list(logical_tracks)
    for row in suppressed_rows:
        key_group = None
        for group in duplicate_groups:
            if group["frame_id"] == str(frame_id(row)) and normalized_track_id(row["raw_track_id"]) in group["suppressed_raw_track_ids"].split("|"):
                key_group = group
                break
        if key_group is None:
            continue
        representative_row = accepted_by_frame_raw.get((key_group["frame_id"], key_group["representative_raw_track_id"]))
        if representative_row is None:
            continue
        output.append(
            logical_row_from_detection(
                row,
                representative_row["logical_vehicle_id"],
                representative_row["tracklet_id"],
                status="duplicate_suppressed",
            )
        )
    return sorted(output, key=lambda item: (int(item["frame_id"]), item["logical_vehicle_id"], item["association_status"], item["raw_track_id"]))


def validate_logical_tracks(logical_rows: list[dict]) -> list[dict]:
    report = []
    accepted = [row for row in logical_rows if row.get("association_status") in {"accepted", "manual_review_applied"}]
    by_logical_frame: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in accepted:
        by_logical_frame[(row["logical_vehicle_id"], int(float(row["frame_id"])))].append(row)
    failures = 0
    for (logical_id, current_frame), rows in sorted(by_logical_frame.items()):
        if len(rows) <= 1:
            continue
        failures += 1
        report.append(
            {
                "check_name": "one_bbox_per_logical_vehicle_per_frame",
                "logical_vehicle_id": logical_id,
                "frame_id": str(current_frame),
                "status": "FAIL",
                "message": f"{len(rows)} accepted rows share one logical vehicle and frame",
            }
        )
    if failures == 0:
        report.append(
            {
                "check_name": "one_bbox_per_logical_vehicle_per_frame",
                "logical_vehicle_id": "__all__",
                "frame_id": "",
                "status": "PASS",
                "message": "accepted rows have at most one bbox per logical vehicle per frame",
            }
        )
    missing_status = [row for row in logical_rows if not row.get("association_status")]
    report.append(
        {
            "check_name": "retained_detection_state",
            "logical_vehicle_id": "__all__",
            "frame_id": "",
            "status": "FAIL" if missing_status else "PASS",
            "message": f"{len(missing_status)} rows missing association_status",
        }
    )
    return report


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def bbox_area(row: dict) -> float:
    return max(0.01, (as_float(row, "x2") - as_float(row, "x1")) * (as_float(row, "y2") - as_float(row, "y1")))


def frame_span(rows: list[dict]) -> tuple[int, int, int]:
    frames = [frame_id(row) for row in rows]
    start = min(frames)
    end = max(frames)
    return start, end, end - start + 1


def border_touch_ratio(rows: list[dict], frame_width: float = 1920.0, frame_height: float = 1080.0, margin_px: float = 2.0) -> float:
    if not rows:
        return 0.0
    touches = [
        row
        for row in rows
        if as_float(row, "x1") <= margin_px
        or as_float(row, "y1") <= margin_px
        or as_float(row, "x2") >= frame_width - margin_px
        or as_float(row, "y2") >= frame_height - margin_px
    ]
    return len(touches) / len(rows)


def target_quality_metrics(rows: list[dict]) -> dict:
    ordered = sorted(rows, key=lambda row: (frame_id(row), row["raw_track_id"]))
    start, end, duration = frame_span(ordered)
    frame_gaps = [
        frame_id(current) - frame_id(previous) - 1
        for previous, current in zip(ordered, ordered[1:])
        if frame_id(current) - frame_id(previous) > 1
    ]
    low_confidence_count = sum(1 for row in ordered if as_float(row, "confidence") < 0.4)
    area_median = median([bbox_area(row) for row in ordered])
    total_displacement = distance(center(ordered[0]), center(ordered[-1])) if len(ordered) > 1 else 0.0
    return {
        "start_frame": start,
        "end_frame": end,
        "duration_frames": duration,
        "coverage_ratio": len(ordered) / duration if duration else 0.0,
        "tracklet_count": len({row["tracklet_id"] for row in ordered}),
        "gap_count": len(frame_gaps),
        "low_confidence_count": low_confidence_count,
        "low_confidence_ratio": low_confidence_count / len(ordered),
        "median_bbox_area": area_median,
        "total_displacement_px": total_displacement,
        "border_touch_ratio": border_touch_ratio(ordered),
    }


def target_quality_reasons(metrics: dict, detected_frame_count: int) -> list[str]:
    reasons = []
    if detected_frame_count <= 5:
        reasons.append("very_short_track")
    elif detected_frame_count < 60:
        reasons.append("short_track")
    if metrics["coverage_ratio"] < 0.75:
        reasons.append("sparse_track")
    if metrics["tracklet_count"] >= 8 or metrics["gap_count"] >= 5:
        reasons.append("flicker_fragmented")
    if metrics["low_confidence_ratio"] >= 0.30:
        reasons.append("low_confidence_many")
    if metrics["median_bbox_area"] < 1200:
        reasons.append("small_area")
    if detected_frame_count < 80 and metrics["border_touch_ratio"] >= 0.30:
        reasons.append("border_short_target")
    if detected_frame_count < 80 and metrics["total_displacement_px"] < 8.0:
        reasons.append("static_like")
    return reasons


def build_target_quality_report(logical_rows: list[dict]) -> list[dict]:
    by_logical: dict[str, list[dict]] = defaultdict(list)
    for row in logical_rows:
        if row.get("association_status") == "accepted":
            by_logical[row["logical_vehicle_id"]].append(row)
    report = []
    for logical_id, rows in sorted(by_logical.items()):
        metrics = target_quality_metrics(rows)
        reasons = target_quality_reasons(metrics, len(rows))
        report.append(
            {
                "logical_vehicle_id": logical_id,
                "raw_track_ids": raw_track_ids_for_rows(rows),
                "detected_frame_count": str(len(rows)),
                "start_frame": str(metrics["start_frame"]),
                "end_frame": str(metrics["end_frame"]),
                "duration_frames": str(metrics["duration_frames"]),
                "coverage_ratio": f"{metrics['coverage_ratio']:.4f}",
                "tracklet_count": str(metrics["tracklet_count"]),
                "gap_count": str(metrics["gap_count"]),
                "low_confidence_count": str(metrics["low_confidence_count"]),
                "low_confidence_ratio": f"{metrics['low_confidence_ratio']:.4f}",
                "median_bbox_area": f"{metrics['median_bbox_area']:.2f}",
                "total_displacement_px": f"{metrics['total_displacement_px']:.2f}",
                "border_touch_ratio": f"{metrics['border_touch_ratio']:.4f}",
                "quality_status": "RISK_REVIEW" if reasons else "QUALITY_PASS",
                "risk_reasons": "|".join(reasons),
            }
        )
    return report


def build_target_validity_report(logical_rows: list[dict]) -> list[dict]:
    by_logical: dict[str, list[dict]] = defaultdict(list)
    for row in logical_rows:
        if row.get("association_status") != "accepted":
            continue
        by_logical[row["logical_vehicle_id"]].append(row)
    report = []
    for logical_id, rows in sorted(by_logical.items()):
        class_votes = Counter(row.get("class_name", "") for row in rows)
        dominant_class = class_votes.most_common(1)[0][0] if class_votes else ""
        area_median = median([bbox_area(row) for row in rows])
        height_median = median([as_float(row, "y2") - as_float(row, "y1") for row in rows])
        metrics = target_quality_metrics(rows)
        quality_reasons = set(target_quality_reasons(metrics, len(rows)))
        status = "AUTO_KEEP"
        exclude_reason = ""
        review_reason = ""
        if dominant_class in {"motorcycle", "bicycle", "person"}:
            status = "AUTO_EXCLUDE"
            exclude_reason = "two_wheeler_or_person_class"
        elif "very_short_track" in quality_reasons and (
            "static_like" in quality_reasons or "low_confidence_many" in quality_reasons
        ):
            status = "AUTO_EXCLUDE"
            exclude_reason = "short_static_false_positive"
        elif "border_short_target" in quality_reasons and "small_area" in quality_reasons:
            status = "REVIEW_ONLY_IF_UNCERTAIN"
            review_reason = "border_short_target"
        elif dominant_class == "car" and len(rows) < 50 and area_median < 800 and height_median < 26:
            status = "REVIEW_ONLY_IF_UNCERTAIN"
            review_reason = "small_short_vehicle_like_target"
        elif "short_track" in quality_reasons and "small_area" in quality_reasons:
            status = "REVIEW_ONLY_IF_UNCERTAIN"
            review_reason = "short_small_target"
        report.append(
            {
                "logical_vehicle_id": logical_id,
                "class_votes": "|".join(f"{key}:{value}" for key, value in sorted(class_votes.items())),
                "detected_frame_count": str(len(rows)),
                "median_bbox_area": f"{area_median:.2f}",
                "vehicle_validity_status": status,
                "exclude_reason": exclude_reason,
                "review_reason": review_reason,
            }
        )
    return report


def build_identity_purity_report(logical_rows: list[dict]) -> list[dict]:
    by_logical: dict[str, list[dict]] = defaultdict(list)
    for row in logical_rows:
        if row.get("association_status") != "accepted":
            continue
        by_logical[row["logical_vehicle_id"]].append(row)
    report = []
    for logical_id, rows in sorted(by_logical.items()):
        ordered = sorted(rows, key=lambda row: (int(float(row["frame_id"])), row["raw_track_id"]))
        max_speed = 0.0
        max_area_ratio = 1.0
        raw_switch_count = 0
        review_reasons = []
        for previous, current in zip(ordered, ordered[1:]):
            frame_gap = int(float(current["frame_id"])) - int(float(previous["frame_id"]))
            if frame_gap <= 0:
                continue
            max_speed = max(max_speed, distance(center(previous), center(current)) / frame_gap)
            ratio = max(bbox_area(previous), bbox_area(current)) / max(min(bbox_area(previous), bbox_area(current)), 0.01)
            max_area_ratio = max(max_area_ratio, ratio)
            if previous["raw_track_id"] != current["raw_track_id"]:
                raw_switch_count += 1
                if ratio > 2.0:
                    review_reasons.append("raw_switch_area_jump")
                if frame_gap > 5:
                    review_reasons.append("raw_switch_long_gap")
            if ratio > 2.2:
                review_reasons.append("bbox_area_jump")
        status = "PURITY_REVIEW" if review_reasons else "PURITY_PASS"
        report.append(
            {
                "logical_vehicle_id": logical_id,
                "purity_status": status,
                "review_reason": "|".join(sorted(set(review_reasons))),
                "max_speed_px_per_frame": f"{max_speed:.2f}",
                "max_bbox_area_ratio": f"{max_area_ratio:.2f}",
                "raw_switch_count": str(raw_switch_count),
            }
        )
    return report


def build_final_target_gate(
    summary_rows: list[dict],
    validity_rows: list[dict],
    purity_rows: list[dict],
) -> list[dict]:
    validity_by_id = {row["logical_vehicle_id"]: row for row in validity_rows}
    purity_by_id = {row["logical_vehicle_id"]: row for row in purity_rows}
    gate_rows = []
    for row in summary_rows:
        logical_id = row["logical_vehicle_id"]
        validity = validity_by_id.get(logical_id, {"vehicle_validity_status": "REVIEW_ONLY_IF_UNCERTAIN", "exclude_reason": "missing_validity"})
        purity = purity_by_id.get(logical_id, {"purity_status": "PURITY_REVIEW", "review_reason": "missing_purity"})
        validity_status = validity["vehicle_validity_status"]
        purity_status = purity["purity_status"]
        if validity_status == "AUTO_KEEP" and purity_status == "PURITY_PASS":
            final_status = "AUTO_KEEP"
            reason = ""
        elif validity_status == "AUTO_EXCLUDE":
            final_status = "AUTO_EXCLUDE"
            reason = validity.get("exclude_reason", "")
        else:
            final_status = "REVIEW_ONLY_IF_UNCERTAIN"
            reason = "|".join(filter(None, [validity.get("review_reason", ""), purity.get("review_reason", "")]))
        gate_rows.append(
            {
                "logical_vehicle_id": logical_id,
                "vehicle_validity_status": validity_status,
                "purity_status": purity_status,
                "final_gate_status": final_status,
                "exclude_reason": reason,
            }
        )
    return gate_rows


def apply_final_gate_to_tracks(logical_rows: list[dict], gate_rows: list[dict]) -> list[dict]:
    gate_by_id = {row["logical_vehicle_id"]: row for row in gate_rows}
    output = []
    for row in logical_rows:
        copy = dict(row)
        gate = gate_by_id.get(row["logical_vehicle_id"], {})
        copy["vehicle_validity_status"] = gate.get("vehicle_validity_status", "")
        copy["purity_status"] = gate.get("purity_status", "")
        copy["final_gate_status"] = gate.get("final_gate_status", "REVIEW_ONLY_IF_UNCERTAIN")
        output.append(copy)
    return output


def build_raw_track_split_review(logical_rows: list[dict]) -> list[dict]:
    accepted = [row for row in logical_rows if row.get("association_status") == "accepted"]
    by_raw_lv: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in accepted:
        by_raw_lv[(row["raw_track_id"], row["logical_vehicle_id"])].append(row)
    by_raw: dict[str, list[tuple[str, list[dict]]]] = defaultdict(list)
    for (raw_id, logical_id), rows in by_raw_lv.items():
        by_raw[raw_id].append((logical_id, sorted(rows, key=lambda item: int(float(item["frame_id"])))))
    output = []
    for raw_id, chunks in sorted(by_raw.items()):
        if len(chunks) <= 1:
            continue
        chunks = sorted(chunks, key=lambda item: int(float(item[1][0]["frame_id"])))
        for (from_id, from_rows), (to_id, to_rows) in zip(chunks, chunks[1:]):
            gap = int(float(to_rows[0]["frame_id"])) - int(float(from_rows[-1]["frame_id"])) - 1
            center_distance = distance(center(from_rows[-1]), center(to_rows[0]))
            center_distance_per_frame = center_distance / max(1, gap + 1)
            ratio = row_size_ratio(from_rows[-1], to_rows[0])
            compatible = class_compatible(dominant_class(from_rows), dominant_class(to_rows))
            reason = ""
            if gap <= 3 and center_distance <= 2.0:
                action = "AUTO_MERGE_SAME_RAW_CONTINUITY"
                reason = "same_raw_low_gap_continuity"
            elif (
                4 <= gap <= 25
                and len(from_rows) >= 20
                and len(to_rows) >= 20
                and center_distance_per_frame <= 2.0
                and ratio <= 1.35
                and compatible
            ):
                action = "AUTO_MERGE_SAME_RAW_OCCLUSION"
                reason = "same_raw_occlusion_recovery"
            elif gap <= 30 and center_distance <= 30.0:
                action = "REVIEW_RAW_SPLIT"
            else:
                action = "KEEP_SPLIT_ID_SWITCH_RISK"
            output.append(
                {
                    "raw_track_id": raw_id,
                    "from_logical_vehicle_id": from_id,
                    "to_logical_vehicle_id": to_id,
                    "gap_frames": str(gap),
                    "center_distance_px": f"{center_distance:.2f}",
                    "center_distance_per_frame": f"{center_distance_per_frame:.2f}",
                    "bbox_size_ratio": f"{ratio:.2f}",
                    "class_compatible": "yes" if compatible else "no",
                    "suggested_action": action,
                    "merge_reason": reason,
                }
            )
    return output


def logical_ids_temporally_overlap(logical_rows: list[dict], from_logical_id: str, to_logical_id: str) -> bool:
    from_frames = {
        int(float(row["frame_id"]))
        for row in logical_rows
        if row.get("association_status") == "accepted" and row["logical_vehicle_id"] == from_logical_id
    }
    to_frames = {
        int(float(row["frame_id"]))
        for row in logical_rows
        if row.get("association_status") == "accepted" and row["logical_vehicle_id"] == to_logical_id
    }
    return bool(from_frames & to_frames)


def rewrite_logical_id(logical_rows: list[dict], from_logical_id: str, to_logical_id: str) -> list[dict]:
    rewritten = []
    for row in logical_rows:
        copy = dict(row)
        if copy["logical_vehicle_id"] == to_logical_id:
            copy["logical_vehicle_id"] = from_logical_id
        rewritten.append(copy)
    return sorted(rewritten, key=lambda item: (int(float(item["frame_id"])), item["logical_vehicle_id"], item["raw_track_id"]))


def duplicate_like_overlap(row_a: dict, row_b: dict, min_iou: float = 0.45, max_center_distance_px: float = 12.0) -> bool:
    return bbox_iou(row_a, row_b) >= min_iou and distance(center(row_a), center(row_b)) <= max_center_distance_px


def component_overlap_is_duplicate_like(logical_rows: list[dict], logical_ids: set[str]) -> bool:
    by_frame: dict[int, list[dict]] = defaultdict(list)
    for row in logical_rows:
        if row.get("association_status") != "accepted" or row["logical_vehicle_id"] not in logical_ids:
            continue
        by_frame[int(float(row["frame_id"]))].append(row)
    for rows in by_frame.values():
        owners = {row["logical_vehicle_id"] for row in rows}
        if len(owners) <= 1:
            continue
        for index, row_a in enumerate(rows):
            for row_b in rows[index + 1 :]:
                if row_a["logical_vehicle_id"] == row_b["logical_vehicle_id"]:
                    continue
                if not duplicate_like_overlap(row_a, row_b):
                    return False
    return True


def suppress_same_logical_duplicate_like_rows(logical_rows: list[dict]) -> list[dict]:
    by_logical_frame: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, row in enumerate(logical_rows):
        if row.get("association_status") != "accepted":
            continue
        by_logical_frame[(row["logical_vehicle_id"], int(float(row["frame_id"])))].append(index)
    output = [dict(row) for row in logical_rows]
    for indexes in by_logical_frame.values():
        if len(indexes) <= 1:
            continue
        rows = [output[index] for index in indexes]
        if not all(duplicate_like_overlap(row_a, row_b) for idx, row_a in enumerate(rows) for row_b in rows[idx + 1 :]):
            continue
        representative_index = max(indexes, key=lambda index: (bbox_area(output[index]), as_float(output[index], "confidence")))
        for index in indexes:
            if index == representative_index:
                continue
            output[index]["association_status"] = "duplicate_suppressed"
    return sorted(output, key=lambda item: (int(float(item["frame_id"])), item["logical_vehicle_id"], item["association_status"], item["raw_track_id"]))


def apply_same_raw_continuity_merges(logical_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Apply only low-risk same-raw splits that cannot create same-frame duplicates."""
    review_rows = build_raw_track_split_review(logical_rows)
    merge_actions = {"AUTO_MERGE_SAME_RAW_CONTINUITY", "AUTO_MERGE_SAME_RAW_OCCLUSION"}
    merge_candidates = [row for row in review_rows if row["suggested_action"] in merge_actions]
    parent: dict[str, str] = {}

    def find(logical_id: str) -> str:
        parent.setdefault(logical_id, logical_id)
        while parent[logical_id] != logical_id:
            parent[logical_id] = parent[parent[logical_id]]
            logical_id = parent[logical_id]
        return logical_id

    def union(left: str, right: str) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left == root_right:
            return
        canonical = min(root_left, root_right)
        other = root_right if canonical == root_left else root_left
        parent[other] = canonical

    for candidate in merge_candidates:
        union(candidate["from_logical_vehicle_id"], candidate["to_logical_vehicle_id"])

    components: dict[str, set[str]] = defaultdict(set)
    for logical_id in list(parent):
        components[find(logical_id)].add(logical_id)

    mergeable_components: set[str] = set()
    for canonical, logical_ids in components.items():
        if component_overlap_is_duplicate_like(logical_rows, logical_ids):
            mergeable_components.add(canonical)

    current_rows = []
    for row in logical_rows:
        copy = dict(row)
        logical_id = copy["logical_vehicle_id"]
        if logical_id in parent:
            canonical = find(logical_id)
            if canonical in mergeable_components:
                copy["logical_vehicle_id"] = canonical
        current_rows.append(copy)
    current_rows.sort(key=lambda item: (int(float(item["frame_id"])), item["logical_vehicle_id"], item["raw_track_id"]))
    current_rows = suppress_same_logical_duplicate_like_rows(current_rows)

    applied_review_rows = []
    for review_row in review_rows:
        review_copy = dict(review_row)
        if review_row["suggested_action"] not in merge_actions:
            applied_review_rows.append(review_copy)
            continue
        canonical = find(review_row["from_logical_vehicle_id"])
        if canonical not in mergeable_components:
            review_copy["suggested_action"] = "KEEP_SPLIT_TEMPORAL_OVERLAP"
            applied_review_rows.append(review_copy)
            continue
        review_copy["suggested_action"] = "AUTO_MERGE_APPLIED"
        applied_review_rows.append(review_copy)
    return current_rows, applied_review_rows


def dominant_class(rows: list[dict]) -> str:
    votes = Counter(row.get("class_name", "") for row in rows)
    return votes.most_common(1)[0][0] if votes else ""


def row_size_ratio(row_a: dict, row_b: dict) -> float:
    return size_ratio(size(row_a), size(row_b))


def raw_track_ids_for_rows(rows: list[dict]) -> str:
    return "|".join(sorted({row["raw_track_id"] for row in rows}))


def build_cross_raw_recovery_review(
    logical_rows: list[dict],
    max_gap_frames: int = 60,
    max_center_distance_per_frame: float = 3.0,
    max_size_ratio: float = 1.6,
    min_segment_frames: int = 20,
) -> list[dict]:
    by_logical: dict[str, list[dict]] = defaultdict(list)
    for row in logical_rows:
        if row.get("association_status") == "accepted":
            by_logical[row["logical_vehicle_id"]].append(row)

    ordered_by_logical = {
        logical_id: sorted(rows, key=lambda item: int(float(item["frame_id"])))
        for logical_id, rows in by_logical.items()
        if len(rows) >= min_segment_frames
    }
    candidates_by_from: dict[str, list[dict]] = defaultdict(list)
    for from_id, from_rows in sorted(ordered_by_logical.items()):
        from_raw_ids = {row["raw_track_id"] for row in from_rows}
        from_last = from_rows[-1]
        for to_id, to_rows in sorted(ordered_by_logical.items()):
            if from_id == to_id:
                continue
            to_raw_ids = {row["raw_track_id"] for row in to_rows}
            if from_raw_ids & to_raw_ids:
                continue
            to_first = to_rows[0]
            gap = frame_id(to_first) - frame_id(from_last) - 1
            if gap < 0 or gap > max_gap_frames:
                continue
            center_distance = distance(center(from_last), center(to_first))
            center_distance_per_frame = center_distance / max(1, gap + 1)
            ratio = row_size_ratio(from_last, to_first)
            compatible = class_compatible(dominant_class(from_rows), dominant_class(to_rows))
            if not compatible:
                continue
            if center_distance_per_frame > max_center_distance_per_frame:
                continue
            if ratio > max_size_ratio:
                continue
            candidates_by_from[from_id].append(
                {
                    "from_logical_vehicle_id": from_id,
                    "to_logical_vehicle_id": to_id,
                    "from_raw_track_ids": raw_track_ids_for_rows(from_rows),
                    "to_raw_track_ids": raw_track_ids_for_rows(to_rows),
                    "from_end_frame": str(frame_id(from_last)),
                    "to_start_frame": str(frame_id(to_first)),
                    "gap_frames": str(gap),
                    "center_distance_px": f"{center_distance:.2f}",
                    "center_distance_per_frame": f"{center_distance_per_frame:.2f}",
                    "bbox_size_ratio": f"{ratio:.2f}",
                    "class_compatible": "yes",
                    "review_status": "REVIEW_CROSS_RAW_RECOVERY",
                    "reason": "cross_raw_occlusion_recovery_candidate",
                }
            )

    output = []
    for from_id, candidates in sorted(candidates_by_from.items()):
        ranked = sorted(
            candidates,
            key=lambda row: (
                float(row["center_distance_per_frame"]),
                float(row["center_distance_px"]),
                int(float(row["gap_frames"])),
                row["to_logical_vehicle_id"],
            ),
        )
        for rank, row in enumerate(ranked, start=1):
            copy = dict(row)
            copy["candidate_rank"] = str(rank)
            if rank > 1:
                copy["reason"] = f"{copy['reason']}|competing_candidate"
            output.append(copy)
    return output


def rows_by_int_frame(rows: list[dict]) -> dict[int, dict]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[int(float(row["frame_id"]))].append(row)
    return {
        current_frame: max(items, key=lambda item: (bbox_area(item), as_float(item, "confidence")))
        for current_frame, items in grouped.items()
    }


def fragment_absorption_candidate(
    fragment_id: str,
    fragment_rows: list[dict],
    mature_id: str,
    mature_rows: list[dict],
    min_overlap_ratio: float,
    min_median_iou: float,
    max_median_center_distance_px: float,
    max_median_size_ratio: float,
) -> dict | None:
    if not class_compatible(dominant_class(fragment_rows), dominant_class(mature_rows)):
        return None
    mature_by_frame = rows_by_int_frame(mature_rows)
    overlap_metrics = []
    gap_fill_frames = 0
    for fragment_row in fragment_rows:
        current_frame = int(float(fragment_row["frame_id"]))
        mature_row = mature_by_frame.get(current_frame)
        if mature_row is None:
            gap_fill_frames += 1
            continue
        overlap_metrics.append(
            {
                "iou": bbox_iou(fragment_row, mature_row),
                "center_distance": distance(center(fragment_row), center(mature_row)),
                "size_ratio": row_size_ratio(fragment_row, mature_row),
            }
        )
    overlap_count = len(overlap_metrics)
    fragment_count = len(fragment_rows)
    overlap_ratio = overlap_count / fragment_count if fragment_count else 0.0
    median_iou = median([row["iou"] for row in overlap_metrics])
    median_center_distance = median([row["center_distance"] for row in overlap_metrics])
    median_ratio = median([row["size_ratio"] for row in overlap_metrics])
    if overlap_ratio < min_overlap_ratio:
        return None
    if median_iou < min_median_iou:
        return None
    if median_center_distance > max_median_center_distance_px:
        return None
    if median_ratio > max_median_size_ratio:
        return None
    return {
        "fragment_logical_vehicle_id": fragment_id,
        "mature_logical_vehicle_id": mature_id,
        "fragment_raw_track_ids": raw_track_ids_for_rows(fragment_rows),
        "mature_raw_track_ids": raw_track_ids_for_rows(mature_rows),
        "fragment_frame_count": str(fragment_count),
        "mature_frame_count": str(len(mature_rows)),
        "overlap_frame_count": str(overlap_count),
        "gap_fill_frame_count": str(gap_fill_frames),
        "overlap_ratio": f"{overlap_ratio:.4f}",
        "median_iou": f"{median_iou:.4f}",
        "median_center_distance_px": f"{median_center_distance:.2f}",
        "median_size_ratio": f"{median_ratio:.2f}",
        "action": "AUTO_ABSORB_FRAGMENT_PATH",
        "reason": "short_fragment_near_mature_path",
    }


def apply_fragment_path_absorption(
    logical_rows: list[dict],
    mature_min_frames: int = 100,
    fragment_max_frames: int = 50,
    min_overlap_ratio: float = 0.70,
    min_median_iou: float = 0.75,
    max_median_center_distance_px: float = 10.0,
    max_median_size_ratio: float = 1.35,
) -> tuple[list[dict], list[dict]]:
    accepted_by_logical: dict[str, list[dict]] = defaultdict(list)
    for row in logical_rows:
        if row.get("association_status") == "accepted":
            accepted_by_logical[row["logical_vehicle_id"]].append(row)

    mature_paths = {
        logical_id: sorted(rows, key=lambda item: int(float(item["frame_id"])))
        for logical_id, rows in accepted_by_logical.items()
        if len(rows) >= mature_min_frames
    }
    fragment_paths = {
        logical_id: sorted(rows, key=lambda item: int(float(item["frame_id"])))
        for logical_id, rows in accepted_by_logical.items()
        if 0 < len(rows) <= fragment_max_frames
    }

    review_rows = []
    absorption_by_fragment: dict[str, str] = {}
    for fragment_id, fragment_rows in sorted(fragment_paths.items()):
        candidates = []
        for mature_id, mature_rows in sorted(mature_paths.items()):
            if fragment_id == mature_id:
                continue
            candidate = fragment_absorption_candidate(
                fragment_id,
                fragment_rows,
                mature_id,
                mature_rows,
                min_overlap_ratio,
                min_median_iou,
                max_median_center_distance_px,
                max_median_size_ratio,
            )
            if candidate is not None:
                candidates.append(candidate)
        if len(candidates) == 1:
            review_rows.append(candidates[0])
            absorption_by_fragment[fragment_id] = candidates[0]["mature_logical_vehicle_id"]
        elif len(candidates) > 1:
            best = sorted(
                candidates,
                key=lambda row: (
                    -float(row["overlap_ratio"]),
                    -float(row["median_iou"]),
                    float(row["median_center_distance_px"]),
                    row["mature_logical_vehicle_id"],
                ),
            )[0]
            best["action"] = "REVIEW_AMBIGUOUS_FRAGMENT_MATCH"
            best["reason"] = "fragment_matches_multiple_mature_paths"
            review_rows.append(best)

    mature_frames = {
        mature_id: set(rows_by_int_frame(rows))
        for mature_id, rows in mature_paths.items()
    }
    output = []
    for row in logical_rows:
        copy = dict(row)
        fragment_id = copy["logical_vehicle_id"]
        if fragment_id in absorption_by_fragment:
            mature_id = absorption_by_fragment[fragment_id]
            copy["logical_vehicle_id"] = mature_id
            if (
                copy.get("association_status") == "accepted"
                and int(float(copy["frame_id"])) in mature_frames.get(mature_id, set())
            ):
                copy["association_status"] = "fragment_suppressed"
        output.append(copy)
    output.sort(key=lambda item: (int(float(item["frame_id"])), item["logical_vehicle_id"], item["association_status"], item["raw_track_id"]))
    return output, sorted(review_rows, key=lambda item: (item["fragment_logical_vehicle_id"], item["mature_logical_vehicle_id"]))


def build_logical_summary_from_rows(logical_rows: list[dict]) -> list[dict]:
    by_logical: dict[str, list[dict]] = defaultdict(list)
    for row in logical_rows:
        if row.get("association_status") != "accepted":
            continue
        by_logical[row["logical_vehicle_id"]].append(row)
    rows = []
    for logical_id, items in sorted(by_logical.items()):
        frames = [int(float(row["frame_id"])) for row in items]
        rows.append(
            {
                "logical_vehicle_id": logical_id,
                "raw_track_ids": "|".join(sorted({row["raw_track_id"] for row in items})),
                "tracklet_ids": "|".join(sorted({row["tracklet_id"] for row in items})),
                "detected_frame_count": str(len(frames)),
                "start_frame": str(min(frames)),
                "end_frame": str(max(frames)),
                "association_status": "accepted",
            }
        )
    return rows


def build_raw_mapping_from_rows(logical_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in logical_rows:
        if row.get("association_status") != "accepted":
            continue
        grouped[(row["raw_track_id"], row["logical_vehicle_id"])].append(row)
    rows = []
    for (raw_track_id, logical_id), items in sorted(grouped.items()):
        rows.append(
            {
                "raw_track_id": raw_track_id,
                "logical_vehicle_id": logical_id,
                "tracklet_ids": "|".join(sorted({row["tracklet_id"] for row in items})),
                "detected_frame_count": str(len(items)),
                "association_status": "accepted",
            }
        )
    return rows


def build_risky_accepted_link_review(accepted_links: list[dict]) -> list[dict]:
    rows = []
    for link in accepted_links:
        reasons = []
        cross_raw = link["from_raw_track_id"] != link["to_raw_track_id"]
        gap = int(float(link["gap_frames"]))
        predicted_distance = float(link["predicted_distance_px"])
        ratio = float(link["size_ratio"])
        cost = float(link["link_cost"])
        if cross_raw:
            reasons.append("cross_raw")
        if gap > 5:
            reasons.append("long_gap")
        if predicted_distance > 20:
            reasons.append("large_predicted_distance")
        if ratio > 1.5:
            reasons.append("large_size_ratio")
        if cost > 25:
            reasons.append("high_link_cost")
        if not reasons:
            continue
        rows.append(
            {
                "from_tracklet_id": link["from_tracklet_id"],
                "to_tracklet_id": link["to_tracklet_id"],
                "from_raw_track_id": link["from_raw_track_id"],
                "to_raw_track_id": link["to_raw_track_id"],
                "gap_frames": link["gap_frames"],
                "predicted_distance_px": link["predicted_distance_px"],
                "size_ratio": link["size_ratio"],
                "link_cost": link["link_cost"],
                "risk_reason": "|".join(reasons),
            }
        )
    return rows


def build_logical_vehicle_consistency(
    detections: list[dict],
    final_targets: list[dict],
    fps: float = 50.0,
    max_gap_frames: int = 10,
    max_link_distance_px: float = 80.0,
    max_iou: float = 0.85,
) -> ConsistencyOutputs:
    allowed = final_track_ids(final_targets)
    filtered = [
        normalized_detection(row)
        for row in detections
        if normalized_track_id(row["track_id"]) in allowed
    ]
    duplicate_result = group_same_frame_duplicates(filtered, iou_threshold=max_iou)
    tracklets = build_tracklets(duplicate_result.representative_rows, allowed_track_ids=allowed, max_gap_frames=1)
    association = associate_tracklets(
        tracklets,
        max_gap_frames=max_gap_frames,
        max_link_distance_px=max_link_distance_px,
    )
    logical_tracks = append_suppressed_duplicate_rows(
        association.logical_tracks,
        duplicate_result.suppressed_rows,
        duplicate_result.groups,
    )
    logical_tracks, raw_split_review = apply_same_raw_continuity_merges(logical_tracks)
    logical_tracks, fragment_absorption_review = apply_fragment_path_absorption(logical_tracks)
    logical_vehicle_summary = build_logical_summary_from_rows(logical_tracks)
    raw_track_mapping = build_raw_mapping_from_rows(logical_tracks)
    validity_report = build_target_validity_report(logical_tracks)
    target_quality_report = build_target_quality_report(logical_tracks)
    cross_raw_recovery_review = build_cross_raw_recovery_review(logical_tracks)
    purity_report = build_identity_purity_report(logical_tracks)
    final_gate = build_final_target_gate(logical_vehicle_summary, validity_report, purity_report)
    logical_tracks = apply_final_gate_to_tracks(logical_tracks, final_gate)
    return ConsistencyOutputs(
        logical_tracks=logical_tracks,
        logical_vehicle_summary=logical_vehicle_summary,
        raw_track_to_logical_vehicle=raw_track_mapping,
        duplicate_groups=duplicate_result.groups,
        tracklets=[tracklet.to_row() for tracklet in tracklets],
        tracklet_link_candidates=association.link_candidates,
        tracklet_links_accepted=association.accepted_links,
        ambiguous_link_review=association.ambiguous_links,
        consistency_validation_report=validate_logical_tracks(logical_tracks),
        target_validity_report=validity_report,
        identity_purity_report=purity_report,
        final_target_gate=final_gate,
        raw_track_split_review=raw_split_review,
        risky_accepted_link_review=build_risky_accepted_link_review(association.accepted_links),
        fragment_path_absorption_review=fragment_absorption_review,
        target_quality_report=target_quality_report,
        cross_raw_recovery_review=cross_raw_recovery_review,
    )


def write_consistency_outputs(output_dir: Path, outputs: ConsistencyOutputs) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "logical_vehicle_tracks.csv", outputs.logical_tracks, LOGICAL_TRACK_FIELDS)
    write_csv(output_dir / "logical_vehicle_summary.csv", outputs.logical_vehicle_summary, SUMMARY_FIELDS)
    write_csv(output_dir / "raw_track_to_logical_vehicle.csv", outputs.raw_track_to_logical_vehicle, RAW_MAPPING_FIELDS)
    write_csv(output_dir / "duplicate_groups.csv", outputs.duplicate_groups, DUPLICATE_GROUP_FIELDS)
    write_csv(output_dir / "tracklets.csv", outputs.tracklets, TRACKLET_FIELDS)
    write_csv(output_dir / "tracklet_link_candidates.csv", outputs.tracklet_link_candidates, LINK_FIELDS)
    write_csv(output_dir / "tracklet_links_accepted.csv", outputs.tracklet_links_accepted, LINK_FIELDS)
    write_csv(output_dir / "ambiguous_link_review.csv", outputs.ambiguous_link_review, LINK_FIELDS)
    write_csv(output_dir / "consistency_validation_report.csv", outputs.consistency_validation_report, VALIDATION_FIELDS)
    write_csv(output_dir / "target_validity_report.csv", outputs.target_validity_report, TARGET_VALIDITY_FIELDS)
    write_csv(output_dir / "identity_purity_report.csv", outputs.identity_purity_report, IDENTITY_PURITY_FIELDS)
    write_csv(output_dir / "final_target_gate.csv", outputs.final_target_gate, FINAL_GATE_FIELDS)
    write_csv(output_dir / "raw_track_split_review.csv", outputs.raw_track_split_review, RAW_SPLIT_REVIEW_FIELDS)
    write_csv(output_dir / "risky_accepted_link_review.csv", outputs.risky_accepted_link_review, RISKY_LINK_REVIEW_FIELDS)
    write_csv(output_dir / "fragment_path_absorption_review.csv", outputs.fragment_path_absorption_review, FRAGMENT_PATH_ABSORPTION_FIELDS)
    write_csv(output_dir / "target_quality_report.csv", outputs.target_quality_report, TARGET_QUALITY_FIELDS)
    write_csv(output_dir / "cross_raw_recovery_review.csv", outputs.cross_raw_recovery_review, CROSS_RAW_RECOVERY_FIELDS)
