#!/usr/bin/env python3
"""Suppress duplicate-overlap logical tracks before direction OD use."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


DUPLICATE_FIELDS = [
    "logical_vehicle_id",
    "duplicate_review_status",
    "overlap_partner_count",
    "total_duplicate_overlap_frames",
    "partners",
    "review_note",
]

PAIR_FIELDS = [
    "logical_vehicle_id_a",
    "logical_vehicle_id_b",
    "overlap_frames",
    "start_frame",
    "end_frame",
    "mean_iou",
    "max_iou",
    "raw_track_ids_a",
    "raw_track_ids_b",
]


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


def bbox(row: dict) -> tuple[float, float, float, float]:
    return tuple(float(row[key]) for key in ("x1", "y1", "x2", "y2"))


def area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def find_duplicate_overlap_pairs(
    logical_track_rows: list[dict],
    iou_threshold: float = 0.85,
    min_overlap_frames: int = 10,
) -> list[dict]:
    by_frame: dict[int, list[dict]] = defaultdict(list)
    for row in logical_track_rows:
        by_frame[int(float(row["frame_id"]))].append(row)

    pair_values: dict[tuple[str, str], list[tuple[int, float, dict, dict]]] = defaultdict(list)
    for frame_id, rows in by_frame.items():
        for index, row_a in enumerate(rows):
            for row_b in rows[index + 1:]:
                id_a = row_a["logical_vehicle_id"]
                id_b = row_b["logical_vehicle_id"]
                if id_a == id_b:
                    continue
                value = bbox_iou(bbox(row_a), bbox(row_b))
                if value < iou_threshold:
                    continue
                key = tuple(sorted((id_a, id_b)))
                pair_values[key].append((frame_id, value, row_a, row_b))

    pair_rows = []
    for (id_a, id_b), values in sorted(pair_values.items()):
        if len(values) < min_overlap_frames:
            continue
        frames = [item[0] for item in values]
        raw_ids_a = sorted({row_a["track_id"] if row_a["logical_vehicle_id"] == id_a else row_b["track_id"] for _, _, row_a, row_b in values})
        raw_ids_b = sorted({row_b["track_id"] if row_b["logical_vehicle_id"] == id_b else row_a["track_id"] for _, _, row_a, row_b in values})
        ious = [item[1] for item in values]
        pair_rows.append(
            {
                "logical_vehicle_id_a": id_a,
                "logical_vehicle_id_b": id_b,
                "overlap_frames": str(len(values)),
                "start_frame": str(min(frames)),
                "end_frame": str(max(frames)),
                "mean_iou": f"{sum(ious) / len(ious):.3f}",
                "max_iou": f"{max(ious):.3f}",
                "raw_track_ids_a": "|".join(raw_ids_a),
                "raw_track_ids_b": "|".join(raw_ids_b),
            }
        )
    return pair_rows


def duplicate_decision(logical_id: str, affected_partner_counts: dict[str, int], direction_ready: dict[str, str]) -> str:
    partner_count = affected_partner_counts.get(logical_id, 0)
    if partner_count == 0:
        return "NO_DUPLICATE_OVERLAP"
    if direction_ready.get(logical_id) == "yes" and partner_count >= 2:
        return "CONTAMINATED_DUPLICATE"
    return "DUPLICATE_SUPPRESSED"


def build_duplicate_review_rows(pair_rows: list[dict], logical_targets: list[dict]) -> list[dict]:
    partners: dict[str, set[str]] = defaultdict(set)
    overlap_counts: Counter[str] = Counter()
    for row in pair_rows:
        id_a = row["logical_vehicle_id_a"]
        id_b = row["logical_vehicle_id_b"]
        frames = int(row["overlap_frames"])
        partners[id_a].add(id_b)
        partners[id_b].add(id_a)
        overlap_counts[id_a] += frames
        overlap_counts[id_b] += frames

    direction_ready = {row["logical_vehicle_id"]: row.get("keep_for_direction_od", "no") for row in logical_targets}
    rows = []
    for row in logical_targets:
        logical_id = row["logical_vehicle_id"]
        status = duplicate_decision(
            logical_id,
            {key: len(value) for key, value in partners.items()},
            direction_ready,
        )
        if status == "NO_DUPLICATE_OVERLAP":
            continue
        note = "Long direction-ready track overlaps multiple logical vehicles; remove from OD." if status == "CONTAMINATED_DUPLICATE" else "Persistent same-frame duplicate; suppress unless manually selected."
        rows.append(
            {
                "logical_vehicle_id": logical_id,
                "duplicate_review_status": status,
                "overlap_partner_count": str(len(partners[logical_id])),
                "total_duplicate_overlap_frames": str(overlap_counts[logical_id]),
                "partners": "|".join(sorted(partners[logical_id])),
                "review_note": note,
            }
        )
    return rows


def suppress_direction_for_duplicates(direction_rows: list[dict], duplicate_rows: list[dict]) -> list[dict]:
    duplicate_status = {
        row["logical_vehicle_id"]: row["duplicate_review_status"]
        for row in duplicate_rows
        if row["duplicate_review_status"] in {"CONTAMINATED_DUPLICATE", "DUPLICATE_SUPPRESSED"}
    }
    output = []
    for row in direction_rows:
        copy = dict(row)
        status = duplicate_status.get(row["track_id"])
        if status:
            copy["origin_direction"] = "unknown"
            copy["destination_direction"] = "unknown"
            copy["result_direction"] = "unknown"
            copy["confidence_level"] = "low"
            copy["review_status"] = "DUPLICATE_REVIEW"
            copy["evidence_note"] = f"Suppressed by duplicate-overlap review: {status}."
        output.append(copy)
    return output


def suppress_targets_for_duplicates(target_rows: list[dict], duplicate_rows: list[dict]) -> list[dict]:
    duplicate_status = {
        row["logical_vehicle_id"]: row["duplicate_review_status"]
        for row in duplicate_rows
    }
    output = []
    for row in target_rows:
        copy = dict(row)
        status = duplicate_status.get(row["logical_vehicle_id"])
        if status:
            copy["keep_for_direction_od"] = "no"
            copy["review_note"] = f"Suppressed by duplicate-overlap review: {status}."
        output.append(copy)
    return output


def duplicate_components(duplicate_rows: list[dict]) -> list[set[str]]:
    graph: dict[str, set[str]] = defaultdict(set)
    for row in duplicate_rows:
        logical_id = row["logical_vehicle_id"]
        graph[logical_id]
        for partner in [item for item in row.get("partners", "").split("|") if item]:
            graph[logical_id].add(partner)
            graph[partner].add(logical_id)

    components = []
    seen = set()
    for logical_id in sorted(graph):
        if logical_id in seen:
            continue
        stack = [logical_id]
        component = set()
        seen.add(logical_id)
        while stack:
            current = stack.pop()
            component.add(current)
            for partner in graph[current]:
                if partner not in seen:
                    seen.add(partner)
                    stack.append(partner)
        components.append(component)
    return components


def representative_ids_for_video_review(duplicate_rows: list[dict], target_rows: list[dict]) -> set[str]:
    status_by_id = {row["logical_vehicle_id"]: row["duplicate_review_status"] for row in duplicate_rows}
    frames_by_id = {
        row["logical_vehicle_id"]: int(float(row.get("detected_frame_count") or 0))
        for row in target_rows
    }
    representatives = set()
    for component in duplicate_components(duplicate_rows):
        candidates = sorted(component)
        representatives.add(
            max(
                candidates,
                key=lambda logical_id: (
                    status_by_id.get(logical_id) == "CONTAMINATED_DUPLICATE",
                    frames_by_id.get(logical_id, 0),
                    logical_id,
                ),
            )
        )
    return representatives


def suppress_track_rows_for_duplicates(
    track_rows: list[dict],
    duplicate_rows: list[dict],
    target_rows: list[dict],
) -> list[dict]:
    representative_ids = representative_ids_for_video_review(duplicate_rows, target_rows)
    duplicate_ids = {
        row["logical_vehicle_id"]
        for row in duplicate_rows
        if row["duplicate_review_status"] in {"CONTAMINATED_DUPLICATE", "DUPLICATE_SUPPRESSED"}
    }
    suppressed_ids = duplicate_ids - representative_ids
    return [row for row in track_rows if row["logical_vehicle_id"] not in suppressed_ids]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logical-tracks", required=True)
    parser.add_argument("--logical-targets", required=True)
    parser.add_argument("--logical-od", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.85)
    parser.add_argument("--min-overlap-frames", type=int, default=10)
    args = parser.parse_args()

    track_rows = read_csv(Path(args.logical_tracks))
    target_rows = read_csv(Path(args.logical_targets))
    direction_rows = read_csv(Path(args.logical_od))
    pair_rows = find_duplicate_overlap_pairs(track_rows, args.iou_threshold, args.min_overlap_frames)
    duplicate_rows = build_duplicate_review_rows(pair_rows, target_rows)

    output_dir = Path(args.output_dir)
    write_csv(output_dir / "duplicate_overlap_pairs.csv", pair_rows, PAIR_FIELDS)
    write_csv(output_dir / "duplicate_overlap_review.csv", duplicate_rows, DUPLICATE_FIELDS)
    write_csv(output_dir / "logical_vehicle_targets_deduped.csv", suppress_targets_for_duplicates(target_rows, duplicate_rows), list(target_rows[0].keys()))
    write_csv(output_dir / "logical_direction_od_deduped.csv", suppress_direction_for_duplicates(direction_rows, duplicate_rows), list(direction_rows[0].keys()))
    write_csv(output_dir / "logical_vehicle_tracks_deduped.csv", suppress_track_rows_for_duplicates(track_rows, duplicate_rows, target_rows), list(track_rows[0].keys()))
    print(f"duplicate_pairs={len(pair_rows)}")
    print(f"duplicate_review_rows={len(duplicate_rows)}")
    print(f"contaminated={sum(row['duplicate_review_status'] == 'CONTAMINATED_DUPLICATE' for row in duplicate_rows)}")
    print(f"suppressed={sum(row['duplicate_review_status'] == 'DUPLICATE_SUPPRESSED' for row in duplicate_rows)}")


if __name__ == "__main__":
    main()
