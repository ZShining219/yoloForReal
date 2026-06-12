#!/usr/bin/env python3
"""Apply the human false-positive review gate to MOT target-track summaries.

This script stops at the target-track layer. It removes reviewed non-focus
tracks and never adds vehicles, lanes, movements, SUMO routes, or events.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


FINAL_FIELDS = [
    "track_id",
    "raw_track_id",
    "class_name",
    "class_votes",
    "start_time",
    "end_time",
    "duration_sec",
    "frame_count",
    "detection_count",
    "mean_confidence",
    "max_displacement_px",
    "track_status",
    "exclude_reason",
    "target_final_status",
    "false_positive_gate_status",
    "source_evidence",
    "manual_annotation_added",
]

AFTER_FP_FIELDS = [
    "track_id",
    "raw_track_id",
    "class_name",
    "class_votes",
    "start_time",
    "end_time",
    "duration_sec",
    "frame_count",
    "detection_count",
    "mean_confidence",
    "max_displacement_px",
    "track_status",
    "exclude_reason",
    "precision_filter_status",
    "precision_filter_note",
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


def review_status_by_track(false_positive_gate_rows: list[dict]) -> dict[str, str]:
    statuses = {}
    for row in false_positive_gate_rows:
        track_id = row["track_id"]
        status = row.get("review_status", "").strip().upper()
        if not status:
            raise ValueError(f"Missing review_status for {track_id}")
        if status not in {"KEEP", "EXCLUDE"}:
            raise ValueError(f"Unresolved false-positive review status for {track_id}: {status}")
        statuses[track_id] = status
    return statuses


def build_after_fp_filter_rows(target_summary_rows: list[dict], false_positive_gate_rows: list[dict]) -> list[dict]:
    gate = review_status_by_track(false_positive_gate_rows)
    rows = []
    for summary in target_summary_rows:
        if summary.get("track_status") != "target_review_candidate":
            continue
        if gate.get(summary["track_id"]) == "EXCLUDE":
            continue
        row = dict(summary)
        row["precision_filter_status"] = "KEPT_AFTER_FALSE_POSITIVE_GATE"
        row["precision_filter_note"] = "Not excluded by current false-positive review gate."
        rows.append(row)
    return rows


def false_positive_status(track_id: str, gate: dict[str, str]) -> str:
    status = gate.get(track_id)
    if status == "KEEP":
        return "KEEP"
    if status == "EXCLUDE":
        return "EXCLUDE_BY_USER_GATE"
    return "NOT_FLAGGED_FOR_FALSE_POSITIVE_REVIEW"


def build_final_target_rows(target_summary_rows: list[dict], false_positive_gate_rows: list[dict]) -> list[dict]:
    gate = review_status_by_track(false_positive_gate_rows)
    rows = []
    for summary in target_summary_rows:
        if summary.get("track_status") != "target_review_candidate":
            continue
        if gate.get(summary["track_id"]) == "EXCLUDE":
            continue
        row = dict(summary)
        row["target_final_status"] = "FINAL_TARGET_TRACK"
        row["false_positive_gate_status"] = false_positive_status(summary["track_id"], gate)
        row["source_evidence"] = "yolo11m_bytetrack_target_extraction"
        row["manual_annotation_added"] = "no"
        rows.append(row)
    return rows


def build_overall_review_gate(final_rows: list[dict], false_positive_gate_rows: list[dict]) -> list[dict]:
    counts = Counter(row.get("review_status", "").strip().upper() for row in false_positive_gate_rows)
    return [
        {
            "case_id": "mvi0866",
            "review_layer": "target_extraction",
            "review_status": "TARGET_EXTRACTION_REVIEW_COMPLETE",
            "recall_status": "COARSE_PASS_BY_USER",
            "precision_status": f"FALSE_POSITIVE_GATE_COMPLETE_EXCLUDE_{counts.get('EXCLUDE', 0)}_KEEP_{counts.get('KEEP', 0)}",
            "next_action": "use target_tracks_final.csv for anchor/event reconstruction; do not use old 0.5s IOU candidate events",
            "note": (
                "Motor-vehicle target extraction coarse recall passed. User excluded "
                f"{counts.get('EXCLUDE', 0)} non-focus target tracks and kept remaining false-positive review candidates. "
                "No vehicles were manually added."
            ),
        }
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-summary", required=True)
    parser.add_argument("--false-positive-gate", required=True)
    parser.add_argument("--output-after-fp-filter", required=True)
    parser.add_argument("--output-final", required=True)
    parser.add_argument("--output-overall-gate", required=True)
    args = parser.parse_args()

    summaries = read_csv(Path(args.target_summary))
    gate_rows = read_csv(Path(args.false_positive_gate))
    after_fp_rows = build_after_fp_filter_rows(summaries, gate_rows)
    final_rows = build_final_target_rows(summaries, gate_rows)
    write_csv(Path(args.output_after_fp_filter), after_fp_rows, AFTER_FP_FIELDS)
    write_csv(Path(args.output_final), final_rows, FINAL_FIELDS)
    write_csv(
        Path(args.output_overall_gate),
        build_overall_review_gate(final_rows, gate_rows),
        ["case_id", "review_layer", "review_status", "recall_status", "precision_status", "next_action", "note"],
    )
    print(f"target_tracks_after_fp_filter={len(after_fp_rows)}")
    print(f"target_tracks_final={len(final_rows)}")


if __name__ == "__main__":
    main()
