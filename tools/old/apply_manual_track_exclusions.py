#!/usr/bin/env python3
"""Create a versioned target-track table after manual track-id exclusions."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


BASE_FIELDS = [
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
]

FINAL_FIELDS = BASE_FIELDS + [
    "target_final_status",
    "manual_filter_status",
    "source_evidence",
    "manual_annotation_added",
]

FILTERED_FIELDS = BASE_FIELDS + [
    "manual_filter_status",
    "manual_filter_note",
]

GATE_FIELDS = [
    "version",
    "track_id",
    "input_id",
    "track_status",
    "manual_filter_action",
    "class_name",
    "start_time",
    "end_time",
    "duration_sec",
    "mean_confidence",
    "max_displacement_px",
    "note",
]

SUMMARY_FIELDS = [
    "version",
    "input_exclusion_ids",
    "normalized_exclusion_ids",
    "target_review_candidates_before_filter",
    "candidate_excluded_by_manual_filter",
    "already_auto_excluded_ids",
    "missing_ids",
    "final_target_tracks",
    "note",
]


@dataclass
class ManualFilterOutputs:
    final_rows: list[dict]
    filtered_rows: list[dict]
    gate_rows: list[dict]
    summary: dict


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


def normalize_exclusion_ids(ids: list[str]) -> list[str]:
    normalized = []
    for raw in ids:
        value = raw.strip()
        if not value:
            continue
        if value.startswith("mot_"):
            normalized.append(value)
        else:
            normalized.append(f"mot_{int(float(value)):04d}")
    return normalized


def manual_status(version: str) -> str:
    suffix = version.upper()
    if suffix.startswith("MANUAL_FILTER_"):
        suffix = suffix.replace("MANUAL_FILTER_", "")
    return f"KEPT_AFTER_MANUAL_FILTER_{suffix}"


def build_gate_row(version: str, input_id: str, track_id: str, row: dict | None, action: str, note: str) -> dict:
    return {
        "version": version,
        "input_id": input_id,
        "track_id": track_id,
        "track_status": row.get("track_status", "") if row else "",
        "manual_filter_action": action,
        "class_name": row.get("class_name", "") if row else "",
        "start_time": row.get("start_time", "") if row else "",
        "end_time": row.get("end_time", "") if row else "",
        "duration_sec": row.get("duration_sec", "") if row else "",
        "mean_confidence": row.get("mean_confidence", "") if row else "",
        "max_displacement_px": row.get("max_displacement_px", "") if row else "",
        "note": note,
    }


def build_manual_filter_outputs(
    target_summary_rows: list[dict],
    exclusion_ids: list[str],
    version: str,
    note: str,
) -> ManualFilterOutputs:
    by_id = {row["track_id"]: row for row in target_summary_rows}
    normalized_ids = normalize_exclusion_ids(exclusion_ids)
    exclusion_set = set(normalized_ids)
    keep_status = manual_status(version)

    gate_rows = []
    candidate_excluded = []
    already_auto_excluded = []
    missing = []
    for input_id, track_id in zip([item.strip() for item in exclusion_ids if item.strip()], normalized_ids):
        row = by_id.get(track_id)
        if row is None:
            missing.append(track_id)
            gate_rows.append(build_gate_row(version, input_id, track_id, None, "ID_NOT_FOUND", note))
        elif row.get("track_status") == "target_review_candidate":
            candidate_excluded.append(track_id)
            gate_rows.append(build_gate_row(version, input_id, track_id, row, "EXCLUDE_FROM_FINAL", note))
        else:
            already_auto_excluded.append(track_id)
            gate_rows.append(build_gate_row(version, input_id, track_id, row, "ALREADY_AUTO_EXCLUDED", note))

    filtered_rows = []
    final_rows = []
    for row in target_summary_rows:
        if row.get("track_status") != "target_review_candidate":
            continue
        if row["track_id"] in exclusion_set:
            continue
        filtered = dict(row)
        filtered["manual_filter_status"] = keep_status
        filtered["manual_filter_note"] = f"Kept after {version}; not listed in manual exclusion ids."
        filtered_rows.append(filtered)

        final = dict(row)
        final["target_final_status"] = "FINAL_TARGET_TRACK"
        final["manual_filter_status"] = keep_status
        final["source_evidence"] = "yolo26x_bytetrack_manual_filter"
        final["manual_annotation_added"] = "no"
        final_rows.append(final)

    summary = {
        "version": version,
        "input_exclusion_ids": ",".join([item.strip() for item in exclusion_ids if item.strip()]),
        "normalized_exclusion_ids": ",".join(normalized_ids),
        "target_review_candidates_before_filter": str(sum(row.get("track_status") == "target_review_candidate" for row in target_summary_rows)),
        "candidate_excluded_by_manual_filter": str(len(set(candidate_excluded))),
        "already_auto_excluded_ids": ",".join(already_auto_excluded),
        "missing_ids": ",".join(missing),
        "final_target_tracks": str(len(final_rows)),
        "note": note,
    }
    return ManualFilterOutputs(final_rows=final_rows, filtered_rows=filtered_rows, gate_rows=gate_rows, summary=summary)


def write_version_note(path: Path, outputs: ManualFilterOutputs) -> None:
    summary = outputs.summary
    lines = [
        f"# {summary['version']} Manual Track Exclusion",
        "",
        f"- input_exclusion_ids: `{summary['input_exclusion_ids']}`",
        f"- normalized_exclusion_ids: `{summary['normalized_exclusion_ids']}`",
        f"- target_review_candidates_before_filter: `{summary['target_review_candidates_before_filter']}`",
        f"- candidate_excluded_by_manual_filter: `{summary['candidate_excluded_by_manual_filter']}`",
        f"- already_auto_excluded_ids: `{summary['already_auto_excluded_ids']}`",
        f"- missing_ids: `{summary['missing_ids']}`",
        f"- final_target_tracks: `{summary['final_target_tracks']}`",
        "",
        "This version only removes manually listed track ids from the yolo26x target-review candidates.",
        "It does not add vehicles, merge fragments, infer lanes, or create SUMO events.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-summary", required=True)
    parser.add_argument("--exclude-ids", nargs="+", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--note", default="")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    outputs = build_manual_filter_outputs(
        target_summary_rows=read_csv(Path(args.target_summary)),
        exclusion_ids=args.exclude_ids,
        version=args.version,
        note=args.note,
    )
    output_dir = Path(args.output_dir)
    write_csv(output_dir / "manual_exclusion_gate.csv", outputs.gate_rows, GATE_FIELDS)
    write_csv(output_dir / "target_tracks_after_manual_filter.csv", outputs.filtered_rows, FILTERED_FIELDS)
    write_csv(output_dir / "target_tracks_final.csv", outputs.final_rows, FINAL_FIELDS)
    write_csv(output_dir / "manual_filter_summary.csv", [outputs.summary], SUMMARY_FIELDS)
    write_version_note(output_dir / "VERSION_NOTE.md", outputs)
    print(f"version={args.version}")
    print(f"candidate_excluded_by_manual_filter={outputs.summary['candidate_excluded_by_manual_filter']}")
    print(f"already_auto_excluded_ids={outputs.summary['already_auto_excluded_ids']}")
    print(f"missing_ids={outputs.summary['missing_ids']}")
    print(f"final_target_tracks={outputs.summary['final_target_tracks']}")


if __name__ == "__main__":
    main()
