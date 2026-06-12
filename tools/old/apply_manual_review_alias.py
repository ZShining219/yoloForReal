#!/usr/bin/env python3
"""Apply manual logical-ID aliases for a bounded review-video version."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


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


def alias_map(alias_rows: list[dict]) -> dict[str, str]:
    return {
        row["alias_logical_vehicle_id"]: row["canonical_logical_vehicle_id"]
        for row in alias_rows
    }


def apply_manual_aliases_to_tracks(
    track_rows: list[dict],
    alias_rows: list[dict],
    frame_start: int,
    frame_end: int,
) -> list[dict]:
    aliases = alias_map(alias_rows)
    output = []
    for row in track_rows:
        frame_id = int(float(row["frame_id"]))
        if frame_id < frame_start or frame_id > frame_end:
            continue
        if row["logical_vehicle_id"] in aliases:
            continue
        output.append(dict(row))
    return output


def apply_manual_aliases_to_targets(target_rows: list[dict], alias_rows: list[dict]) -> list[dict]:
    aliases = alias_map(alias_rows)
    output = []
    for row in target_rows:
        copy = dict(row)
        canonical = aliases.get(row["logical_vehicle_id"])
        if canonical:
            copy["keep_for_direction_od"] = "no"
            copy["review_note"] = f"Manual review alias to {canonical}; suppress duplicate logical ID in bounded review video."
        output.append(copy)
    return output


def apply_manual_aliases_to_od(od_rows: list[dict], alias_rows: list[dict]) -> list[dict]:
    aliases = alias_map(alias_rows)
    output = []
    for row in od_rows:
        copy = dict(row)
        canonical = aliases.get(row["track_id"])
        if canonical:
            copy["origin_direction"] = "unknown"
            copy["destination_direction"] = "unknown"
            copy["result_direction"] = "unknown"
            copy["confidence_level"] = "low"
            copy["review_status"] = "MANUAL_ALIAS_SUPPRESSED"
            copy["evidence_note"] = f"Manual review alias to {canonical}; duplicate logical ID hidden from bounded review video."
        output.append(copy)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logical-tracks", required=True)
    parser.add_argument("--logical-targets", required=True)
    parser.add_argument("--logical-od", required=True)
    parser.add_argument("--manual-aliases", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-end", type=int, required=True)
    args = parser.parse_args()

    track_rows = read_csv(Path(args.logical_tracks))
    target_rows = read_csv(Path(args.logical_targets))
    od_rows = read_csv(Path(args.logical_od))
    aliases = read_csv(Path(args.manual_aliases))
    output_dir = Path(args.output_dir)

    filtered_tracks = apply_manual_aliases_to_tracks(track_rows, aliases, args.frame_start, args.frame_end)
    updated_targets = apply_manual_aliases_to_targets(target_rows, aliases)
    updated_od = apply_manual_aliases_to_od(od_rows, aliases)

    write_csv(output_dir / "logical_vehicle_tracks_manual_review.csv", filtered_tracks, list(track_rows[0].keys()))
    write_csv(output_dir / "logical_vehicle_targets_manual_review.csv", updated_targets, list(target_rows[0].keys()))
    write_csv(output_dir / "logical_direction_od_manual_review.csv", updated_od, list(od_rows[0].keys()))
    print(f"track_rows={len(filtered_tracks)}")
    print(f"target_rows={len(updated_targets)}")
    print(f"od_rows={len(updated_od)}")
    print(f"manual_aliases={len(aliases)}")


if __name__ == "__main__":
    main()
