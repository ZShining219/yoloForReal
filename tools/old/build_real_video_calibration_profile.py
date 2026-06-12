#!/usr/bin/env python3
"""Build a real-video traffic calibration profile for SUMO-RL demand generation."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
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


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def usable_rows(rows: list[dict]) -> list[dict]:
    return [
        row
        for row in rows
        if row.get("readiness_status") not in {"", "EXCLUDED", "MISSING_LANE_ASSIGNMENT"}
        and row.get("route_edges")
    ]


def share_map(counter: Counter, total: int, extra_by_key: dict[str, dict] | None = None) -> dict:
    result = {}
    extra_by_key = extra_by_key or {}
    for key in sorted(counter):
        result[key] = {
            **extra_by_key.get(key, {}),
            "count": counter[key],
            "share": round(counter[key] / total, 4) if total else 0.0,
        }
    return result


def depart_time(row: dict) -> float:
    for key in ("depart_time_sec", "video_depart_time_sec", "video_gate_entry_time_sec"):
        value = row.get(key, "")
        if value != "":
            return float(value)
    return 0.0


def build_flow_by_approach_rows(rows: list[dict], source_window_sec: float) -> list[dict]:
    source_rows = usable_rows(rows)
    counts = Counter(row["from_edge"] for row in source_rows)
    output = []
    for edge in sorted(counts):
        count = counts[edge]
        output.append(
            {
                "from_edge": edge,
                "vehicle_count": str(count),
                "vehicles_per_hour": f"{count / source_window_sec * 3600.0:.2f}",
                "share": f"{count / len(source_rows):.4f}" if source_rows else "0.0000",
            }
        )
    return output


def build_turning_ratio_rows(rows: list[dict]) -> list[dict]:
    source_rows = usable_rows(rows)
    counts = Counter(row["result_direction"] for row in source_rows if row.get("result_direction"))
    total = sum(counts.values())
    return [
        {
            "result_direction": direction,
            "vehicle_count": str(counts[direction]),
            "share": f"{counts[direction] / total:.4f}" if total else "0.0000",
        }
        for direction in sorted(counts)
    ]


def build_vehicle_type_ratio_rows(rows: list[dict]) -> list[dict]:
    source_rows = usable_rows(rows)
    counts = Counter(row.get("vtype", "passenger") or "passenger" for row in source_rows)
    total = sum(counts.values())
    return [
        {"vtype": vtype, "vehicle_count": str(counts[vtype]), "share": f"{counts[vtype] / total:.4f}" if total else "0.0000"}
        for vtype in sorted(counts)
    ]


def build_initial_queue_profile_rows(rows: list[dict]) -> list[dict]:
    source_rows = [
        row
        for row in usable_rows(rows)
        if row.get("initial_state_type") == "red_light_waiting_queue"
    ]
    counts = Counter((row.get("from_edge", ""), row.get("depart_lane", "")) for row in source_rows)
    return [
        {
            "from_edge": edge,
            "depart_lane": lane,
            "queue_vehicle_count": str(count),
        }
        for (edge, lane), count in sorted(counts.items())
    ]


def build_warmup_vehicle_profile_rows(rows: list[dict]) -> list[dict]:
    source_rows = [
        row
        for row in usable_rows(rows)
        if row.get("initial_state_type") in {"manual_warmup_anchor", "window_start_partial_exit"}
    ]
    counts = Counter(row.get("route_edges", "") for row in source_rows)
    return [
        {
            "route_edges": route,
            "warmup_vehicle_count": str(counts[route]),
        }
        for route in sorted(counts)
    ]


def build_arrival_profile_rows(rows: list[dict], bin_size_sec: float, source_window_sec: float) -> list[dict]:
    source_rows = usable_rows(rows)
    bin_count = max(1, int(source_window_sec // bin_size_sec))
    counts = [0 for _ in range(bin_count)]
    for row in source_rows:
        time_sec = max(0.0, depart_time(row))
        index = min(bin_count - 1, int(time_sec // bin_size_sec))
        counts[index] += 1
    return [
        {
            "bin_start_sec": f"{index * bin_size_sec:.2f}",
            "bin_end_sec": f"{(index + 1) * bin_size_sec:.2f}",
            "vehicle_count": str(count),
            "vehicles_per_hour": f"{count / bin_size_sec * 3600.0:.2f}",
        }
        for index, count in enumerate(counts)
    ]


def build_route_distribution(rows: list[dict]) -> dict:
    source_rows = usable_rows(rows)
    counts = Counter(row["route_edges"] for row in source_rows)
    extra = {}
    for row in source_rows:
        extra[row["route_edges"]] = {
            "from_edge": row.get("from_edge", ""),
            "to_edge": row.get("to_edge", ""),
        }
    return share_map(counts, len(source_rows), extra)


def build_calibration_profile(rows: list[dict], source_window_sec: float, bin_size_sec: float) -> dict:
    source_rows = usable_rows(rows)
    status_counts = Counter(row.get("readiness_status", "") for row in source_rows)
    initial_counts = Counter(row.get("initial_state_type", "") for row in source_rows)
    vtype_counts = Counter(row.get("vtype", "passenger") or "passenger" for row in source_rows)
    return {
        "schema_version": "real_video_calibration_profile_v1",
        "source_window_sec": source_window_sec,
        "bin_size_sec": bin_size_sec,
        "observed_vehicle_count": len(source_rows),
        "source_note": "Current profile is derived from the reviewed 0-30s video evidence window. It calibrates demand statistics and does not claim full-video trajectory reconstruction.",
        "training_policy": {
            "uses_replay_control": False,
            "uses_move_to_xy": False,
            "uses_spatial_init": False,
            "demand_generation": "statistical_route_departure_generation",
        },
        "evidence_status_counts": dict(sorted(status_counts.items())),
        "initial_state_counts": dict(sorted(initial_counts.items())),
        "route_distribution": build_route_distribution(rows),
        "vehicle_type_distribution": share_map(vtype_counts, len(source_rows)),
        "flow_by_approach": build_flow_by_approach_rows(rows, source_window_sec),
        "turning_ratio": build_turning_ratio_rows(rows),
        "arrival_profile": build_arrival_profile_rows(rows, bin_size_sec, source_window_sec),
        "initial_queue_profile": build_initial_queue_profile_rows(rows),
        "warmup_vehicle_profile": build_warmup_vehicle_profile_rows(rows),
    }


def write_summary(path: Path, profile: dict) -> None:
    path.write_text(
        "\n".join(
            [
                "# Real Video Calibration Profile v1",
                "",
                f"- source_window_sec: `{profile['source_window_sec']}`",
                f"- observed_vehicle_count: `{profile['observed_vehicle_count']}`",
                f"- evidence_status_counts: `{profile['evidence_status_counts']}`",
                f"- initial_state_counts: `{profile['initial_state_counts']}`",
                "",
                "This profile is for SUMO-RL demand calibration. It must not be used as a per-vehicle replay policy.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def build_package(input_csv: Path, output_dir: Path, source_window_sec: float, bin_size_sec: float) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    rows = read_csv(input_csv)
    profile = build_calibration_profile(rows, source_window_sec, bin_size_sec)
    data_dir = output_dir / "data"
    audit_dir = output_dir / "audit"
    output_dir.mkdir(parents=True)
    data_dir.mkdir()
    audit_dir.mkdir()
    write_json(data_dir / "calibration_profile.json", profile)
    write_csv(data_dir / "flow_by_approach.csv", build_flow_by_approach_rows(rows, source_window_sec), ["from_edge", "vehicle_count", "vehicles_per_hour", "share"])
    write_csv(data_dir / "turning_ratio.csv", build_turning_ratio_rows(rows), ["result_direction", "vehicle_count", "share"])
    write_csv(data_dir / "vehicle_type_ratio.csv", build_vehicle_type_ratio_rows(rows), ["vtype", "vehicle_count", "share"])
    write_csv(data_dir / "arrival_time_profile.csv", build_arrival_profile_rows(rows, bin_size_sec, source_window_sec), ["bin_start_sec", "bin_end_sec", "vehicle_count", "vehicles_per_hour"])
    write_csv(data_dir / "initial_queue_profile.csv", build_initial_queue_profile_rows(rows), ["from_edge", "depart_lane", "queue_vehicle_count"])
    write_csv(data_dir / "warmup_vehicle_profile.csv", build_warmup_vehicle_profile_rows(rows), ["route_edges", "warmup_vehicle_count"])
    write_summary(audit_dir / "calibration_summary.md", profile)
    return profile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-window-sec", type=float, default=30.0)
    parser.add_argument("--bin-size-sec", type=float, default=5.0)
    args = parser.parse_args()
    profile = build_package(Path(args.input), Path(args.output_dir), args.source_window_sec, args.bin_size_sec)
    print(f"observed_vehicle_count={profile['observed_vehicle_count']}")
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
