#!/usr/bin/env python3
"""Build first-pass SUMO readiness and route candidate artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from xml.dom import minidom


READINESS_FIELDS = [
    "track_id",
    "readiness_status",
    "initial_state_type",
    "result_direction",
    "in_lane_code",
    "out_lane_code",
    "from_edge",
    "to_edge",
    "route_edges",
    "depart_lane",
    "arrival_lane",
    "depart_time_sec",
    "depart_time_source",
    "entry_time_sec",
    "exit_time_sec",
    "anchor_source",
    "class_name",
    "vtype",
    "first_seen_time_sec",
    "last_seen_time_sec",
    "exclude_reason",
    "review_note",
]

SEED_FIELDS = [
    "sumo_vehicle_id",
    "track_id",
    "vtype",
    "route_edges",
    "from_edge",
    "to_edge",
    "depart_time_sec",
    "depart_lane",
    "arrival_lane",
    "initial_state_type",
    "readiness_status",
    "review_note",
]

VTYPE_BY_CLASS = {
    "car": "passenger",
    "truck": "truck",
    "bus": "bus",
}

VTYPE_DEFS = {
    "passenger": {"vClass": "passenger", "length": "4.5", "maxSpeed": "13.9", "accel": "2.6", "decel": "4.5"},
    "truck": {"vClass": "truck", "length": "8.0", "maxSpeed": "11.0", "accel": "1.3", "decel": "4.0"},
    "bus": {"vClass": "bus", "length": "12.0", "maxSpeed": "10.0", "accel": "1.2", "decel": "4.0"},
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


def lane_edge(lane_code: str) -> str:
    return lane_code.rsplit("_", 1)[0]


def lane_index(lane_code: str) -> str:
    return lane_code.rsplit("_", 1)[1]


def rows_by_track(track_rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in track_rows:
        grouped[row["logical_vehicle_id"]].append(row)
    return {track_id: sorted(rows, key=lambda row: int(float(row["frame_id"]))) for track_id, rows in grouped.items()}


def dominant_class(rows: list[dict]) -> str:
    counts = Counter(row.get("class_name", "car") for row in rows)
    return counts.most_common(1)[0][0] if counts else "car"


def track_time_range(rows: list[dict]) -> tuple[str, str]:
    if not rows:
        return "", ""
    frames = [int(float(row["frame_id"])) for row in rows]
    return f"{min(frames) / 50.0:.2f}", f"{max(frames) / 50.0:.2f}"


def build_index(rows: list[dict], key: str = "track_id") -> dict[str, dict]:
    return {row[key]: row for row in rows}


def classify_anchor(anchor: dict) -> tuple[str, str, str, str]:
    route_source = anchor.get("route_source", "")
    entry_method = anchor.get("entry_estimation_method", "")
    entry_time = anchor.get("estimated_entry_time_sec", "")
    if route_source == "manual_review_supplement":
        return "READY_REVIEW_SPECIAL", "left_turn_waiting_area_review", entry_time, "manual_review_supplement"
    if route_source == "observed_complete_od" or entry_method == "observed_gate_crossing":
        return "READY_STRONG", "normal_crossing_anchor", entry_time, "observed_gate_crossing"
    if entry_time and float(entry_time) < 0:
        return "READY_WARMUP", "manual_warmup_anchor", entry_time, entry_method
    return "READY_ANCHORED_REVIEW", "anchored_review", entry_time, entry_method


def build_readiness_rows(
    track_rows: list[dict],
    assignment_rows: list[dict],
    exclusion_rows: list[dict],
    anchor_rows: list[dict],
    override_rows: list[dict],
) -> list[dict]:
    tracks_by_id = rows_by_track(track_rows)
    assignments = build_index(assignment_rows)
    exclusions = build_index(exclusion_rows)
    anchors = build_index(anchor_rows)
    overrides = build_index(override_rows)
    all_track_ids = sorted(set(tracks_by_id) | set(assignments) | set(exclusions))
    rows = []
    for track_id in all_track_ids:
        track_history = tracks_by_id.get(track_id, [])
        first_seen, last_seen = track_time_range(track_history)
        class_name = dominant_class(track_history)
        vtype = VTYPE_BY_CLASS.get(class_name, "passenger")
        base = {
            "track_id": track_id,
            "class_name": class_name,
            "vtype": vtype,
            "first_seen_time_sec": first_seen,
            "last_seen_time_sec": last_seen,
        }
        if track_id in exclusions:
            exclusion = exclusions[track_id]
            rows.append(
                {
                    **base,
                    "readiness_status": "EXCLUDED",
                    "initial_state_type": "excluded",
                    "exclude_reason": exclusion.get("exclude_reason", ""),
                    "review_note": exclusion.get("note", ""),
                }
            )
            continue
        assignment = assignments.get(track_id)
        if not assignment:
            rows.append({**base, "readiness_status": "MISSING_LANE_ASSIGNMENT", "initial_state_type": "missing_lane_assignment"})
            continue
        from_edge = lane_edge(assignment["in_lane_code"])
        to_edge = lane_edge(assignment["out_lane_code"])
        row = {
            **base,
            "result_direction": assignment.get("result_direction", ""),
            "in_lane_code": assignment["in_lane_code"],
            "out_lane_code": assignment["out_lane_code"],
            "from_edge": from_edge,
            "to_edge": to_edge,
            "route_edges": f"{from_edge} {to_edge}",
            "depart_lane": lane_index(assignment["in_lane_code"]),
            "arrival_lane": lane_index(assignment["out_lane_code"]),
            "review_note": assignment.get("note", ""),
        }
        override = overrides.get(track_id, {})
        anchor = anchors.get(track_id)
        if override:
            depart_time = override.get("depart_time_sec") or first_seen or "0.00"
            row.update(
                {
                    "readiness_status": "READY_WITH_INITIAL_STATE_REVIEW",
                    "initial_state_type": override.get("initial_state_type", ""),
                    "depart_time_sec": f"{float(depart_time):.2f}",
                    "depart_time_source": "manual_initial_state_override",
                    "entry_time_sec": override.get("entry_time_sec", ""),
                    "exit_time_sec": override.get("exit_time_sec", ""),
                    "anchor_source": "manual_initial_state_override",
                    "review_note": override.get("note", row.get("review_note", "")),
                }
            )
        elif anchor:
            status, initial_state, depart_time, source = classify_anchor(anchor)
            row.update(
                {
                    "readiness_status": status,
                    "initial_state_type": initial_state,
                    "depart_time_sec": f"{float(depart_time):.2f}" if depart_time else first_seen,
                    "depart_time_source": source,
                    "entry_time_sec": anchor.get("estimated_entry_time_sec", ""),
                    "exit_time_sec": anchor.get("observed_exit_time_sec", ""),
                    "anchor_source": anchor.get("route_source", ""),
                    "review_note": anchor.get("evidence_note", row.get("review_note", "")),
                }
            )
        else:
            depart_time = first_seen or "0.00"
            row.update(
                {
                    "readiness_status": "READY_WITH_TIME_FALLBACK",
                    "initial_state_type": "first_seen_time_fallback",
                    "depart_time_sec": f"{float(depart_time):.2f}",
                    "depart_time_source": "first_seen_time_fallback",
                    "anchor_source": "manual_lane_assignment_only",
                }
            )
        rows.append(row)
    return rows


def build_seed_rows(readiness_rows: list[dict]) -> list[dict]:
    seed_rows = []
    for row in readiness_rows:
        if row.get("readiness_status") in {"EXCLUDED", "MISSING_LANE_ASSIGNMENT"}:
            continue
        seed_rows.append(
            {
                "sumo_vehicle_id": f"veh_{row['track_id']}",
                "track_id": row["track_id"],
                "vtype": row["vtype"],
                "route_edges": row["route_edges"],
                "from_edge": row["from_edge"],
                "to_edge": row["to_edge"],
                "depart_time_sec": row["depart_time_sec"],
                "depart_lane": row["depart_lane"],
                "arrival_lane": row["arrival_lane"],
                "initial_state_type": row["initial_state_type"],
                "readiness_status": row["readiness_status"],
                "review_note": row.get("review_note", ""),
            }
        )
    return sorted(seed_rows, key=lambda row: (float(row["depart_time_sec"]), row["sumo_vehicle_id"]))


def pretty_xml(element: ET.Element) -> str:
    rough = ET.tostring(element, encoding="utf-8")
    return minidom.parseString(rough).toprettyxml(indent="  ")


def render_routes_xml(seed_rows: list[dict]) -> str:
    root = ET.Element("routes")
    for vtype_id, attrs in VTYPE_DEFS.items():
        ET.SubElement(root, "vType", {"id": vtype_id, **attrs})
    for row in sorted(seed_rows, key=lambda item: (float(item["depart_time_sec"]), item["sumo_vehicle_id"])):
        vehicle = ET.SubElement(
            root,
            "vehicle",
            {
                "id": row["sumo_vehicle_id"],
                "type": row["vtype"],
                "depart": f"{float(row['depart_time_sec']):.2f}",
                "departLane": row["depart_lane"],
                "arrivalLane": row["arrival_lane"],
            },
        )
        if row.get("initial_state_type") == "red_light_waiting_queue":
            vehicle.set("departSpeed", "0")
        ET.SubElement(vehicle, "route", {"edges": row["route_edges"]})
    return pretty_xml(root)


def render_sumocfg(net_file: str, route_file: str, begin: float, end: float) -> str:
    root = ET.Element("configuration")
    input_el = ET.SubElement(root, "input")
    ET.SubElement(input_el, "net-file", {"value": net_file})
    ET.SubElement(input_el, "route-files", {"value": route_file})
    time_el = ET.SubElement(root, "time")
    ET.SubElement(time_el, "begin", {"value": f"{begin:.2f}"})
    ET.SubElement(time_el, "end", {"value": f"{end:.2f}"})
    return pretty_xml(root)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--lane-assignments", required=True)
    parser.add_argument("--lane-exclusions", required=True)
    parser.add_argument("--direction-anchors", required=True)
    parser.add_argument("--initial-state-overrides", required=True)
    parser.add_argument("--net-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--end-sec", type=float, default=40.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    data_dir = output_dir / "data"
    sumo_dir = output_dir / "sumo"
    readiness_rows = build_readiness_rows(
        track_rows=read_csv(Path(args.tracks)),
        assignment_rows=read_csv(Path(args.lane_assignments)),
        exclusion_rows=read_csv(Path(args.lane_exclusions)),
        anchor_rows=read_csv(Path(args.direction_anchors)),
        override_rows=read_csv(Path(args.initial_state_overrides)),
    )
    seed_rows = build_seed_rows(readiness_rows)
    write_csv(data_dir / "sumo_readiness_review.csv", readiness_rows, READINESS_FIELDS)
    write_csv(data_dir / "sumo_vehicle_seed_plan.csv", seed_rows, SEED_FIELDS)
    sumo_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(args.net_file), sumo_dir / "real_scene.net.xml")
    (sumo_dir / "routes.rou.xml").write_text(render_routes_xml(seed_rows), encoding="utf-8")
    min_depart = min(float(row["depart_time_sec"]) for row in seed_rows) if seed_rows else 0.0
    begin = min(0.0, min_depart)
    (sumo_dir / "simulation.sumocfg").write_text(render_sumocfg("real_scene.net.xml", "routes.rou.xml", begin, args.end_sec), encoding="utf-8")
    status_counts = Counter(row["readiness_status"] for row in readiness_rows)
    write_json(
        output_dir / "sumo_candidate_manifest.json",
        {
            "readiness_rows": len(readiness_rows),
            "seed_vehicle_rows": len(seed_rows),
            "readiness_status_counts": dict(status_counts),
            "net_file": "sumo/real_scene.net.xml",
            "route_file": "sumo/routes.rou.xml",
            "sumocfg": "sumo/simulation.sumocfg",
        },
    )
    print(f"readiness_rows={len(readiness_rows)}")
    print(f"seed_vehicle_rows={len(seed_rows)}")
    print(f"readiness_status_counts={dict(status_counts)}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
