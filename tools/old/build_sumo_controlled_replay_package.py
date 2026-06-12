#!/usr/bin/env python3
"""Build a non-negative-time SUMO controlled replay package."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from xml.dom import minidom


CONTROLLED_REPLAY_FIELDS = [
    "sumo_vehicle_id",
    "track_id",
    "vtype",
    "route_edges",
    "from_edge",
    "depart_lane",
    "arrival_lane",
    "video_depart_time_sec",
    "sim_depart_time_sec",
    "time_shift_sec",
    "control_mode",
    "hold_lane_id",
    "hold_end_pos_m",
    "hold_until_sim_sec",
    "readiness_status",
    "initial_state_type",
    "review_note",
]

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


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def pretty_xml(element: ET.Element) -> str:
    rough = ET.tostring(element, encoding="utf-8")
    return minidom.parseString(rough).toprettyxml(indent="  ")


def parse_lane_lengths(net_file: Path) -> dict[str, float]:
    root = ET.parse(net_file).getroot()
    lengths: dict[str, float] = {}
    for lane in root.findall(".//lane"):
        lane_id = lane.attrib.get("id", "")
        if lane_id.startswith(":"):
            continue
        if "length" in lane.attrib:
            lengths[lane_id] = float(lane.attrib["length"])
    return lengths


def control_mode_for(row: dict) -> str:
    initial_state = row.get("initial_state_type", "")
    if initial_state == "red_light_waiting_queue":
        return "hold_until_window_end"
    if initial_state == "window_start_partial_exit":
        return "warmup_partial_exit"
    if initial_state == "manual_warmup_anchor":
        return "warmup_route"
    return "route_replay"


def shifted_depart_time(row: dict, time_shift_sec: float) -> tuple[float, float]:
    mode = control_mode_for(row)
    video_depart = float(row.get("depart_time_sec") or 0.0)
    if mode == "warmup_partial_exit":
        return -time_shift_sec, 0.0
    return video_depart, max(0.0, video_depart + time_shift_sec)


def hold_position(lane_length: float, queue_rank: int) -> float:
    pos = lane_length - 8.0 - queue_rank * 7.0
    return max(5.0, pos)


def build_controlled_replay_rows(
    readiness_rows: list[dict],
    lane_lengths: dict[str, float],
    time_shift_sec: float,
    sim_end_sec: float,
) -> list[dict]:
    playable = [
        row
        for row in readiness_rows
        if row.get("readiness_status") not in {"", "EXCLUDED", "MISSING_LANE_ASSIGNMENT"}
    ]
    hold_groups: dict[str, list[dict]] = defaultdict(list)
    for row in playable:
        if control_mode_for(row) == "hold_until_window_end":
            hold_groups[f"{row['from_edge']}_{row['depart_lane']}"].append(row)
    hold_rank_by_track = {}
    for lane_id, rows in hold_groups.items():
        for rank, row in enumerate(sorted(rows, key=lambda item: (float(item.get("depart_time_sec") or 0.0), item["track_id"]))):
            hold_rank_by_track[row["track_id"]] = rank

    controlled_rows = []
    for row in playable:
        mode = control_mode_for(row)
        video_depart, sim_depart = shifted_depart_time(row, time_shift_sec)
        lane_id = f"{row['from_edge']}_{row['depart_lane']}"
        hold_lane_id = ""
        hold_end_pos = ""
        hold_until = ""
        if mode == "hold_until_window_end":
            hold_lane_id = lane_id
            lane_length = lane_lengths.get(lane_id, 100.0)
            hold_end_pos = f"{hold_position(lane_length, hold_rank_by_track[row['track_id']]):.2f}"
            hold_until = f"{sim_end_sec:.2f}"
        controlled_rows.append(
            {
                "sumo_vehicle_id": f"veh_{row['track_id']}",
                "track_id": row["track_id"],
                "vtype": row.get("vtype", "passenger"),
                "route_edges": row["route_edges"],
                "from_edge": row["from_edge"],
                "depart_lane": row["depart_lane"],
                "arrival_lane": row["arrival_lane"],
                "video_depart_time_sec": f"{video_depart:.2f}",
                "sim_depart_time_sec": f"{sim_depart:.2f}",
                "time_shift_sec": f"{time_shift_sec:.2f}",
                "control_mode": mode,
                "hold_lane_id": hold_lane_id,
                "hold_end_pos_m": hold_end_pos,
                "hold_until_sim_sec": hold_until,
                "readiness_status": row.get("readiness_status", ""),
                "initial_state_type": row.get("initial_state_type", ""),
                "review_note": row.get("review_note", ""),
            }
        )
    return sorted(controlled_rows, key=lambda item: (float(item["sim_depart_time_sec"]), item["sumo_vehicle_id"]))


def render_controlled_routes_xml(controlled_rows: list[dict]) -> str:
    root = ET.Element("routes")
    for vtype_id, attrs in VTYPE_DEFS.items():
        ET.SubElement(root, "vType", {"id": vtype_id, **attrs})
    for row in controlled_rows:
        attrs = {
            "id": row["sumo_vehicle_id"],
            "type": row["vtype"],
            "depart": row["sim_depart_time_sec"],
            "departLane": row["depart_lane"],
            "arrivalLane": row["arrival_lane"],
        }
        if row["control_mode"] == "hold_until_window_end":
            attrs["departSpeed"] = "0"
        vehicle = ET.SubElement(root, "vehicle", attrs)
        ET.SubElement(vehicle, "route", {"edges": row["route_edges"]})
        if row["control_mode"] == "hold_until_window_end":
            ET.SubElement(
                vehicle,
                "stop",
                {
                    "lane": row["hold_lane_id"],
                    "endPos": row["hold_end_pos_m"],
                    "until": row["hold_until_sim_sec"],
                },
            )
    return pretty_xml(root)


def render_controlled_sumocfg(net_file: str, route_file: str, begin: float, end: float) -> str:
    root = ET.Element("configuration")
    input_el = ET.SubElement(root, "input")
    ET.SubElement(input_el, "net-file", {"value": net_file})
    ET.SubElement(input_el, "route-files", {"value": route_file})
    time_el = ET.SubElement(root, "time")
    ET.SubElement(time_el, "begin", {"value": f"{begin:.2f}"})
    ET.SubElement(time_el, "end", {"value": f"{end:.2f}"})
    return pretty_xml(root)


def render_traci_controller() -> str:
    return '''#!/usr/bin/env python3
"""Run SUMO controlled replay and keep review-marked waiting vehicles stopped."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def load_plan(path: Path) -> dict[str, dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["sumo_vehicle_id"]: row for row in csv.DictReader(handle)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sumo-binary", default="sumo")
    parser.add_argument("--sumocfg", default="simulation_controlled.sumocfg")
    parser.add_argument("--plan", default="../data/controlled_replay_plan.csv")
    parser.add_argument("--step-length", default="0.02")
    args = parser.parse_args()

    try:
        import traci
    except ImportError as exc:
        raise SystemExit("TraCI is not available. Install SUMO tools or set PYTHONPATH to SUMO_HOME/tools.") from exc

    plan = load_plan(Path(args.plan))
    hold_rows = {
        vehicle_id: row
        for vehicle_id, row in plan.items()
        if row.get("control_mode") == "hold_until_window_end"
    }
    traci.start([args.sumo_binary, "-c", args.sumocfg, "--step-length", args.step_length])
    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            now = traci.simulation.getTime()
            active = set(traci.vehicle.getIDList())
            for vehicle_id, row in hold_rows.items():
                if vehicle_id not in active:
                    continue
                until = float(row["hold_until_sim_sec"])
                if now <= until:
                    traci.vehicle.setSpeed(vehicle_id, 0.0)
                else:
                    traci.vehicle.setSpeed(vehicle_id, -1.0)
    finally:
        traci.close()


if __name__ == "__main__":
    main()
'''


def build_package(source_dir: Path, output_dir: Path, time_shift_sec: float, video_end_sec: float) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    sim_end_sec = time_shift_sec + video_end_sec
    data_dir = output_dir / "data"
    sumo_dir = output_dir / "sumo"
    snapshot_dir = output_dir / "inputs_snapshot"
    source_sumo_dir = source_dir / "sumo"
    readiness_path = source_dir / "data" / "sumo_readiness_review.csv"
    source_net = source_sumo_dir / "real_scene.net.xml"
    readiness_rows = read_csv(readiness_path)
    lane_lengths = parse_lane_lengths(source_net)
    controlled_rows = build_controlled_replay_rows(readiness_rows, lane_lengths, time_shift_sec, sim_end_sec)

    output_dir.mkdir(parents=True)
    data_dir.mkdir()
    sumo_dir.mkdir()
    snapshot_dir.mkdir()
    write_csv(data_dir / "controlled_replay_plan.csv", controlled_rows, CONTROLLED_REPLAY_FIELDS)
    write_json(data_dir / "controlled_replay_plan.json", {"vehicles": controlled_rows})
    shutil.copy2(source_net, sumo_dir / "real_scene.net.xml")
    shutil.copy2(readiness_path, snapshot_dir / "sumo_readiness_review.csv")
    if (source_dir / "data" / "sumo_vehicle_seed_plan.csv").exists():
        shutil.copy2(source_dir / "data" / "sumo_vehicle_seed_plan.csv", snapshot_dir / "sumo_vehicle_seed_plan.csv")
    (sumo_dir / "routes_controlled.rou.xml").write_text(render_controlled_routes_xml(controlled_rows), encoding="utf-8")
    (sumo_dir / "simulation_controlled.sumocfg").write_text(
        render_controlled_sumocfg("real_scene.net.xml", "routes_controlled.rou.xml", 0.0, sim_end_sec),
        encoding="utf-8",
    )
    controller_path = sumo_dir / "run_controlled_replay.py"
    controller_path.write_text(render_traci_controller(), encoding="utf-8")
    controller_path.chmod(0o755)
    mode_counts = Counter(row["control_mode"] for row in controlled_rows)
    manifest = {
        "version_id": output_dir.name,
        "source_version": str(source_dir),
        "time_shift_sec": f"{time_shift_sec:.2f}",
        "video_zero_sim_time_sec": f"{time_shift_sec:.2f}",
        "sim_begin_sec": "0.00",
        "sim_end_sec": f"{sim_end_sec:.2f}",
        "vehicle_count": len(controlled_rows),
        "control_mode_counts": dict(mode_counts),
        "sumo_files": {
            "net_file": "sumo/real_scene.net.xml",
            "route_file": "sumo/routes_controlled.rou.xml",
            "sumocfg": "sumo/simulation_controlled.sumocfg",
            "traci_controller": "sumo/run_controlled_replay.py",
        },
        "run_commands": [
            "cd sumo && sumo -c simulation_controlled.sumocfg",
            "cd sumo && python3 run_controlled_replay.py --sumo-binary sumo",
            "cd sumo && python3 run_controlled_replay.py --sumo-binary sumo-gui",
        ],
        "note": "SUMO time is shifted so video_time = sim_time - time_shift_sec. Red-light waiting vehicles are held through the review window.",
    }
    write_json(output_dir / "sumo_controlled_replay_manifest.json", manifest)
    (output_dir / "VERSION_LOCK.md").write_text(
        "\n".join(
            [
                f"# SUMO Controlled Replay Version: {output_dir.name}",
                "",
                "This package is derived from the locked SUMO candidate v1 package.",
                "It uses a non-negative SUMO time axis and fixed vehicle control for review.",
                "",
                f"- video_zero_sim_time_sec: `{time_shift_sec:.2f}`",
                f"- sim_begin_sec: `0.00`",
                f"- sim_end_sec: `{sim_end_sec:.2f}`",
                "- overwrite_policy: locked_do_not_overwrite_create_new_version_for_changes",
                "",
                "Run with TraCI control:",
                "",
                "```bash",
                "cd sumo",
                "python3 run_controlled_replay.py --sumo-binary sumo-gui",
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--time-shift-sec", type=float, default=6.0)
    parser.add_argument("--video-end-sec", type=float, default=40.0)
    args = parser.parse_args()
    manifest = build_package(Path(args.source_dir), Path(args.output_dir), args.time_shift_sec, args.video_end_sec)
    print(f"vehicle_count={manifest['vehicle_count']}")
    print(f"control_mode_counts={manifest['control_mode_counts']}")
    print(f"video_zero_sim_time_sec={manifest['video_zero_sim_time_sec']}")
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
