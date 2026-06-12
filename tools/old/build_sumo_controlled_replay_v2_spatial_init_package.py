#!/usr/bin/env python3
"""Build gate-aligned SUMO controlled replay with spatial initialization.

Version v1 used video gate/junction times directly as SUMO depart times.
SUMO interprets depart as insertion at the upstream edge start, so moving
vehicles arrived at the junction late. This builder corrects route replay
departures and marks warm-up/window-start vehicles for explicit video-zero
spatial initialization.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from xml.dom import minidom


REPLAY_V2_FIELDS = [
    "sumo_vehicle_id",
    "track_id",
    "vtype",
    "route_edges",
    "from_edge",
    "to_edge",
    "depart_lane",
    "arrival_lane",
    "video_depart_time_sec",
    "video_gate_entry_time_sec",
    "video_gate_exit_time_sec",
    "sim_depart_time_sec",
    "time_shift_sec",
    "incoming_travel_time_sec",
    "control_mode",
    "spatial_init_lane_id",
    "spatial_init_pos_m",
    "hold_lane_id",
    "hold_end_pos_m",
    "hold_until_sim_sec",
    "readiness_status",
    "initial_state_type",
    "result_direction",
    "timing_alignment_status",
    "review_note",
]

SPATIAL_INIT_FIELDS = [
    "sumo_vehicle_id",
    "track_id",
    "init_method",
    "init_sim_time_sec",
    "init_video_time_sec",
    "lane_id",
    "pos_m",
    "reason",
]

TRACEABILITY_FIELDS = [
    "logical_vehicle_id",
    "sumo_vehicle_id",
    "readiness_status",
    "initial_state_type",
    "control_mode",
    "result_direction",
    "route_edges",
    "from_edge",
    "to_edge",
    "depart_lane",
    "arrival_lane",
    "video_depart_time_sec",
    "video_gate_entry_time_sec",
    "video_gate_exit_time_sec",
    "sim_depart_time_sec",
    "time_shift_sec",
    "video_sumo_time_mapping",
    "timing_alignment_status",
    "review_note",
]

VTYPE_DEFS = {
    "passenger": {"vClass": "passenger", "length": "4.5", "maxSpeed": "13.89", "accel": "2.6", "decel": "4.5"},
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


def parse_lane_shapes(net_file: Path) -> dict[str, list[tuple[float, float]]]:
    root = ET.parse(net_file).getroot()
    shapes: dict[str, list[tuple[float, float]]] = {}
    for lane in root.findall(".//lane"):
        lane_id = lane.attrib.get("id", "")
        shape = lane.attrib.get("shape", "")
        if lane_id.startswith(":") or not shape:
            continue
        points = []
        for item in shape.split():
            x, y = item.split(",")[:2]
            points.append((float(x), float(y)))
        if points:
            shapes[lane_id] = points
    return shapes


def vtype_speed(vtype: str) -> float:
    return float(VTYPE_DEFS.get(vtype, VTYPE_DEFS["passenger"])["maxSpeed"])


def lane_id(row: dict) -> str:
    return f"{row['from_edge']}_{row['depart_lane']}"


def hold_position(lane_length: float, queue_rank: int) -> float:
    pos = lane_length - 8.0 - queue_rank * 7.0
    return max(5.0, pos)


def control_mode_for(row: dict) -> str:
    initial_state = row.get("initial_state_type", "")
    if initial_state == "red_light_waiting_queue":
        return "hold_until_window_end"
    if initial_state in {"manual_warmup_anchor", "window_start_partial_exit"}:
        return "spatial_init_at_video_zero"
    return "corrected_route_replay"


def incoming_travel_time(row: dict, lane_lengths: dict[str, float]) -> float:
    length = lane_lengths.get(lane_id(row), 100.0)
    return length / vtype_speed(row.get("vtype", "passenger"))


def gate_entry_time(row: dict) -> float:
    value = row.get("entry_time_sec") or row.get("depart_time_sec") or "0"
    return float(value)


def route_depart_time(row: dict, lane_lengths: dict[str, float], time_shift_sec: float) -> float:
    return max(0.0, gate_entry_time(row) + time_shift_sec - incoming_travel_time(row, lane_lengths))


def build_replay_v2_rows(
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
            hold_groups[lane_id(row)].append(row)
    hold_rank_by_track = {}
    for _, rows in hold_groups.items():
        for rank, row in enumerate(sorted(rows, key=lambda item: (float(item.get("depart_time_sec") or 0.0), item["track_id"]))):
            hold_rank_by_track[row["track_id"]] = rank

    replay_rows = []
    for row in playable:
        mode = control_mode_for(row)
        travel_time = incoming_travel_time(row, lane_lengths)
        in_lane = lane_id(row)
        lane_length = lane_lengths.get(in_lane, 100.0)
        sim_depart = 0.0
        timing_status = ""
        spatial_lane = ""
        spatial_pos = ""
        hold_lane = ""
        hold_end_pos = ""
        hold_until = ""
        if mode == "corrected_route_replay":
            sim_depart = route_depart_time(row, lane_lengths, time_shift_sec)
            timing_status = "GATE_ALIGNED_DEPART"
        elif mode == "hold_until_window_end":
            sim_depart = max(0.0, float(row.get("depart_time_sec") or 0.0) + time_shift_sec)
            rank = hold_rank_by_track.get(row["track_id"], 0)
            hold_lane = in_lane
            hold_end_pos = f"{hold_position(lane_length, rank):.2f}"
            hold_until = f"{sim_end_sec:.2f}"
            timing_status = "HOLD_CONTROL_RETAINED"
        else:
            sim_depart = 0.0
            spatial_lane = in_lane
            spatial_pos = f"{max(0.0, lane_length - 1.0):.2f}"
            timing_status = "SPATIAL_INIT_REQUIRED"
        video_gate_entry = row.get("entry_time_sec") or row.get("depart_time_sec") or ""
        replay_rows.append(
            {
                "sumo_vehicle_id": f"veh_{row['track_id']}",
                "track_id": row["track_id"],
                "vtype": row.get("vtype", "passenger"),
                "route_edges": row["route_edges"],
                "from_edge": row["from_edge"],
                "to_edge": row.get("to_edge", ""),
                "depart_lane": row["depart_lane"],
                "arrival_lane": row["arrival_lane"],
                "video_depart_time_sec": f"{float(row.get('depart_time_sec') or 0.0):.2f}",
                "video_gate_entry_time_sec": f"{float(video_gate_entry):.2f}" if video_gate_entry != "" else "",
                "video_gate_exit_time_sec": f"{float(row['exit_time_sec']):.2f}" if row.get("exit_time_sec", "") else "",
                "sim_depart_time_sec": f"{sim_depart:.2f}",
                "time_shift_sec": f"{time_shift_sec:.2f}",
                "incoming_travel_time_sec": f"{travel_time:.2f}",
                "control_mode": mode,
                "spatial_init_lane_id": spatial_lane,
                "spatial_init_pos_m": spatial_pos,
                "hold_lane_id": hold_lane,
                "hold_end_pos_m": hold_end_pos,
                "hold_until_sim_sec": hold_until,
                "readiness_status": row.get("readiness_status", ""),
                "initial_state_type": row.get("initial_state_type", ""),
                "result_direction": row.get("result_direction", ""),
                "timing_alignment_status": timing_status,
                "review_note": row.get("review_note", ""),
            }
        )
    return sorted(replay_rows, key=lambda item: (float(item["sim_depart_time_sec"]), item["sumo_vehicle_id"]))


def build_spatial_initialization_rows(replay_rows: list[dict]) -> list[dict]:
    rows = []
    for row in replay_rows:
        if row.get("control_mode") != "spatial_init_at_video_zero":
            continue
        rows.append(
            {
                "sumo_vehicle_id": row["sumo_vehicle_id"],
                "track_id": row["track_id"],
                "init_method": "moveTo_lane_end_at_video_zero",
                "init_sim_time_sec": "0.00",
                "init_video_time_sec": "0.00",
                "lane_id": row["spatial_init_lane_id"],
                "pos_m": row["spatial_init_pos_m"],
                "reason": row["initial_state_type"],
            }
        )
    return rows


def build_traceability_rows(replay_rows: list[dict]) -> list[dict]:
    rows = []
    for row in replay_rows:
        rows.append(
            {
                "logical_vehicle_id": row["track_id"],
                "sumo_vehicle_id": row["sumo_vehicle_id"],
                "readiness_status": row["readiness_status"],
                "initial_state_type": row["initial_state_type"],
                "control_mode": row["control_mode"],
                "result_direction": row["result_direction"],
                "route_edges": row["route_edges"],
                "from_edge": row["from_edge"],
                "to_edge": row["to_edge"],
                "depart_lane": row["depart_lane"],
                "arrival_lane": row["arrival_lane"],
                "video_depart_time_sec": row["video_depart_time_sec"],
                "video_gate_entry_time_sec": row["video_gate_entry_time_sec"],
                "video_gate_exit_time_sec": row["video_gate_exit_time_sec"],
                "sim_depart_time_sec": row["sim_depart_time_sec"],
                "time_shift_sec": row["time_shift_sec"],
                "video_sumo_time_mapping": f"video_time=sim_time-{row['time_shift_sec']}",
                "timing_alignment_status": row["timing_alignment_status"],
                "review_note": row["review_note"],
            }
        )
    return rows


def render_routes_xml(replay_rows: list[dict]) -> str:
    root = ET.Element("routes")
    for vtype_id, attrs in VTYPE_DEFS.items():
        ET.SubElement(root, "vType", {"id": vtype_id, **attrs})
    for row in replay_rows:
        attrs = {
            "id": row["sumo_vehicle_id"],
            "type": row["vtype"],
            "depart": row["sim_depart_time_sec"],
            "departLane": row["depart_lane"],
            "arrivalLane": row["arrival_lane"],
        }
        if row["control_mode"] == "hold_until_window_end":
            attrs["departSpeed"] = "0"
        if row["control_mode"] == "spatial_init_at_video_zero":
            attrs["departPos"] = row["spatial_init_pos_m"]
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


def render_sumocfg(net_file: str, route_file: str, begin: float, end: float) -> str:
    root = ET.Element("configuration")
    input_el = ET.SubElement(root, "input")
    ET.SubElement(input_el, "net-file", {"value": net_file})
    ET.SubElement(input_el, "route-files", {"value": route_file})
    time_el = ET.SubElement(root, "time")
    ET.SubElement(time_el, "begin", {"value": f"{begin:.2f}"})
    ET.SubElement(time_el, "end", {"value": f"{end:.2f}"})
    processing = ET.SubElement(root, "processing")
    ET.SubElement(processing, "ignore-route-errors", {"value": "true"})
    return pretty_xml(root)


def render_traci_controller() -> str:
    return '''#!/usr/bin/env python3
"""Run SUMO controlled replay v2 and write audit trajectories."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def add_known_traci_paths() -> None:
    candidates = [
        Path.home() / ".local/share/eclipse-sumo-venv/lib/python3.13/site-packages",
        Path.home() / ".local/share/eclipse-sumo-venv/lib/python3.13/site-packages/sumo/tools",
    ]
    sumo_home = os.environ.get("SUMO_HOME")
    if sumo_home:
        candidates.append(Path(sumo_home) / "tools")
    for path in candidates:
        if path.exists():
            sys.path.insert(0, str(path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sumo-binary", default="sumo")
    parser.add_argument("--sumocfg", default="simulation_controlled_v2.sumocfg")
    parser.add_argument("--plan", default="../data/controlled_replay_v2_plan.csv")
    parser.add_argument("--spatial-init-plan", default="../data/spatial_initialization_plan.csv")
    parser.add_argument("--trajectory-output", default="../data/sumo_replay_trajectory.csv")
    parser.add_argument("--step-length", default="0.20")
    parser.add_argument("--traci-port", type=int, default=8873)
    args = parser.parse_args()

    add_known_traci_paths()
    try:
        import traci
    except ImportError as exc:
        raise SystemExit("TraCI is not available. Set SUMO_HOME or use the SUMO venv Python.") from exc

    plan = {row["sumo_vehicle_id"]: row for row in load_rows(Path(args.plan))}
    spatial_rows = {row["sumo_vehicle_id"]: row for row in load_rows(Path(args.spatial_init_plan))}
    hold_rows = {
        vehicle_id: row
        for vehicle_id, row in plan.items()
        if row.get("control_mode") == "hold_until_window_end"
    }
    trajectory_path = Path(args.trajectory_output)
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)

    traci.start(
        [args.sumo_binary, "-c", args.sumocfg, "--step-length", args.step_length, "--duration-log.disable", "true"],
        port=args.traci_port,
    )
    initialized = set()
    with trajectory_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sim_time_sec", "video_time_sec", "sumo_vehicle_id", "track_id", "x", "y", "angle", "speed", "lane_id", "road_id"],
        )
        writer.writeheader()
        try:
            while traci.simulation.getMinExpectedNumber() > 0:
                traci.simulationStep()
                now = traci.simulation.getTime()
                active = set(traci.vehicle.getIDList())
                for vehicle_id, row in spatial_rows.items():
                    if vehicle_id in active and vehicle_id not in initialized:
                        try:
                            traci.vehicle.moveTo(vehicle_id, row["lane_id"], float(row["pos_m"]))
                        except Exception:
                            pass
                        initialized.add(vehicle_id)
                for vehicle_id, row in hold_rows.items():
                    if vehicle_id not in active:
                        continue
                    until = float(row["hold_until_sim_sec"])
                    if now <= until:
                        traci.vehicle.setSpeed(vehicle_id, 0.0)
                    else:
                        traci.vehicle.setSpeed(vehicle_id, -1.0)
                for vehicle_id in sorted(active):
                    row = plan.get(vehicle_id, {})
                    x, y = traci.vehicle.getPosition(vehicle_id)
                    writer.writerow(
                        {
                            "sim_time_sec": f"{now:.2f}",
                            "video_time_sec": f"{now - float(row.get('time_shift_sec') or 0.0):.2f}",
                            "sumo_vehicle_id": vehicle_id,
                            "track_id": row.get("track_id", ""),
                            "x": f"{x:.2f}",
                            "y": f"{y:.2f}",
                            "angle": f"{traci.vehicle.getAngle(vehicle_id):.2f}",
                            "speed": f"{traci.vehicle.getSpeed(vehicle_id):.2f}",
                            "lane_id": traci.vehicle.getLaneID(vehicle_id),
                            "road_id": traci.vehicle.getRoadID(vehicle_id),
                        }
                    )
        finally:
            traci.close()


if __name__ == "__main__":
    main()
'''


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksums(output_dir: Path) -> None:
    rows = []
    for path in sorted(item for item in output_dir.rglob("*") if item.is_file() and item.name != "SHA256SUMS.txt"):
        rows.append(f"{sha256(path)}  {path.relative_to(output_dir)}")
    (output_dir / "SHA256SUMS.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")


def run_validation(sumo_binary: str, sumo_dir: Path, controller_path: Path) -> dict:
    summary = {
        "sumo_binary": sumo_binary,
        "static_sumo_exit_code": None,
        "traci_replay_exit_code": None,
        "static_sumo_stdout_tail": "",
        "static_sumo_stderr_tail": "",
        "traci_stdout_tail": "",
        "traci_stderr_tail": "",
    }
    static = subprocess.run(
        [sumo_binary, "-c", "simulation_controlled_v2.sumocfg", "--duration-log.disable", "true"],
        cwd=sumo_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    summary["static_sumo_exit_code"] = static.returncode
    summary["static_sumo_stdout_tail"] = static.stdout[-2000:]
    summary["static_sumo_stderr_tail"] = static.stderr[-2000:]
    traci = subprocess.run(
        ["python3", controller_invocation_path(controller_path), "--sumo-binary", sumo_binary],
        cwd=sumo_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    summary["traci_replay_exit_code"] = traci.returncode
    summary["traci_stdout_tail"] = traci.stdout[-2000:]
    summary["traci_stderr_tail"] = traci.stderr[-2000:]
    return summary


def controller_invocation_path(controller_path: Path) -> str:
    """Return the controller path to invoke with cwd already set to sumo_dir."""
    return controller_path.name


def build_package(
    source_dir: Path,
    diagnostics_dir: Path,
    output_dir: Path,
    time_shift_sec: float,
    video_end_sec: float,
    sumo_binary: str | None = None,
    run_sumo_validation_flag: bool = False,
) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    sim_end_sec = time_shift_sec + video_end_sec
    data_dir = output_dir / "data"
    sumo_dir = output_dir / "sumo"
    snapshot_dir = output_dir / "inputs_snapshot"
    source_sumo_dir = source_dir / "sumo"
    readiness_path = source_dir / "data" / "sumo_readiness_review.csv"
    diagnostics_path = diagnostics_dir / "sumo_timing_diagnostics.csv"
    correction_path = diagnostics_dir / "sumo_timing_correction_recommendations.csv"
    source_net = source_sumo_dir / "real_scene.net.xml"
    readiness_rows = read_csv(readiness_path)
    lane_lengths = parse_lane_lengths(source_net)
    replay_rows = build_replay_v2_rows(readiness_rows, lane_lengths, time_shift_sec, sim_end_sec)
    spatial_rows = build_spatial_initialization_rows(replay_rows)
    traceability_rows = build_traceability_rows(replay_rows)

    output_dir.mkdir(parents=True)
    data_dir.mkdir()
    sumo_dir.mkdir()
    snapshot_dir.mkdir()
    write_csv(data_dir / "controlled_replay_v2_plan.csv", replay_rows, REPLAY_V2_FIELDS)
    write_csv(data_dir / "spatial_initialization_plan.csv", spatial_rows, SPATIAL_INIT_FIELDS)
    write_csv(data_dir / "video_sumo_traceability.csv", traceability_rows, TRACEABILITY_FIELDS)
    write_json(data_dir / "controlled_replay_v2_plan.json", {"vehicles": replay_rows})
    shutil.copy2(source_net, sumo_dir / "real_scene.net.xml")
    shutil.copy2(readiness_path, snapshot_dir / "sumo_readiness_review.csv")
    if (source_dir / "data" / "sumo_vehicle_seed_plan.csv").exists():
        shutil.copy2(source_dir / "data" / "sumo_vehicle_seed_plan.csv", snapshot_dir / "sumo_vehicle_seed_plan.csv")
    if diagnostics_path.exists():
        shutil.copy2(diagnostics_path, snapshot_dir / "sumo_timing_diagnostics_v1.csv")
    if correction_path.exists():
        shutil.copy2(correction_path, snapshot_dir / "sumo_timing_correction_recommendations_v1.csv")

    (sumo_dir / "routes_controlled_v2.rou.xml").write_text(render_routes_xml(replay_rows), encoding="utf-8")
    (sumo_dir / "simulation_controlled_v2.sumocfg").write_text(
        render_sumocfg("real_scene.net.xml", "routes_controlled_v2.rou.xml", 0.0, sim_end_sec),
        encoding="utf-8",
    )
    controller_path = sumo_dir / "run_controlled_replay_v2.py"
    controller_path.write_text(render_traci_controller(), encoding="utf-8")
    controller_path.chmod(0o755)

    mode_counts = Counter(row["control_mode"] for row in replay_rows)
    status_counts = Counter(row["timing_alignment_status"] for row in replay_rows)
    validation_summary = {}
    if run_sumo_validation_flag and sumo_binary:
        validation_summary = run_validation(sumo_binary, sumo_dir, controller_path)
        write_json(data_dir / "run_validation_summary.json", validation_summary)
    else:
        write_json(data_dir / "run_validation_summary.json", {"status": "not_run"})

    manifest = {
        "version_id": output_dir.name,
        "source_version": str(source_dir),
        "diagnostics_source": str(diagnostics_dir),
        "time_shift_sec": f"{time_shift_sec:.2f}",
        "video_zero_sim_time_sec": f"{time_shift_sec:.2f}",
        "sim_begin_sec": "0.00",
        "sim_end_sec": f"{sim_end_sec:.2f}",
        "vehicle_count": len(replay_rows),
        "control_mode_counts": dict(mode_counts),
        "timing_alignment_status_counts": dict(status_counts),
        "spatial_init_vehicle_count": len(spatial_rows),
        "sumo_files": {
            "net_file": "sumo/real_scene.net.xml",
            "route_file": "sumo/routes_controlled_v2.rou.xml",
            "sumocfg": "sumo/simulation_controlled_v2.sumocfg",
            "traci_controller": "sumo/run_controlled_replay_v2.py",
        },
        "data_files": {
            "replay_plan": "data/controlled_replay_v2_plan.csv",
            "spatial_init_plan": "data/spatial_initialization_plan.csv",
            "traceability": "data/video_sumo_traceability.csv",
            "validation_summary": "data/run_validation_summary.json",
        },
        "validation_summary": validation_summary,
        "note": "Route replay vehicles use gate-aligned depart correction. Warm-up/window-start vehicles are explicitly marked as video-zero spatial initialization rather than claimed as precise automatic projection.",
    }
    write_json(output_dir / "sumo_controlled_replay_v2_manifest.json", manifest)
    (output_dir / "VERSION_LOCK.md").write_text(
        "\n".join(
            [
                f"# SUMO Controlled Replay Version: {output_dir.name}",
                "",
                "This package is derived from the locked SUMO candidate v1 package and v1 timing diagnostics.",
                "It corrects route replay depart times from video gate anchors to SUMO edge-start insertion semantics.",
                "Warm-up/window-start vehicles are retained as auditable video-zero spatial initialization rows.",
                "",
                f"- vehicle_count: `{len(replay_rows)}`",
                f"- control_mode_counts: `{dict(mode_counts)}`",
                f"- timing_alignment_status_counts: `{dict(status_counts)}`",
                f"- video_zero_sim_time_sec: `{time_shift_sec:.2f}`",
                f"- sim_begin_sec: `0.00`",
                f"- sim_end_sec: `{sim_end_sec:.2f}`",
                "- overwrite_policy: locked_do_not_overwrite_create_new_version_for_changes",
                "",
                "Run with TraCI control:",
                "",
                "```bash",
                "cd sumo",
                "python3 run_controlled_replay_v2.py --sumo-binary /Users/zfh/.local/bin/sumo",
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )
    write_checksums(output_dir)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--diagnostics-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--time-shift-sec", type=float, default=6.0)
    parser.add_argument("--video-end-sec", type=float, default=40.0)
    parser.add_argument("--sumo-binary", default="")
    parser.add_argument("--run-sumo-validation", action="store_true")
    args = parser.parse_args()
    manifest = build_package(
        Path(args.source_dir),
        Path(args.diagnostics_dir),
        Path(args.output_dir),
        args.time_shift_sec,
        args.video_end_sec,
        sumo_binary=args.sumo_binary or None,
        run_sumo_validation_flag=args.run_sumo_validation,
    )
    print(f"vehicle_count={manifest['vehicle_count']}")
    print(f"control_mode_counts={manifest['control_mode_counts']}")
    print(f"timing_alignment_status_counts={manifest['timing_alignment_status_counts']}")
    print(f"spatial_init_vehicle_count={manifest['spatial_init_vehicle_count']}")
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
