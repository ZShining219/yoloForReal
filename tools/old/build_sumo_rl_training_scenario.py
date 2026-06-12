#!/usr/bin/env python3
"""Build a clean SUMO-RL training scenario from a calibration profile."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from xml.dom import minidom


VTYPE_DEFS = {
    "passenger": {"vClass": "passenger", "length": "4.5", "maxSpeed": "13.89", "accel": "2.6", "decel": "4.5"},
    "truck": {"vClass": "truck", "length": "8.0", "maxSpeed": "11.0", "accel": "1.3", "decel": "4.0"},
    "bus": {"vClass": "bus", "length": "12.0", "maxSpeed": "10.0", "accel": "1.2", "decel": "4.0"},
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def pretty_xml(element: ET.Element) -> str:
    rough = ET.tostring(element, encoding="utf-8")
    return minidom.parseString(rough).toprettyxml(indent="  ")


def vehicle_count_for(profile: dict, episode_duration_sec: float, min_vehicle_count: int, demand_scale: float = 1.0) -> int:
    source_window_sec = float(profile["source_window_sec"])
    observed = int(profile["observed_vehicle_count"])
    return max(min_vehicle_count, int(round(observed / source_window_sec * episode_duration_sec * demand_scale)))


def allocate_counts(total: int, distribution: dict) -> dict[str, int]:
    keys = sorted(distribution)
    raw = {key: total * float(distribution[key].get("share", 0.0)) for key in keys}
    counts = {key: int(math.floor(raw[key])) for key in keys}
    remaining = total - sum(counts.values())
    order = sorted(keys, key=lambda key: (raw[key] - counts[key], key), reverse=True)
    for key in order[:remaining]:
        counts[key] += 1
    return counts


def ordered_route_sequence(profile: dict, total: int) -> list[str]:
    counts = allocate_counts(total, profile["route_distribution"])
    routes = []
    for route in sorted(counts):
        routes.extend([route] * counts[route])
    return routes


def ordered_vtype_sequence(profile: dict, total: int) -> list[str]:
    distribution = profile.get("vehicle_type_distribution") or {"passenger": {"share": 1.0}}
    counts = allocate_counts(total, distribution)
    vtypes = []
    for vtype in sorted(counts):
        vtypes.extend([vtype] * counts[vtype])
    if not vtypes:
        vtypes = ["passenger"] * total
    return vtypes


def generate_training_vehicles(
    profile: dict,
    episode_duration_sec: float,
    episode_id: str = "ep001",
    seed: int = 20260608,
    demand_scale: float = 1.0,
    min_vehicle_count: int = 1,
) -> list[dict]:
    total = vehicle_count_for(profile, episode_duration_sec, min_vehicle_count, demand_scale=demand_scale)
    rng = random.Random(seed)
    routes = ordered_route_sequence(profile, total)
    vtypes = ordered_vtype_sequence(profile, total)
    rng.shuffle(routes)
    rng.shuffle(vtypes)
    if len(vtypes) < total:
        vtypes.extend(["passenger"] * (total - len(vtypes)))
    vehicles = []
    interval = episode_duration_sec / total if total else episode_duration_sec
    for index in range(total):
        depart = min(episode_duration_sec - 0.01, index * interval + rng.uniform(0.0, interval * 0.35))
        vehicles.append(
            {
                "vehicle_id": f"train_{episode_id}_{index:04d}",
                "vtype": vtypes[index],
                "depart": f"{depart:.2f}",
                "route_edges": routes[index],
            }
        )
    return sorted(vehicles, key=lambda row: float(row["depart"]))


def render_routes_xml(vehicles: list[dict]) -> str:
    root = ET.Element("routes")
    for vtype_id, attrs in VTYPE_DEFS.items():
        ET.SubElement(root, "vType", {"id": vtype_id, **attrs})
    for row in vehicles:
        vehicle = ET.SubElement(
            root,
            "vehicle",
            {
                "id": row["vehicle_id"],
                "type": row["vtype"],
                "depart": row["depart"],
                "departLane": "best",
            },
        )
        # Training demand uses native SUMO routing only: no stop, moveTo, or departPos.
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
    processing = ET.SubElement(root, "processing")
    ET.SubElement(processing, "ignore-route-errors", {"value": "true"})
    return pretty_xml(root)


def detect_tls_ids(net_file: Path) -> list[str]:
    root = ET.parse(net_file).getroot()
    ids = [node.attrib["id"] for node in root.findall(".//junction") if node.attrib.get("type") == "traffic_light"]
    ids.extend(tl.attrib["id"] for tl in root.findall(".//tlLogic") if tl.attrib.get("id"))
    return sorted(set(ids))


def signalize_plain_nodes_xml(xml_text: str, junction_id: str) -> str:
    root = ET.fromstring(xml_text)
    found = False
    for node in root.findall("node"):
        if node.attrib.get("id") == junction_id:
            node.set("type", "traffic_light")
            found = True
    if not found:
        raise ValueError(f"Junction {junction_id!r} not found in plain nodes XML")
    return pretty_xml(root)


def build_signalized_net(source_net: Path, output_net: Path, junction_id: str, netconvert_binary: str) -> None:
    output_net.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sumo_signalized_net_") as tmpdir:
        prefix = Path(tmpdir) / "plain"
        subprocess.run(
            [
                netconvert_binary,
                "--sumo-net-file",
                str(source_net),
                "--plain-output-prefix",
                str(prefix),
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        nodes_path = prefix.with_suffix(".nod.xml")
        nodes_path.write_text(signalize_plain_nodes_xml(nodes_path.read_text(encoding="utf-8"), junction_id), encoding="utf-8")
        subprocess.run(
            [
                netconvert_binary,
                "--node-files",
                str(nodes_path),
                "--edge-files",
                str(prefix.with_suffix(".edg.xml")),
                "--connection-files",
                str(prefix.with_suffix(".con.xml")),
                "--output-file",
                str(output_net),
                "--no-turnarounds",
                "true",
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )


def build_rl_configs(
    profile: dict,
    tls_ids: list[str],
    episode_duration_sec: float,
    demand_scale: float,
    net_file: str = "sumo/real_scene.net.xml",
) -> dict[str, dict]:
    env_config = {
        "scenario_id": "sumo_rl_training_scenario_v1",
        "net_file": net_file,
        "route_file": "sumo/routes_train.rou.xml",
        "sumocfg": "sumo/simulation_train.sumocfg",
        "episode_duration_sec": episode_duration_sec,
        "demand_scale": demand_scale,
        "traffic_signal_ids": tls_ids,
        "traffic_signal_control_ready": bool(tls_ids),
        "calibration_profile": "data/calibration_profile.json",
        "note": "Demand is statistically calibrated from real-video evidence. It is not a per-vehicle replay.",
    }
    dqn_config = {
        "algorithm": "DQN",
        "environment": "SUMO-RL-compatible",
        "episodes": 100,
        "gamma": 0.99,
        "learning_rate": 0.0005,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay_steps": 5000,
        "replay_buffer_size": 50000,
        "batch_size": 64,
        "training_ready_requires_tls": True,
        "traffic_signal_control_ready": bool(tls_ids),
    }
    reward_config = {
        "primary_reward": "negative_total_waiting_time_delta",
        "secondary_metrics": ["queue_length", "throughput", "average_delay"],
    }
    return {
        "env_config": env_config,
        "dqn_training_config": dqn_config,
        "reward_config": reward_config,
        "action_space": {"type": "traffic_signal_phase_selection", "traffic_signal_ids": tls_ids},
        "observation_space": {"features": ["lane_vehicle_count", "lane_queue_length", "lane_waiting_time"]},
    }


def build_package(
    profile_path: Path,
    net_file: Path,
    output_dir: Path,
    episode_duration_sec: float,
    seed: int,
    demand_scale: float,
    signalize_junction_id: str = "",
    netconvert_binary: str = "/Users/zfh/.local/share/eclipse-sumo-venv/bin/netconvert",
) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    profile = read_json(profile_path)
    vehicles = generate_training_vehicles(profile, episode_duration_sec=episode_duration_sec, seed=seed, demand_scale=demand_scale)
    sumo_dir = output_dir / "sumo"
    rl_dir = output_dir / "rl"
    data_dir = output_dir / "data"
    output_dir.mkdir(parents=True)
    sumo_dir.mkdir()
    rl_dir.mkdir()
    data_dir.mkdir()
    net_filename = "real_scene.net.xml"
    if signalize_junction_id:
        net_filename = "real_scene_tls.net.xml"
        build_signalized_net(net_file, sumo_dir / net_filename, signalize_junction_id, netconvert_binary)
        shutil.copy2(net_file, sumo_dir / "real_scene_source_priority.net.xml")
    else:
        shutil.copy2(net_file, sumo_dir / net_filename)
    shutil.copy2(profile_path, data_dir / "calibration_profile.json")
    (sumo_dir / "routes_train.rou.xml").write_text(render_routes_xml(vehicles), encoding="utf-8")
    (sumo_dir / "simulation_train.sumocfg").write_text(
        render_sumocfg(net_filename, "routes_train.rou.xml", 0.0, episode_duration_sec),
        encoding="utf-8",
    )
    tls_ids = detect_tls_ids(sumo_dir / net_filename)
    configs = build_rl_configs(profile, tls_ids, episode_duration_sec, demand_scale, net_file=f"sumo/{net_filename}")
    for name, payload in configs.items():
        write_json(rl_dir / f"{name}.json", payload)
    route_counts = Counter(row["route_edges"] for row in vehicles)
    vtype_counts = Counter(row["vtype"] for row in vehicles)
    write_csv(
        data_dir / "generated_episode_manifest.csv",
        vehicles,
        ["vehicle_id", "vtype", "depart", "route_edges"],
    )
    summary = {
        "scenario_id": "sumo_rl_training_scenario_v1",
        "episode_duration_sec": episode_duration_sec,
        "vehicle_count": len(vehicles),
        "seed": seed,
        "demand_scale": demand_scale,
        "route_counts": dict(sorted(route_counts.items())),
        "vtype_counts": dict(sorted(vtype_counts.items())),
        "uses_replay_control": False,
        "uses_move_to_xy": False,
        "uses_spatial_init": False,
        "traffic_signal_ids": tls_ids,
        "traffic_signal_control_ready": bool(tls_ids),
        "net_file": f"sumo/{net_filename}",
        "signalized_junction_id": signalize_junction_id,
    }
    write_json(data_dir / "demand_generation_summary.json", summary)
    (output_dir / "VERSION_LOCK.md").write_text(
        "\n".join(
            [
                "# SUMO-RL Training Scenario v1",
                "",
                "This package is generated from a real-video calibration profile.",
                "It uses clean statistical SUMO demand, not per-vehicle replay control.",
                "",
                f"- episode_duration_sec: `{episode_duration_sec:.2f}`",
                f"- vehicle_count: `{len(vehicles)}`",
                f"- demand_scale: `{demand_scale:.2f}`",
                f"- uses_replay_control: `False`",
                f"- uses_move_to_xy: `False`",
                f"- traffic_signal_control_ready: `{bool(tls_ids)}`",
                f"- net_file: `sumo/{net_filename}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--net-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episode-duration-sec", type=float, default=600.0)
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument("--demand-scale", type=float, default=0.5)
    parser.add_argument("--signalize-junction-id", default="")
    parser.add_argument("--netconvert-binary", default="/Users/zfh/.local/share/eclipse-sumo-venv/bin/netconvert")
    args = parser.parse_args()
    summary = build_package(
        Path(args.profile),
        Path(args.net_file),
        Path(args.output_dir),
        args.episode_duration_sec,
        args.seed,
        args.demand_scale,
        signalize_junction_id=args.signalize_junction_id,
        netconvert_binary=args.netconvert_binary,
    )
    print(f"vehicle_count={summary['vehicle_count']}")
    print(f"traffic_signal_control_ready={summary['traffic_signal_control_ready']}")
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
