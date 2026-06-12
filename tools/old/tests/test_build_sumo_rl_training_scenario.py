from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT / "tools"))

from build_sumo_rl_training_scenario import (
    build_rl_configs,
    generate_training_vehicles,
    signalize_plain_nodes_xml,
    render_routes_xml,
    render_sumocfg,
)


class BuildSumoRlTrainingScenarioTest(unittest.TestCase):
    def test_generated_training_vehicles_use_statistical_routes_not_replay_ids(self):
        profile = {
            "source_window_sec": 30.0,
            "observed_vehicle_count": 3,
            "route_distribution": {
                "E2J J2W": {"count": 2, "share": 0.6667, "from_edge": "E2J", "to_edge": "J2W"},
                "S2J J2E": {"count": 1, "share": 0.3333, "from_edge": "S2J", "to_edge": "J2E"},
            },
            "vehicle_type_distribution": {
                "passenger": {"count": 3, "share": 1.0},
            },
        }

        vehicles = generate_training_vehicles(profile, episode_duration_sec=60.0, episode_id="ep001", min_vehicle_count=1)

        self.assertEqual(len(vehicles), 6)
        self.assertTrue(all(row["vehicle_id"].startswith("train_ep001_") for row in vehicles))
        self.assertFalse(any("cf_" in row["vehicle_id"] for row in vehicles))
        self.assertEqual(sum(1 for row in vehicles if row["route_edges"] == "E2J J2W"), 4)
        self.assertEqual(sum(1 for row in vehicles if row["route_edges"] == "S2J J2E"), 2)

    def test_demand_scale_reduces_generated_training_vehicle_count(self):
        profile = {
            "source_window_sec": 30.0,
            "observed_vehicle_count": 10,
            "route_distribution": {
                "E2J J2W": {"count": 10, "share": 1.0, "from_edge": "E2J", "to_edge": "J2W"},
            },
            "vehicle_type_distribution": {
                "passenger": {"count": 10, "share": 1.0},
            },
        }

        vehicles = generate_training_vehicles(profile, episode_duration_sec=60.0, episode_id="ep001", demand_scale=0.5, min_vehicle_count=1)

        self.assertEqual(len(vehicles), 10)

    def test_routes_xml_has_no_replay_or_spatial_control_attributes(self):
        vehicles = [
            {
                "vehicle_id": "train_ep001_0001",
                "vtype": "passenger",
                "depart": "10.00",
                "route_edges": "E2J J2W",
            }
        ]

        root = ET.fromstring(render_routes_xml(vehicles))
        vehicle = root.find("vehicle")

        self.assertEqual(vehicle.get("id"), "train_ep001_0001")
        self.assertIsNone(vehicle.get("departPos"))
        self.assertIsNone(vehicle.get("departSpeed"))
        self.assertIsNone(vehicle.find("stop"))
        self.assertEqual(vehicle.find("route").get("edges"), "E2J J2W")

    def test_sumocfg_references_training_route_file(self):
        root = ET.fromstring(render_sumocfg("real_scene.net.xml", "routes_train.rou.xml", 0.0, 60.0))

        self.assertEqual(root.find("./input/route-files").get("value"), "routes_train.rou.xml")
        self.assertEqual(root.find("./time/end").get("value"), "60.00")

    def test_signalize_plain_nodes_changes_only_requested_junction(self):
        xml_text = """<?xml version="1.0"?><nodes><node id="J" type="priority"/><node id="E" type="priority"/></nodes>"""

        updated = signalize_plain_nodes_xml(xml_text, "J")
        root = ET.fromstring(updated)
        by_id = {node.get("id"): node for node in root.findall("node")}

        self.assertEqual(by_id["J"].get("type"), "traffic_light")
        self.assertEqual(by_id["E"].get("type"), "priority")

    def test_rl_configs_are_signal_control_ready_when_tls_exists(self):
        configs = build_rl_configs({}, ["J"], 60.0, 0.5, net_file="sumo/real_scene_tls.net.xml")

        self.assertEqual(configs["env_config"]["traffic_signal_ids"], ["J"])
        self.assertEqual(configs["env_config"]["traffic_signal_control_ready"], True)
        self.assertEqual(configs["env_config"]["net_file"], "sumo/real_scene_tls.net.xml")
        self.assertEqual(configs["dqn_training_config"]["traffic_signal_control_ready"], True)


if __name__ == "__main__":
    unittest.main()
