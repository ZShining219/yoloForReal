from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT / "tools"))

from build_sumo_candidate_package import (
    build_readiness_rows,
    lane_edge,
    lane_index,
    render_routes_xml,
)


class BuildSumoCandidatePackageTest(unittest.TestCase):
    def test_lane_edge_and_index_parse_sumo_lane_ids(self):
        self.assertEqual(lane_edge("S2J_2"), "S2J")
        self.assertEqual(lane_index("S2J_2"), "2")
        self.assertEqual(lane_edge("J2E_0"), "J2E")
        self.assertEqual(lane_index("J2E_0"), "0")

    def test_readiness_marks_red_light_waiting_queue_from_override(self):
        rows = build_readiness_rows(
            track_rows=[
                {"logical_vehicle_id": "cf_0010", "frame_id": "0", "time_sec": "0.00", "class_name": "car"},
                {"logical_vehicle_id": "cf_0010", "frame_id": "1499", "time_sec": "29.98", "class_name": "car"},
            ],
            assignment_rows=[
                {"track_id": "cf_0010", "in_lane_code": "N2J_1", "out_lane_code": "J2S_0", "result_direction": "N_to_S"}
            ],
            exclusion_rows=[],
            anchor_rows=[],
            override_rows=[
                {"track_id": "cf_0010", "initial_state_type": "red_light_waiting_queue", "depart_time_sec": "0.00", "note": "waiting at red light"}
            ],
        )

        row = rows[0]
        self.assertEqual(row["readiness_status"], "READY_WITH_INITIAL_STATE_REVIEW")
        self.assertEqual(row["initial_state_type"], "red_light_waiting_queue")
        self.assertEqual(row["depart_time_sec"], "0.00")
        self.assertEqual(row["depart_lane"], "1")
        self.assertEqual(row["arrival_lane"], "0")

    def test_readiness_marks_window_start_partial_exit(self):
        rows = build_readiness_rows(
            track_rows=[
                {"logical_vehicle_id": "cf_0008", "frame_id": "0", "time_sec": "0.00", "class_name": "car"},
                {"logical_vehicle_id": "cf_0008", "frame_id": "82", "time_sec": "1.64", "class_name": "car"},
            ],
            assignment_rows=[
                {"track_id": "cf_0008", "in_lane_code": "S2J_2", "out_lane_code": "J2E_0", "result_direction": "S_to_E"}
            ],
            exclusion_rows=[],
            anchor_rows=[],
            override_rows=[
                {"track_id": "cf_0008", "initial_state_type": "window_start_partial_exit", "depart_time_sec": "0.00", "note": "near east exit at window start"}
            ],
        )

        row = rows[0]
        self.assertEqual(row["readiness_status"], "READY_WITH_INITIAL_STATE_REVIEW")
        self.assertEqual(row["initial_state_type"], "window_start_partial_exit")
        self.assertEqual(row["route_edges"], "S2J J2E")

    def test_readiness_marks_excluded_tracks(self):
        rows = build_readiness_rows(
            track_rows=[{"logical_vehicle_id": "cf_0026", "frame_id": "1259", "time_sec": "25.18", "class_name": "car"}],
            assignment_rows=[],
            exclusion_rows=[{"track_id": "cf_0026", "exclude_reason": "user_marked_ignore"}],
            anchor_rows=[],
            override_rows=[],
        )

        self.assertEqual(rows[0]["readiness_status"], "EXCLUDED")
        self.assertEqual(rows[0]["exclude_reason"], "user_marked_ignore")

    def test_render_routes_xml_uses_lane_indices_and_route_edges(self):
        xml_text = render_routes_xml(
            [
                {
                    "track_id": "cf_0015",
                    "sumo_vehicle_id": "veh_cf_0015",
                    "vtype": "passenger",
                    "depart_time_sec": "11.76",
                    "depart_lane": "2",
                    "arrival_lane": "0",
                    "route_edges": "S2J J2E",
                    "readiness_status": "READY_STRONG",
                    "initial_state_type": "normal_crossing_anchor",
                }
            ]
        )
        root = ET.fromstring(xml_text)
        vehicle = root.find("vehicle")

        self.assertEqual(vehicle.get("depart"), "11.76")
        self.assertEqual(vehicle.get("departLane"), "2")
        self.assertEqual(vehicle.find("route").get("edges"), "S2J J2E")


if __name__ == "__main__":
    unittest.main()
