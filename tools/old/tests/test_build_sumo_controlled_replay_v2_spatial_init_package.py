from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT / "tools"))

from build_sumo_controlled_replay_v2_spatial_init_package import (
    build_replay_v2_rows,
    build_spatial_initialization_rows,
    build_traceability_rows,
    controller_invocation_path,
    render_routes_xml,
)


class BuildSumoControlledReplayV2SpatialInitPackageTest(unittest.TestCase):
    def test_route_replay_depart_is_advanced_to_match_gate_anchor(self):
        rows = build_replay_v2_rows(
            readiness_rows=[
                {
                    "track_id": "cf_0015",
                    "vtype": "passenger",
                    "route_edges": "S2J J2E",
                    "from_edge": "S2J",
                    "to_edge": "J2E",
                    "depart_lane": "2",
                    "arrival_lane": "0",
                    "depart_time_sec": "11.76",
                    "entry_time_sec": "11.76",
                    "exit_time_sec": "16.12",
                    "initial_state_type": "normal_crossing_anchor",
                    "readiness_status": "READY_STRONG",
                    "result_direction": "S_to_E",
                    "anchor_source": "observed_complete_od",
                }
            ],
            lane_lengths={"S2J_2": 106.4},
            time_shift_sec=6.0,
            sim_end_sec=46.0,
        )

        row = rows[0]
        self.assertEqual(row["control_mode"], "corrected_route_replay")
        self.assertEqual(row["video_gate_entry_time_sec"], "11.76")
        self.assertEqual(row["incoming_travel_time_sec"], "7.66")
        self.assertEqual(row["sim_depart_time_sec"], "10.10")
        self.assertEqual(row["timing_alignment_status"], "GATE_ALIGNED_DEPART")

    def test_warmup_rows_are_spatially_initialized_at_video_zero(self):
        rows = build_replay_v2_rows(
            readiness_rows=[
                {
                    "track_id": "cf_0002",
                    "vtype": "passenger",
                    "route_edges": "E2J J2W",
                    "from_edge": "E2J",
                    "to_edge": "J2W",
                    "depart_lane": "1",
                    "arrival_lane": "0",
                    "depart_time_sec": "-6.00",
                    "entry_time_sec": "-6.00",
                    "exit_time_sec": "0.00",
                    "initial_state_type": "manual_warmup_anchor",
                    "readiness_status": "READY_WARMUP",
                    "result_direction": "E_to_W",
                    "anchor_source": "user_manual_alignment",
                }
            ],
            lane_lengths={"E2J_1": 106.4},
            time_shift_sec=6.0,
            sim_end_sec=46.0,
        )

        row = rows[0]
        self.assertEqual(row["control_mode"], "spatial_init_at_video_zero")
        self.assertEqual(row["sim_depart_time_sec"], "0.00")
        self.assertEqual(row["spatial_init_lane_id"], "E2J_1")
        self.assertEqual(row["spatial_init_pos_m"], "105.40")
        self.assertEqual(row["timing_alignment_status"], "SPATIAL_INIT_REQUIRED")

        spatial_rows = build_spatial_initialization_rows(rows)
        self.assertEqual(len(spatial_rows), 1)
        self.assertEqual(spatial_rows[0]["sumo_vehicle_id"], "veh_cf_0002")
        self.assertEqual(spatial_rows[0]["init_method"], "moveTo_lane_end_at_video_zero")

    def test_waiting_queue_keeps_hold_control(self):
        rows = build_replay_v2_rows(
            readiness_rows=[
                {
                    "track_id": "cf_0012",
                    "vtype": "passenger",
                    "route_edges": "S2J J2N",
                    "from_edge": "S2J",
                    "to_edge": "J2N",
                    "depart_lane": "1",
                    "arrival_lane": "1",
                    "depart_time_sec": "2.80",
                    "initial_state_type": "red_light_waiting_queue",
                    "readiness_status": "READY_WITH_INITIAL_STATE_REVIEW",
                    "result_direction": "S_to_N",
                }
            ],
            lane_lengths={"S2J_1": 106.4},
            time_shift_sec=6.0,
            sim_end_sec=46.0,
        )

        row = rows[0]
        self.assertEqual(row["control_mode"], "hold_until_window_end")
        self.assertEqual(row["sim_depart_time_sec"], "8.80")
        self.assertEqual(row["hold_lane_id"], "S2J_1")
        self.assertEqual(row["hold_until_sim_sec"], "46.00")

    def test_traceability_links_video_vehicle_to_sumo_vehicle(self):
        replay_rows = build_replay_v2_rows(
            readiness_rows=[
                {
                    "track_id": "cf_0025",
                    "vtype": "passenger",
                    "route_edges": "E2J J2W",
                    "from_edge": "E2J",
                    "to_edge": "J2W",
                    "depart_lane": "0",
                    "arrival_lane": "0",
                    "depart_time_sec": "25.06",
                    "entry_time_sec": "25.06",
                    "exit_time_sec": "29.48",
                    "initial_state_type": "normal_crossing_anchor",
                    "readiness_status": "READY_STRONG",
                    "result_direction": "E_to_W",
                    "anchor_source": "observed_complete_od",
                }
            ],
            lane_lengths={"E2J_0": 106.4},
            time_shift_sec=6.0,
            sim_end_sec=46.0,
        )
        trace_rows = build_traceability_rows(replay_rows)

        self.assertEqual(trace_rows[0]["logical_vehicle_id"], "cf_0025")
        self.assertEqual(trace_rows[0]["sumo_vehicle_id"], "veh_cf_0025")
        self.assertEqual(trace_rows[0]["video_sumo_time_mapping"], "video_time=sim_time-6.00")
        self.assertEqual(trace_rows[0]["route_edges"], "E2J J2W")

    def test_routes_xml_contains_depart_position_for_spatial_init_and_stop_for_hold(self):
        rows = build_replay_v2_rows(
            readiness_rows=[
                {
                    "track_id": "cf_0002",
                    "vtype": "passenger",
                    "route_edges": "E2J J2W",
                    "from_edge": "E2J",
                    "to_edge": "J2W",
                    "depart_lane": "1",
                    "arrival_lane": "0",
                    "depart_time_sec": "-6.00",
                    "entry_time_sec": "-6.00",
                    "exit_time_sec": "0.00",
                    "initial_state_type": "manual_warmup_anchor",
                    "readiness_status": "READY_WARMUP",
                    "result_direction": "E_to_W",
                },
                {
                    "track_id": "cf_0012",
                    "vtype": "passenger",
                    "route_edges": "S2J J2N",
                    "from_edge": "S2J",
                    "to_edge": "J2N",
                    "depart_lane": "1",
                    "arrival_lane": "1",
                    "depart_time_sec": "2.80",
                    "initial_state_type": "red_light_waiting_queue",
                    "readiness_status": "READY_WITH_INITIAL_STATE_REVIEW",
                    "result_direction": "S_to_N",
                },
            ],
            lane_lengths={"E2J_1": 106.4, "S2J_1": 106.4},
            time_shift_sec=6.0,
            sim_end_sec=46.0,
        )

        root = ET.fromstring(render_routes_xml(rows))
        vehicles = {vehicle.get("id"): vehicle for vehicle in root.findall("vehicle")}
        self.assertEqual(vehicles["veh_cf_0002"].get("depart"), "0.00")
        self.assertEqual(vehicles["veh_cf_0002"].get("departPos"), "105.40")
        self.assertEqual(vehicles["veh_cf_0012"].find("stop").get("until"), "46.00")

    def test_controller_invocation_uses_filename_when_running_inside_sumo_dir(self):
        path = Path("outputs/version/sumo/run_controlled_replay_v2.py")

        self.assertEqual(controller_invocation_path(path), "run_controlled_replay_v2.py")


if __name__ == "__main__":
    unittest.main()
