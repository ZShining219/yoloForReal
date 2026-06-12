from pathlib import Path
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT / "tools"))

from build_sumo_controlled_replay_package import (
    build_controlled_replay_rows,
    render_controlled_routes_xml,
    render_controlled_sumocfg,
)


class BuildSumoControlledReplayPackageTest(unittest.TestCase):
    def test_controlled_replay_shifts_negative_video_times_to_nonnegative_sumo_times(self):
        rows = build_controlled_replay_rows(
            readiness_rows=[
                {
                    "track_id": "cf_0002",
                    "vtype": "passenger",
                    "route_edges": "E2J J2W",
                    "from_edge": "E2J",
                    "depart_lane": "1",
                    "arrival_lane": "0",
                    "depart_time_sec": "-6.00",
                    "initial_state_type": "manual_warmup_anchor",
                    "readiness_status": "READY_WARMUP",
                },
                {
                    "track_id": "cf_0015",
                    "vtype": "passenger",
                    "route_edges": "S2J J2E",
                    "from_edge": "S2J",
                    "depart_lane": "2",
                    "arrival_lane": "0",
                    "depart_time_sec": "11.76",
                    "initial_state_type": "normal_crossing_anchor",
                    "readiness_status": "READY_STRONG",
                },
            ],
            lane_lengths={"E2J_1": 106.4, "S2J_2": 106.4},
            time_shift_sec=6.0,
            sim_end_sec=46.0,
        )

        by_id = {row["track_id"]: row for row in rows}
        self.assertEqual(by_id["cf_0002"]["sim_depart_time_sec"], "0.00")
        self.assertEqual(by_id["cf_0002"]["video_depart_time_sec"], "-6.00")
        self.assertEqual(by_id["cf_0015"]["sim_depart_time_sec"], "17.76")
        self.assertEqual(by_id["cf_0015"]["time_shift_sec"], "6.00")

    def test_red_light_waiting_queue_gets_hold_control_and_stop(self):
        rows = build_controlled_replay_rows(
            readiness_rows=[
                {
                    "track_id": "cf_0012",
                    "vtype": "passenger",
                    "route_edges": "S2J J2N",
                    "from_edge": "S2J",
                    "depart_lane": "1",
                    "arrival_lane": "1",
                    "depart_time_sec": "2.80",
                    "initial_state_type": "red_light_waiting_queue",
                    "readiness_status": "READY_WITH_INITIAL_STATE_REVIEW",
                }
            ],
            lane_lengths={"S2J_1": 106.4},
            time_shift_sec=6.0,
            sim_end_sec=46.0,
        )

        row = rows[0]
        self.assertEqual(row["control_mode"], "hold_until_window_end")
        self.assertEqual(row["sim_depart_time_sec"], "8.80")
        self.assertEqual(row["hold_until_sim_sec"], "46.00")
        self.assertEqual(row["hold_lane_id"], "S2J_1")

        root = ET.fromstring(render_controlled_routes_xml(rows))
        vehicle = root.find("vehicle")
        stop = vehicle.find("stop")
        self.assertEqual(vehicle.get("depart"), "8.80")
        self.assertEqual(vehicle.get("departSpeed"), "0")
        self.assertEqual(stop.get("lane"), "S2J_1")
        self.assertEqual(stop.get("until"), "46.00")

    def test_window_start_partial_exit_is_warmed_before_video_zero(self):
        rows = build_controlled_replay_rows(
            readiness_rows=[
                {
                    "track_id": "cf_0008",
                    "vtype": "passenger",
                    "route_edges": "S2J J2E",
                    "from_edge": "S2J",
                    "depart_lane": "2",
                    "arrival_lane": "0",
                    "depart_time_sec": "0.00",
                    "initial_state_type": "window_start_partial_exit",
                    "readiness_status": "READY_WITH_INITIAL_STATE_REVIEW",
                }
            ],
            lane_lengths={"S2J_2": 106.4},
            time_shift_sec=6.0,
            sim_end_sec=46.0,
        )

        self.assertEqual(rows[0]["control_mode"], "warmup_partial_exit")
        self.assertEqual(rows[0]["sim_depart_time_sec"], "0.00")
        self.assertEqual(rows[0]["video_depart_time_sec"], "-6.00")

    def test_sumocfg_starts_at_zero_even_when_video_window_is_shifted(self):
        root = ET.fromstring(render_controlled_sumocfg("real_scene.net.xml", "routes_controlled.rou.xml", 0.0, 46.0))

        self.assertEqual(root.find("./time/begin").get("value"), "0.00")
        self.assertEqual(root.find("./time/end").get("value"), "46.00")


if __name__ == "__main__":
    unittest.main()
