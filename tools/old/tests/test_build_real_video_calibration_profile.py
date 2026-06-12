from pathlib import Path
import sys
import unittest

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT / "tools"))

from build_real_video_calibration_profile import (
    build_arrival_profile_rows,
    build_calibration_profile,
    build_flow_by_approach_rows,
    build_turning_ratio_rows,
)


class BuildRealVideoCalibrationProfileTest(unittest.TestCase):
    def test_profile_keeps_evidence_confidence_layers_separate(self):
        rows = [
            {
                "track_id": "cf_0002",
                "readiness_status": "READY_WARMUP",
                "initial_state_type": "manual_warmup_anchor",
                "result_direction": "E_to_W",
                "from_edge": "E2J",
                "to_edge": "J2W",
                "route_edges": "E2J J2W",
                "depart_time_sec": "-6.00",
                "vtype": "passenger",
            },
            {
                "track_id": "cf_0010",
                "readiness_status": "READY_WITH_INITIAL_STATE_REVIEW",
                "initial_state_type": "red_light_waiting_queue",
                "result_direction": "N_to_S",
                "from_edge": "N2J",
                "to_edge": "J2S",
                "route_edges": "N2J J2S",
                "depart_time_sec": "0.00",
                "vtype": "passenger",
            },
            {
                "track_id": "cf_0015",
                "readiness_status": "READY_STRONG",
                "initial_state_type": "normal_crossing_anchor",
                "result_direction": "S_to_E",
                "from_edge": "S2J",
                "to_edge": "J2E",
                "route_edges": "S2J J2E",
                "depart_time_sec": "11.76",
                "vtype": "passenger",
            },
        ]

        profile = build_calibration_profile(rows, source_window_sec=30.0, bin_size_sec=10.0)

        self.assertEqual(profile["observed_vehicle_count"], 3)
        self.assertEqual(profile["source_window_sec"], 30.0)
        self.assertEqual(profile["evidence_status_counts"]["READY_WARMUP"], 1)
        self.assertEqual(profile["initial_state_counts"]["red_light_waiting_queue"], 1)
        self.assertEqual(profile["training_policy"]["uses_replay_control"], False)
        self.assertEqual(profile["training_policy"]["uses_move_to_xy"], False)
        self.assertEqual(profile["route_distribution"]["S2J J2E"]["count"], 1)

    def test_flow_turning_and_arrival_rows_are_rate_or_ratio_outputs(self):
        rows = [
            {"track_id": "a", "from_edge": "E2J", "route_edges": "E2J J2W", "result_direction": "E_to_W", "depart_time_sec": "-6.00", "vtype": "passenger", "readiness_status": "READY_WARMUP", "initial_state_type": "manual_warmup_anchor"},
            {"track_id": "b", "from_edge": "E2J", "route_edges": "E2J J2N", "result_direction": "E_to_N", "depart_time_sec": "12.00", "vtype": "bus", "readiness_status": "READY_STRONG", "initial_state_type": "normal_crossing_anchor"},
            {"track_id": "c", "from_edge": "S2J", "route_edges": "S2J J2N", "result_direction": "S_to_N", "depart_time_sec": "21.00", "vtype": "passenger", "readiness_status": "READY_WITH_INITIAL_STATE_REVIEW", "initial_state_type": "red_light_waiting_queue"},
        ]

        flow_rows = build_flow_by_approach_rows(rows, source_window_sec=30.0)
        turn_rows = build_turning_ratio_rows(rows)
        arrival_rows = build_arrival_profile_rows(rows, bin_size_sec=10.0, source_window_sec=30.0)

        self.assertEqual({row["from_edge"]: row["vehicle_count"] for row in flow_rows}, {"E2J": "2", "S2J": "1"})
        self.assertEqual({row["result_direction"]: row["share"] for row in turn_rows}, {"E_to_N": "0.3333", "E_to_W": "0.3333", "S_to_N": "0.3333"})
        self.assertEqual([row["vehicle_count"] for row in arrival_rows], ["1", "1", "1"])


if __name__ == "__main__":
    unittest.main()
