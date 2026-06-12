from pathlib import Path
import sys
import unittest

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT / "tools"))

from build_direction_ready_v1 import (
    build_quality_review_row,
    link_decision,
    partial_window_status,
    review_case_groups,
    selected_track_rows,
    should_keep_for_direction_od,
)


class BuildDirectionReadyV1Test(unittest.TestCase):
    def test_static_track_without_crossing_is_excluded_from_direction_od(self):
        final_row = {
            "track_id": "mot_0010",
            "duration_sec": "39.98",
            "max_displacement_px": "4.12",
            "class_name": "car",
        }
        od_row = {
            "crossing_count": "0",
            "review_status": "UNKNOWN",
            "result_direction": "unknown",
        }

        review = build_quality_review_row(final_row, od_row)

        self.assertEqual(review["static_or_parked_flag"], "yes")
        self.assertEqual(review["keep_for_direction_od"], "no")
        self.assertEqual(review["exclude_reason"], "static_or_parked_without_crossing")

    def test_moving_track_with_accepted_od_is_kept_pending_motor_review(self):
        final_row = {
            "track_id": "mot_0189",
            "duration_sec": "8.92",
            "max_displacement_px": "1053.19",
            "class_name": "car",
        }
        od_row = {
            "crossing_count": "2",
            "review_status": "ACCEPTED",
            "result_direction": "S_to_E",
        }

        review = build_quality_review_row(final_row, od_row)

        self.assertEqual(review["static_or_parked_flag"], "no")
        self.assertEqual(review["motor_vehicle_review_status"], "PENDING_MOTOR_REVIEW")
        self.assertEqual(review["keep_for_direction_od"], "yes")

    def test_low_risk_cross_raw_id_link_is_auto_accepted_for_review_layer(self):
        link = {
            "from_raw_track_id": "mot_0384",
            "to_raw_track_id": "mot_0389",
            "gap_frames": "6",
            "predicted_distance_px": "0.59",
            "size_ratio": "1.01",
        }

        self.assertEqual(link_decision(link), "ACCEPT_LOW_RISK_CONTINUITY")

    def test_keep_for_direction_od_requires_quality_yes_and_motor_not_rejected(self):
        self.assertTrue(should_keep_for_direction_od("yes", "PENDING_MOTOR_REVIEW"))
        self.assertFalse(should_keep_for_direction_od("yes", "NON_MOTOR_REVIEW_EXCLUDE"))
        self.assertFalse(should_keep_for_direction_od("no", "PENDING_MOTOR_REVIEW"))

    def test_window_start_track_with_only_exit_is_partial_not_failed(self):
        final_row = {
            "track_id": "mot_0002",
            "start_frame": "0",
            "end_frame": "330",
            "duration_sec": "6.60",
            "max_displacement_px": "752.78",
            "class_name": "car",
        }
        od_row = {
            "crossing_count": "1",
            "review_status": "UNKNOWN",
            "first_crossing_gate": "",
            "last_crossing_gate": "E",
        }

        self.assertEqual(partial_window_status(final_row, od_row, video_last_frame=1999), "WINDOW_START_PARTIAL_EXIT_ONLY")

    def test_window_end_track_with_only_entry_is_partial_not_failed(self):
        final_row = {
            "track_id": "mot_0621",
            "start_frame": "1710",
            "end_frame": "1999",
            "duration_sec": "5.78",
            "max_displacement_px": "162.99",
            "class_name": "car",
        }
        od_row = {
            "crossing_count": "1",
            "review_status": "UNKNOWN",
            "first_crossing_gate": "E",
            "last_crossing_gate": "",
        }

        self.assertEqual(partial_window_status(final_row, od_row, video_last_frame=1999), "WINDOW_END_PARTIAL_ENTRY_ONLY")

    def test_selected_track_rows_returns_first_middle_last(self):
        rows = [
            {"frame_id": "10"},
            {"frame_id": "20"},
            {"frame_id": "30"},
            {"frame_id": "40"},
            {"frame_id": "50"},
        ]

        selected = selected_track_rows(rows)

        self.assertEqual([row["frame_id"] for row in selected], ["10", "30", "50"])

    def test_review_case_groups_prioritizes_visual_review_categories(self):
        quality_rows = [
            {"track_id": "mot_0010", "static_or_parked_flag": "yes", "window_partial_status": "", "motor_vehicle_review_status": "PENDING_MOTOR_REVIEW", "keep_for_direction_od": "no"},
            {"track_id": "mot_0002", "static_or_parked_flag": "no", "window_partial_status": "WINDOW_START_PARTIAL_EXIT_ONLY", "motor_vehicle_review_status": "PENDING_MOTOR_REVIEW", "keep_for_direction_od": "no"},
            {"track_id": "mot_0189", "static_or_parked_flag": "no", "window_partial_status": "", "motor_vehicle_review_status": "PENDING_MOTOR_REVIEW", "keep_for_direction_od": "yes"},
        ]
        link_rows = [
            {"from_raw_track_id": "mot_0384", "to_raw_track_id": "mot_0389", "review_decision": "ACCEPT_LOW_RISK_CONTINUITY"},
            {"from_raw_track_id": "mot_0011", "to_raw_track_id": "mot_0011", "review_decision": "ACCEPT_SAME_RAW_GAP"},
        ]

        groups = review_case_groups(quality_rows, link_rows)

        self.assertEqual([row["track_id"] for row in groups["static_or_parked"]], ["mot_0010"])
        self.assertEqual([row["track_id"] for row in groups["window_partial"]], ["mot_0002"])
        self.assertEqual([row["track_id"] for row in groups["motor_vehicle"]], ["mot_0189"])
        self.assertEqual(len(groups["link_candidates"]), 1)


if __name__ == "__main__":
    unittest.main()
