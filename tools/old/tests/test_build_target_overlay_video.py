from pathlib import Path
import sys
import unittest

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT / "tools"))

from build_target_overlay_video import (
    build_isolated_display_suppression,
    build_track_statuses,
    detection_label,
    filter_rows_for_mode,
    normalized_track_id,
    suppress_rows_for_display,
    trajectory_points_by_id,
)


class BuildTargetOverlayVideoTest(unittest.TestCase):
    def test_normalized_track_id_matches_module_id_format(self):
        self.assertEqual(normalized_track_id("7"), "mot_0007")
        self.assertEqual(normalized_track_id("7.0"), "mot_0007")

    def test_build_track_statuses_prioritizes_human_review_gate(self):
        statuses = build_track_statuses(
            summary_rows=[
                {"track_id": "mot_0001", "track_status": "target_review_candidate"},
                {"track_id": "mot_0002", "track_status": "target_review_candidate"},
                {"track_id": "mot_0003", "track_status": "target_review_candidate"},
                {"track_id": "mot_0004", "track_status": "excluded"},
            ],
            false_positive_rows=[
                {"track_id": "mot_0002", "review_status": "KEEP"},
                {"track_id": "mot_0003", "review_status": "EXCLUDE"},
            ],
            final_rows=[
                {"track_id": "mot_0001"},
                {"track_id": "mot_0002"},
            ],
        )

        self.assertEqual(statuses["mot_0001"], "final_target")
        self.assertEqual(statuses["mot_0002"], "kept_by_user_gate")
        self.assertEqual(statuses["mot_0003"], "excluded_by_user")
        self.assertEqual(statuses["mot_0004"], "summary_excluded")

    def test_filter_rows_for_mode_keeps_only_final_rows_in_final_mode(self):
        rows = [
            {"track_id": "1", "frame_id": "0"},
            {"track_id": "2", "frame_id": "0"},
            {"track_id": "3", "frame_id": "0"},
        ]
        final_ids = {"mot_0001", "mot_0003"}

        selected = filter_rows_for_mode(rows, mode="final", final_ids=final_ids)

        self.assertEqual([row["track_id"] for row in selected], ["1", "3"])
        self.assertEqual(filter_rows_for_mode(rows, mode="all", final_ids=final_ids), rows)

    def test_build_isolated_display_suppression_hides_short_isolated_segments_only(self):
        rows = [
            {"track_id": "1", "frame_id": "1", "x1": "10", "y1": "20", "x2": "30", "y2": "40", "confidence": "0.50", "class_name": "car"},
            {"track_id": "1", "frame_id": "2", "x1": "12", "y1": "22", "x2": "32", "y2": "42", "confidence": "0.70", "class_name": "car"},
            {"track_id": "1", "frame_id": "10", "x1": "14", "y1": "24", "x2": "34", "y2": "44", "confidence": "0.90", "class_name": "car"},
            {"track_id": "2", "frame_id": "20", "x1": "100", "y1": "120", "x2": "140", "y2": "160", "confidence": "0.80", "class_name": "car"},
            {"track_id": "2", "frame_id": "21", "x1": "101", "y1": "121", "x2": "141", "y2": "161", "confidence": "0.80", "class_name": "car"},
            {"track_id": "2", "frame_id": "22", "x1": "102", "y1": "122", "x2": "142", "y2": "162", "confidence": "0.80", "class_name": "car"},
            {"track_id": "2", "frame_id": "23", "x1": "103", "y1": "123", "x2": "143", "y2": "163", "confidence": "0.80", "class_name": "car"},
            {"track_id": "2", "frame_id": "24", "x1": "104", "y1": "124", "x2": "144", "y2": "164", "confidence": "0.80", "class_name": "car"},
            {"track_id": "2", "frame_id": "25", "x1": "105", "y1": "125", "x2": "145", "y2": "165", "confidence": "0.80", "class_name": "car"},
        ]

        suppression = build_isolated_display_suppression(
            rows,
            final_ids={"mot_0001", "mot_0002"},
            max_segment_frames=2,
            min_gap_frames=6,
            fps=50.0,
        )

        self.assertEqual(
            [(row["track_id"], row["start_frame"], row["end_frame"], row["frame_count"]) for row in suppression.segment_rows],
            [("mot_0001", "1", "2", "2"), ("mot_0001", "10", "10", "1")],
        )
        self.assertEqual(
            suppress_rows_for_display(rows, suppression.suppressed_frame_keys),
            rows[3:],
        )

    def test_detection_label_uses_logical_vehicle_id_when_available(self):
        row = {
            "track_id": "7",
            "logical_vehicle_id": "lv_0003",
            "class_name": "car",
            "confidence": "0.8123",
        }

        self.assertEqual(detection_label(row, "FINAL"), "lv_0003/mot_0007 FINAL car 0.81")

    def test_trajectory_points_by_id_uses_logical_vehicle_id_when_available(self):
        rows = [
            {"track_id": "1", "logical_vehicle_id": "lv_0001", "frame_id": "0", "x1": "0", "y1": "0", "x2": "10", "y2": "10"},
            {"track_id": "2", "logical_vehicle_id": "lv_0001", "frame_id": "1", "x1": "10", "y1": "0", "x2": "20", "y2": "10"},
            {"track_id": "3", "logical_vehicle_id": "lv_0002", "frame_id": "1", "x1": "100", "y1": "0", "x2": "110", "y2": "10"},
        ]

        points = trajectory_points_by_id(rows)

        self.assertEqual(points["lv_0001"], [(0, 5.0, 5.0), (1, 15.0, 5.0)])
        self.assertEqual(points["lv_0002"], [(1, 105.0, 5.0)])


if __name__ == "__main__":
    unittest.main()
