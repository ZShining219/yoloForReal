from pathlib import Path
import sys
import unittest

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT / "tools"))

from build_logical_consistency_review_video import build_logical_statuses, compact_legend_box, overlay_label


class BuildLogicalConsistencyReviewVideoTest(unittest.TestCase):
    def test_build_logical_statuses_prioritizes_direction_ready(self):
        logical_targets = [
            {"logical_vehicle_id": "cf_0001", "raw_track_ids": "mot_0011", "raw_track_id_count": "1", "keep_for_direction_od": "yes"},
            {"logical_vehicle_id": "cf_0002", "raw_track_ids": "mot_0010", "raw_track_id_count": "1", "keep_for_direction_od": "no"},
        ]
        logical_od = [
            {"track_id": "cf_0001", "review_status": "ACCEPTED", "result_direction": "W_to_N"},
            {"track_id": "cf_0002", "review_status": "UNKNOWN", "result_direction": "unknown"},
        ]
        raw_quality = [
            {"track_id": "mot_0010", "static_or_parked_flag": "yes", "window_partial_status": ""},
        ]

        statuses = build_logical_statuses(logical_targets, logical_od, raw_quality)

        self.assertEqual(statuses["cf_0001"]["status"], "READY")
        self.assertEqual(statuses["cf_0001"]["result_direction"], "W_to_N")
        self.assertEqual(statuses["cf_0002"]["status"], "STATIC")

    def test_build_logical_statuses_marks_duplicate_review_tracks(self):
        logical_targets = [
            {"logical_vehicle_id": "cf_0001", "raw_track_ids": "mot_0011", "raw_track_id_count": "1", "keep_for_direction_od": "no"},
        ]
        logical_od = [
            {"track_id": "cf_0001", "review_status": "DUPLICATE_REVIEW", "result_direction": "unknown"},
        ]
        raw_quality = []

        statuses = build_logical_statuses(logical_targets, logical_od, raw_quality)

        self.assertEqual(statuses["cf_0001"]["status"], "DUPLICATE")

    def test_overlay_label_includes_logical_raw_status_and_direction(self):
        row = {"logical_vehicle_id": "cf_0001", "track_id": "mot_0011", "class_name": "car", "confidence": "0.8"}
        status = {"status": "READY", "result_direction": "W_to_N"}

        self.assertEqual(overlay_label(row, status), "cf_0001/mot_0011 READY W_to_N car 0.80")

    def test_compact_legend_uses_limited_vertical_space(self):
        box = compact_legend_box()

        self.assertLessEqual(box[3] - box[1], 72)


if __name__ == "__main__":
    unittest.main()
