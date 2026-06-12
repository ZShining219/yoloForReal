from pathlib import Path
import sys
import unittest

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT / "tools"))

from apply_duplicate_overlap_suppression import (
    duplicate_decision,
    find_duplicate_overlap_pairs,
    suppress_direction_for_duplicates,
    suppress_track_rows_for_duplicates,
)


def row(frame, logical_id, raw_id, x1, y1, x2, y2):
    return {
        "frame_id": str(frame),
        "time_sec": f"{frame / 50:.2f}",
        "logical_vehicle_id": logical_id,
        "track_id": raw_id,
        "x1": str(x1),
        "y1": str(y1),
        "x2": str(x2),
        "y2": str(y2),
    }


class ApplyDuplicateOverlapSuppressionTest(unittest.TestCase):
    def test_finds_persistent_high_iou_duplicate_pair(self):
        rows = []
        for frame in range(20):
            rows.append(row(frame, "cf_0001", "mot_0011", 0, 0, 10, 10))
            rows.append(row(frame, "cf_0013", "mot_0134", 0, 0, 10, 10))

        pairs = find_duplicate_overlap_pairs(rows, iou_threshold=0.85, min_overlap_frames=10)

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["logical_vehicle_id_a"], "cf_0001")
        self.assertEqual(pairs[0]["logical_vehicle_id_b"], "cf_0013")
        self.assertEqual(pairs[0]["overlap_frames"], "20")

    def test_marks_multi_duplicate_ready_track_as_contaminated(self):
        affected_counts = {"cf_0001": 3, "cf_0013": 1}
        direction_ready = {"cf_0001": "yes", "cf_0013": "no"}

        self.assertEqual(duplicate_decision("cf_0001", affected_counts, direction_ready), "CONTAMINATED_DUPLICATE")
        self.assertEqual(duplicate_decision("cf_0013", affected_counts, direction_ready), "DUPLICATE_SUPPRESSED")

    def test_suppresses_direction_for_contaminated_duplicate(self):
        direction_rows = [
            {"track_id": "cf_0001", "review_status": "ACCEPTED", "result_direction": "W_to_N", "confidence_level": "high"},
            {"track_id": "cf_0015", "review_status": "ACCEPTED", "result_direction": "S_to_E", "confidence_level": "high"},
        ]
        duplicate_rows = [
            {"logical_vehicle_id": "cf_0001", "duplicate_review_status": "CONTAMINATED_DUPLICATE"},
        ]

        suppressed = suppress_direction_for_duplicates(direction_rows, duplicate_rows)

        by_id = {row["track_id"]: row for row in suppressed}
        self.assertEqual(by_id["cf_0001"]["review_status"], "DUPLICATE_REVIEW")
        self.assertEqual(by_id["cf_0001"]["result_direction"], "unknown")
        self.assertEqual(by_id["cf_0015"]["review_status"], "ACCEPTED")

    def test_retains_contaminated_representative_for_video_review(self):
        track_rows = [
            row(1, "cf_0001", "mot_0011", 0, 0, 10, 10),
            row(1, "cf_0013", "mot_0134", 0, 0, 10, 10),
        ]
        duplicate_rows = [
            {"logical_vehicle_id": "cf_0001", "duplicate_review_status": "CONTAMINATED_DUPLICATE", "partners": "cf_0013"},
            {"logical_vehicle_id": "cf_0013", "duplicate_review_status": "DUPLICATE_SUPPRESSED", "partners": "cf_0001"},
        ]
        target_rows = [
            {"logical_vehicle_id": "cf_0001", "detected_frame_count": "1814"},
            {"logical_vehicle_id": "cf_0013", "detected_frame_count": "594"},
        ]

        output = suppress_track_rows_for_duplicates(track_rows, duplicate_rows, target_rows)

        self.assertEqual([row["logical_vehicle_id"] for row in output], ["cf_0001"])

    def test_retains_longest_representative_when_duplicate_pair_has_no_contaminated_track(self):
        track_rows = [
            row(1, "cf_0009", "mot_0009", 0, 0, 10, 10),
            row(1, "cf_0011", "mot_0028", 0, 0, 10, 10),
        ]
        duplicate_rows = [
            {"logical_vehicle_id": "cf_0009", "duplicate_review_status": "DUPLICATE_SUPPRESSED", "partners": "cf_0011"},
            {"logical_vehicle_id": "cf_0011", "duplicate_review_status": "DUPLICATE_SUPPRESSED", "partners": "cf_0009"},
        ]
        target_rows = [
            {"logical_vehicle_id": "cf_0009", "detected_frame_count": "330"},
            {"logical_vehicle_id": "cf_0011", "detected_frame_count": "68"},
        ]

        output = suppress_track_rows_for_duplicates(track_rows, duplicate_rows, target_rows)

        self.assertEqual([row["logical_vehicle_id"] for row in output], ["cf_0009"])


if __name__ == "__main__":
    unittest.main()
