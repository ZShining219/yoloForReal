from pathlib import Path
import sys
import unittest

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT / "tools"))

from apply_target_false_positive_gate import build_final_target_rows


def summary(track_id, status="target_review_candidate"):
    return {
        "track_id": track_id,
        "raw_track_id": str(int(track_id.split("_")[1])),
        "class_name": "car",
        "class_votes": '{"car": 3}',
        "start_time": "0.00",
        "end_time": "2.00",
        "duration_sec": "2.00",
        "frame_count": "3",
        "detection_count": "3",
        "mean_confidence": "0.8000",
        "max_displacement_px": "120.00",
        "track_status": status,
        "exclude_reason": "" if status == "target_review_candidate" else "single_sample_track",
    }


class ApplyTargetFalsePositiveGateTest(unittest.TestCase):
    def test_build_final_target_rows_keeps_only_review_accepted_tracks(self):
        summaries = [
            summary("mot_0001"),
            summary("mot_0002"),
            summary("mot_0003"),
            summary("mot_0004", status="excluded"),
        ]
        false_positive_gate = [
            {"track_id": "mot_0002", "review_status": "EXCLUDE"},
            {"track_id": "mot_0003", "review_status": "KEEP"},
        ]

        rows = build_final_target_rows(summaries, false_positive_gate)

        self.assertEqual([row["track_id"] for row in rows], ["mot_0001", "mot_0003"])
        self.assertEqual(rows[0]["target_final_status"], "FINAL_TARGET_TRACK")
        self.assertEqual(rows[0]["false_positive_gate_status"], "NOT_FLAGGED_FOR_FALSE_POSITIVE_REVIEW")
        self.assertEqual(rows[1]["false_positive_gate_status"], "KEEP")
        self.assertEqual({row["manual_annotation_added"] for row in rows}, {"no"})


if __name__ == "__main__":
    unittest.main()
