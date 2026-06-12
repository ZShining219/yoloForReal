from pathlib import Path
import sys
import unittest

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT / "tools"))

from mot_target_extraction import (
    build_review_gate_rows,
    build_track_summary_rows,
    keep_track_for_review,
)


def det(track_id, time_sec, x1=100, y1=100, x2=150, y2=150, conf=0.8, cls="car"):
    return {
        "track_id": str(track_id),
        "frame_id": str(int(time_sec * 50)),
        "time_sec": f"{time_sec:.2f}",
        "class_name": cls,
        "confidence": str(conf),
        "x1": str(x1),
        "y1": str(y1),
        "x2": str(x2),
        "y2": str(y2),
    }


class MotTargetExtractionTest(unittest.TestCase):
    def test_build_track_summary_rows_counts_frames_and_confidence(self):
        rows = [
            det(7, 0.0, conf=0.6),
            det(7, 0.5, x1=140, x2=190, conf=0.8),
            det(7, 1.0, x1=180, x2=230, conf=1.0),
        ]

        summary = build_track_summary_rows(rows)[0]

        self.assertEqual(summary["track_id"], "mot_0007")
        self.assertEqual(summary["frame_count"], "3")
        self.assertEqual(summary["start_time"], "0.00")
        self.assertEqual(summary["end_time"], "1.00")
        self.assertEqual(summary["class_name"], "car")
        self.assertEqual(summary["mean_confidence"], "0.8000")
        self.assertGreater(float(summary["max_displacement_px"]), 70.0)

    def test_keep_track_for_review_rejects_short_single_sample(self):
        row = {
            "class_name": "car",
            "frame_count": "1",
            "duration_sec": "0.00",
            "mean_confidence": "0.90",
            "max_displacement_px": "0.00",
        }

        decision = keep_track_for_review(row)

        self.assertEqual(decision["track_status"], "excluded")
        self.assertEqual(decision["exclude_reason"], "single_sample_track")

    def test_keep_track_for_review_keeps_stable_vehicle_track(self):
        row = {
            "class_name": "car",
            "frame_count": "8",
            "duration_sec": "3.50",
            "mean_confidence": "0.50",
            "max_displacement_px": "90.00",
        }

        decision = keep_track_for_review(row)

        self.assertEqual(decision["track_status"], "target_review_candidate")
        self.assertEqual(decision["exclude_reason"], "")

    def test_build_review_gate_rows_uses_group_gate_language(self):
        summaries = [
            {
                "track_id": "mot_0001",
                "track_status": "target_review_candidate",
                "start_time": "0.00",
                "end_time": "2.00",
                "frame_count": "5",
            },
            {
                "track_id": "mot_0002",
                "track_status": "excluded",
                "start_time": "0.00",
                "end_time": "0.00",
                "frame_count": "1",
            },
        ]

        rows = build_review_gate_rows(summaries)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["review_unit"], "target_track_overlay")
        self.assertEqual(rows[0]["status"], "PENDING_RECALL_REVIEW")
        self.assertIn("do not add vehicles manually", rows[0]["next_action"])


if __name__ == "__main__":
    unittest.main()
