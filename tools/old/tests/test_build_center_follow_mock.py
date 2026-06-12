from pathlib import Path
import sys
import unittest

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT / "tools"))

from build_center_follow_mock import build_center_follow_mock


def detection(frame_id, track_id, cx, cy=100.0, class_name="car"):
    return {
        "frame_id": str(frame_id),
        "time_sec": f"{frame_id / 50.0:.2f}",
        "track_id": str(track_id),
        "class_name": class_name,
        "confidence": "0.9000",
        "x1": f"{cx - 5:.2f}",
        "y1": f"{cy - 5:.2f}",
        "x2": f"{cx + 5:.2f}",
        "y2": f"{cy + 5:.2f}",
    }


class BuildCenterFollowMockTest(unittest.TestCase):
    def test_follows_center_motion_across_raw_id_change(self):
        detections = [
            detection(0, 1, 0),
            detection(1, 1, 5),
            detection(3, 2, 15),
            detection(4, 2, 20),
        ]

        outputs = build_center_follow_mock(
            detection_rows=detections,
            allowed_track_ids={"mot_0001", "mot_0002"},
            fps=50.0,
            max_gap_frames=3,
            max_prediction_distance_px=8.0,
        )

        logical_ids = {row["logical_vehicle_id"] for row in outputs.overlay_rows}
        self.assertEqual(len(logical_ids), 1)
        self.assertEqual(outputs.links[0]["from_raw_track_id"], "mot_0001")
        self.assertEqual(outputs.links[0]["to_raw_track_id"], "mot_0002")
        self.assertEqual(outputs.summary_rows[0]["raw_track_ids"], "mot_0001|mot_0002")

    def test_keeps_two_crossing_targets_separate_by_nearest_prediction(self):
        detections = [
            detection(0, 1, 0),
            detection(1, 1, 5),
            detection(0, 2, 100),
            detection(1, 2, 95),
            detection(3, 8, 15),
            detection(3, 9, 85),
        ]

        outputs = build_center_follow_mock(
            detection_rows=detections,
            allowed_track_ids={"mot_0001", "mot_0002", "mot_0008", "mot_0009"},
            fps=50.0,
            max_gap_frames=3,
            max_prediction_distance_px=8.0,
        )

        by_raw_id = {}
        for row in outputs.overlay_rows:
            by_raw_id.setdefault(row["track_id"], set()).add(row["logical_vehicle_id"])

        self.assertEqual(by_raw_id["mot_0001"], by_raw_id["mot_0008"])
        self.assertEqual(by_raw_id["mot_0002"], by_raw_id["mot_0009"])
        self.assertNotEqual(next(iter(by_raw_id["mot_0001"])), next(iter(by_raw_id["mot_0002"])))


if __name__ == "__main__":
    unittest.main()
