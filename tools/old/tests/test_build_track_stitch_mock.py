from pathlib import Path
import sys
import unittest

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT / "tools"))

from build_track_stitch_mock import build_track_stitch_mock, normalized_track_id


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


class BuildTrackStitchMockTest(unittest.TestCase):
    def test_normalized_track_id_accepts_prefixed_and_numeric_ids(self):
        self.assertEqual(normalized_track_id("7"), "mot_0007")
        self.assertEqual(normalized_track_id("7.0"), "mot_0007")
        self.assertEqual(normalized_track_id("mot_0007"), "mot_0007")

    def test_stitches_short_gap_motion_continuity_and_interpolates_missing_points(self):
        detections = [
            detection(0, 1, 0),
            detection(1, 1, 5),
            detection(2, 1, 10),
            detection(5, 2, 25),
            detection(6, 2, 30),
        ]

        outputs = build_track_stitch_mock(
            detection_rows=detections,
            final_ids={"mot_0001", "mot_0002"},
            fps=50.0,
            max_gap_frames=5,
            max_link_distance_px=12.0,
            max_speed_change_ratio=2.0,
        )

        logical_by_track = {
            row["raw_track_id"]: row["logical_vehicle_id"]
            for row in outputs.logical_tracks
            if row["source"] == "detected"
        }
        self.assertEqual(logical_by_track["mot_0001"], logical_by_track["mot_0002"])
        self.assertEqual(len(outputs.links), 1)
        self.assertEqual(outputs.links[0]["from_track_id"], "mot_0001")
        self.assertEqual(outputs.links[0]["to_track_id"], "mot_0002")
        self.assertEqual(
            [(row["frame_id"], row["center_x"]) for row in outputs.interpolated_points],
            [("3", "15.00"), ("4", "20.00")],
        )

    def test_rejects_candidate_with_large_motion_jump(self):
        detections = [
            detection(0, 1, 0),
            detection(1, 1, 5),
            detection(2, 1, 10),
            detection(4, 3, 200),
            detection(5, 3, 205),
        ]

        outputs = build_track_stitch_mock(
            detection_rows=detections,
            final_ids={"mot_0001", "mot_0003"},
            fps=50.0,
            max_gap_frames=5,
            max_link_distance_px=12.0,
            max_speed_change_ratio=2.0,
        )

        logical_by_track = {
            row["raw_track_id"]: row["logical_vehicle_id"]
            for row in outputs.logical_tracks
            if row["source"] == "detected"
        }
        self.assertNotEqual(logical_by_track["mot_0001"], logical_by_track["mot_0003"])
        self.assertEqual(outputs.links, [])
        self.assertEqual(outputs.rejections[0]["reject_reason"], "predicted_distance_too_large")

    def test_rejects_ambiguous_cross_id_swap(self):
        detections = [
            detection(0, 1, 0),
            detection(1, 1, 5),
            detection(0, 2, 100),
            detection(1, 2, 95),
            detection(3, 1, 85),
            detection(4, 1, 80),
            detection(3, 2, 15),
            detection(4, 2, 20),
        ]

        outputs = build_track_stitch_mock(
            detection_rows=detections,
            final_ids={"mot_0001", "mot_0002"},
            fps=50.0,
            max_gap_frames=5,
            max_link_distance_px=12.0,
            max_speed_change_ratio=2.0,
        )

        self.assertEqual(outputs.links, [])
        self.assertIn("ambiguous_cross_id_swap", {row["reject_reason"] for row in outputs.rejections})


if __name__ == "__main__":
    unittest.main()
