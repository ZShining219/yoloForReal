from pathlib import Path
import sys
import unittest

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT / "tools"))

from build_target_false_positive_review import overlay_frame_name, select_overlay_aligned_rows


def det(frame_id, candidate_id):
    return {
        "frame_id": str(frame_id),
        "time_sec": f"{frame_id / 50:.2f}",
        "candidate_id": candidate_id,
        "x1": "100",
        "y1": "100",
        "x2": "150",
        "y2": "150",
    }


class TargetFalsePositiveReviewTest(unittest.TestCase):
    def test_overlay_frame_name_matches_target_overlay_naming(self):
        self.assertEqual(overlay_frame_name(25), "target_track_f000025.jpg")
        self.assertEqual(overlay_frame_name("125"), "target_track_f000125.jpg")

    def test_select_overlay_aligned_rows_prefers_existing_overlay_frames(self):
        rows = [
            det(1, "skip_1"),
            det(25, "first"),
            det(50, "middle"),
            det(75, "last"),
            det(76, "skip_76"),
        ]

        selected = select_overlay_aligned_rows(rows, overlay_step=25)

        self.assertEqual([row["candidate_id"] for row in selected], ["first", "middle", "last"])

    def test_select_overlay_aligned_rows_falls_back_when_no_overlay_rows_exist(self):
        rows = [det(1, "first"), det(2, "middle"), det(3, "last")]

        selected = select_overlay_aligned_rows(rows, overlay_step=25)

        self.assertEqual([row["candidate_id"] for row in selected], ["first", "middle", "last"])


if __name__ == "__main__":
    unittest.main()
