from pathlib import Path
import sys
import unittest

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT / "tools"))

from build_logical_vehicle_consistency_v1 import equal_frame_windows, logical_video_output_paths, logical_window_output_paths


class BuildLogicalVehicleConsistencyV1Test(unittest.TestCase):
    def test_declares_three_required_video_outputs(self):
        output_dir = Path("outputs/logical_vehicle_consistency_v1")

        paths = logical_video_output_paths(output_dir)

        self.assertEqual(
            {path.name for path in paths.values()},
            {
                "logical_vehicle_id_final.mp4",
                "logical_vehicle_id_debug.mp4",
                "logical_vehicle_id_review.mp4",
            },
        )
        self.assertEqual(set(paths), {"final", "debug", "review"})

    def test_declares_three_window_slice_final_outputs(self):
        output_dir = Path("outputs/logical_vehicle_consistency_v1")

        paths = logical_window_output_paths(output_dir, window_count=3)

        self.assertEqual(set(paths), {"window_01", "window_02", "window_03"})
        self.assertEqual(paths["window_01"], output_dir / "window_01" / "logical_vehicle_id_final.mp4")
        self.assertEqual(paths["window_03"], output_dir / "window_03" / "logical_vehicle_id_final.mp4")

    def test_equal_frame_windows_cover_full_clip_without_overlap(self):
        windows = equal_frame_windows(total_frames=2000, window_count=3)

        self.assertEqual(windows, [(0, 667), (667, 1334), (1334, 2000)])


if __name__ == "__main__":
    unittest.main()
