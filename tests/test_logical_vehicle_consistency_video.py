from pathlib import Path
import sys
import unittest

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT / "tools"))

from build_logical_vehicle_id_video import final_color_map, final_detection_label, rows_by_frame


class LogicalVehicleConsistencyVideoTest(unittest.TestCase):
    def test_final_label_shows_only_logical_vehicle_id(self):
        row = {
            "logical_vehicle_id": "lv_0014",
            "raw_track_id": "mot_0384",
            "class_name": "car",
            "confidence": "0.8123",
        }

        self.assertEqual(final_detection_label(row), "lv_0014")

    def test_rows_by_frame_keeps_all_supplied_final_rows(self):
        rows = [
            {
                "frame_id": "1",
                "logical_vehicle_id": "lv_0001",
                "raw_track_id": "mot_0001",
                "association_status": "accepted",
                "final_gate_status": "AUTO_KEEP",
            },
            {
                "frame_id": "1",
                "logical_vehicle_id": "lv_0002",
                "raw_track_id": "mot_0002",
                "association_status": "accepted",
                "final_gate_status": "AUTO_EXCLUDE",
            },
        ]

        grouped = rows_by_frame(rows)

        self.assertEqual([row["logical_vehicle_id"] for row in grouped[1]], ["lv_0001", "lv_0002"])

    def test_final_color_map_assigns_unique_highlight_colors(self):
        rows = [
            {"logical_vehicle_id": "lv_0001"},
            {"logical_vehicle_id": "lv_0002"},
            {"logical_vehicle_id": "lv_0001"},
        ]

        colors = final_color_map(rows)

        self.assertEqual(set(colors), {"lv_0001", "lv_0002"})
        self.assertNotEqual(colors["lv_0001"], colors["lv_0002"])
        self.assertTrue(all(max(color) >= 180 for color in colors.values()))


if __name__ == "__main__":
    unittest.main()
