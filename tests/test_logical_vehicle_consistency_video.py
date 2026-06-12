from pathlib import Path
import sys
import unittest

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT / "tools"))

from build_logical_vehicle_id_video import detection_label, rows_by_frame


class LogicalVehicleConsistencyVideoTest(unittest.TestCase):
    def test_debug_label_shows_logical_and_raw_ids(self):
        row = {
            "logical_vehicle_id": "lv_0014",
            "raw_track_id": "mot_0384",
            "class_name": "car",
            "confidence": "0.8123",
        }

        self.assertEqual(detection_label(row, mode="debug"), "lv_0014/mot_0384 DEBUG car 0.81")

    def test_final_label_prioritizes_logical_id_only(self):
        row = {
            "logical_vehicle_id": "lv_0014",
            "raw_track_id": "mot_0384",
            "class_name": "car",
            "confidence": "0.8123",
        }

        self.assertEqual(detection_label(row, mode="final"), "lv_0014 FINAL car 0.81")

    def test_final_mode_hides_rows_not_kept_by_final_gate(self):
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

        grouped = rows_by_frame(rows, mode="final")

        self.assertEqual([row["logical_vehicle_id"] for row in grouped[1]], ["lv_0001"])


if __name__ == "__main__":
    unittest.main()
