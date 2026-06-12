from pathlib import Path
import sys
import unittest

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT / "tools"))

from build_single_track_direction_evidence import (
    Gate,
    TrackPoint,
    build_direction_result,
    detect_gate_crossings,
    load_direction_semantics,
    side_label,
)


class BuildSingleTrackDirectionEvidenceTest(unittest.TestCase):
    def test_side_label_uses_mapping_signed_side_rule(self):
        gate = Gate(
            approach_id="S",
            approach_name="south",
            x1=0.0,
            y1=0.0,
            x2=10.0,
            y2=0.0,
            intersection_side="right",
            approach_outside_side="left",
            entering_transition="left_to_right",
            exiting_transition="right_to_left",
        )

        self.assertEqual(side_label(gate, 5.0, 1.0), "left")
        self.assertEqual(side_label(gate, 5.0, -1.0), "right")

    def test_detects_entering_crossing_after_stable_side_change(self):
        gate = Gate(
            approach_id="S",
            approach_name="south",
            x1=0.0,
            y1=0.0,
            x2=10.0,
            y2=0.0,
            intersection_side="right",
            approach_outside_side="left",
            entering_transition="left_to_right",
            exiting_transition="right_to_left",
        )
        points = [
            TrackPoint("mot_0001", 0, 0.00, 5.0, 5.0, 0, 0, 0, 0),
            TrackPoint("mot_0001", 1, 0.02, 5.0, 4.0, 0, 0, 0, 0),
            TrackPoint("mot_0001", 2, 0.04, 5.0, -1.0, 0, 0, 0, 0),
            TrackPoint("mot_0001", 3, 0.06, 5.0, -2.0, 0, 0, 0, 0),
        ]

        crossings = detect_gate_crossings(points, [gate], stable_frames=2)

        self.assertEqual(len(crossings), 1)
        self.assertEqual(crossings[0].gate_id, "S")
        self.assertEqual(crossings[0].crossing_type, "entering")
        self.assertEqual(crossings[0].crossing_frame, 2)

    def test_builds_direction_result_from_first_entering_and_last_exiting(self):
        semantics = load_direction_semantics(
            [
                {
                    "first_crossing_approach_id": "S",
                    "last_crossing_approach_id": "E",
                    "origin_direction": "S",
                    "destination_direction": "E",
                    "result_direction": "S_to_E",
                }
            ]
        )

        result = build_direction_result(
            track_id="mot_0001",
            crossings=[
                {
                    "track_id": "mot_0001",
                    "gate_id": "S",
                    "crossing_type": "entering",
                    "crossing_frame": "10",
                    "crossing_time_sec": "0.20",
                },
                {
                    "track_id": "mot_0001",
                    "gate_id": "E",
                    "crossing_type": "exiting",
                    "crossing_frame": "30",
                    "crossing_time_sec": "0.60",
                },
            ],
            semantics=semantics,
        )

        self.assertEqual(result["origin_direction"], "S")
        self.assertEqual(result["destination_direction"], "E")
        self.assertEqual(result["result_direction"], "S_to_E")
        self.assertEqual(result["confidence_level"], "high")
        self.assertEqual(result["review_status"], "ACCEPTED")


if __name__ == "__main__":
    unittest.main()
