from pathlib import Path
import sys
import unittest

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT / "tools"))

from build_single_track_direction_evidence import Gate, TrackPoint
from build_warmup_direction_anchor import (
    build_warmup_anchor_row,
    movement_duration_priors,
    ray_segment_intersection,
)


class BuildWarmupDirectionAnchorTest(unittest.TestCase):
    def test_ray_segment_intersection_returns_distance_for_hit(self):
        hit = ray_segment_intersection((5, 5), (-1, 0), (0, 0), (0, 10))

        self.assertIsNotNone(hit)
        self.assertAlmostEqual(hit.distance_px, 5.0)
        self.assertAlmostEqual(hit.segment_position, 0.5)

    def test_builds_geometric_warmup_anchor_from_manual_route_and_observed_exit(self):
        gates = {
            "E": Gate("E", "east", 10, 0, 10, 10, "right", "left", "left_to_right", "right_to_left"),
            "W": Gate("W", "west", 0, 0, 0, 10, "right", "left", "right_to_left", "left_to_right"),
        }
        points = [
            TrackPoint("cf_test", 0, 0.0, 5.0, 5.0, 4, 4, 6, 5),
            TrackPoint("cf_test", 10, 0.2, 4.0, 5.0, 3, 4, 5, 5),
        ]
        crossings = [
            {"track_id": "cf_test", "gate_id": "W", "crossing_type": "exiting", "crossing_frame": "50", "crossing_time_sec": "1.00"}
        ]
        route = {"track_id": "cf_test", "origin_direction": "E", "destination_direction": "W", "route_source": "manual"}

        row = build_warmup_anchor_row(route, points, gates, crossings, duration_priors={}, fps=50.0)

        self.assertEqual(row["result_direction"], "E_to_W")
        self.assertEqual(row["observed_exit_direction"], "W")
        self.assertEqual(row["observed_exit_time_sec"], "1.00")
        self.assertEqual(row["entry_estimation_method"], "backward_ray_to_origin_gate")
        self.assertEqual(row["estimated_entry_time_sec"], "-1.00")

    def test_uses_same_movement_duration_prior_when_backward_ray_misses(self):
        gates = {
            "E": Gate("E", "east", 10, 0, 10, 10, "right", "left", "left_to_right", "right_to_left"),
            "N": Gate("N", "north", 0, 10, 10, 10, "left", "right", "right_to_left", "left_to_right"),
        }
        points = [
            TrackPoint("cf_turn", 0, 0.0, 5.0, 5.0, 4, 4, 6, 5),
            TrackPoint("cf_turn", 10, 0.2, 6.0, 5.0, 5, 4, 7, 5),
        ]
        crossings = [
            {"track_id": "cf_turn", "gate_id": "N", "crossing_type": "exiting", "crossing_frame": "100", "crossing_time_sec": "2.00"}
        ]
        route = {"track_id": "cf_turn", "origin_direction": "E", "destination_direction": "N", "route_source": "manual"}

        row = build_warmup_anchor_row(route, points, gates, crossings, duration_priors={"E_to_N": 4.0}, fps=50.0)

        self.assertEqual(row["entry_estimation_method"], "same_movement_duration_prior")
        self.assertEqual(row["estimated_entry_time_sec"], "-2.00")
        self.assertEqual(row["confidence_level"], "medium")

    def test_manual_entry_time_override_takes_priority_over_backward_ray(self):
        gates = {
            "E": Gate("E", "east", 10, 0, 10, 10, "right", "left", "left_to_right", "right_to_left"),
            "W": Gate("W", "west", 0, 0, 0, 10, "right", "left", "right_to_left", "left_to_right"),
        }
        points = [
            TrackPoint("cf_test", 0, 0.0, 5.0, 5.0, 4, 4, 6, 5),
            TrackPoint("cf_test", 10, 0.2, 4.0, 5.0, 3, 4, 5, 5),
        ]
        route = {
            "track_id": "cf_test",
            "origin_direction": "E",
            "destination_direction": "W",
            "route_source": "user_manual_alignment",
            "manual_estimated_entry_time_sec": "-6.00",
        }

        row = build_warmup_anchor_row(route, points, gates, crossings=[], duration_priors={}, fps=50.0)

        self.assertEqual(row["estimated_entry_time_sec"], "-6.00")
        self.assertEqual(row["estimated_entry_frame"], "-300.0")
        self.assertEqual(row["entry_estimation_method"], "manual_entry_time_override")
        self.assertEqual(row["confidence_level"], "high")
        self.assertIn("Manual entry time override", row["evidence_note"])

    def test_marks_destination_as_already_outside_when_frame0_is_on_destination_outside_side(self):
        gates = {
            "E": Gate("E", "east", 10, 0, 10, 10, "right", "left", "left_to_right", "right_to_left"),
            "W": Gate("W", "west", 0, 0, 0, 10, "right", "left", "right_to_left", "left_to_right"),
        }
        points = [
            TrackPoint("cf_test", 0, 0.0, -1.0, 5.0, -2, 4, 0, 5),
            TrackPoint("cf_test", 10, 0.2, -2.0, 5.0, -3, 4, -1, 5),
        ]
        route = {"track_id": "cf_test", "origin_direction": "E", "destination_direction": "W", "route_source": "manual"}

        row = build_warmup_anchor_row(route, points, gates, crossings=[], duration_priors={}, fps=50.0)

        self.assertEqual(row["observed_exit_direction"], "W")
        self.assertEqual(row["observed_exit_time_sec"], "0.00")
        self.assertEqual(row["exit_time_source"], "already_outside_at_window_start")

    def test_movement_duration_priors_extracts_complete_od_durations(self):
        od_rows = [
            {"result_direction": "E_to_N", "review_status": "ACCEPTED", "first_crossing_frame": "100", "last_crossing_frame": "300"},
            {"result_direction": "E_to_N", "review_status": "ACCEPTED", "first_crossing_frame": "200", "last_crossing_frame": "500"},
        ]

        priors = movement_duration_priors(od_rows, fps=50.0)

        self.assertEqual(priors["E_to_N"], 5.0)


if __name__ == "__main__":
    unittest.main()
