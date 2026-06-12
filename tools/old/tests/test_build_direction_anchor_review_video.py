from pathlib import Path
import sys
import unittest

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT / "tools"))

from build_direction_anchor_review_video import (
    anchor_event_labels,
    anchor_box_label,
    anchor_event_highlight,
    anchor_summary_label,
    build_combined_anchor_rows,
    build_manual_supplement_anchor_row,
    build_observed_od_anchor_row,
    build_anchor_statuses,
)


class BuildDirectionAnchorReviewVideoTest(unittest.TestCase):
    def test_anchor_summary_label_includes_direction_entry_exit_and_method(self):
        row = {
            "track_id": "cf_0002",
            "result_direction": "E_to_W",
            "estimated_entry_direction": "E",
            "estimated_entry_time_sec": "-6.00",
            "entry_estimation_method": "manual_entry_time_override",
            "observed_exit_direction": "W",
            "observed_exit_time_sec": "0.00",
            "exit_time_source": "already_outside_at_window_start",
            "confidence_level": "high",
        }

        label = anchor_summary_label(row)

        self.assertEqual(label, "cf_0002 E_to_W 入东@-6.00s 出西@0.00s manual high")

    def test_anchor_event_labels_show_window_start_warmup_entry(self):
        row = {
            "track_id": "cf_0002",
            "estimated_entry_direction": "E",
            "estimated_entry_frame": "-300.0",
            "estimated_entry_time_sec": "-6.00",
            "entry_estimation_method": "manual_entry_time_override",
            "observed_exit_direction": "W",
            "observed_exit_frame": "0",
            "observed_exit_time_sec": "0.00",
            "exit_time_source": "already_outside_at_window_start",
        }

        labels = anchor_event_labels(row, frame_id=0, hold_frames=120)

        self.assertIn("cf_0002 入东 @ -6.00s manual warmup", labels)
        self.assertIn("cf_0002 出西 @ 0.00s window-start", labels)

    def test_anchor_event_labels_show_observed_exit_near_crossing_frame(self):
        row = {
            "track_id": "cf_0003",
            "estimated_entry_direction": "E",
            "estimated_entry_frame": "-300.0",
            "estimated_entry_time_sec": "-6.00",
            "entry_estimation_method": "manual_entry_time_override",
            "observed_exit_direction": "W",
            "observed_exit_frame": "28",
            "observed_exit_time_sec": "0.56",
            "exit_time_source": "observed_gate_crossing",
        }

        self.assertIn("cf_0003 出西 @ 0.56s observed", anchor_event_labels(row, frame_id=28, hold_frames=120))
        self.assertEqual(anchor_event_labels(row, frame_id=200, hold_frames=120), [])

    def test_anchor_box_label_stays_next_to_each_vehicle_box(self):
        row = {
            "track_id": "cf_0003",
            "result_direction": "E_to_W",
            "estimated_entry_direction": "E",
            "estimated_entry_time_sec": "-6.00",
            "observed_exit_direction": "W",
            "observed_exit_time_sec": "0.56",
            "entry_estimation_method": "manual_entry_time_override",
        }

        self.assertEqual(anchor_box_label(row), "E_to_W | 入东 -6.00s | 出西 0.56s | manual")

    def test_anchor_box_label_marks_observed_complete_od(self):
        row = {
            "track_id": "cf_0015",
            "result_direction": "S_to_E",
            "estimated_entry_direction": "S",
            "estimated_entry_time_sec": "11.76",
            "observed_exit_direction": "E",
            "observed_exit_time_sec": "16.12",
            "entry_estimation_method": "observed_gate_crossing",
        }

        self.assertEqual(anchor_box_label(row), "S_to_E | 入南 11.76s | 出东 16.12s | observed")

    def test_anchor_box_label_marks_manual_review_supplement(self):
        row = {
            "track_id": "cf_0001",
            "result_direction": "W_to_N",
            "estimated_entry_direction": "W",
            "estimated_entry_time_sec": "21.96",
            "observed_exit_direction": "N",
            "observed_exit_time_sec": "34.44",
            "entry_estimation_method": "manual_review_supplement",
        }

        self.assertEqual(anchor_box_label(row), "W_to_N | 入西 21.96s | 出北 34.44s | review")

    def test_anchor_event_highlight_returns_current_time_and_event_text_near_crossing(self):
        row = {
            "track_id": "cf_0003",
            "estimated_entry_direction": "E",
            "estimated_entry_frame": "-300.0",
            "estimated_entry_time_sec": "-6.00",
            "entry_estimation_method": "manual_entry_time_override",
            "observed_exit_direction": "W",
            "observed_exit_frame": "28",
            "observed_exit_time_sec": "0.56",
            "exit_time_source": "observed_gate_crossing",
        }

        highlight = anchor_event_highlight(row, frame_id=28, fps=50.0, hold_frames=20)

        self.assertTrue(highlight["active"])
        self.assertEqual(highlight["label"], "当前0.56s | 出西 @ 0.56s observed")
        self.assertEqual(highlight["event_kind"], "exit")

    def test_anchor_event_highlight_marks_manual_supplement_entry_as_review(self):
        row = {
            "track_id": "cf_0001",
            "estimated_entry_direction": "W",
            "estimated_entry_frame": "1098.0",
            "estimated_entry_time_sec": "21.96",
            "entry_estimation_method": "manual_review_supplement",
            "observed_exit_direction": "N",
            "observed_exit_frame": "1722",
            "observed_exit_time_sec": "34.44",
            "exit_time_source": "outside_current_window_manual_review",
        }

        highlight = anchor_event_highlight(row, frame_id=1098, fps=50.0, hold_frames=45)

        self.assertTrue(highlight["active"])
        self.assertEqual(highlight["label"], "当前21.96s | 入西 @ 21.96s review")
        self.assertEqual(highlight["event_kind"], "entry")

    def test_build_observed_od_anchor_row_from_accepted_complete_od(self):
        od_row = {
            "track_id": "cf_0015",
            "origin_direction": "S",
            "destination_direction": "E",
            "result_direction": "S_to_E",
            "first_crossing_gate": "S",
            "first_crossing_frame": "588",
            "last_crossing_gate": "E",
            "last_crossing_frame": "806",
            "confidence_level": "high",
            "review_status": "ACCEPTED",
        }

        row = build_observed_od_anchor_row(od_row, fps=50.0)

        self.assertEqual(row["track_id"], "cf_0015")
        self.assertEqual(row["estimated_entry_direction"], "S")
        self.assertEqual(row["estimated_entry_time_sec"], "11.76")
        self.assertEqual(row["observed_exit_direction"], "E")
        self.assertEqual(row["observed_exit_time_sec"], "16.12")
        self.assertEqual(row["entry_estimation_method"], "observed_gate_crossing")

    def test_build_manual_supplement_anchor_row_for_cf0001_left_turn_waiting_area(self):
        supplement = {
            "track_id": "cf_0001",
            "origin_direction": "W",
            "destination_direction": "N",
            "result_direction": "W_to_N",
            "entry_frame": "1098",
            "exit_frame": "1722",
            "confidence_level": "review",
            "manual_note": "left-turn waiting area; exit is outside the 0-30s window",
        }

        row = build_manual_supplement_anchor_row(supplement, fps=50.0)

        self.assertEqual(row["track_id"], "cf_0001")
        self.assertEqual(row["estimated_entry_direction"], "W")
        self.assertEqual(row["estimated_entry_time_sec"], "21.96")
        self.assertEqual(row["observed_exit_direction"], "N")
        self.assertEqual(row["observed_exit_time_sec"], "34.44")
        self.assertEqual(row["entry_estimation_method"], "manual_review_supplement")
        self.assertIn("left-turn waiting area", row["evidence_note"])

    def test_build_combined_anchor_rows_adds_observed_od_without_overriding_warmup(self):
        warmup_rows = [
            {
                "track_id": "cf_0003",
                "result_direction": "E_to_W",
                "estimated_entry_direction": "E",
                "estimated_entry_time_sec": "-6.00",
            }
        ]
        od_rows = [
            {
                "track_id": "cf_0003",
                "review_status": "ACCEPTED",
                "result_direction": "E_to_W",
                "origin_direction": "E",
                "destination_direction": "W",
                "first_crossing_frame": "10",
                "last_crossing_frame": "20",
            },
            {
                "track_id": "cf_0015",
                "review_status": "ACCEPTED",
                "result_direction": "S_to_E",
                "origin_direction": "S",
                "destination_direction": "E",
                "first_crossing_frame": "588",
                "last_crossing_frame": "806",
            },
        ]

        rows = build_combined_anchor_rows(warmup_rows, od_rows, track_ids={"cf_0003", "cf_0015"}, fps=50.0)

        self.assertEqual([row["track_id"] for row in rows], ["cf_0003", "cf_0015"])
        self.assertEqual(rows[0]["estimated_entry_time_sec"], "-6.00")
        self.assertEqual(rows[1]["entry_estimation_method"], "observed_gate_crossing")

    def test_build_anchor_statuses_marks_high_confidence_direction_anchors(self):
        statuses = build_anchor_statuses(
            [
                {
                    "track_id": "cf_0002",
                    "result_direction": "E_to_W",
                    "confidence_level": "high",
                    "entry_estimation_method": "manual_entry_time_override",
                }
            ]
        )

        self.assertEqual(statuses["cf_0002"]["status"], "ANCHOR_HIGH")
        self.assertEqual(statuses["cf_0002"]["result_direction"], "E_to_W")


if __name__ == "__main__":
    unittest.main()
