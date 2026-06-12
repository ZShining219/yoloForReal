from pathlib import Path
import sys
import unittest

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT / "tools"))

from apply_manual_track_exclusions import build_manual_filter_outputs, normalize_exclusion_ids


def summary(track_id, status="target_review_candidate"):
    return {
        "track_id": track_id,
        "raw_track_id": str(int(track_id.split("_")[1])),
        "class_name": "car",
        "class_votes": '{"car": 10}',
        "start_time": "0.00",
        "end_time": "2.00",
        "duration_sec": "2.00",
        "frame_count": "10",
        "detection_count": "10",
        "mean_confidence": "0.8000",
        "max_displacement_px": "100.00",
        "track_status": status,
        "exclude_reason": "" if status == "target_review_candidate" else "too_short_for_target_review",
    }


class ApplyManualTrackExclusionsTest(unittest.TestCase):
    def test_normalize_exclusion_ids_accepts_short_and_prefixed_ids(self):
        self.assertEqual(
            normalize_exclusion_ids(["0013", "13", "mot_0080", "80.0"]),
            ["mot_0013", "mot_0013", "mot_0080", "mot_0080"],
        )

    def test_build_manual_filter_outputs_excludes_review_candidates_and_records_auto_excluded_ids(self):
        summaries = [
            summary("mot_0001"),
            summary("mot_0002"),
            summary("mot_0003", status="excluded"),
        ]

        outputs = build_manual_filter_outputs(
            target_summary_rows=summaries,
            exclusion_ids=["0002", "0003", "0099"],
            version="manual_filter_v1",
            note="user pass",
        )

        self.assertEqual([row["track_id"] for row in outputs.final_rows], ["mot_0001"])
        self.assertEqual(outputs.final_rows[0]["manual_filter_status"], "KEPT_AFTER_MANUAL_FILTER_V1")
        self.assertEqual(outputs.final_rows[0]["manual_annotation_added"], "no")

        gate_by_id = {row["track_id"]: row for row in outputs.gate_rows}
        self.assertEqual(gate_by_id["mot_0002"]["manual_filter_action"], "EXCLUDE_FROM_FINAL")
        self.assertEqual(gate_by_id["mot_0003"]["manual_filter_action"], "ALREADY_AUTO_EXCLUDED")
        self.assertEqual(gate_by_id["mot_0099"]["manual_filter_action"], "ID_NOT_FOUND")
        self.assertEqual(outputs.summary["candidate_excluded_by_manual_filter"], "1")
        self.assertEqual(outputs.summary["final_target_tracks"], "1")


if __name__ == "__main__":
    unittest.main()
