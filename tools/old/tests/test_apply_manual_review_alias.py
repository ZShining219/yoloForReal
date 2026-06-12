from pathlib import Path
import sys
import unittest

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT / "tools"))

from apply_manual_review_alias import apply_manual_aliases_to_tracks, apply_manual_aliases_to_od, apply_manual_aliases_to_targets


class ApplyManualReviewAliasTest(unittest.TestCase):
    def test_filters_alias_track_and_limits_frame_window(self):
        tracks = [
            {"frame_id": "700", "logical_vehicle_id": "cf_0010", "track_id": "mot_0010"},
            {"frame_id": "701", "logical_vehicle_id": "cf_0018", "track_id": "mot_0226"},
            {"frame_id": "1500", "logical_vehicle_id": "cf_0010", "track_id": "mot_0010"},
        ]
        aliases = [{"alias_logical_vehicle_id": "cf_0018", "canonical_logical_vehicle_id": "cf_0010"}]

        output = apply_manual_aliases_to_tracks(tracks, aliases, frame_start=0, frame_end=1499)

        self.assertEqual([row["logical_vehicle_id"] for row in output], ["cf_0010"])
        self.assertEqual(output[0]["frame_id"], "700")

    def test_marks_alias_target_as_manual_suppressed(self):
        targets = [
            {"logical_vehicle_id": "cf_0010", "review_note": "keep", "keep_for_direction_od": "no"},
            {"logical_vehicle_id": "cf_0018", "review_note": "review", "keep_for_direction_od": "no"},
        ]
        aliases = [{"alias_logical_vehicle_id": "cf_0018", "canonical_logical_vehicle_id": "cf_0010"}]

        output = apply_manual_aliases_to_targets(targets, aliases)

        by_id = {row["logical_vehicle_id"]: row for row in output}
        self.assertEqual(by_id["cf_0018"]["keep_for_direction_od"], "no")
        self.assertIn("Manual review alias to cf_0010", by_id["cf_0018"]["review_note"])
        self.assertEqual(by_id["cf_0010"]["review_note"], "keep")

    def test_marks_alias_od_as_manual_review_alias(self):
        od_rows = [
            {"track_id": "cf_0010", "review_status": "UNKNOWN", "result_direction": "unknown", "evidence_note": ""},
            {"track_id": "cf_0018", "review_status": "UNKNOWN", "result_direction": "unknown", "evidence_note": ""},
        ]
        aliases = [{"alias_logical_vehicle_id": "cf_0018", "canonical_logical_vehicle_id": "cf_0010"}]

        output = apply_manual_aliases_to_od(od_rows, aliases)

        by_id = {row["track_id"]: row for row in output}
        self.assertEqual(by_id["cf_0018"]["review_status"], "MANUAL_ALIAS_SUPPRESSED")
        self.assertEqual(by_id["cf_0018"]["result_direction"], "unknown")
        self.assertIn("Manual review alias to cf_0010", by_id["cf_0018"]["evidence_note"])


if __name__ == "__main__":
    unittest.main()
