from pathlib import Path
import sys
import unittest

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT / "tools"))

from logical_vehicle_consistency import (
    apply_fragment_path_absorption,
    apply_same_raw_continuity_merges,
    associate_tracklets,
    build_cross_raw_recovery_review,
    build_logical_vehicle_consistency,
    build_final_target_gate,
    build_identity_purity_report,
    build_target_quality_report,
    build_tracklets,
    build_target_validity_report,
    group_same_frame_duplicates,
    validate_logical_tracks,
)


def detection(frame_id, track_id, cx, cy=100.0, class_name="car", confidence="0.9000"):
    return {
        "frame_id": str(frame_id),
        "time_sec": f"{frame_id / 50.0:.2f}",
        "track_id": str(track_id),
        "class_name": class_name,
        "confidence": confidence,
        "x1": f"{cx - 5:.2f}",
        "y1": f"{cy - 5:.2f}",
        "x2": f"{cx + 5:.2f}",
        "y2": f"{cy + 5:.2f}",
    }


class LogicalVehicleConsistencyTest(unittest.TestCase):
    def test_duplicate_groups_choose_a_single_representative(self):
        rows = [
            detection(10, 1, 50, confidence="0.81"),
            detection(10, 2, 51, confidence="0.94"),
            detection(10, 3, 150, confidence="0.93"),
        ]

        result = group_same_frame_duplicates(rows, iou_threshold=0.80)

        self.assertEqual(len(result.groups), 1)
        self.assertEqual(result.representative_rows[0]["track_id"], "mot_0002")
        self.assertEqual({row["track_id"] for row in result.suppressed_rows}, {"mot_0001"})

    def test_build_tracklets_and_association_keep_split_raw_track_as_one_vehicle(self):
        rows = [
            detection(0, 1, 0),
            detection(1, 1, 5),
            detection(4, 2, 20),
            detection(5, 2, 25),
        ]

        tracklets = build_tracklets(rows, allowed_track_ids={"mot_0001", "mot_0002"}, max_gap_frames=1)
        assoc = associate_tracklets(tracklets, max_gap_frames=4, max_link_distance_px=30.0)

        self.assertEqual(len(tracklets), 2)
        self.assertEqual(len(assoc.accepted_links), 1)
        logical_ids = {row["logical_vehicle_id"] for row in assoc.logical_tracks if row["source"] == "detected"}
        self.assertEqual(len(logical_ids), 1)

    def test_same_raw_low_risk_link_overrides_ambiguous_competitor(self):
        rows = [
            detection(0, 1, 0),
            detection(1, 1, 5),
            detection(3, 1, 15),
            detection(4, 1, 20),
            detection(2, 2, 11),
            detection(3, 2, 16),
        ]

        tracklets = build_tracklets(rows, allowed_track_ids={"mot_0001", "mot_0002"}, max_gap_frames=1)
        assoc = associate_tracklets(
            tracklets,
            max_gap_frames=4,
            max_link_distance_px=30.0,
            ambiguous_cost_margin=5.0,
        )

        accepted_pairs = {(row["from_tracklet_id"], row["to_tracklet_id"]) for row in assoc.accepted_links}
        self.assertIn(("mot_0001_tl01", "mot_0001_tl02"), accepted_pairs)

    def test_validation_flags_same_frame_duplicate_rows_inside_one_logical_vehicle(self):
        logical_rows = [
            {
                "frame_id": "10",
                "time_sec": "0.20",
                "logical_vehicle_id": "lv_0001",
                "raw_track_id": "mot_0001",
                "tracklet_id": "trk_0001",
                "source": "detected",
                "class_name": "car",
                "confidence": "0.9",
                "x1": "0",
                "y1": "0",
                "x2": "10",
                "y2": "10",
                "center_x": "5",
                "center_y": "5",
                "association_status": "accepted",
            },
            {
                "frame_id": "10",
                "time_sec": "0.20",
                "logical_vehicle_id": "lv_0001",
                "raw_track_id": "mot_0002",
                "tracklet_id": "trk_0002",
                "source": "detected",
                "class_name": "car",
                "confidence": "0.8",
                "x1": "1",
                "y1": "1",
                "x2": "11",
                "y2": "11",
                "center_x": "6",
                "center_y": "6",
                "association_status": "accepted",
            },
        ]

        report = validate_logical_tracks(logical_rows)

        self.assertTrue(any(row["status"] == "FAIL" for row in report))

    def test_end_to_end_pipeline_builds_outputs(self):
        detections = [
            detection(0, 1, 0),
            detection(1, 1, 5),
            detection(4, 2, 20),
            detection(5, 2, 25),
            detection(10, 3, 100),
            detection(10, 4, 101, confidence="0.95"),
        ]
        final_targets = [
            {"track_id": "mot_0001"},
            {"track_id": "mot_0002"},
            {"track_id": "mot_0003"},
            {"track_id": "mot_0004"},
        ]

        outputs = build_logical_vehicle_consistency(
            detections=detections,
            final_targets=final_targets,
            fps=50.0,
            max_gap_frames=4,
            max_link_distance_px=30.0,
            max_iou=0.8,
        )

        self.assertTrue(outputs.logical_tracks)
        self.assertIn("duplicate_groups", outputs.as_dict())
        self.assertIn("consistency_validation_report", outputs.as_dict())

    def test_short_fragment_path_is_absorbed_into_nearby_mature_path(self):
        detections = []
        for frame in range(120):
            if frame == 62:
                continue
            detections.append(
                {
                    **detection(frame, 10, 130, cy=120, confidence="0.8500"),
                    "x1": "100.00",
                    "y1": "100.00",
                    "x2": "160.00",
                    "y2": "140.00",
                }
            )
        for frame in [50, 51, 52, 53, 54, 55, 62, 70, 71, 72, 73, 74, 75]:
            detections.append(
                {
                    **detection(frame, 226, 136.5, cy=120, confidence="0.3500"),
                    "x1": "100.00",
                    "y1": "100.00",
                    "x2": "173.00",
                    "y2": "140.00",
                }
            )

        outputs = build_logical_vehicle_consistency(
            detections=detections,
            final_targets=[{"track_id": "mot_0010"}, {"track_id": "mot_0226"}],
            max_gap_frames=10,
            max_link_distance_px=80.0,
            max_iou=0.85,
        )

        accepted_rows = [row for row in outputs.logical_tracks if row["association_status"] == "accepted"]
        accepted_ids = {row["logical_vehicle_id"] for row in accepted_rows}
        self.assertEqual(len(accepted_ids), 1)
        canonical_id = next(iter(accepted_ids))
        self.assertTrue(
            any(
                row["frame_id"] == "62"
                and row["raw_track_id"] == "mot_0226"
                and row["logical_vehicle_id"] == canonical_id
                and row["association_status"] == "accepted"
                for row in outputs.logical_tracks
            )
        )
        self.assertTrue(
            all(
                row["association_status"] == "fragment_suppressed"
                for row in outputs.logical_tracks
                if row["raw_track_id"] == "mot_0226" and row["frame_id"] != "62"
            )
        )
        self.assertEqual(len(outputs.fragment_path_absorption_review), 2)
        self.assertTrue(
            all(row["action"] == "AUTO_ABSORB_FRAGMENT_PATH" for row in outputs.fragment_path_absorption_review)
        )

    def test_short_adjacent_path_is_not_absorbed_into_mature_path(self):
        detections = []
        for frame in range(120):
            detections.append(
                {
                    **detection(frame, 10, 130, cy=120, confidence="0.8500"),
                    "x1": "100.00",
                    "y1": "100.00",
                    "x2": "160.00",
                    "y2": "140.00",
                }
            )
        for frame in range(50, 63):
            detections.append(
                {
                    **detection(frame, 226, 185, cy=120, confidence="0.7000"),
                    "x1": "155.00",
                    "y1": "100.00",
                    "x2": "215.00",
                    "y2": "140.00",
                }
            )

        outputs = build_logical_vehicle_consistency(
            detections=detections,
            final_targets=[{"track_id": "mot_0010"}, {"track_id": "mot_0226"}],
            max_gap_frames=10,
            max_link_distance_px=80.0,
            max_iou=0.85,
        )

        accepted_ids = {
            row["logical_vehicle_id"]
            for row in outputs.logical_tracks
            if row["association_status"] == "accepted"
        }
        self.assertEqual(len(accepted_ids), 2)
        self.assertEqual(outputs.fragment_path_absorption_review, [])

    def test_fragment_matching_multiple_mature_paths_requires_review(self):
        logical_rows = []
        for frame in range(120):
            for logical_id, raw_track_id, x1, x2 in [
                ("lv_0001", "mot_0001", 100.0, 160.0),
                ("lv_0002", "mot_0002", 108.0, 168.0),
            ]:
                logical_rows.append(
                    {
                        **detection(frame, raw_track_id, (x1 + x2) / 2.0, cy=120, confidence="0.8500"),
                        "logical_vehicle_id": logical_id,
                        "raw_track_id": raw_track_id,
                        "tracklet_id": f"{raw_track_id}_tl01",
                        "source": "detected",
                        "x1": f"{x1:.2f}",
                        "y1": "100.00",
                        "x2": f"{x2:.2f}",
                        "y2": "140.00",
                        "association_status": "accepted",
                    }
                )
        for frame in range(50, 56):
            logical_rows.append(
                {
                    **detection(frame, 226, 134, cy=120, confidence="0.3500"),
                    "logical_vehicle_id": "lv_0003",
                    "raw_track_id": "mot_0226",
                    "tracklet_id": "mot_0226_tl01",
                    "source": "detected",
                    "x1": "104.00",
                    "y1": "100.00",
                    "x2": "164.00",
                    "y2": "140.00",
                    "association_status": "accepted",
                }
            )

        output_rows, review_rows = apply_fragment_path_absorption(logical_rows)

        self.assertEqual(
            {row["logical_vehicle_id"] for row in output_rows if row["association_status"] == "accepted"},
            {"lv_0001", "lv_0002", "lv_0003"},
        )
        self.assertEqual(review_rows[0]["action"], "REVIEW_AMBIGUOUS_FRAGMENT_MATCH")

    def test_vehicle_validity_auto_excludes_two_wheeler_class(self):
        logical_rows = [
            {
                **detection(0, 7, 50, class_name="motorcycle"),
                "logical_vehicle_id": "lv_0007",
                "raw_track_id": "mot_0007",
                "tracklet_id": "mot_0007_tl01",
                "source": "detected",
                "association_status": "accepted",
            },
            {
                **detection(1, 7, 55, class_name="motorcycle"),
                "logical_vehicle_id": "lv_0007",
                "raw_track_id": "mot_0007",
                "tracklet_id": "mot_0007_tl01",
                "source": "detected",
                "association_status": "accepted",
            },
        ]

        report = build_target_validity_report(logical_rows)

        self.assertEqual(report[0]["vehicle_validity_status"], "AUTO_EXCLUDE")
        self.assertEqual(report[0]["exclude_reason"], "two_wheeler_or_person_class")

    def test_vehicle_validity_marks_small_short_car_like_target_uncertain(self):
        logical_rows = []
        for frame in range(6):
            logical_rows.append(
                {
                    **detection(frame, 8, 50 + frame, class_name="car"),
                    "x1": "10",
                    "y1": "10",
                    "x2": "34",
                    "y2": "28",
                    "logical_vehicle_id": "lv_0008",
                    "raw_track_id": "mot_0008",
                    "tracklet_id": "mot_0008_tl01",
                    "source": "detected",
                    "association_status": "accepted",
                }
            )

        report = build_target_validity_report(logical_rows)

        self.assertEqual(report[0]["vehicle_validity_status"], "REVIEW_ONLY_IF_UNCERTAIN")
        self.assertEqual(report[0]["review_reason"], "small_short_vehicle_like_target")

    def test_vehicle_validity_marks_short_small_car_target_uncertain(self):
        logical_rows = []
        for frame in range(40):
            logical_rows.append(
                {
                    **detection(frame, 402, 560 + frame * 0.5, cy=40, confidence="0.5500"),
                    "x1": f"{545 + frame * 0.5:.2f}",
                    "y1": "25.00",
                    "x2": f"{575 + frame * 0.5:.2f}",
                    "y2": "58.00",
                    "logical_vehicle_id": "lv_0023",
                    "raw_track_id": "mot_0402",
                    "tracklet_id": "mot_0402_tl01",
                    "source": "detected",
                    "association_status": "accepted",
                }
            )

        report = build_target_validity_report(logical_rows)

        self.assertEqual(report[0]["vehicle_validity_status"], "REVIEW_ONLY_IF_UNCERTAIN")
        self.assertEqual(report[0]["review_reason"], "short_small_target")

    def test_vehicle_validity_excludes_very_short_static_low_confidence_target(self):
        logical_rows = []
        for frame in range(3):
            logical_rows.append(
                {
                    **detection(frame, 27, 80, confidence="0.3100"),
                    "x1": "60.00",
                    "y1": "60.00",
                    "x2": "100.00",
                    "y2": "95.00",
                    "logical_vehicle_id": "lv_0027",
                    "raw_track_id": "mot_0027",
                    "tracklet_id": "mot_0027_tl01",
                    "source": "detected",
                    "association_status": "accepted",
                }
            )

        report = build_target_validity_report(logical_rows)

        self.assertEqual(report[0]["vehicle_validity_status"], "AUTO_EXCLUDE")
        self.assertEqual(report[0]["exclude_reason"], "short_static_false_positive")

    def test_target_quality_report_flags_short_border_and_fragmented_tracks(self):
        logical_rows = []
        for frame in range(40):
            logical_rows.append(
                {
                    **detection(frame, 402, 560 + frame * 0.5, cy=15, confidence="0.5500"),
                    "x1": f"{545 + frame * 0.5:.2f}",
                    "y1": "0.50",
                    "x2": f"{575 + frame * 0.5:.2f}",
                    "y2": "30.00",
                    "logical_vehicle_id": "lv_0023",
                    "raw_track_id": "mot_0402",
                    "tracklet_id": "mot_0402_tl01",
                    "source": "detected",
                    "association_status": "accepted",
                }
            )
        for frame in [0, 1, 5, 6, 12, 13, 20, 21]:
            logical_rows.append(
                {
                    **detection(frame, 622, 100 + frame, cy=150, confidence="0.3500"),
                    "logical_vehicle_id": "lv_0041",
                    "raw_track_id": "mot_0622",
                    "tracklet_id": f"mot_0622_tl{frame:02d}",
                    "source": "detected",
                    "association_status": "accepted",
                }
            )

        report = build_target_quality_report(logical_rows)
        by_id = {row["logical_vehicle_id"]: row for row in report}

        self.assertEqual(by_id["lv_0023"]["quality_status"], "RISK_REVIEW")
        self.assertIn("border_short_target", by_id["lv_0023"]["risk_reasons"])
        self.assertIn("small_area", by_id["lv_0023"]["risk_reasons"])
        self.assertEqual(by_id["lv_0041"]["quality_status"], "RISK_REVIEW")
        self.assertIn("sparse_track", by_id["lv_0041"]["risk_reasons"])
        self.assertIn("flicker_fragmented", by_id["lv_0041"]["risk_reasons"])

    def test_cross_raw_recovery_review_reports_smooth_candidate(self):
        logical_rows = []
        for frame in range(30):
            logical_rows.append(
                {
                    **detection(frame, 392, 100 + frame, cy=200, confidence="0.8500"),
                    "logical_vehicle_id": "lv_0021",
                    "raw_track_id": "mot_0392",
                    "tracklet_id": "mot_0392_tl01",
                    "source": "detected",
                    "association_status": "accepted",
                }
            )
        for frame in range(70, 100):
            logical_rows.append(
                {
                    **detection(frame, 570, 100 + frame * 0.9, cy=200, confidence="0.8000"),
                    "logical_vehicle_id": "lv_0036",
                    "raw_track_id": "mot_0570",
                    "tracklet_id": "mot_0570_tl01",
                    "source": "detected",
                    "association_status": "accepted",
                }
            )

        report = build_cross_raw_recovery_review(logical_rows)

        self.assertEqual(len(report), 1)
        self.assertEqual(report[0]["from_logical_vehicle_id"], "lv_0021")
        self.assertEqual(report[0]["to_logical_vehicle_id"], "lv_0036")
        self.assertEqual(report[0]["review_status"], "REVIEW_CROSS_RAW_RECOVERY")

    def test_final_gate_requires_validity_and_purity_pass(self):
        summary_rows = [
            {"logical_vehicle_id": "lv_0001"},
            {"logical_vehicle_id": "lv_0002"},
        ]
        validity_rows = [
            {"logical_vehicle_id": "lv_0001", "vehicle_validity_status": "AUTO_KEEP", "exclude_reason": ""},
            {"logical_vehicle_id": "lv_0002", "vehicle_validity_status": "AUTO_EXCLUDE", "exclude_reason": "two_wheeler_or_person_class"},
        ]
        purity_rows = [
            {"logical_vehicle_id": "lv_0001", "purity_status": "PURITY_PASS", "review_reason": ""},
            {"logical_vehicle_id": "lv_0002", "purity_status": "PURITY_PASS", "review_reason": ""},
        ]

        gate = build_final_target_gate(summary_rows, validity_rows, purity_rows)

        by_id = {row["logical_vehicle_id"]: row for row in gate}
        self.assertEqual(by_id["lv_0001"]["final_gate_status"], "AUTO_KEEP")
        self.assertEqual(by_id["lv_0002"]["final_gate_status"], "AUTO_EXCLUDE")

    def test_same_raw_continuity_merge_rewrites_fragmented_logical_id(self):
        logical_rows = []
        for frame, cx, logical_id in [
            (0, 10, "lv_0001"),
            (1, 12, "lv_0001"),
            (3, 13, "lv_0002"),
            (4, 15, "lv_0002"),
        ]:
            logical_rows.append(
                {
                    **detection(frame, 9, cx),
                    "logical_vehicle_id": logical_id,
                    "raw_track_id": "mot_0009",
                    "tracklet_id": f"mot_0009_{logical_id}",
                    "source": "detected",
                    "association_status": "accepted",
                }
            )

        merged_rows, review_rows = apply_same_raw_continuity_merges(logical_rows)

        self.assertEqual({row["logical_vehicle_id"] for row in merged_rows}, {"lv_0001"})
        self.assertEqual(review_rows[0]["suggested_action"], "AUTO_MERGE_APPLIED")

    def test_same_raw_occlusion_recovery_merges_medium_gap_smooth_motion(self):
        logical_rows = []
        for frame in range(30):
            logical_rows.append(
                {
                    **detection(frame, 622, 100 + frame, cy=150),
                    "x1": f"{82 + frame:.2f}",
                    "y1": "133.00",
                    "x2": f"{118 + frame:.2f}",
                    "y2": "167.00",
                    "logical_vehicle_id": "lv_0001",
                    "raw_track_id": "mot_0622",
                    "tracklet_id": "mot_0622_lv_0001",
                    "source": "detected",
                    "association_status": "accepted",
                }
            )
        for frame in range(41, 71):
            cx = 100 + frame
            logical_rows.append(
                {
                    **detection(frame, 622, cx, cy=150),
                    "x1": f"{cx - 18:.2f}",
                    "y1": "133.00",
                    "x2": f"{cx + 18:.2f}",
                    "y2": "167.00",
                    "logical_vehicle_id": "lv_0002",
                    "raw_track_id": "mot_0622",
                    "tracklet_id": "mot_0622_lv_0002",
                    "source": "detected",
                    "association_status": "accepted",
                }
            )

        merged_rows, review_rows = apply_same_raw_continuity_merges(logical_rows)

        self.assertEqual({row["logical_vehicle_id"] for row in merged_rows}, {"lv_0001"})
        self.assertEqual(review_rows[0]["suggested_action"], "AUTO_MERGE_APPLIED")
        self.assertEqual(review_rows[0]["merge_reason"], "same_raw_occlusion_recovery")

    def test_same_raw_continuity_merge_skips_temporal_overlap(self):
        logical_rows = [
            {
                **detection(0, 9, 10),
                "logical_vehicle_id": "lv_0001",
                "raw_track_id": "mot_0009",
                "tracklet_id": "mot_0009_a",
                "source": "detected",
                "association_status": "accepted",
            },
            {
                **{
                    **detection(2, 9, 12),
                    "x1": "2",
                    "y1": "98",
                    "x2": "22",
                    "y2": "102",
                },
                "logical_vehicle_id": "lv_0001",
                "raw_track_id": "mot_0009",
                "tracklet_id": "mot_0009_a",
                "source": "detected",
                "association_status": "accepted",
            },
            {
                **detection(1, 9, 11),
                "logical_vehicle_id": "lv_0002",
                "raw_track_id": "mot_0009",
                "tracklet_id": "mot_0009_b",
                "source": "detected",
                "association_status": "accepted",
            },
            {
                **{
                    **detection(2, 9, 12),
                    "x1": "10",
                    "y1": "90",
                    "x2": "14",
                    "y2": "110",
                },
                "logical_vehicle_id": "lv_0002",
                "raw_track_id": "mot_0009",
                "tracklet_id": "mot_0009_b",
                "source": "detected",
                "association_status": "accepted",
            },
        ]

        merged_rows, review_rows = apply_same_raw_continuity_merges(logical_rows)

        self.assertEqual({row["logical_vehicle_id"] for row in merged_rows}, {"lv_0001", "lv_0002"})
        self.assertEqual(review_rows[0]["suggested_action"], "KEEP_SPLIT_TEMPORAL_OVERLAP")

    def test_same_raw_continuity_merge_suppresses_duplicate_like_overlap(self):
        logical_rows = []
        for frame, raw_track_id, logical_id, cx, area_scale in [
            (0, "mot_0009", "lv_0001", 10, "large"),
            (1, "mot_0009", "lv_0001", 11, "large"),
            (3, "mot_0009", "lv_0003", 13, "large"),
            (4, "mot_0009", "lv_0003", 14, "large"),
            (3, "mot_0010", "lv_0002", 13, "small"),
            (4, "mot_0010", "lv_0002", 14, "small"),
            (6, "mot_0010", "lv_0003", 16, "small"),
            (7, "mot_0010", "lv_0003", 17, "small"),
        ]:
            row = detection(frame, raw_track_id, cx)
            if area_scale == "large":
                row.update({"x1": f"{cx - 8:.2f}", "y1": "92", "x2": f"{cx + 8:.2f}", "y2": "108"})
            else:
                row.update({"x1": f"{cx - 6:.2f}", "y1": "94", "x2": f"{cx + 6:.2f}", "y2": "106"})
            logical_rows.append(
                {
                    **row,
                    "logical_vehicle_id": logical_id,
                    "raw_track_id": raw_track_id,
                    "tracklet_id": f"{raw_track_id}_{logical_id}",
                    "source": "detected",
                    "association_status": "accepted",
                }
            )

        merged_rows, review_rows = apply_same_raw_continuity_merges(logical_rows)

        accepted_by_frame = {}
        for row in merged_rows:
            if row["association_status"] != "accepted":
                continue
            accepted_by_frame.setdefault(int(float(row["frame_id"])), []).append(row)
        self.assertEqual({row["logical_vehicle_id"] for row in merged_rows}, {"lv_0001"})
        self.assertEqual(len(accepted_by_frame[3]), 1)
        self.assertEqual(len(accepted_by_frame[4]), 1)
        self.assertTrue(all(row["suggested_action"] == "AUTO_MERGE_APPLIED" for row in review_rows))

    def test_same_raw_continuity_merge_applies_connected_components(self):
        logical_rows = []
        for frame, raw_track_id, logical_id in [
            (0, "mot_0009", "lv_0001"),
            (1, "mot_0009", "lv_0001"),
            (3, "mot_0009", "lv_0003"),
            (4, "mot_0009", "lv_0003"),
            (6, "mot_0010", "lv_0002"),
            (7, "mot_0010", "lv_0002"),
            (9, "mot_0010", "lv_0003"),
            (10, "mot_0010", "lv_0003"),
        ]:
            logical_rows.append(
                {
                    **detection(frame, raw_track_id, 10 + frame),
                    "logical_vehicle_id": logical_id,
                    "raw_track_id": raw_track_id,
                    "tracklet_id": f"{raw_track_id}_{logical_id}",
                    "source": "detected",
                    "association_status": "accepted",
                }
            )

        merged_rows, review_rows = apply_same_raw_continuity_merges(logical_rows)

        self.assertEqual({row["logical_vehicle_id"] for row in merged_rows}, {"lv_0001"})
        self.assertEqual(
            [row["suggested_action"] for row in review_rows],
            ["AUTO_MERGE_APPLIED", "AUTO_MERGE_APPLIED"],
        )


if __name__ == "__main__":
    unittest.main()
