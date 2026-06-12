# Target Consistency Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a logical vehicle consistency layer for the MVI_0866 clip that assigns each vehicle a unique stable logical ID and exports reviewable CSVs and three ID-bearing videos.

**Architecture:** Keep YOLO detections as evidence only. Add a new logical-vehicle pipeline that groups same-frame duplicates, segments raw tracks into tracklets, links tracklets with constrained association, validates consistency, and renders three review videos with stable logical IDs. Reuse the current mock association and overlay code where it helps, but replace their local-only behavior with a deterministic logical-ID layer.

**Tech Stack:** Python 3.9+, csv, pathlib, dataclasses, PIL, ffmpeg, existing YOLO review CSVs and video clip.

---

### Task 1: Add core logical-consistency module

**Files:**
- Create: `tools/logical_vehicle_consistency.py`
- Test: `tests/test_logical_vehicle_consistency.py`

- [ ] **Step 1: Write the failing test**

```python
def test_group_same_frame_duplicates_picks_one_representative():
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest discover -s tests -p 'test_logical_vehicle_consistency.py'`

Expected: FAIL because `logical_vehicle_consistency` is not defined yet.

- [ ] **Step 3: Write minimal implementation**

```python
def group_same_frame_duplicates(rows, iou_threshold=0.85):
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest discover -s tests -p 'test_logical_vehicle_consistency.py'`

Expected: PASS for duplicate grouping and tracklet construction tests.

### Task 2: Add tracklet association and validation

**Files:**
- Modify: `tools/logical_vehicle_consistency.py`
- Test: `tests/test_logical_vehicle_consistency.py`

- [ ] **Step 1: Write the failing test**

```python
def test_association_links_tracklets_into_one_logical_vehicle():
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest discover -s tests -p 'test_logical_vehicle_consistency.py'`

Expected: FAIL on missing association behavior.

- [ ] **Step 3: Write minimal implementation**

```python
def associate_tracklets(tracklets):
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest discover -s tests -p 'test_logical_vehicle_consistency.py'`

Expected: PASS for unique logical IDs, duplicate suppression, and validation report.

### Task 3: Add logical-ID video renderer and pipeline entrypoint

**Files:**
- Create: `tools/build_logical_vehicle_id_video.py`
- Create: `tools/build_logical_vehicle_consistency_v1.py`
- Test: `tests/test_logical_vehicle_consistency_video.py`

- [ ] **Step 1: Write the failing test**

```python
def test_label_format_uses_logical_id_and_raw_id():
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest discover -s tests -p 'test_logical_vehicle_consistency_video.py'`

Expected: FAIL before renderer exists.

- [ ] **Step 3: Write minimal implementation**

```python
def detection_label(row, mode):
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest discover -s tests -p 'test_logical_vehicle_consistency_video.py'`

Expected: PASS for final/debug/review label handling.

### Task 4: Wire pipeline outputs and verify on packaged clip

**Files:**
- Modify: `tools/build_logical_vehicle_consistency_v1.py`
- Modify: existing helper scripts only if needed for reuse

- [ ] **Step 1: Run the pipeline on the packaged MVI_0866 inputs**

Run: `python tools/build_logical_vehicle_consistency_v1.py ...`

- [ ] **Step 2: Verify required CSV outputs exist**

Check:
- `logical_vehicle_tracks.csv`
- `logical_vehicle_summary.csv`
- `raw_track_to_logical_vehicle.csv`
- `duplicate_groups.csv`
- `tracklets.csv`
- `tracklet_link_candidates.csv`
- `tracklet_links_accepted.csv`
- `ambiguous_link_review.csv`
- `consistency_validation_report.csv`

- [ ] **Step 3: Verify three videos exist**

Check:
- `logical_vehicle_id_final.mp4`
- `logical_vehicle_id_debug.mp4`
- `logical_vehicle_id_review.mp4`

- [ ] **Step 4: Verify invariants**

Confirm:
- one bbox per logical ID per frame
- retained detections have final states
- ambiguous links remain explicit
- output is deterministic and auditable

- [ ] **Step 5: Final review and report**

Summarize changed files, output paths, and any ambiguous cases left for manual review.

