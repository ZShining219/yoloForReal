# Target Consistency Tracking Design

## Scope

This design covers only vehicle target consistency tracking for the
`MVI_0866_520_560.mp4` clip. The goal is to assign each visible vehicle a
unique and stable `logical_vehicle_id` across the video.

This design does not cover lane assignment, turning movement, OD inference,
SUMO demand generation, route generation, or calibration correction.

## Problem Statement

The current YOLO/ByteTrack outputs are useful detection evidence, but the raw
`mot_xxxx` IDs are not sufficient as final vehicle identities. The current
versions show three target-consistency risks:

- One real vehicle can be split into multiple raw tracks.
- One real vehicle can appear as duplicate overlapping logical tracks.
- Local center-follow or track-stitching rules can create reviewable outputs,
  but they do not enforce global one-vehicle-one-ID consistency.

The required fix is an explicit logical vehicle consistency layer above YOLO
detections. YOLO detections remain evidence; `logical_vehicle_id` becomes the
final target identity.

## Reusable Current Assets

The following current assets should be reused.

- `outputs/yolo26x_trial/target_tracks/detections_tracked.csv`
  - Raw YOLO26X tracked detections.
  - Used as the detection evidence source.
- `outputs/yolo26x_manual_filter_v1/target_tracks_final.csv`
  - Reviewed target candidates after manual false-positive filtering.
  - Used as the allowed raw-track set for the first implementation.
- `inputs/video_clips/MVI_0866_520_560.mp4`
  - Source video for overlay rendering and review crops.
- `tools/build_center_follow_mock.py`
  - Reuse center prediction, normalized ID handling, and logical-track output
    fields.
  - Do not reuse its greedy association as final authority.
- `tools/build_track_stitch_mock.py`
  - Reuse tracklet segmentation and link metric ideas.
  - Replace local greedy linking with constrained global association.
- `tools/apply_duplicate_overlap_suppression.py`
  - Reuse bbox IoU and duplicate-pair detection.
  - Move duplicate handling before final association instead of applying it
    only after OD output.
- `tools/build_target_overlay_video.py`
  - Reuse the video decode/encode and drawing pattern.
  - Extend it into a logical-ID video renderer with stronger label styling.

## Proposed Pipeline

### 1. Input Normalization

Read YOLO detection rows and the reviewed raw-track list. Normalize all raw
track IDs to `mot_0000` format.

The initial implementation should only consume raw tracks present in
`target_tracks_final.csv`. Excluded tracks remain available for audit but do not
participate in logical vehicle construction.

### 2. Same-Frame Duplicate Grouping

Before association, detect duplicate boxes in the same frame.

Two boxes are duplicate candidates when they satisfy all of the following:

- Same vehicle-compatible class group: `car`, `truck`, or `bus`.
- IoU above a configurable threshold, initially `0.85`.
- Center distance below a configurable threshold, initially relative to bbox
  size.
- Frame ID is identical.

For each duplicate group, choose one representative detection for tracking.
Selection priority:

1. Higher confidence.
2. Longer raw-track support near the frame.
3. More stable bbox size across neighboring frames.

Non-representative duplicates must not create separate final vehicle IDs. They
are written to `duplicate_groups.csv` for audit.

### 3. Tracklet Construction

Build short, internally consistent tracklets from representative detections.

A tracklet is a continuous or near-continuous sequence from one raw track. A new
tracklet starts when any of the following occurs:

- Frame gap exceeds the small internal gap threshold.
- Motion jump exceeds the internal continuity threshold.
- Bbox size changes sharply.
- The raw track contains duplicate-contaminated detections that were removed.

Each tracklet stores:

- `tracklet_id`
- `raw_track_id`
- `start_frame`, `end_frame`
- frame count
- representative class
- mean confidence
- first and last bbox
- estimated start and end velocity
- representative crop paths

### 4. Global Tracklet Association

Build a directed graph from tracklets. A candidate edge means the two tracklets
may belong to the same real vehicle.

Edge features:

- Temporal gap.
- Predicted center distance.
- Bottom-center displacement.
- Bbox size ratio.
- Velocity direction consistency.
- Raw ID continuity bonus.
- Class compatibility.
- Simple appearance similarity from representative crops.

The first implementation may use deterministic image features such as HSV color
histograms. The interface should allow replacing this with ReID embeddings
later.

Association must enforce these hard constraints:

- A tracklet can have at most one predecessor and at most one successor.
- A logical vehicle cannot contain two detections in the same frame.
- Tracklets that overlap in time and are spatially distinct cannot be merged.
- Long same-frame duplicate overlap is treated as duplicate evidence, not as two
  vehicles.
- Edges above the uncertainty threshold are not auto-accepted.

Accepted edges form logical vehicle paths. Each path receives one
`logical_vehicle_id`.

### 5. Ambiguity Handling

The pipeline should not force uncertain links.

Ambiguous links are written to `ambiguous_link_review.csv` when:

- Multiple candidate successors have similar costs.
- Appearance and motion disagree.
- The gap is large.
- The candidate involves known duplicate-contaminated raw tracks.
- The merge would recover a vehicle but cannot be proven from geometry alone.

The review file should include before/after crop paths, link metrics, and a
default `PENDING_REVIEW` status. Later manual decisions can be applied in a
separate gate file without changing raw detections.

### 6. Consistency Validation

After logical IDs are assigned, validate each logical vehicle.

Required checks:

- No same-frame duplicate bbox within one `logical_vehicle_id`.
- Frames are strictly increasing within each logical vehicle.
- No impossible large jumps unless the segment is explicitly interpolated.
- No logical vehicle contains two spatially distinct simultaneous tracklets.
- Every raw detection row is either assigned, suppressed as duplicate, or
  excluded by the reviewed target gate.

Validation results are written to `consistency_validation_report.csv`.

### 7. Target Validity And Final Gate

The final review video is treated as downstream data evidence. Therefore it
must display only effective vehicle boxes, not all boxes kept for engineering
audit.

After logical ID assignment, build a target validity gate per
`logical_vehicle_id`:

- `motorcycle`, `bicycle`, and `person` dominant classes are `AUTO_EXCLUDE`.
- Small and short car-like targets are `REVIEW_ONLY_IF_UNCERTAIN`, not shown in
  the final video.
- Normal car/truck/bus logical vehicles remain `AUTO_KEEP` only if identity
  purity also passes.

Build an identity purity gate per `logical_vehicle_id`:

- Large bbox-area jumps across raw-track switches trigger `PURITY_REVIEW`.
- Long raw-switch gaps trigger `PURITY_REVIEW`.
- Same-frame duplicate-like overlaps may be merged only when bbox IoU and
  center distance show they are duplicate observations of the same target.

The final gate is conservative:

- `AUTO_KEEP`: shown in `logical_vehicle_id_final.mp4` and window-slice final
  videos.
- `REVIEW_ONLY_IF_UNCERTAIN`: retained in CSV/debug/review outputs but hidden
  from final videos.
- `AUTO_EXCLUDE`: retained only as audit evidence and hidden from final videos.

### 8. Same-Raw Continuity Post-Merge

Some raw tracks can be split across multiple logical IDs after the first graph
association pass. A deterministic post-merge module reviews same-raw splits and
applies only low-risk merges.

Automatic merge condition:

- Same `raw_track_id` appears in adjacent logical IDs.
- Gap is at most 3 frames.
- Center distance is at most 2 px.
- Any temporal overlap in the resulting connected component is duplicate-like:
  IoU at least 0.45 and center distance at most 12 px.

When duplicate-like overlap exists inside the merged component, the frame keeps
one accepted bbox and marks the smaller duplicate-like row as
`duplicate_suppressed`. If overlap is spatially distinct, the merge is blocked
as `KEEP_SPLIT_TEMPORAL_OVERLAP`.

## New Or Modified Tools

### `tools/build_logical_vehicle_consistency_v1.py`

Main pipeline.

Inputs:

- `--detections`
- `--final-targets`
- `--video`
- `--output-dir`
- optional threshold arguments

Outputs:

- normalized detections
- duplicate groups
- tracklets
- candidate links
- accepted links
- ambiguous links
- logical vehicle tracks
- validation report

### `tools/build_logical_vehicle_review_assets.py`

Review asset generator.

Responsibilities:

- Build crop pairs for ambiguous links.
- Build duplicate group contact sheets.
- Build per-logical-vehicle first/mid/last evidence images.
- Optionally build a static HTML review board.

### `tools/build_logical_vehicle_id_video.py`

Logical ID video renderer.

Responsibilities:

- Render boxes from `logical_vehicle_tracks.csv`.
- Draw stable, high-contrast colors per `logical_vehicle_id`.
- Draw large ID labels that remain readable in the output video.
- Support three output modes described below.

## Required Outputs

The output directory should be versioned, for example:

```text
outputs/logical_vehicle_consistency_v1/
```

Required CSV outputs:

- `logical_vehicle_tracks.csv`
- `logical_vehicle_summary.csv`
- `raw_track_to_logical_vehicle.csv`
- `duplicate_groups.csv`
- `tracklets.csv`
- `tracklet_link_candidates.csv`
- `tracklet_links_accepted.csv`
- `ambiguous_link_review.csv`
- `consistency_validation_report.csv`
- `target_validity_report.csv`
- `identity_purity_report.csv`
- `final_target_gate.csv`
- `raw_track_split_review.csv`
- `risky_accepted_link_review.csv`

Required review assets:

- `review_assets/duplicate_groups/`
- `review_assets/ambiguous_links/`
- `review_assets/logical_vehicle_triplets/`

Required video outputs:

### 1. Final ID Video

```text
logical_vehicle_id_final.mp4
```

Purpose:

- Human-facing final review.
- Shows only final `logical_vehicle_id`.
- Does not show raw `mot_xxxx` unless configured.

Visual requirements:

- Thick bbox border.
- Stable color per logical vehicle.
- Large label such as `LV-001`.
- Label background with high contrast.
- Label placed near bbox without covering the vehicle more than necessary.

### 2. Debug ID Video

```text
logical_vehicle_id_debug.mp4
```

Purpose:

- Engineering review of ID switches and raw-track merging.

Visual requirements:

- Show `logical_vehicle_id`.
- Show raw `mot_xxxx`.
- Mark interpolated frames if present.
- Optionally draw short trailing trajectory.

Example label:

```text
LV-014 / mot_0384
```

### 3. Review ID Video

```text
logical_vehicle_id_review.mp4
```

Purpose:

- Focused review of uncertainty and duplicate handling.

Visual requirements:

- Highlight duplicate-suppressed detections.
- Highlight ambiguous links.
- Mark cases with `DUP`, `LINK?`, or `REVIEW`.
- Use clear color semantics:
  - accepted logical vehicle: green or stable ID color
  - duplicate suppressed: red or gray
  - ambiguous link: amber
  - interpolated display-only segment: dashed or distinct label

### 4. Three Window-Slice Final Videos

The goal-level validation requires three different video/window slices from the
packaged project video asset. The current implementation splits
`inputs/video_clips/MVI_0866_520_560.mp4` into three contiguous frame windows
and renders final-gated overlays for each:

```text
window_01/logical_vehicle_id_final.mp4
window_02/logical_vehicle_id_final.mp4
window_03/logical_vehicle_id_final.mp4
```

These are not debug/review variants. They are three separate temporal slices of
the source asset and should show only rows whose `final_gate_status` is
`AUTO_KEEP`.

## Output Semantics

`logical_vehicle_tracks.csv` should contain one row per displayed vehicle box or
interpolated point.

Required fields:

- `frame_id`
- `time_sec`
- `logical_vehicle_id`
- `raw_track_id`
- `tracklet_id`
- `source`
- `class_name`
- `confidence`
- `x1`
- `y1`
- `x2`
- `y2`
- `center_x`
- `center_y`
- `association_status`
- `vehicle_validity_status`
- `purity_status`
- `final_gate_status`

`source` values:

- `detected`
- `interpolated`

`association_status` values:

- `accepted`
- `duplicate_suppressed`
- `ambiguous_review`
- `manual_review_applied`

## Boundaries

This module may:

- Assign stable logical vehicle IDs.
- Suppress duplicate boxes.
- Link raw track fragments when evidence is strong.
- Output uncertain links for manual review.
- Interpolate short display gaps, explicitly marked as interpolation.
- Render logical-ID review videos.
- Hide uncertain or invalid targets from final videos while preserving them in
  audit CSVs.
- Produce three final-gated temporal window slices from the packaged project
  video asset.

This module must not:

- Infer lane membership.
- Infer route, direction, or OD.
- Generate SUMO vehicles.
- Add manually imagined vehicles with no detection evidence.
- Treat interpolated points as YOLO detections.
- Hide uncertainty in final CSVs.
- Show review-only or excluded targets in final evidence videos.

## Acceptance Criteria

The design is considered implemented when all of the following are true:

- Every retained detection is assigned to exactly one final state: accepted
  logical vehicle, duplicate-suppressed, ambiguous review, or excluded by the
  target gate.
- No `logical_vehicle_id` has more than one bbox in the same frame.
- Known duplicate overlaps do not produce multiple final logical vehicle IDs.
- Fragment links are reproducible from `tracklet_links_accepted.csv`.
- Ambiguous links are visible in `ambiguous_link_review.csv`.
- Three full-clip view videos are generated:
  - `logical_vehicle_id_final.mp4`
  - `logical_vehicle_id_debug.mp4`
  - `logical_vehicle_id_review.mp4`
- Three window-slice final videos are generated:
  - `window_01/logical_vehicle_id_final.mp4`
  - `window_02/logical_vehicle_id_final.mp4`
  - `window_03/logical_vehicle_id_final.mp4`
- The final video has large, readable, stable IDs for every accepted logical
  vehicle.
- The final video and three window slices show only `AUTO_KEEP` targets.
- The pipeline can be rerun from packaged inputs without manual GUI steps.

## Initial Implementation Boundary

The first version should target the existing packaged clip and current YOLO26X
reviewed outputs only. It should not generalize to arbitrary videos until the
MVI_0866 consistency layer passes visual review.

Manual review can be introduced as a CSV gate after automatic candidate
generation. The automatic pipeline should remain deterministic and auditable.
