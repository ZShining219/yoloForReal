# SUMO Controlled Replay V2 Spatial Init Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fixed, auditable SUMO replay package that aligns video gate times with SUMO edge-start semantics and handles warm-up/window-start vehicles through explicit spatial initialization.

**Architecture:** Add a new builder script that consumes the locked v1 SUMO candidate package and v1 diagnostics. It writes a new versioned package with corrected route departure times, a spatial initialization plan, traceability tables, SUMO route/config files, a TraCI controller, validation outputs, and review video artifacts.

**Tech Stack:** Python 3.13 standard library, SUMO CLI/SUMO-GUI, TraCI, ffmpeg.

---

### Task 1: Corrected Replay Planning

**Files:**
- Create: `tools/build_sumo_controlled_replay_v2_spatial_init_package.py`
- Create: `tests/test_build_sumo_controlled_replay_v2_spatial_init_package.py`

- [ ] **Step 1: Write failing tests**
  - Verify `READY_STRONG`/route replay rows use corrected SUMO depart times by subtracting incoming edge travel time from video gate entry time.
  - Verify warm-up/window-start rows are classified as `spatial_init_at_video_zero`.
  - Verify red-light queue rows preserve `hold_until_window_end`.

- [ ] **Step 2: Run failing tests**
  - Run: `python3 -m unittest tests/test_build_sumo_controlled_replay_v2_spatial_init_package.py`
  - Expected: import failure because the builder does not exist yet.

- [ ] **Step 3: Implement the minimal builder API**
  - Implement CSV/JSON helpers, lane length parsing, control mode classification, corrected depart calculation, replay row creation, spatial init row creation, traceability row creation, route XML rendering, sumocfg rendering, and TraCI controller rendering.

- [ ] **Step 4: Run focused tests**
  - Run: `python3 -m unittest tests/test_build_sumo_controlled_replay_v2_spatial_init_package.py`
  - Expected: all tests pass.

### Task 2: Package Generation

**Files:**
- Modify: `tools/build_sumo_controlled_replay_v2_spatial_init_package.py`

- [ ] **Step 1: Add CLI**
  - Inputs: source candidate dir, diagnostics dir, output dir, time shift, video end, SUMO binary.
  - Outputs: `data/controlled_replay_v2_plan.csv`, `data/spatial_initialization_plan.csv`, `data/video_sumo_traceability.csv`, `data/run_validation_summary.json`, `sumo/routes_controlled_v2.rou.xml`, `sumo/simulation_controlled_v2.sumocfg`, `sumo/run_controlled_replay_v2.py`, `VERSION_LOCK.md`, manifest, checksums.

- [ ] **Step 2: Generate package**
  - Run builder against `outputs/manual_review_versions/yolo26x_0_30s_sumo_candidate_v1`.
  - Expected: new fixed output directory under `outputs/manual_review_versions/yolo26x_0_30s_sumo_controlled_replay_v2_spatial_init`.

### Task 3: SUMO Runtime Validation And Video

**Files:**
- Use generated package files only.

- [ ] **Step 1: Run non-GUI SUMO validation**
  - Run: `/Users/zfh/.local/bin/sumo -c sumo/simulation_controlled_v2.sumocfg --duration-log.disable true`
  - Expected: exits 0.

- [ ] **Step 2: Run TraCI controller validation**
  - Run: `python3 sumo/run_controlled_replay_v2.py --sumo-binary /Users/zfh/.local/bin/sumo`
  - Expected: exits 0 and writes trajectory/replay logs.

- [ ] **Step 3: Generate review video**
  - Use SUMO-GUI screenshot/video generation if available; otherwise render SUMO trajectory frames from TraCI/FCD-style logs with ffmpeg.
  - Expected: fixed review MP4 under the v2 package, plus a paired evidence note.

### Task 4: Final Verification

**Files:**
- Use tests and generated artifacts.

- [ ] **Step 1: Run full unit test suite**
  - Run: `python3 -m unittest discover -s tests`
  - Expected: all tests pass.

- [ ] **Step 2: Verify artifact inventory**
  - Confirm package path, fixed SUMO config, traceability table, validation summary, and test video exist.

- [ ] **Step 3: Report exact paths**
  - Provide the fixed package path and review video path to the user.
