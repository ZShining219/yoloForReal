# Agent Guide

## Identity And Reply Rules
- Every final result reply to the user must start with `Dear Z`.
- Keep replies concise, warm, and action-oriented. Explain what changed, how it was verified, and what remains.
- If a task is ambiguous, make the safest useful assumption and state it briefly before acting.

## Project Context
- Goal: maintain an independent logical vehicle consistency module for the `MVI_0866_520_560.mp4` real-video clip.
- The active pipeline consumes YOLO/ByteTrack detection evidence and reviewed raw target tracks, then builds stable `logical_vehicle_id` trajectories.
- Scope is target identity consistency only: duplicate suppression, tracklet construction, tracklet association, same-raw continuity, fragment-to-mature path absorption, validity/purity/final gates, and logical-ID review videos.
- This repository does not infer lanes, lane crossings, OD, turning movements, SUMO routes, or traffic demand.
- Prefer reproducible scripts and documented parameters over one-off notebook-only workflows.

## Active Project Surface
- Active tools live at the top of `tools/`:
  - `tools/logical_vehicle_consistency.py`
  - `tools/build_logical_vehicle_consistency_v1.py`
  - `tools/build_logical_vehicle_id_video.py`
- Archived tools live under `tools/old/` and are kept for traceability only.
- Active tests live under `tests/`.
- Archived tests for old tools live under `tools/old/tests/` and are not part of the default test suite.

## Output Convention
- Current generated outputs use:
  - `outputs/logical_vehicle_consistency/v3/`
- Historical, upstream, and experimental outputs use:
  - `outputs/old/`
- Keep this layout stable for future versions:
  - `outputs/<module_name>/<version>/`
- Current v3 CSV reports and small manifest files may be committed when they are useful audit evidence.
- Do not commit generated videos, large images, model weights, raw video clips, cache folders, or archived bulk outputs.

## Working Conventions
- Read existing files before editing and keep changes narrowly scoped.
- Preserve user data and generated artifacts unless explicitly asked to clean them.
- Use `rg`/`rg --files` for searches when available.
- Prefer simple, local, inspectable solutions before introducing new frameworks or services.
- Keep code and workflow decisions documented only where they help future execution.
- When behavior changes, add or update focused unit tests before changing production logic.

## File Hygiene
- Do not create loose documents, scratch files, or ad hoc notes in the project root.
- Durable documentation belongs in `docs/`, `README.md`, `outputs/README.md`, or a nearby README such as `tools/old/README.md`.
- Temporary exploration belongs in `.scratch/`, `tmp/`, or `/tmp`; clean it up before finishing unless it is useful evidence.
- Before adding a new top-level directory, check whether an existing directory already fits the purpose.
- If a file is no longer useful after a task, remove it or explicitly mention why it is being kept.

## File Lifecycle And Source Of Truth
- Prefer updating the current authoritative file over creating a parallel replacement.
- Do not create files named like `new`, `final`, `v2`, `backup`, `copy`, or similar version-by-filename variants.
- For active outputs, use directory versioning such as `outputs/logical_vehicle_consistency/v3/` instead of scattered filename variants.
- If old implementation files must be retained, move them under `tools/old/` and update `tools/old/README.md` when the boundary changes.
- Do not allow multiple long-lived implementations of the same active pipeline step unless their boundaries are documented.

## Documentation Creation Gate
- Do not create Markdown documents during conversation by default.
- First explain documentation recommendations in the chat and ask for user approval before writing a new Markdown document.
- Only create documentation after the user agrees to turn the recommendation into a file.
- When updating existing documentation, edit the authoritative file in place and avoid creating duplicate summaries.
- Root Markdown files should be limited to recognized project entrypoints such as `README.md`, `AGENTS.md`, or `CHANGELOG.md`.

## Optional Task Tracking
- Use `docs/tasks/TASKS.md` only when the user explicitly asks to track tasks in a persistent project task list.
- Do not create or update task tracking files automatically.
- If task tracking is enabled, keep current open tasks and the two most recent closed tasks, and ask the user before closing a task.

## Git Discipline
- Check `git status --short --branch` before and after meaningful changes.
- Commit at the end of each coherent unit of work when the workspace is in a verified state.
- Use small, descriptive commits that make rollback easy.
- Before risky edits, create a checkpoint commit if there are valuable uncommitted changes.
- Never rewrite, delete, or revert user changes unless the user explicitly asks.
- Do not commit ignored/generated artifacts, local secrets, large videos, model weights, or cache directories.
- Good commit messages for this repo include `adapt agent guide`, `fix fragment absorption`, or `standardize output layout`.

## Expected Structure
- `tools/`: active v3 logical vehicle consistency scripts.
- `tools/old/`: archived scripts from earlier target extraction, direction, review, and SUMO experiments.
- `tests/`: active unit tests for the v3 logical consistency module.
- `outputs/logical_vehicle_consistency/v3/`: canonical current v3 CSV/manifests and locally generated videos.
- `outputs/old/`: archived historical, upstream, and experimental outputs.
- `inputs/`: local input videos/model weights; large files are ignored by Git.
- `docs/`: durable notes, assumptions, and workflow documentation.
- `.scratch/` or `tmp/`: temporary local exploration if needed; ignored by Git.

## Verification
- Default unit test command:
  ```bash
  python3 -m unittest discover tests
  ```
- Lightweight rebuild check without video rendering:
  ```bash
  python3 tools/build_logical_vehicle_consistency_v1.py \
    --detections outputs/old/yolo26x_trial/target_tracks/detections_tracked.csv \
    --final-targets outputs/old/yolo26x_manual_filter_v1/target_tracks_final.csv \
    --clip inputs/video_clips/MVI_0866_520_560.mp4 \
    --output-dir /tmp/logical_vehicle_consistency_check \
    --fps 50.0 \
    --max-gap-frames 10 \
    --max-link-distance-px 80.0 \
    --max-iou 0.85 \
    --skip-videos
  ```
- Full v3 rebuild with videos uses `--output-dir outputs/logical_vehicle_consistency/v3 --window-slices`.
- Report tests or checks that were run. If checks were skipped, explain why.
