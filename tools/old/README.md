# Archived Tools

This directory stores earlier or downstream helper scripts that are not part of
the active logical vehicle consistency v3 toolchain.

Archived categories include:

- YOLO target extraction and false-positive review
- center-follow and track-stitch mock experiments
- direction anchor and direction-ready helpers
- SUMO candidate, replay, and training scenario builders
- manual alias and exclusion helpers
- older review video builders

The active project surface remains in the parent `tools/` directory:

- `logical_vehicle_consistency.py`
- `build_logical_vehicle_consistency_v1.py`
- `build_logical_vehicle_id_video.py`

Old tests are archived under `tools/old/tests/` and are not included in the
default `python3 -m unittest discover tests` run.
