#!/usr/bin/env python3
"""Build logical vehicle consistency outputs for YOLO target tracks.

This command is target-consistency only. It does not infer lanes, OD, turns,
SUMO routes, or demand.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from build_logical_vehicle_id_video import probe_video_frame_count, render_logical_vehicle_video
from logical_vehicle_consistency import (
    build_logical_vehicle_consistency,
    read_csv,
    write_consistency_outputs,
)


def logical_video_output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "final": output_dir / "logical_vehicle_id_final.mp4",
        "debug": output_dir / "logical_vehicle_id_debug.mp4",
        "review": output_dir / "logical_vehicle_id_review.mp4",
    }


def logical_window_output_paths(output_dir: Path, window_count: int = 3) -> dict[str, Path]:
    return {
        f"window_{index:02d}": output_dir / f"window_{index:02d}" / "logical_vehicle_id_final.mp4"
        for index in range(1, window_count + 1)
    }


def equal_frame_windows(total_frames: int, window_count: int = 3) -> list[tuple[int, int]]:
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    if window_count <= 0:
        raise ValueError("window_count must be positive")
    boundaries = [0] + [math.ceil(total_frames * index / window_count) for index in range(1, window_count)] + [total_frames]
    return [(boundaries[index], boundaries[index + 1]) for index in range(window_count)]


def render_window_slice_videos(
    clip_path: Path,
    logical_rows: list[dict],
    output_dir: Path,
    fps: float,
    window_count: int = 3,
) -> dict[str, Path]:
    total_frames = probe_video_frame_count(clip_path)
    windows = equal_frame_windows(total_frames, window_count=window_count)
    paths = logical_window_output_paths(output_dir, window_count=window_count)
    for (window_name, path), (start_frame, end_frame) in zip(paths.items(), windows):
        render_logical_vehicle_video(
            clip_path=clip_path,
            logical_rows=logical_rows,
            output_path=path,
            mode="final",
            fps=fps,
            start_frame=start_frame,
            end_frame=end_frame,
        )
        (path.parent / "window_manifest.txt").write_text(
            f"window={window_name}\nstart_frame={start_frame}\nend_frame={end_frame}\n",
            encoding="utf-8",
        )
    return paths


def ensure_review_asset_dirs(output_dir: Path) -> None:
    for name in ["duplicate_groups", "ambiguous_links", "logical_vehicle_triplets"]:
        (output_dir / "review_assets" / name).mkdir(parents=True, exist_ok=True)


def write_version_note(output_dir: Path, outputs, video_paths: dict[str, Path]) -> None:
    lines = [
        "# Logical Vehicle Consistency V3",
        "",
        "This output only addresses target consistency tracking.",
        "It does not infer lanes, OD, turns, SUMO routes, or demand.",
        "",
        f"- logical_vehicle_track_rows: `{len(outputs.logical_tracks)}`",
        f"- logical_vehicle_summary_rows: `{len(outputs.logical_vehicle_summary)}`",
        f"- duplicate_groups: `{len(outputs.duplicate_groups)}`",
        f"- accepted_links: `{len(outputs.tracklet_links_accepted)}`",
        f"- ambiguous_links: `{len(outputs.ambiguous_link_review)}`",
        "",
        "Video outputs:",
    ]
    for mode, path in video_paths.items():
        lines.append(f"- `{mode}`: `{path.name}`")
    (output_dir / "LOGICAL_VEHICLE_CONSISTENCY_V3_NOTE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_outputs(
    detections_path: Path,
    final_targets_path: Path,
    clip_path: Path,
    output_dir: Path,
    fps: float,
    max_gap_frames: int,
    max_link_distance_px: float,
    max_iou: float,
    render_videos: bool = True,
    render_window_slices: bool = False,
    window_slice_count: int = 3,
) -> None:
    detections = read_csv(detections_path)
    final_targets = read_csv(final_targets_path)
    outputs = build_logical_vehicle_consistency(
        detections=detections,
        final_targets=final_targets,
        fps=fps,
        max_gap_frames=max_gap_frames,
        max_link_distance_px=max_link_distance_px,
        max_iou=max_iou,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_review_asset_dirs(output_dir)
    write_consistency_outputs(output_dir, outputs)
    video_paths = logical_video_output_paths(output_dir)
    if render_videos:
        for mode, path in video_paths.items():
            render_logical_vehicle_video(
                clip_path=clip_path,
                logical_rows=outputs.logical_tracks,
                output_path=path,
                mode=mode,
                fps=fps,
            )
        if render_window_slices:
            render_window_slice_videos(
                clip_path=clip_path,
                logical_rows=outputs.logical_tracks,
                output_dir=output_dir,
                fps=fps,
                window_count=window_slice_count,
            )
    write_version_note(output_dir, outputs, video_paths)
    print(f"logical_vehicle_tracks={len(outputs.logical_tracks)}")
    print(f"logical_vehicle_summary={len(outputs.logical_vehicle_summary)}")
    print(f"duplicate_groups={len(outputs.duplicate_groups)}")
    print(f"accepted_links={len(outputs.tracklet_links_accepted)}")
    print(f"ambiguous_links={len(outputs.ambiguous_link_review)}")
    print(f"output_dir={output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections", required=True)
    parser.add_argument("--final-targets", required=True)
    parser.add_argument("--clip", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--max-gap-frames", type=int, default=10)
    parser.add_argument("--max-link-distance-px", type=float, default=80.0)
    parser.add_argument("--max-iou", type=float, default=0.85)
    parser.add_argument("--skip-videos", action="store_true")
    parser.add_argument("--window-slices", action="store_true")
    parser.add_argument("--window-slice-count", type=int, default=3)
    args = parser.parse_args()

    build_outputs(
        detections_path=Path(args.detections),
        final_targets_path=Path(args.final_targets),
        clip_path=Path(args.clip),
        output_dir=Path(args.output_dir),
        fps=args.fps,
        max_gap_frames=args.max_gap_frames,
        max_link_distance_px=args.max_link_distance_px,
        max_iou=args.max_iou,
        render_videos=not args.skip_videos,
        render_window_slices=args.window_slices,
        window_slice_count=args.window_slice_count,
    )


if __name__ == "__main__":
    main()
