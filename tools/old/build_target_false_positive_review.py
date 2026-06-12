#!/usr/bin/env python3
"""Build a focused false-positive review page for MOT target tracks."""

from __future__ import annotations

import argparse
import csv
import html
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(row: dict, key: str) -> float:
    return float(row[key])


def overlay_frame_name(frame_id) -> str:
    return f"target_track_f{int(float(frame_id)):06d}.jpg"


def select_overlay_aligned_rows(rows: list[dict], overlay_step: int = 25) -> list[dict]:
    ordered = sorted(rows, key=lambda row: int(float(row["frame_id"])))
    aligned = [row for row in ordered if int(float(row["frame_id"])) % overlay_step == 0]
    candidates = aligned if aligned else ordered
    if len(candidates) <= 3:
        return candidates
    first = candidates[0]
    last = candidates[-1]
    mid_frame = (int(float(first["frame_id"])) + int(float(last["frame_id"]))) / 2.0
    middle = min(candidates, key=lambda row: abs(int(float(row["frame_id"])) - mid_frame))
    selected = [first, middle, last]
    deduped = []
    seen = set()
    for row in selected:
        key = row["frame_id"]
        if key not in seen:
            deduped.append(row)
            seen.add(key)
    return deduped


def crop_review_image(
    overlay_dir: Path,
    crop_dir: Path,
    track_id: str,
    row: dict,
    label: str,
    crop_size: tuple[int, int] = (520, 360),
) -> str:
    frame_id = int(float(row["frame_id"]))
    source_path = overlay_dir / overlay_frame_name(frame_id)
    if not source_path.exists():
        return ""

    image = Image.open(source_path).convert("RGB")
    width, height = image.size
    x1, y1, x2, y2 = [as_float(row, key) for key in ("x1", "y1", "x2", "y2")]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    crop_w, crop_h = crop_size
    left = int(max(0, min(width - crop_w, cx - crop_w / 2)))
    top = int(max(0, min(height - crop_h, cy - crop_h / 2)))
    cropped = image.crop((left, top, left + crop_w, top + crop_h))

    draw = ImageDraw.Draw(cropped)
    draw.rectangle([x1 - left, y1 - top, x2 - left, y2 - top], outline=(255, 70, 45), width=5)
    draw.text((12, 12), f"{track_id} {label} t={as_float(row, 'time_sec'):.1f}s", fill=(255, 255, 255))

    crop_dir.mkdir(parents=True, exist_ok=True)
    output_name = f"{track_id}_{label}_f{frame_id:06d}.jpg"
    cropped.save(crop_dir / output_name, quality=92)
    return output_name


def detections_by_track(detections: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in detections:
        track_id = f"mot_{int(float(row['track_id'])):04d}"
        grouped[track_id].append(row)
    return grouped


def build_cards(review_rows: list[dict], grouped_detections: dict[str, list[dict]], overlay_dir: Path, crop_dir: Path) -> str:
    cards = []
    for review in review_rows:
        track_id = review["track_id"]
        selected = select_overlay_aligned_rows(grouped_detections.get(track_id, []))
        figures = []
        for label, row in zip(["first", "middle", "last"], selected):
            crop_name = crop_review_image(overlay_dir, crop_dir, track_id, row, label)
            full_name = overlay_frame_name(row["frame_id"])
            if crop_name:
                figures.append(
                    f"""
                    <figure>
                      <img src="../target_tracks/false_positive_review_crops/{html.escape(crop_name)}" alt="{html.escape(track_id)} {label}">
                      <figcaption>{label} | t={as_float(row, "time_sec"):.1f}s | <a href="../target_tracks/target_track_overlay_frames/{html.escape(full_name)}">完整 overlay</a></figcaption>
                    </figure>
                    """
                )
        cards.append(
            f"""
            <article class="fp-card">
              <div class="card-head">
                <h3>{html.escape(track_id)} | {html.escape(review["class_name"])}</h3>
                <span>{html.escape(review["risk_labels"])}</span>
              </div>
              <div class="metrics">
                <span>t={html.escape(review["start_time"])}-{html.escape(review["end_time"])}s</span>
                <span>frames={html.escape(review["frame_count"])}</span>
                <span>conf={html.escape(review["mean_confidence"])}</span>
                <span>disp={html.escape(review["max_displacement_px"])}px</span>
              </div>
              <div class="decision-guide">
                <b>KEEP</b>：主车流机动车；<b>EXCLUDE</b>：非机动车、路边停车、非路面/非主车流目标；<b>FLAG</b>：看不准。
              </div>
              <div class="crop-grid">{''.join(figures) or '<p>没有可用 crop，请打开完整 overlay 检查。</p>'}</div>
            </article>
            """
        )
    return "\n".join(cards)


def build_html(review_rows: list[dict], grouped_detections: dict[str, list[dict]], overlay_dir: Path, crop_dir: Path) -> str:
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MVI_0866 false-positive 目标审计</title>
  <style>
    :root {{
      --ink: #1f2933;
      --muted: #657381;
      --line: #d7dde5;
      --paper: #f4f1ea;
      --panel: #fff;
      --blue: #17364c;
      --green: #236b4f;
      --amber: #97650f;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--paper);
      color: var(--ink);
      font-family: "Songti SC", "Noto Serif CJK SC", "Times New Roman", serif;
    }}
    header {{
      background: var(--blue);
      color: white;
      padding: 22px 28px;
      border-bottom: 4px solid #c59c55;
    }}
    header h1 {{ margin: 0 0 8px; font-size: 28px; }}
    header p {{ margin: 0; color: #dce7ef; line-height: 1.6; }}
    main {{ max-width: 1500px; margin: 0 auto; padding: 18px; }}
    section, .fp-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 16px;
      margin-bottom: 16px;
      box-shadow: 0 10px 24px rgba(22, 32, 42, 0.10);
    }}
    .guide {{ border-left: 6px solid var(--green); background: #eef8f2; }}
    .card-head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      border-bottom: 1px solid var(--line);
      padding-bottom: 10px;
      margin-bottom: 10px;
    }}
    h2 {{ margin: 0 0 10px; }}
    h3 {{ margin: 0; }}
    .card-head span, .metrics span {{
      color: var(--muted);
      font: 13px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }}
    .metrics {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 10px;
    }}
    .metrics span {{
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 5px 9px;
      background: #f6f8fa;
    }}
    .decision-guide {{
      border: 1px solid #d8c28b;
      border-left: 5px solid var(--amber);
      background: #fff9e8;
      border-radius: 6px;
      padding: 10px;
      margin-bottom: 12px;
      line-height: 1.55;
    }}
    .crop-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
    }}
    figure {{
      margin: 0;
      background: #101820;
      border: 1px solid #263541;
      border-radius: 5px;
      overflow: hidden;
    }}
    figure img {{
      display: block;
      width: 100%;
      aspect-ratio: 13 / 9;
      object-fit: cover;
    }}
    figcaption {{
      background: #eef3f7;
      padding: 7px 9px;
      font: 13px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }}
    a {{ color: #1d5b7a; font-weight: 700; text-decoration: none; }}
    @media (max-width: 900px) {{
      .crop-grid {{ grid-template-columns: 1fr; }}
      .card-head {{ display: block; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>MVI_0866 false-positive 目标审计</h1>
    <p>本页只处理误检/非关注目标剔除：非机动车、路边停车、非路面或非主车流目标。不要补车、不要画框、不要改 track id。</p>
  </header>
  <main>
    <section class="guide">
      <h2>你要做什么</h2>
      <ol>
        <li>每张卡看 first / middle / last 三个局部图。</li>
        <li>如果是主车流机动车，记 KEEP。</li>
        <li>如果是非机动车、路边停车、非路面车辆或非主车流目标，记 EXCLUDE。</li>
        <li>看不准记 FLAG。不要人工新增任何车辆。</li>
      </ol>
      <p>候选数：<b>{len(review_rows)}</b></p>
    </section>
    {build_cards(review_rows, grouped_detections, overlay_dir, crop_dir)}
  </main>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-gate", required=True)
    parser.add_argument("--detections", required=True)
    parser.add_argument("--overlay-dir", required=True)
    parser.add_argument("--crop-dir", required=True)
    parser.add_argument("--output-html", required=True)
    args = parser.parse_args()

    review_rows = read_csv(Path(args.review_gate))
    detections = read_csv(Path(args.detections))
    grouped = detections_by_track(detections)
    crop_dir = Path(args.crop_dir)
    output_html = Path(args.output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(build_html(review_rows, grouped, Path(args.overlay_dir), crop_dir), encoding="utf-8")
    print(f"false_positive_candidates={len(review_rows)}")
    print(f"html={output_html}")


if __name__ == "__main__":
    main()
