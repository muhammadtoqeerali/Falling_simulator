#!/usr/bin/env python3

import csv
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ==========================================================
# CONFIG
# ==========================================================

PROJECT_ROOT = Path("/mnt/hdd16T/ToqeerHomeBackup/mujoco_project")

RUNS_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "_highrate_overnight"
    / "campaign_highrate_truth_v2_extended35"
    / "runs"
)

OUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "paper_scenario_media_selected"
)

ARCHIVE_ROOT = (
    PROJECT_ROOT
    / "archives"
)

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

SCENARIO_MAP = {
    "stairs_ladder": {
        "tasks": [41, 42, 43],
        "label": "stairs / ladder related"
    },
    "height": {
        "tasks": [39, 40, 44],
        "label": "fall from height / obstacle"
    },
    "chair": {
        "tasks": [20, 21, 22, 23, 24],
        "label": "chair / sit-stand fall"
    },
    "sitting": {
        "tasks": [25, 26, 27],
        "label": "seated / fainting while sitting"
    },
}

FRAME_FRACTIONS = [0.10, 0.25, 0.40, 0.60, 0.80]
MAX_VIDEOS_PER_CATEGORY = 1

# ==========================================================
# HELPERS
# ==========================================================

def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


def ensure_tools():
    for tool in ["ffmpeg", "ffprobe"]:
        if shutil.which(tool) is None:
            raise SystemExit(f"ERROR: required tool not found: {tool}")


def parse_task(path: Path):
    s = str(path)
    m = re.search(r"task[_\-](\d+)", s, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def get_duration(video_path: Path):
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path)
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    return float(out)


def extract_frame(video_path: Path, time_sec: float, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", f"{time_sec:.3f}",
        "-i", str(video_path),
        "-frames:v", "1",
        "-q:v", "2",
        str(out_path)
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def find_all_videos(root: Path):
    videos = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            videos.append(p)
    return sorted(videos)


def select_videos(videos):
    categorized = {k: [] for k in SCENARIO_MAP}

    for v in videos:
        task = parse_task(v)
        if task is None:
            continue
        for cat, cfg in SCENARIO_MAP.items():
            if task in cfg["tasks"]:
                categorized[cat].append(v)

    selected = {}

    for cat, items in categorized.items():
        # Prefer larger files in case they are the real full render videos
        items = sorted(
            items,
            key=lambda p: (-p.stat().st_size, str(p))
        )
        selected[cat] = items[:MAX_VIDEOS_PER_CATEGORY]

    return selected


# ==========================================================
# MAIN
# ==========================================================

def main():
    ensure_tools()

    if not RUNS_ROOT.exists():
        raise SystemExit(f"ERROR: RUNS_ROOT not found: {RUNS_ROOT}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    log("Searching for scenario videos...")
    videos = find_all_videos(RUNS_ROOT)
    log(f"Total videos found: {len(videos)}")

    if not videos:
        raise SystemExit("ERROR: no videos found under runs directory")

    selected = select_videos(videos)

    summary_rows = []
    frame_rows = []

    for category, picked in selected.items():
        cat_dir = OUT_ROOT / category
        cat_dir.mkdir(parents=True, exist_ok=True)

        if not picked:
            log(f"No video found for category: {category}")
            continue

        for idx, video in enumerate(picked, start=1):
            task = parse_task(video)
            duration = get_duration(video)

            vid_dir = cat_dir / f"task_{task:02d}_video_{idx}"
            frames_dir = vid_dir / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)

            # Copy the source video for traceability
            copied_video = vid_dir / video.name
            shutil.copy2(video, copied_video)

            summary_rows.append({
                "category": category,
                "category_label": SCENARIO_MAP[category]["label"],
                "task": task,
                "source_video": str(video),
                "copied_video": str(copied_video),
                "duration_sec": round(duration, 3),
            })

            for frame_id, frac in enumerate(FRAME_FRACTIONS, start=1):
                t = max(0.0, min(duration - 0.05, duration * frac))
                out_img = frames_dir / f"{category}_task{task:02d}_frame{frame_id}_{t:.2f}s.jpg"
                extract_frame(video, t, out_img)

                frame_rows.append({
                    "category": category,
                    "task": task,
                    "source_video": str(video),
                    "frame_file": str(out_img),
                    "time_sec": round(t, 3),
                })

            log(f"Extracted frames for {category} | task {task} | {video.name}")

    # Save inventories
    with open(OUT_ROOT / "selected_videos.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "category",
                "category_label",
                "task",
                "source_video",
                "copied_video",
                "duration_sec",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    with open(OUT_ROOT / "extracted_frames.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "category",
                "task",
                "source_video",
                "frame_file",
                "time_sec",
            ],
        )
        writer.writeheader()
        writer.writerows(frame_rows)

    with open(OUT_ROOT / "selection_summary.json", "w") as f:
        json.dump(
            {
                "runs_root": str(RUNS_ROOT),
                "output_root": str(OUT_ROOT),
                "selected_categories": list(selected.keys()),
                "selected_videos_count": len(summary_rows),
                "extracted_frames_count": len(frame_rows),
            },
            f,
            indent=2,
        )

    # Create ZIP
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    zip_base = ARCHIVE_ROOT / f"paper_scenario_media_selected_{timestamp}"
    shutil.make_archive(str(zip_base), "zip", root_dir=OUT_ROOT)

    log("DONE")
    log(f"Output folder: {OUT_ROOT}")
    log(f"ZIP archive  : {zip_base}.zip")
    log(f"Selected vids: {len(summary_rows)}")
    log(f"Frames       : {len(frame_rows)}")


if __name__ == "__main__":
    main()
