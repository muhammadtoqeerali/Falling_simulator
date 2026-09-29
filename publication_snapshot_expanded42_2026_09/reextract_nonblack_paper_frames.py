#!/usr/bin/env python3

from pathlib import Path
import csv
import math
import time
import shutil

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path("/mnt/hdd16T/ToqeerHomeBackup/mujoco_project")
OUT = ROOT / "outputs" / "PAPER_VIDEO_FRAME_LIBRARY_FILTERED"
ARCH = ROOT / "archives"

VIDEOS = {
    "01_chair_getting_up": ROOT / "outputs" / "scenario23_age25_h1p62_sex_female_w58p0_20260824_083429" / "fall_scenario23_age25_20260824_083833_highrate_truth_display.mp4",
    "02_sitting_fainting": ROOT / "outputs" / "scenario25_age25_h1p62_sex_female_w58p0_20260824_083843" / "fall_scenario25_age25_20260824_084000_highrate_truth_display.mp4",
    "03_height_fall": ROOT / "outputs" / "scenario39_age25_h1p62_sex_female_w58p0_20260824_084007" / "fall_forward_height_age25_20260824_084213_highrate_truth_display.mp4",
    "04_ladder_fall": ROOT / "outputs" / "scenario43_age25_h1p62_sex_female_w58p0_20260824_084238" / "fall_forward_ladder_age25_20260824_084402_highrate_truth_display.mp4",
}

TARGET_EXTRACT_FPS = 2.0
MAX_KEEP_PER_VIDEO = 400

# Black-frame filtering thresholds
MIN_MEAN_BRIGHTNESS = 18.0
MIN_STD_BRIGHTNESS = 6.0

# Duplicate filtering
MIN_FRAME_DIFF = 2.0

CONTACT_SHEET_COLS = 4
CONTACT_SHEET_ROWS = 4
THUMB_W = 320
THUMB_H = 180

def ensure_dir(p):
    p.mkdir(parents=True, exist_ok=True)

def frame_stats(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return float(gray.mean()), float(gray.std()), gray

def is_black_or_empty(frame_bgr):
    mean_b, std_b, _ = frame_stats(frame_bgr)
    if mean_b < MIN_MEAN_BRIGHTNESS:
        return True
    if std_b < MIN_STD_BRIGHTNESS:
        return True
    return False

def small_gray_signature(gray):
    small = cv2.resize(gray, (64, 36), interpolation=cv2.INTER_AREA)
    return small.astype(np.float32)

def mean_abs_diff(a, b):
    return float(np.mean(np.abs(a - b)))

def save_contact_sheets(image_paths, out_dir, label):
    if not image_paths:
        return 0

    per_sheet = CONTACT_SHEET_COLS * CONTACT_SHEET_ROWS
    total = len(image_paths)
    n_sheets = math.ceil(total / per_sheet)

    sheet_count = 0

    for sheet_idx in range(n_sheets):
        chunk = image_paths[sheet_idx * per_sheet : (sheet_idx + 1) * per_sheet]

        canvas_w = CONTACT_SHEET_COLS * THUMB_W
        canvas_h = CONTACT_SHEET_ROWS * (THUMB_H + 28)

        canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
        draw = ImageDraw.Draw(canvas)

        for i, img_path in enumerate(chunk):
            row = i // CONTACT_SHEET_COLS
            col = i % CONTACT_SHEET_COLS

            x = col * THUMB_W
            y = row * (THUMB_H + 28)

            img = Image.open(img_path).convert("RGB")
            img = img.resize((THUMB_W, THUMB_H))

            canvas.paste(img, (x, y))
            draw.text((x + 6, y + THUMB_H + 4), img_path.stem, fill="black")

        out_file = out_dir / f"{label}__contact_sheet_{sheet_idx+1:02d}.jpg"
        canvas.save(out_file, quality=92)
        sheet_count += 1

    return sheet_count

def process_video(label, video_path):
    print("\n" + "=" * 78)
    print(label)
    print("=" * 78)
    print("Source:", video_path)

    if not video_path.exists():
        print("MISSING VIDEO")
        return {
            "label": label,
            "video_path": str(video_path),
            "exists": False,
            "duration_s": "",
            "fps": "",
            "frames_total": 0,
            "frames_sampled": 0,
            "frames_black_skipped": 0,
            "frames_duplicate_skipped": 0,
            "frames_saved": 0,
            "contact_sheets": 0,
        }

    out_dir = OUT / label
    frames_dir = out_dir / "frames"
    ensure_dir(frames_dir)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("FAILED TO OPEN VIDEO")
        return {
            "label": label,
            "video_path": str(video_path),
            "exists": True,
            "duration_s": "",
            "fps": "",
            "frames_total": 0,
            "frames_sampled": 0,
            "frames_black_skipped": 0,
            "frames_duplicate_skipped": 0,
            "frames_saved": 0,
            "contact_sheets": 0,
        }

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = (total_frames / fps) if fps and fps > 0 else 0.0

    sample_every = max(int(round(fps / TARGET_EXTRACT_FPS)), 1) if fps and fps > 0 else 15

    print(f"FPS: {fps:.3f}")
    print(f"Total frames: {total_frames}")
    print(f"Duration: {duration_s:.3f} s")
    print(f"Sampling every {sample_every} frames (~{TARGET_EXTRACT_FPS} fps target)")

    manifest_rows = []
    saved_paths = []

    idx = 0
    sampled = 0
    black_skipped = 0
    duplicate_skipped = 0
    saved = 0

    prev_sig = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if idx % sample_every != 0:
            idx += 1
            continue

        sampled += 1

        if is_black_or_empty(frame):
            black_skipped += 1
            idx += 1
            continue

        mean_b, std_b, gray = frame_stats(frame)
        sig = small_gray_signature(gray)

        if prev_sig is not None:
            diff = mean_abs_diff(sig, prev_sig)
            if diff < MIN_FRAME_DIFF:
                duplicate_skipped += 1
                idx += 1
                continue

        prev_sig = sig

        t = idx / fps if fps and fps > 0 else 0.0

        out_name = f"frame_{saved+1:04d}_t{t:07.2f}s.jpg"
        out_path = frames_dir / out_name

        cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

        manifest_rows.append({
            "frame_index": idx,
            "time_s": round(t, 3),
            "mean_brightness": round(mean_b, 3),
            "std_brightness": round(std_b, 3),
            "file": out_name,
        })
        saved_paths.append(out_path)
        saved += 1

        if saved >= MAX_KEEP_PER_VIDEO:
            break

        idx += 1

    cap.release()

    manifest_file = out_dir / "frame_manifest.csv"
    with open(manifest_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["frame_index", "time_s", "mean_brightness", "std_brightness", "file"]
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    sheet_count = save_contact_sheets(saved_paths, out_dir, label)

    print(f"Sampled          : {sampled}")
    print(f"Black skipped    : {black_skipped}")
    print(f"Duplicate skipped: {duplicate_skipped}")
    print(f"Saved frames     : {saved}")
    print(f"Contact sheets   : {sheet_count}")

    return {
        "label": label,
        "video_path": str(video_path),
        "exists": True,
        "duration_s": round(duration_s, 3),
        "fps": round(fps, 3) if fps else "",
        "frames_total": total_frames,
        "frames_sampled": sampled,
        "frames_black_skipped": black_skipped,
        "frames_duplicate_skipped": duplicate_skipped,
        "frames_saved": saved,
        "contact_sheets": sheet_count,
    }

def main():
    print("=" * 78)
    print("RE-EXTRACTING NON-BLACK PAPER VIDEO FRAMES")
    print("=" * 78)

    if OUT.exists():
        shutil.rmtree(OUT)
    ensure_dir(OUT)
    ensure_dir(ARCH)

    summary = []

    for label, video_path in VIDEOS.items():
        result = process_video(label, video_path)
        summary.append(result)

    summary_file = OUT / "summary.csv"
    with open(summary_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "label",
                "video_path",
                "exists",
                "duration_s",
                "fps",
                "frames_total",
                "frames_sampled",
                "frames_black_skipped",
                "frames_duplicate_skipped",
                "frames_saved",
                "contact_sheets",
            ]
        )
        writer.writeheader()
        writer.writerows(summary)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    zipbase = ARCH / f"PAPER_VIDEO_FRAME_LIBRARY_FILTERED_{timestamp}"
    zipfile = shutil.make_archive(str(zipbase), "zip", root_dir=OUT)

    print("\n" + "=" * 78)
    print("COMPLETE")
    print("=" * 78)
    print("Output folder:", OUT)
    print("Summary file :", summary_file)
    print("ZIP          :", zipfile)

if __name__ == "__main__":
    main()
