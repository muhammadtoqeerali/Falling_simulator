#!/usr/bin/env python3

from pathlib import Path
import os
import re
import csv
import json
import shutil
import subprocess
import time

ROOT = Path("/mnt/hdd16T/ToqeerHomeBackup/mujoco_project")
OUT = ROOT / "outputs" / "paper_scenario_frame_candidates"
ARCHIVES = ROOT / "archives"

VIDEO_EXTS = {
    ".mp4", ".avi", ".mov", ".mkv",
    ".webm", ".m4v", ".mpg", ".mpeg"
}

IMAGE_EXTS = {
    ".png", ".jpg", ".jpeg"
}

# Do NOT use the broad word "high", because "highrate"
# occurs throughout this project and would generate false matches.
CATEGORIES = {
    "01_ladder_stairs": [
        "ladder",
        "stairs",
        "stair",
        "staircase",
        "step_down",
        "stepdown",
        "step_up",
        "stepup",
    ],

    "02_height_platform": [
        "height",
        "platform",
        "elevated",
        "ledge",
        "fall_from_height",
        "fallfromheight",
        "drop_from",
    ],

    "03_chair": [
        "chair",
        "stool",
        "armchair",
        "sofa",
        "couch",
    ],

    "04_sitting": [
        "sitting",
        "seated",
        "sit_down",
        "sitdown",
        "sit-to",
        "sit_to",
    ],
}

SKIP_DIR_NAMES = {
    ".git",
    ".venv",
    "__pycache__",
    "archives",
    "paper_scenario_frame_candidates",
}

MAX_VIDEOS_PER_CATEGORY = 3

FRAME_FRACTIONS = [
    0.08,
    0.20,
    0.35,
    0.50,
    0.65,
    0.80,
    0.92,
]


def log(msg):
    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}",
        flush=True
    )


def norm(s):
    return str(s).lower().replace("-", "_").replace(" ", "_")


def path_score(path, keywords):
    p = norm(path)

    score = 0

    for kw in keywords:
        kw = norm(kw)

        if kw in p:
            score += 10

            # Extra preference when keyword appears in
            # filename or immediate directory.
            if kw in norm(path.name):
                score += 8

            if path.parent and kw in norm(path.parent.name):
                score += 5

    return score


def walk_files():

    for root, dirs, files in os.walk(ROOT):

        dirs[:] = [
            d for d in dirs
            if d not in SKIP_DIR_NAMES
        ]

        rootp = Path(root)

        for fn in files:
            yield rootp / fn


def ffprobe_duration(video):

    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video),
    ]

    try:
        out = subprocess.check_output(
            cmd,
            text=True,
            stderr=subprocess.DEVNULL
        ).strip()

        return float(out)

    except Exception:
        return None


def extract_frame(video, sec, outfile):

    outfile.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel", "error",
        "-ss", f"{sec:.4f}",
        "-i", str(video),
        "-frames:v", "1",
        "-q:v", "2",
        str(outfile),
    ]

    result = subprocess.run(cmd)

    return (
        result.returncode == 0
        and outfile.exists()
        and outfile.stat().st_size > 0
    )


def write_csv(path, rows):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if not rows:
        path.write_text("")
        return

    keys = list(rows[0].keys())

    with open(
        path,
        "w",
        newline=""
    ) as f:

        w = csv.DictWriter(
            f,
            fieldnames=keys
        )

        w.writeheader()
        w.writerows(rows)


def create_contact_sheet(category_dir):

    try:
        from PIL import Image, ImageOps, ImageDraw
    except Exception:
        return None

    imgs = sorted(
        category_dir.rglob("*.jpg")
    )

    imgs = [
        p for p in imgs
        if "contact_sheet" not in p.name
    ]

    if not imgs:
        return None

    thumb_w = 420
    thumb_h = 300
    margin = 15
    label_h = 35
    cols = 3

    rows = (
        len(imgs) + cols - 1
    ) // cols

    canvas_w = (
        cols * thumb_w
        + (cols + 1) * margin
    )

    canvas_h = (
        rows * (thumb_h + label_h)
        + (rows + 1) * margin
    )

    canvas = Image.new(
        "RGB",
        (canvas_w, canvas_h),
        "white"
    )

    draw = ImageDraw.Draw(canvas)

    for i, img_path in enumerate(imgs):

        r = i // cols
        c = i % cols

        x = (
            margin
            + c * (thumb_w + margin)
        )

        y = (
            margin
            + r * (
                thumb_h
                + label_h
                + margin
            )
        )

        try:
            im = Image.open(
                img_path
            ).convert("RGB")

            im = ImageOps.contain(
                im,
                (thumb_w, thumb_h)
            )

            bx = (
                x
                + (thumb_w - im.width) // 2
            )

            by = (
                y
                + (thumb_h - im.height) // 2
            )

            canvas.paste(
                im,
                (bx, by)
            )

            label = img_path.name[:60]

            draw.text(
                (x, y + thumb_h + 5),
                label,
                fill="black"
            )

        except Exception:
            pass

    outfile = (
        category_dir
        / "contact_sheet.jpg"
    )

    canvas.save(
        outfile,
        quality=92
    )

    return outfile


def scan_rendering_code():

    terms = [
        ".mp4",
        "VideoWriter",
        "imageio",
        "mediapy",
        "ffmpeg",
        "render(",
        "renderer",
        "write_video",
        "save_video",
    ]

    matches = []

    for p in ROOT.rglob("*.py"):

        if any(
            part in SKIP_DIR_NAMES
            for part in p.parts
        ):
            continue

        try:
            lines = p.read_text(
                errors="ignore"
            ).splitlines()
        except Exception:
            continue

        for idx, line in enumerate(
            lines,
            start=1
        ):

            if any(
                t.lower() in line.lower()
                for t in terms
            ):
                matches.append(
                    (
                        str(p.relative_to(ROOT)),
                        idx,
                        line.strip()
                    )
                )

    return matches


def scan_scenario_paths():

    matches = []

    for p in ROOT.rglob("*"):

        if any(
            part in SKIP_DIR_NAMES
            for part in p.parts
        ):
            continue

        text = norm(p)

        found = []

        for cat, kws in CATEGORIES.items():

            if any(
                norm(k) in text
                for k in kws
            ):
                found.append(cat)

        if found:
            matches.append(
                (
                    str(p.relative_to(ROOT)),
                    ",".join(found)
                )
            )

    return matches


def main():

    if OUT.exists():
        shutil.rmtree(OUT)

    OUT.mkdir(
        parents=True,
        exist_ok=True
    )

    log("Scanning entire MuJoCo project for videos...")

    all_files = list(
        walk_files()
    )

    videos = [
        p for p in all_files
        if p.suffix.lower() in VIDEO_EXTS
    ]

    existing_images = [
        p for p in all_files
        if p.suffix.lower() in IMAGE_EXTS
    ]

    log(
        f"Video files discovered: {len(videos)}"
    )

    log(
        f"Existing image files discovered: {len(existing_images)}"
    )

    inventory_rows = []

    for v in sorted(videos):

        category_scores = {
            cat: path_score(
                v,
                keywords
            )
            for cat, keywords
            in CATEGORIES.items()
        }

        best_cat = max(
            category_scores,
            key=category_scores.get
        )

        best_score = (
            category_scores[best_cat]
        )

        inventory_rows.append({
            "video": str(v),
            "size_mb":
                round(
                    v.stat().st_size
                    / 1024 / 1024,
                    3
                ),
            "best_category":
                best_cat
                if best_score > 0
                else "",
            "match_score":
                best_score,
        })

    write_csv(
        OUT / "all_video_inventory.csv",
        inventory_rows
    )

    # ------------------------------------------------------
    # Build verified candidates
    # ------------------------------------------------------

    selected = {}

    for category, keywords in CATEGORIES.items():

        candidates = []

        for v in videos:

            score = path_score(
                v,
                keywords
            )

            if score <= 0:
                continue

            candidates.append(
                (
                    score,
                    v.stat().st_size,
                    v
                )
            )

        candidates.sort(
            key=lambda x: (
                -x[0],
                -x[1],
                str(x[2])
            )
        )

        selected[category] = [
            x[2]
            for x in candidates[
                :MAX_VIDEOS_PER_CATEGORY
            ]
        ]

    selected_count = sum(
        len(v)
        for v in selected.values()
    )

    log(
        f"Verified keyword-matched scenario videos: {selected_count}"
    )

    # ------------------------------------------------------
    # If no useful video: produce diagnostic report
    # ------------------------------------------------------

    if not videos or selected_count == 0:

        log(
            "No verified scenario videos available for automatic extraction."
        )

        log(
            "Creating diagnostic discovery report instead."
        )

        scenario_paths = (
            scan_scenario_paths()
        )

        rendering_code = (
            scan_rendering_code()
        )

        report = (
            OUT
            / "VIDEO_DISCOVERY_REPORT.txt"
        )

        with open(
            report,
            "w"
        ) as f:

            f.write(
                "MUJOCO PAPER SCENARIO MEDIA DISCOVERY\n"
            )

            f.write(
                "=" * 70 + "\n\n"
            )

            f.write(
                f"Project root: {ROOT}\n"
            )

            f.write(
                f"Video files found: {len(videos)}\n"
            )

            f.write(
                f"Existing PNG/JPG images found: "
                f"{len(existing_images)}\n\n"
            )

            f.write(
                "VIDEOS FOUND\n"
            )

            f.write(
                "-" * 70 + "\n"
            )

            for v in videos:
                f.write(
                    str(v) + "\n"
                )

            f.write(
                "\n\nSCENARIO-RELATED PATHS\n"
            )

            f.write(
                "-" * 70 + "\n"
            )

            for path, cats in (
                scenario_paths[:1000]
            ):

                f.write(
                    f"[{cats}] {path}\n"
                )

            f.write(
                "\n\nPOSSIBLE VIDEO/RENDERING CODE\n"
            )

            f.write(
                "-" * 70 + "\n"
            )

            for path, line_no, text in (
                rendering_code[:2000]
            ):

                f.write(
                    f"{path}:{line_no}: {text}\n"
                )

        # Also save image paths that could themselves
        # already be usable renders.
        image_rows = []

        for p in existing_images:

            score_map = {
                cat: path_score(
                    p,
                    kws
                )
                for cat, kws
                in CATEGORIES.items()
            }

            best = max(
                score_map,
                key=score_map.get
            )

            if score_map[best] > 0:

                image_rows.append({
                    "image":
                        str(p),
                    "category":
                        best,
                    "score":
                        score_map[best],
                })

        write_csv(
            OUT
            / "existing_scenario_image_candidates.csv",
            image_rows
        )

        zip_name = (
            ARCHIVES
            / (
                "paper_scenario_media_DISCOVERY_"
                + time.strftime(
                    "%Y%m%d_%H%M%S"
                )
            )
        )

        shutil.make_archive(
            str(zip_name),
            "zip",
            root_dir=OUT
        )

        log(f"Report: {report}")
        log(
            f"Discovery ZIP: {zip_name}.zip"
        )

        return

    # ------------------------------------------------------
    # Extract selected video frames
    # ------------------------------------------------------

    if shutil.which("ffmpeg") is None:
        raise SystemExit(
            "ERROR: ffmpeg is required but not installed."
        )

    if shutil.which("ffprobe") is None:
        raise SystemExit(
            "ERROR: ffprobe is required but not installed."
        )

    selections = []
    extracted = []

    for category, vids in selected.items():

        catdir = (
            OUT
            / category
        )

        catdir.mkdir(
            parents=True,
            exist_ok=True
        )

        for vid_idx, video in enumerate(
            vids,
            start=1
        ):

            duration = ffprobe_duration(
                video
            )

            if not duration or duration <= 0:
                continue

            safe_video_name = re.sub(
                r"[^A-Za-z0-9_.-]+",
                "_",
                video.stem
            )[:100]

            vdir = (
                catdir
                / (
                    f"candidate_{vid_idx:02d}_"
                    f"{safe_video_name}"
                )
            )

            vdir.mkdir(
                parents=True,
                exist_ok=True
            )

            selections.append({
                "category":
                    category,
                "candidate":
                    vid_idx,
                "source_video":
                    str(video),
                "duration_sec":
                    round(duration, 3),
                "size_mb":
                    round(
                        video.stat().st_size
                        / 1024 / 1024,
                        3
                    ),
            })

            for frame_idx, frac in enumerate(
                FRAME_FRACTIONS,
                start=1
            ):

                sec = duration * frac

                sec = min(
                    sec,
                    max(
                        0,
                        duration - 0.05
                    )
                )

                outfile = (
                    vdir
                    / (
                        f"frame_{frame_idx:02d}_"
                        f"{frac:0.2f}_"
                        f"{sec:0.2f}s.jpg"
                    )
                )

                ok = extract_frame(
                    video,
                    sec,
                    outfile
                )

                if ok:

                    extracted.append({
                        "category":
                            category,
                        "candidate":
                            vid_idx,
                        "source_video":
                            str(video),
                        "fraction":
                            frac,
                        "time_sec":
                            round(sec, 3),
                        "frame":
                            str(outfile),
                    })

                    log(
                        f"{category} candidate {vid_idx}: "
                        f"frame {frame_idx}/"
                        f"{len(FRAME_FRACTIONS)}"
                    )

    write_csv(
        OUT / "selected_videos.csv",
        selections
    )

    write_csv(
        OUT / "extracted_frames.csv",
        extracted
    )

    # Contact sheet for fast paper selection
    contact_sheets = []

    for category in CATEGORIES:

        catdir = OUT / category

        if catdir.exists():

            sheet = create_contact_sheet(
                catdir
            )

            if sheet:
                contact_sheets.append(
                    str(sheet)
                )

    summary = {
        "project_root":
            str(ROOT),
        "videos_found_total":
            len(videos),
        "videos_selected":
            len(selections),
        "frames_extracted":
            len(extracted),
        "categories":
            {
                k: len(v)
                for k, v
                in selected.items()
            },
        "contact_sheets":
            contact_sheets,
    }

    (
        OUT
        / "selection_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2
        )
    )

    zip_base = (
        ARCHIVES
        / (
            "paper_scenario_frame_candidates_"
            + time.strftime(
                "%Y%m%d_%H%M%S"
            )
        )
    )

    shutil.make_archive(
        str(zip_base),
        "zip",
        root_dir=OUT
    )

    log("=" * 60)
    log("SCENARIO FRAME EXTRACTION COMPLETE")
    log(f"Selected videos: {len(selections)}")
    log(f"Extracted frames: {len(extracted)}")
    log(f"Output: {OUT}")
    log(f"ZIP: {zip_base}.zip")
    log("=" * 60)


if __name__ == "__main__":
    main()

