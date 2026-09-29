from pathlib import Path
import subprocess
import shutil
import csv
import time
import hashlib

ROOT = Path(
    "/mnt/hdd16T/ToqeerHomeBackup/mujoco_project"
)

OUT = ROOT / "outputs" / "PAPER_VIDEO_FRAME_LIBRARY"
ARCHIVES = ROOT / "archives"

FPS = 10

VIDEOS = {
    "01_chair_getting_up": ROOT / (
        "outputs/"
        "scenario23_age25_h1p62_sex_female_w58p0_20260824_083429/"
        "fall_scenario23_age25_20260824_083833_highrate_truth_display.mp4"
    ),

    "02_sitting_fainting": ROOT / (
        "outputs/"
        "scenario25_age25_h1p62_sex_female_w58p0_20260824_083843/"
        "fall_scenario25_age25_20260824_084000_highrate_truth_display.mp4"
    ),

    "03_height_fall": ROOT / (
        "outputs/"
        "scenario39_age25_h1p62_sex_female_w58p0_20260824_084007/"
        "fall_forward_height_age25_20260824_084213_highrate_truth_display.mp4"
    ),

    "04_ladder_fall": ROOT / (
        "outputs/"
        "scenario43_age25_h1p62_sex_female_w58p0_20260824_084238/"
        "fall_forward_ladder_age25_20260824_084402_highrate_truth_display.mp4"
    ),
}


def ffprobe_duration(path):
    result = subprocess.check_output(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        text=True,
    )
    return float(result.strip())


def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def make_contact_sheets(folder, category):
    try:
        from PIL import Image, ImageOps, ImageDraw
    except Exception:
        return []

    images = sorted(folder.glob("frame_*.jpg"))

    if not images:
        return []

    every = max(
        1,
        round(FPS)
    )

    selected = images[::every]

    per_sheet = 24
    sheets = []

    for sheet_idx in range(
        0,
        len(selected),
        per_sheet,
    ):
        group = selected[
            sheet_idx:
            sheet_idx + per_sheet
        ]

        cols = 4
        rows = (
            len(group) + cols - 1
        ) // cols

        thumb_w = 430
        thumb_h = 240
        label_h = 35
        margin = 10

        canvas = Image.new(
            "RGB",
            (
                cols * thumb_w
                + (cols + 1) * margin,

                rows * (thumb_h + label_h)
                + (rows + 1) * margin,
            ),
            "white",
        )

        draw = ImageDraw.Draw(canvas)

        for i, p in enumerate(group):
            r = i // cols
            c = i % cols

            x = margin + c * (
                thumb_w + margin
            )

            y = margin + r * (
                thumb_h + label_h + margin
            )

            im = Image.open(
                p
            ).convert("RGB")

            im = ImageOps.contain(
                im,
                (thumb_w, thumb_h)
            )

            px = x + (
                thumb_w - im.width
            ) // 2

            py = y + (
                thumb_h - im.height
            ) // 2

            canvas.paste(
                im,
                (px, py),
            )

            n = int(
                p.stem.split("_")[-1]
            )

            t = (n - 1) / FPS

            draw.text(
                (
                    x,
                    y + thumb_h + 5,
                ),
                f"{p.name}   t={t:.1f}s",
                fill="black",
            )

        num = (
            sheet_idx // per_sheet
        ) + 1

        dst = (
            folder
            / f"{category}_CONTACT_{num:02d}.jpg"
        )

        canvas.save(
            dst,
            quality=94,
        )

        sheets.append(dst)

    return sheets


def main():

    if OUT.exists():
        shutil.rmtree(OUT)

    OUT.mkdir(
        parents=True
    )

    ARCHIVES.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = []

    print("=" * 78)
    print("EXTRACTING PAPER VIDEO FRAME LIBRARY")
    print("=" * 78)
    print("Candidate extraction rate:", FPS, "fps")
    print()

    for category, video in VIDEOS.items():

        if not video.exists():
            raise FileNotFoundError(
                video
            )

        folder = OUT / category
        folder.mkdir(
            parents=True
        )

        duration = ffprobe_duration(
            video
        )

        print(category)
        print("  source  :", video)
        print(
            "  duration:",
            f"{duration:.3f} s"
        )

        pattern = (
            folder
            / "frame_%05d.jpg"
        )

        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel", "error",
                "-y",
                "-i", str(video),
                "-vf", f"fps={FPS}",
                "-q:v", "2",
                str(pattern),
            ],
            check=True,
        )

        frames = sorted(
            folder.glob(
                "frame_*.jpg"
            )
        )

        print(
            "  frames  :",
            len(frames)
        )

        for p in frames:

            n = int(
                p.stem.split("_")[-1]
            )

            timestamp = (
                n - 1
            ) / FPS

            manifest.append({
                "category":
                    category,
                "frame":
                    p.name,
                "timestamp_s":
                    f"{timestamp:.3f}",
                "source_video":
                    str(video),
                "frame_path":
                    str(p),
                "sha256":
                    sha256(p),
            })

        sheets = make_contact_sheets(
            folder,
            category,
        )

        print(
            "  contact sheets:",
            len(sheets)
        )

        print()

    manifest_file = (
        OUT
        / "FRAME_INDEX.csv"
    )

    with manifest_file.open(
        "w",
        newline="",
    ) as f:

        fields = [
            "category",
            "frame",
            "timestamp_s",
            "source_video",
            "frame_path",
            "sha256",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(
            manifest
        )

    readme = OUT / "README.txt"

    readme.write_text(
        "\n".join([
            "PAPER VIDEO FRAME LIBRARY",
            "",
            "Source:",
            "Four freshly regenerated MuJoCo scenario MP4 files.",
            "",
            "Scenarios:",
            "01 chair/getting-up fall = task 23",
            "02 sitting/fainting fall = task 25",
            "03 forward fall from height = task 39",
            "04 forward fall climbing ladder = task 43",
            "",
            f"Candidate frame extraction rate = {FPS} fps.",
            "",
            (
                "These JPEG images are intended for visual selection. "
                "After a final frame is chosen, extract the same timestamp "
                "directly from the source MP4 as lossless PNG for publication."
            ),
            "",
            (
                "Task 23 and task 25 video capture succeeded even though "
                "their later impact-score labeling stage failed. "
                "This does not affect their use as scenario visualization."
            ),
        ])
    )

    timestamp = time.strftime(
        "%Y%m%d_%H%M%S"
    )

    zipbase = (
        ARCHIVES
        / (
            "PAPER_VIDEO_FRAME_LIBRARY_"
            + timestamp
        )
    )

    zipfile = shutil.make_archive(
        str(zipbase),
        "zip",
        root_dir=OUT,
    )

    print("=" * 78)
    print("COMPLETE")
    print("=" * 78)

    print(
        "Total candidate frames:",
        len(manifest)
    )

    print(
        "Frame library:",
        OUT
    )

    print(
        "ZIP:",
        zipfile
    )


if __name__ == "__main__":
    main()
