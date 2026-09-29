from pathlib import Path
import re
import csv
import json
import shutil
import hashlib
import time

ROOT = Path("/mnt/hdd16T/ToqeerHomeBackup/mujoco_project")

CAMPAIGN = (
    ROOT
    / "outputs"
    / "_highrate_overnight"
    / "campaign_highrate_truth_v2_extended35"
    / "runs"
)

OUT = (
    ROOT
    / "outputs"
    / "PAPER_SCENARIO_FRAME_CANDIDATES"
)

ARCHIVES = ROOT / "archives"

CATEGORIES = {
    "01_ladder": [
        r"ladder",
    ],

    "02_height": [
        r"height",
        r"elevated",
        r"platform",
        r"ledge",
    ],

    "03_chair": [
        r"chair",
        r"stool",
        r"sofa",
        r"couch",
    ],

    "04_sitting": [
        r"sitting",
        r"seated",
        r"sit_down",
        r"sitdown",
        r"sit_to",
        r"sit-to",
        r"faint.*sit",
        r"sit.*faint",
    ],
}

PREFERRED_VIEWS = [
    "synthetic_pose_preview",
    "sagittal_snapshots",
    "pose_style_views",
    "skeleton_3d_snapshots",
    "openpose_style_fall",
]

MAX_TRIALS_PER_CATEGORY = 6
MAX_VIEWS_PER_TRIAL = 3


def category_for(path):

    s = str(path).lower()

    for category, patterns in CATEGORIES.items():

        for pattern in patterns:

            if re.search(pattern, s):
                return category

    return None


def participant_id(path):

    for part in path.parts:

        if re.fullmatch(r"P\d+", part):
            return part

    return "UNKNOWN"


def task_id(path):

    for part in path.parts:

        m = re.fullmatch(
            r"task_(\d+)",
            part
        )

        if m:
            return int(m.group(1))

    return -1


def view_type(path):

    stem = path.stem.lower()

    for v in PREFERRED_VIEWS:

        if stem.endswith(v):
            return v

    return None


def choose_evenly(items, n):

    if len(items) <= n:
        return items

    if n == 1:
        return [items[len(items)//2]]

    indexes = []

    for i in range(n):

        idx = round(
            i * (len(items)-1)
            / (n-1)
        )

        indexes.append(idx)

    return [
        items[i]
        for i in sorted(set(indexes))
    ]


def sha256(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:

        for chunk in iter(
            lambda: f.read(1024*1024),
            b""
        ):
            h.update(chunk)

    return h.hexdigest()


def make_contact_sheet(category_dir):

    try:
        from PIL import (
            Image,
            ImageOps,
            ImageDraw,
        )
    except Exception:
        return None

    images = sorted([
        p
        for p in category_dir.glob("*.png")
        if "CONTACT_SHEET" not in p.name
    ])

    if not images:
        return None

    thumb_w = 520
    thumb_h = 360
    label_h = 42
    margin = 15
    cols = 2

    rows = (
        len(images)
        + cols - 1
    ) // cols

    width = (
        cols * thumb_w
        + (cols + 1) * margin
    )

    height = (
        rows
        * (thumb_h + label_h)
        + (rows + 1) * margin
    )

    canvas = Image.new(
        "RGB",
        (width, height),
        "white"
    )

    draw = ImageDraw.Draw(
        canvas
    )

    for i, path in enumerate(images):

        r = i // cols
        c = i % cols

        x = (
            margin
            + c * (thumb_w + margin)
        )

        y = (
            margin
            + r
            * (thumb_h + label_h + margin)
        )

        try:

            im = Image.open(
                path
            ).convert("RGB")

            im = ImageOps.contain(
                im,
                (thumb_w, thumb_h)
            )

            px = (
                x
                + (thumb_w - im.width)//2
            )

            py = (
                y
                + (thumb_h - im.height)//2
            )

            canvas.paste(
                im,
                (px, py)
            )

            label = path.stem

            if len(label) > 70:
                label = label[:67] + "..."

            draw.text(
                (
                    x,
                    y + thumb_h + 8
                ),
                label,
                fill="black"
            )

        except Exception:
            pass

    outfile = (
        category_dir
        / f"{category_dir.name}_CONTACT_SHEET.jpg"
    )

    canvas.save(
        outfile,
        quality=94
    )

    return outfile


def main():

    if not CAMPAIGN.exists():
        raise SystemExit(
            f"Campaign not found: {CAMPAIGN}"
        )

    if OUT.exists():
        shutil.rmtree(OUT)

    OUT.mkdir(
        parents=True
    )

    all_pngs = sorted(
        CAMPAIGN.rglob("*.png")
    )

    print(
        "Campaign PNGs found:",
        len(all_pngs)
    )

    candidates = {
        cat: []
        for cat in CATEGORIES
    }

    for p in all_pngs:

        cat = category_for(p)

        if cat is None:
            continue

        view = view_type(p)

        if view is None:
            continue

        candidates[cat].append(p)

    manifest = []

    summary = {}

    for cat in CATEGORIES:

        catdir = OUT / cat

        catdir.mkdir(
            parents=True,
            exist_ok=True
        )

        files = candidates[cat]

        by_trial = {}

        for p in files:

            trial = str(
                p.parent
            )

            by_trial.setdefault(
                trial,
                []
            ).append(p)

        trials = sorted(
            by_trial.keys(),
            key=lambda x: (
                participant_id(
                    Path(x)
                ),
                task_id(
                    Path(x)
                ),
                x
            )
        )

        chosen_trials = choose_evenly(
            trials,
            MAX_TRIALS_PER_CATEGORY
        )

        copied = 0

        for trial_number, trial in enumerate(
            chosen_trials,
            1
        ):

            paths = by_trial[trial]

            selected_views = []

            for preferred in PREFERRED_VIEWS:

                matches = [
                    p
                    for p in paths
                    if view_type(p)
                    == preferred
                ]

                if matches:
                    selected_views.append(
                        matches[0]
                    )

                if (
                    len(selected_views)
                    >= MAX_VIEWS_PER_TRIAL
                ):
                    break

            for p in selected_views:

                pid = participant_id(
                    p
                )

                task = task_id(
                    p
                )

                view = view_type(
                    p
                )

                dest_name = (
                    f"{cat}"
                    f"__{pid}"
                    f"__task{task:02d}"
                    f"__trial{trial_number:02d}"
                    f"__{view}.png"
                )

                dest = (
                    catdir
                    / dest_name
                )

                shutil.copy2(
                    p,
                    dest
                )

                manifest.append({
                    "category":
                        cat,
                    "participant":
                        pid,
                    "task":
                        task,
                    "view":
                        view,
                    "source":
                        str(p),
                    "selected_file":
                        str(dest),
                    "sha256":
                        sha256(dest),
                })

                copied += 1

        sheet = make_contact_sheet(
            catdir
        )

        summary[cat] = {
            "matched_images":
                len(files),
            "matched_trials":
                len(trials),
            "selected_trials":
                len(chosen_trials),
            "selected_images":
                copied,
            "contact_sheet":
                str(sheet)
                if sheet
                else None,
        }

        print()
        print(cat)
        print(
            "  matched images :",
            len(files)
        )
        print(
            "  matched trials :",
            len(trials)
        )
        print(
            "  selected trials:",
            len(chosen_trials)
        )
        print(
            "  copied images  :",
            copied
        )

    with open(
        OUT
        / "SCENARIO_FRAME_MANIFEST.csv",
        "w",
        newline=""
    ) as f:

        fields = [
            "category",
            "participant",
            "task",
            "view",
            "source",
            "selected_file",
            "sha256",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()
        writer.writerows(
            manifest
        )

    (
        OUT
        / "SCENARIO_SELECTION_SUMMARY.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2
        )
    )

    missing = [
        cat
        for cat, data
        in summary.items()
        if data[
            "selected_images"
        ] == 0
    ]

    readme = [
        "PAPER SCENARIO FRAME CANDIDATES",
        "",
        "Purpose:",
        (
            "Curated existing MuJoCo-rendered "
            "scenario images for later paper figure selection."
        ),
        "",
        (
            "No simulation was rerun and no new "
            "synthetic motion was generated."
        ),
        "",
        (
            "Analytical plots such as dynamics, COP, "
            "kinematics and trajectory plots were deliberately "
            "excluded from this visual scenario package."
        ),
        "",
        "Preferred visual products:",
        "- synthetic_pose_preview",
        "- sagittal_snapshots",
        "- pose_style_views",
        "- skeleton_3d_snapshots",
        "- openpose_style_fall",
        "",
        "Categories:",
        "- ladder",
        "- height/elevated fall",
        "- chair",
        "- sitting/seated",
        "",
    ]

    if missing:

        readme += [
            "IMPORTANT:",
            (
                "The following categories were not identified "
                "by explicit filenames and require task-definition "
                "mapping before final selection:"
            ),
            *[
                f"- {x}"
                for x in missing
            ],
            "",
        ]

    (
        OUT
        / "README_SCENARIO_FRAMES.txt"
    ).write_text(
        "\n".join(readme)
    )

    timestamp = time.strftime(
        "%Y%m%d_%H%M%S"
    )

    zipbase = (
        ARCHIVES
        / (
            "PAPER_SCENARIO_FRAME_CANDIDATES_"
            + timestamp
        )
    )

    zipfile = shutil.make_archive(
        str(zipbase),
        "zip",
        root_dir=OUT
    )

    print()
    print("=" * 76)
    print(
        "SCENARIO FRAME CURATION COMPLETE"
    )
    print("=" * 76)

    print(
        "Output:",
        OUT
    )

    print(
        "ZIP   :",
        zipfile
    )

    print(
        "Selected images:",
        len(manifest)
    )

    if missing:

        print()
        print(
            "CATEGORIES STILL NEEDING TASK MAPPING:"
        )

        for x in missing:
            print(
                " ",
                x
            )

    else:

        print()
        print(
            "ALL FOUR TARGET CATEGORIES FOUND."
        )


if __name__ == "__main__":
    main()
