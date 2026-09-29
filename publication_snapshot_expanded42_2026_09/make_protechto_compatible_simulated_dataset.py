from pathlib import Path
import argparse
import re
import shutil
import pandas as pd

TASK_REMAP = {
    250: 45,
    290: 46,
    291: 47,
}

def map_task(task):
    return TASK_REMAP.get(int(task), int(task))

def parse_task_code(x):
    s = str(x)
    m = re.search(r"\((\d+)\)", s)
    if m:
        return int(m.group(1))
    m = re.search(r"F\s*0*(\d+)", s, re.IGNORECASE)
    if m:
        return int(m.group(1))
    nums = re.findall(r"\d+", s)
    if nums:
        return int(nums[-1])
    raise ValueError(f"Cannot parse task code: {x}")

def parse_sensor_filename(name):
    m = re.match(r"S(\d+)T(\d+)R(\d+)\.csv$", name)
    if not m:
        raise ValueError(f"Unexpected sensor filename: {name}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src",
        default="/mnt/hdd16T/ToqeerHomeBackup/mujoco_project/outputs/simulated_dataset_all_subjects/laboratory",
    )
    ap.add_argument(
        "--dst",
        default="/mnt/hdd16T/ToqeerHomeBackup/mujoco_project/outputs/simulated_dataset_all_subjects_protechto/laboratory",
    )
    ap.add_argument("--clean", action="store_true")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    src_sensors = src / "sensors_data"
    src_labels = src / "labels_data"

    dst_sensors = dst / "sensors_data"
    dst_labels = dst / "labels_data"

    if args.clean and dst.parent.exists():
        shutil.rmtree(dst.parent)

    dst_sensors.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    copied = 0
    renamed = 0
    errors = []

    # Copy/rename sensor files.
    for subj_dir in sorted(src_sensors.glob("SA*")):
        if not subj_dir.is_dir():
            continue

        subj_name = subj_dir.name
        out_subj_dir = dst_sensors / subj_name
        out_subj_dir.mkdir(parents=True, exist_ok=True)

        for csv_path in sorted(subj_dir.glob("*.csv")):
            try:
                sid, task, trial = parse_sensor_filename(csv_path.name)

                if sid > 99:
                    raise ValueError(
                        f"Subject ID {sid} is >99. Protechto fixed parser expects 2-digit subject IDs."
                    )

                new_task = map_task(task)

                if new_task > 99:
                    raise ValueError(
                        f"Mapped task ID {new_task} is >99. Protechto fixed parser expects 2-digit task IDs."
                    )

                new_name = f"S{sid:02d}T{new_task:02d}R{trial:02d}.csv"
                out_path = out_subj_dir / new_name

                if out_path.exists():
                    raise ValueError(f"Duplicate output file would be created: {out_path}")

                shutil.copy2(csv_path, out_path)
                copied += 1

                if new_name != csv_path.name:
                    renamed += 1

            except Exception as e:
                errors.append(f"{csv_path}: {e}")

    # Copy/update label files.
    label_files = 0

    for label_file in sorted(src_labels.glob("SA*_label.xlsx")):
        try:
            labels = pd.read_excel(label_file)

            required = [
                "Task Code (Task ID)",
                "Description",
                "Trial ID",
                "Fall_onset_frame",
                "Fall_impact_frame",
            ]

            missing = [c for c in required if c not in labels.columns]
            if missing:
                raise ValueError(f"Missing label columns {missing}")

            new_rows = []

            for _, row in labels.iterrows():
                old_task = parse_task_code(row["Task Code (Task ID)"])
                new_task = map_task(old_task)

                desc = str(row["Description"])

                if old_task != new_task:
                    desc = f"Original scenario {old_task}: {desc}"

                new_rows.append({
                    "Task Code (Task ID)": f"F{new_task:02d} ({new_task})",
                    "Description": desc,
                    "Trial ID": int(row["Trial ID"]),
                    "Fall_onset_frame": int(row["Fall_onset_frame"]),
                    "Fall_impact_frame": int(row["Fall_impact_frame"]),
                })

            out_labels = pd.DataFrame(new_rows, columns=required)
            out_path = dst_labels / label_file.name
            out_labels.to_excel(out_path, index=False)
            label_files += 1

        except Exception as e:
            errors.append(f"{label_file}: {e}")

    readme = dst.parent / "README_PROTECHTO_COMPATIBLE.txt"
    readme.write_text(
        "Protechto-compatible simulated dataset copy.\n"
        "Reason: Protechto metadata_from_filename uses fixed positions and expects two-digit task IDs.\n"
        "Remapping used:\n"
        "  scenario 250 -> task 45\n"
        "  scenario 290 -> task 46\n"
        "  scenario 291 -> task 47\n"
        "Original dataset is unchanged.\n",
        encoding="utf-8",
    )

    print("\nDONE")
    print("Source laboratory:", src)
    print("Output laboratory:", dst)
    print("Copied sensor CSVs:", copied)
    print("Renamed sensor CSVs:", renamed)
    print("Label files:", label_files)
    print("Errors:", len(errors))

    for e in errors[:50]:
        print("  -", e)

    if errors:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
