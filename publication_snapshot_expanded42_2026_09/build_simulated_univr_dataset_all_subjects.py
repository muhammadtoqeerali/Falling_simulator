from pathlib import Path
from collections import defaultdict
import argparse, json, math, re, shutil
import numpy as np
import pandas as pd

G = 9.80665
RAD_TO_MDPS = 180.0 / math.pi * 1000.0

TASK_DESC = {
    20:"Forward fall when trying to sit down",
    21:"Backward fall when trying to sit down",
    22:"Lateral fall when trying to sit down",
    23:"Forward fall when trying to get up",
    24:"Lateral fall when trying to get up",
    25:"Forward fall while sitting, caused by fainting",
    26:"Lateral fall while sitting, caused by fainting",
    27:"Backward fall while sitting, caused by fainting",
    28:"Vertical/forward fall while walking caused by fainting",
    29:"Fall while walking, hands used to dampen",
    30:"Forward fall while walking caused by a trip",
    31:"Forward fall while jogging caused by a trip",
    32:"Forward fall while walking caused by a slip",
    33:"Lateral fall while walking caused by a slip",
    34:"Backward fall while walking caused by a slip",
    37:"Backward fall while slowly moving back",
    38:"Backward fall while quickly moving back",
    39:"Forward fall from height",
    40:"Backward fall from height",
    41:"Backward fall while climbing up the ladder",
    42:"Backward fall while climbing down the ladder",
    43:"Forward fall while climbing up the ladder",
    44:"Vertical fall while climbing up the ladder caused by a slip",
    250:"Forward fall while standing, caused by fainting",
    290:"Fall backward while walking, hands used to dampen",
    291:"Fall lateral while walking, hands used to dampen",
}

BAD_CSV_TOKENS = [
    "backup", "unlabeled", "markers", "segments", "joints",
    "quality", "dynamics", "contacts", "summary", "batch",
    "validation", "report"
]

def parse_folder_info(folder):
    name = Path(folder).name

    m_task = re.search(r"scenario(\d+)_", name)
    m_age = re.search(r"_age(\d+)", name)
    m_height = re.search(r"_h([0-9p]+)", name)
    m_sex = re.search(r"_sex_([^_]+)", name)
    m_weight = re.search(r"_w([^_]+)", name)
    m_stamp = re.search(r"_(\d{8}_\d{6})$", name)

    if not m_task or not m_age:
        return None

    task_id = int(m_task.group(1))
    age = int(m_age.group(1))

    return {
        "task_id": task_id,
        "subject_id": age,
        "age": age,
        "height_token": m_height.group(1) if m_height else "",
        "sex": m_sex.group(1) if m_sex else "",
        "weight_token": m_weight.group(1) if m_weight else "",
        "timestamp": m_stamp.group(1) if m_stamp else "",
    }

def get_col(df, names):
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for n in names:
        key = str(n).lower()
        if key in lookup:
            return lookup[key]
    return None

def vec(df, names, default=0.0):
    c = get_col(df, names)
    if c is None:
        return np.full(len(df), default, float)
    return pd.to_numeric(df[c], errors="coerce").fillna(default).to_numpy(float)

def time_vector(df):
    c = get_col(df, ["Time_s_standard", "t", "time_s", "timestamp", "Time"])
    if c is not None:
        t = pd.to_numeric(df[c], errors="coerce").fillna(0).to_numpy(float)
        if len(t) and t[0] > 100:
            t = t - t[0]
        return t
    return np.arange(len(df), dtype=float) / 100.0

def find_csv(folder):
    folder = Path(folder)

    # Prefer metadata because it points to the labeled main CSV.
    meta = sorted(folder.glob("*.label_metadata.json"), key=lambda x: x.stat().st_mtime)
    for m in meta:
        try:
            js = json.loads(m.read_text(encoding="utf-8"))
            p = Path(js.get("csv", ""))
            if p.exists():
                return p, m
        except Exception:
            pass

    csvs = []
    for p in folder.glob("*.csv"):
        n = p.name.lower()
        if not any(b in n for b in BAD_CSV_TOKENS):
            csvs.append(p)

    if not csvs:
        return None, None

    return sorted(csvs, key=lambda x: x.stat().st_mtime)[-1], None

def label_frames(df):
    pre = vec(df, ["PreImpact_Fall_Label"], 0)
    imp = vec(df, ["Impact_Label"], 0)

    pre_i = np.where(pre > 0)[0]
    imp_i = np.where(imp > 0)[0]

    if len(pre_i):
        onset = int(pre_i[0])
    else:
        fall = vec(df, ["Fall_Label", "fall_detected", "fall"], 0)
        f = np.where(fall > 0)[0]
        onset = int(f[0]) if len(f) else 0

    if len(imp_i):
        impact = int(imp_i[0])
    else:
        mag = vec(df, ["Accel_Raw_Magnitude_mps2", "accel_mag", "impact_magnitude"], 0)
        impact = int(np.nanargmax(mag)) if len(mag) else onset

    onset = max(0, min(onset, len(df)-1))
    impact = max(0, min(impact, len(df)-1))

    if impact < onset:
        impact = onset

    return onset, impact

def has_required_labels(df):
    cols = {str(c).strip() for c in df.columns}
    return ("PreImpact_Fall_Label" in cols) and ("Impact_Label" in cols)

def convert_one(in_csv, out_csv, sign_down=1, sign_right=1, sign_board=1):
    df = pd.read_csv(in_csv)

    if df.empty:
        raise RuntimeError("empty input CSV")

    if not has_required_labels(df):
        raise RuntimeError("CSV is not labeled; missing PreImpact_Fall_Label or Impact_Label")

    t = time_vector(df)
    n = len(df)

    # Simulator axes:
    # sim Y = vertical proper acceleration
    # sim X = right/lateral-like axis
    # sim Z = board-normal / AP-like axis
    sim_x = vec(df, ["accel_raw_x", "accel_x", "ax"], 0)
    sim_y = vec(df, ["accel_raw_y", "accel_y", "ay"], 0)
    sim_z = vec(df, ["accel_raw_z", "accel_z", "az"], 0)

    gx = vec(df, ["gyro_x", "gyr_x", "gx"], 0)
    gy = vec(df, ["gyro_y", "gyr_y", "gy"], 0)
    gz = vec(df, ["gyro_z", "gyr_z", "gz"], 0)

    # UniVrFall oriented axes:
    # X = downward
    # Y = right
    # Z = perpendicular to sensor board
    accX = np.rint(sign_down  * sim_y / G * 1000.0).astype(int)
    accY = np.rint(sign_right * sim_x / G * 1000.0).astype(int)
    accZ = np.rint(sign_board * sim_z / G * 1000.0).astype(int)

    gyrX = np.rint(sign_down  * gy * RAD_TO_MDPS).astype(int)
    gyrY = np.rint(sign_right * gx * RAD_TO_MDPS).astype(int)
    gyrZ = np.rint(sign_board * gz * RAD_TO_MDPS).astype(int)

    out = pd.DataFrame({
        "": np.rint(t * 1000.0).astype(int),
        "FrameCounter": np.arange(n, dtype=int),
        "AccX": accX,
        "AccY": accY,
        "AccZ": accZ,
        "GyrX": gyrX,
        "GyrY": gyrY,
        "GyrZ": gyrZ,
        "EulerX": np.zeros(n),
        "EulerY": np.zeros(n),
        "EulerZ": np.zeros(n),
    })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    onset, impact = label_frames(df)
    return onset, impact, n

def collect_runs(outputs, folder_glob, recursive=False):
    outputs = Path(outputs)
    pattern = f"**/{folder_glob}" if recursive else folder_glob

    runs = []

    for folder in sorted(outputs.glob(pattern)):
        if not folder.is_dir():
            continue

        info = parse_folder_info(folder)
        if info is None:
            continue

        csvp, metap = find_csv(folder)
        if csvp is None:
            continue

        runs.append({
            **info,
            "folder": folder,
            "csv": csvp,
            "metadata_json": metap,
        })

    return runs

def latest_only_filter(runs):
    latest = {}

    for r in runs:
        key = (r["subject_id"], r["task_id"])
        old = latest.get(key)

        if old is None:
            latest[key] = r
            continue

        old_time = old["timestamp"] or old["folder"].name
        new_time = r["timestamp"] or r["folder"].name

        if new_time > old_time:
            latest[key] = r

    return list(latest.values())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-root", default="/mnt/hdd16T/ToqeerHomeBackup/mujoco_project/outputs")
    ap.add_argument("--out-name", default="simulated_dataset_all_subjects")
    ap.add_argument("--folder-glob", default="scenario*_age*_h*_sex_*_w*_*")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--latest-only", action="store_true", help="Keep only latest run per age/task instead of treating duplicates as repetitions.")
    ap.add_argument("--sign-down", type=int, default=1, choices=[-1,1])
    ap.add_argument("--sign-right", type=int, default=1, choices=[-1,1])
    ap.add_argument("--sign-board", type=int, default=1, choices=[-1,1])
    args = ap.parse_args()

    outputs = Path(args.outputs_root)
    dataset = outputs / args.out_name

    if args.clean and dataset.exists():
        shutil.rmtree(dataset)

    sensors_root = dataset / "laboratory" / "sensors_data"
    labels_dir = dataset / "laboratory" / "labels_data"
    reports_dir = dataset / "conversion_reports"

    sensors_root.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    runs = collect_runs(outputs, args.folder_glob, args.recursive)

    if args.latest_only:
        runs = latest_only_filter(runs)

    runs = sorted(runs, key=lambda r: (r["subject_id"], r["task_id"], str(r["folder"])))

    if not runs:
        raise SystemExit("No labeled scenario folders found. Check --outputs-root and --folder-glob.")

    reps = defaultdict(int)
    subject_label_rows = defaultdict(list)
    report_rows = []

    for r in runs:
        sid = int(r["subject_id"])
        task = int(r["task_id"])

        reps[(sid, task)] += 1
        rep = reps[(sid, task)]

        subj = f"S{sid:02d}"
        subjf = f"SA{sid:02d}"

        out_name = f"{subj}T{task:02d}R{rep:02d}.csv"
        out_csv = sensors_root / subjf / out_name

        try:
            onset, impact, rows = convert_one(
                r["csv"],
                out_csv,
                args.sign_down,
                args.sign_right,
                args.sign_board,
            )

            subject_label_rows[sid].append({
                "Task Code (Task ID)": f"F{task:02d} ({task})",
                "Description": TASK_DESC.get(task, f"Simulated task {task}"),
                "Trial ID": rep,
                "Fall_onset_frame": onset,
                "Fall_impact_frame": impact,
            })

            report_rows.append({
                "status": "converted",
                "subject_id": sid,
                "subject_folder": subjf,
                "task_id": task,
                "trial_id": rep,
                "age": r["age"],
                "height_token": r["height_token"],
                "sex": r["sex"],
                "weight_token": r["weight_token"],
                "timestamp": r["timestamp"],
                "input_folder": str(r["folder"]),
                "input_csv": str(r["csv"]),
                "metadata_json": str(r["metadata_json"] or ""),
                "output_csv": str(out_csv),
                "rows": rows,
                "fall_onset_frame": onset,
                "fall_impact_frame": impact,
            })

            print(f"[OK] age/subject {sid} scenario {task} rep {rep}: {out_csv}")

        except Exception as e:
            print(f"[FAIL] age/subject {sid} scenario {task} rep {rep}: {e}")

            report_rows.append({
                "status": "failed",
                "subject_id": sid,
                "subject_folder": f"SA{sid:02d}",
                "task_id": task,
                "trial_id": rep,
                "age": r["age"],
                "height_token": r["height_token"],
                "sex": r["sex"],
                "weight_token": r["weight_token"],
                "timestamp": r["timestamp"],
                "input_folder": str(r["folder"]),
                "input_csv": str(r["csv"]),
                "metadata_json": str(r["metadata_json"] or ""),
                "error": str(e),
            })

    # Write one label XLSX per subject.
    for sid, rows in sorted(subject_label_rows.items()):
        subjf = f"SA{sid:02d}"
        labels = pd.DataFrame(rows, columns=[
            "Task Code (Task ID)",
            "Description",
            "Trial ID",
            "Fall_onset_frame",
            "Fall_impact_frame",
        ])

        labels = labels.sort_values(["Task Code (Task ID)", "Trial ID"])
        label_xlsx = labels_dir / f"{subjf}_label.xlsx"
        labels.to_excel(label_xlsx, index=False)

    report = pd.DataFrame(report_rows)
    report_csv = reports_dir / "conversion_report_all_subjects.csv"
    report_json = reports_dir / "conversion_report_all_subjects.json"

    report.to_csv(report_csv, index=False)
    report_json.write_text(json.dumps(report_rows, indent=2), encoding="utf-8")

    # Subject summary.
    converted = report[report["status"] == "converted"].copy()
    if len(converted):
        summary = converted.groupby("subject_id").agg(
            converted_files=("status", "count"),
            unique_tasks=("task_id", "nunique"),
            first_task=("task_id", "min"),
            last_task=("task_id", "max"),
        ).reset_index()
    else:
        summary = pd.DataFrame(columns=["subject_id", "converted_files", "unique_tasks", "first_task", "last_task"])

    summary.to_csv(reports_dir / "subject_summary.csv", index=False)

    readme = dataset / "README_simulated_conversion_all_subjects.txt"
    readme.write_text(
        "Simulated UniVrFall-style dataset for all generated subjects\n"
        "Subject rule: simulator age is used as UniVr subject number.\n"
        "Example: age32 -> SA32 / S32, age75 -> SA75 / S75.\n"
        "Structure: laboratory/sensors_data/SAxx and laboratory/labels_data/SAxx_label.xlsx\n"
        "Axis mapping: UniVr X/down=sim Y, UniVr Y/right=sim X, UniVr Z/board-normal=sim Z\n"
        "Acceleration scale: m/s^2 to mg-style integer\n"
        "Gyro scale: rad/s to mdps-style integer\n"
        "Label rule: onset=first PreImpact_Fall_Label frame, impact=Impact_Label frame\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 90)
    print("DONE: ALL-SUBJECT SIMULATED UNIVR DATASET CREATED")
    print("=" * 90)
    print("Dataset       :", dataset)
    print("Sensors root  :", sensors_root)
    print("Labels dir    :", labels_dir)
    print("Report CSV    :", report_csv)
    print("Subject summary:", reports_dir / "subject_summary.csv")
    print("Subjects      :", len(subject_label_rows))
    print("Converted     :", sum(1 for r in report_rows if r["status"] == "converted"))
    print("Failed        :", sum(1 for r in report_rows if r["status"] == "failed"))
    print("=" * 90)

if __name__ == "__main__":
    main()
