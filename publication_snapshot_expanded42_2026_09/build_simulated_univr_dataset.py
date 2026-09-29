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

def task_id_from_folder(p):
    m = re.search(r"scenario(\d+)_", str(p))
    return int(m.group(1)) if m else None

def latest_batch_summary(outputs):
    files = sorted(Path(outputs).glob("batch_runs/batch_video_*/batch_summary.csv"), key=lambda x: x.stat().st_mtime)
    return files[-1] if files else None

def get_col(df, names):
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lookup:
            return lookup[n.lower()]
    return None

def vec(df, names, default=0.0):
    c = get_col(df, names)
    if c is None:
        return np.full(len(df), default, float)
    return pd.to_numeric(df[c], errors="coerce").fillna(default).to_numpy(float)

def time_vector(df):
    c = get_col(df, ["Time_s_standard", "t", "time_s", "timestamp"])
    if c is not None:
        t = pd.to_numeric(df[c], errors="coerce").fillna(0).to_numpy(float)
        if len(t) and t[0] > 100:
            t = t - t[0]
        return t
    return np.arange(len(df), dtype=float) / 100.0

def find_csv(folder):
    meta = sorted(Path(folder).glob("*.label_metadata.json"), key=lambda x: x.stat().st_mtime)
    for m in meta:
        try:
            js = json.loads(m.read_text())
            p = Path(js.get("csv", ""))
            if p.exists():
                return p
        except Exception:
            pass

    bad = ["backup","unlabeled","markers","segments","joints","quality","dynamics","contacts","summary","batch"]
    csvs = []
    for p in Path(folder).glob("*.csv"):
        n = p.name.lower()
        if not any(b in n for b in bad):
            csvs.append(p)
    return sorted(csvs, key=lambda x: x.stat().st_mtime)[-1] if csvs else None

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
        impact = int(np.nanargmax(mag))

    if impact < onset:
        impact = onset

    return onset, impact

def convert_one(in_csv, out_csv, sign_down=1, sign_right=1, sign_board=1):
    df = pd.read_csv(in_csv)
    t = time_vector(df)
    n = len(df)

    # Simulator axes:
    # sim Y = vertical proper acceleration
    # sim X = left/right
    # sim Z = board-normal / AP-like axis
    sim_x = vec(df, ["accel_raw_x", "accel_x", "ax"], 0)
    sim_y = vec(df, ["accel_raw_y", "accel_y", "ay"], 0)
    sim_z = vec(df, ["accel_raw_z", "accel_z", "az"], 0)

    gx = vec(df, ["gyro_x", "gyr_x", "gx"], 0)
    gy = vec(df, ["gyro_y", "gyr_y", "gy"], 0)
    gz = vec(df, ["gyro_z", "gyr_z", "gz"], 0)

    # UniVrFall axes:
    # X = downward, Y = right, Z = perpendicular to board
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-root", default="/mnt/hdd16T/ToqeerHomeBackup/mujoco_project/outputs")
    ap.add_argument("--subject-id", type=int, default=32)
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--sign-down", type=int, default=1, choices=[-1,1])
    ap.add_argument("--sign-right", type=int, default=1, choices=[-1,1])
    ap.add_argument("--sign-board", type=int, default=1, choices=[-1,1])
    args = ap.parse_args()

    outputs = Path(args.outputs_root)
    dataset = outputs / "simulated_dataset"

    if args.clean and dataset.exists():
        shutil.rmtree(dataset)

    sid = args.subject_id
    subj = f"S{sid:02d}"
    subjf = f"SA{sid:02d}"

    sensors_dir = dataset / "laboratory" / "sensors_data" / subjf
    labels_dir = dataset / "laboratory" / "labels_data"
    reports_dir = dataset / "conversion_reports"

    sensors_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    summary = latest_batch_summary(outputs)
    runs = []

    if summary:
        b = pd.read_csv(summary)
        for _, r in b.iterrows():
            folder = str(r.get("output_folder","")).strip()
            if not folder:
                continue
            task = int(r.get("task_id"))
            csvp = Path(str(r.get("main_csv","")).strip())
            if not csvp.exists():
                csvp = find_csv(folder)
            if csvp and Path(folder).exists():
                runs.append((task, Path(folder), Path(csvp)))
    else:
        for folder in sorted(outputs.glob("scenario*_age32_h1p70_sex_male_w78p0_*")):
            task = task_id_from_folder(folder)
            csvp = find_csv(folder)
            if task and csvp:
                runs.append((task, folder, csvp))

    runs = sorted(runs, key=lambda x: (x[0], str(x[1])))
    reps = defaultdict(int)
    label_rows = []
    report_rows = []

    for task, folder, csvp in runs:
        reps[task] += 1
        rep = reps[task]
        out_name = f"{subj}T{task:02d}R{rep:02d}.csv"
        out_csv = sensors_dir / out_name

        try:
            onset, impact, rows = convert_one(
                csvp,
                out_csv,
                args.sign_down,
                args.sign_right,
                args.sign_board,
            )

            label_rows.append({
                "Task Code (Task ID)": f"F{task:02d} ({task})",
                "Description": TASK_DESC.get(task, f"Simulated task {task}"),
                "Trial ID": rep,
                "Fall_onset_frame": onset,
                "Fall_impact_frame": impact,
            })

            report_rows.append({
                "status": "converted",
                "task_id": task,
                "trial_id": rep,
                "input_folder": str(folder),
                "input_csv": str(csvp),
                "output_csv": str(out_csv),
                "rows": rows,
                "fall_onset_frame": onset,
                "fall_impact_frame": impact,
            })

            print(f"[OK] scenario {task} rep {rep}: {out_csv}")

        except Exception as e:
            print(f"[FAIL] scenario {task} rep {rep}: {e}")
            report_rows.append({
                "status": "failed",
                "task_id": task,
                "trial_id": rep,
                "input_folder": str(folder),
                "input_csv": str(csvp),
                "error": str(e),
            })

    labels = pd.DataFrame(label_rows, columns=[
        "Task Code (Task ID)",
        "Description",
        "Trial ID",
        "Fall_onset_frame",
        "Fall_impact_frame",
    ])

    label_xlsx = labels_dir / f"{subjf}_label.xlsx"
    labels.to_excel(label_xlsx, index=False)

    report = pd.DataFrame(report_rows)
    report.to_csv(reports_dir / "conversion_report.csv", index=False)

    (reports_dir / "conversion_report.json").write_text(
        json.dumps(report_rows, indent=2),
        encoding="utf-8",
    )

    readme = dataset / "README_simulated_conversion.txt"
    readme.write_text(
        "Simulated UniVrFall-style dataset\n"
        "Structure: laboratory/sensors_data/SAxx and laboratory/labels_data/SAxx_label.xlsx\n"
        "Axis mapping: UniVr X/down=sim Y, UniVr Y/right=sim X, UniVr Z/board-normal=sim Z\n"
        "Acceleration scale: m/s^2 to mg-style integer\n"
        "Gyro scale: rad/s to mdps-style integer\n"
        "Label rule: onset=first PreImpact_Fall_Label frame, impact=Impact_Label frame\n",
        encoding="utf-8",
    )

    print("\nDONE")
    print("Dataset:", dataset)
    print("Sensors:", sensors_dir)
    print("Labels :", label_xlsx)
    print("Report :", reports_dir / "conversion_report.csv")
    print("Converted:", sum(1 for r in report_rows if r["status"] == "converted"))
    print("Failed   :", sum(1 for r in report_rows if r["status"] == "failed"))

if __name__ == "__main__":
    main()
