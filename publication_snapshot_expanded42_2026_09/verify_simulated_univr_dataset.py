from pathlib import Path
import json, math
import numpy as np
import pandas as pd

G = 9.80665
RAD_TO_MDPS = 180.0 / math.pi * 1000.0

DATASET = Path("/mnt/hdd16T/ToqeerHomeBackup/mujoco_project/outputs/simulated_dataset")
SUBJECT = "SA32"

SENSORS = DATASET / "laboratory" / "sensors_data" / SUBJECT
LABEL_XLSX = DATASET / "laboratory" / "labels_data" / f"{SUBJECT}_label.xlsx"
REPORT_CSV = DATASET / "conversion_reports" / "conversion_report.csv"
VERIFY_DIR = DATASET / "verification_reports"

EXPECTED_SENSOR_COLS = [
    "Unnamed: 0",
    "FrameCounter",
    "AccX",
    "AccY",
    "AccZ",
    "GyrX",
    "GyrY",
    "GyrZ",
    "EulerX",
    "EulerY",
    "EulerZ",
]

EXPECTED_LABEL_COLS = [
    "Task Code (Task ID)",
    "Description",
    "Trial ID",
    "Fall_onset_frame",
    "Fall_impact_frame",
]

def get_col(df, names):
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for n in names:
        if str(n).lower() in lookup:
            return lookup[str(n).lower()]
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

def fail(msg):
    return {"status": "FAIL", "message": msg}

def ok(msg="OK"):
    return {"status": "OK", "message": msg}

def check_close_int(name, got, exp, max_abs_diff=1):
    got = np.asarray(got)
    exp = np.asarray(exp)
    if len(got) != len(exp):
        return False, f"{name}: length mismatch {len(got)} vs {len(exp)}"
    diff = np.abs(got.astype(float) - exp.astype(float))
    m = float(np.nanmax(diff)) if len(diff) else 0.0
    if m > max_abs_diff:
        return False, f"{name}: max abs diff {m} > {max_abs_diff}"
    return True, f"{name}: max abs diff {m}"

def verify_one(row):
    out_csv = Path(row["output_csv"])
    in_csv = Path(row["input_csv"])
    task_id = int(row["task_id"])
    trial_id = int(row["trial_id"])

    result = {
        "task_id": task_id,
        "trial_id": trial_id,
        "sensor_csv": str(out_csv),
        "input_csv": str(in_csv),
        "status": "OK",
        "errors": [],
        "warnings": [],
    }

    if not out_csv.exists():
        result["status"] = "FAIL"
        result["errors"].append("Converted sensor CSV missing")
        return result

    if not in_csv.exists():
        result["status"] = "FAIL"
        result["errors"].append("Original labeled simulator CSV missing")
        return result

    df = pd.read_csv(out_csv)
    src = pd.read_csv(in_csv)

    result["rows"] = len(df)
    result["columns"] = list(df.columns)

    if list(df.columns) != EXPECTED_SENSOR_COLS:
        result["status"] = "FAIL"
        result["errors"].append(f"Wrong sensor columns: {list(df.columns)}")

    if len(df) != len(src):
        result["status"] = "FAIL"
        result["errors"].append(f"Row count mismatch converted={len(df)} original={len(src)}")
        return result

    # Basic frame and time checks.
    fc = pd.to_numeric(df["FrameCounter"], errors="coerce").to_numpy()
    if not np.array_equal(fc, np.arange(len(df))):
        result["status"] = "FAIL"
        result["errors"].append("FrameCounter is not 0..N-1")

    ms = pd.to_numeric(df["Unnamed: 0"], errors="coerce").to_numpy()
    dms = np.diff(ms)
    if len(dms):
        med_dt = float(np.nanmedian(dms))
        result["median_dt_ms"] = med_dt
        if not (9 <= med_dt <= 11):
            result["warnings"].append(f"Median timestamp step is {med_dt} ms, expected about 10 ms")

    # Exact conversion check.
    sim_x = vec(src, ["accel_raw_x", "accel_x", "ax"], 0)
    sim_y = vec(src, ["accel_raw_y", "accel_y", "ay"], 0)
    sim_z = vec(src, ["accel_raw_z", "accel_z", "az"], 0)

    gx = vec(src, ["gyro_x", "gyr_x", "gx"], 0)
    gy = vec(src, ["gyro_y", "gyr_y", "gy"], 0)
    gz = vec(src, ["gyro_z", "gyr_z", "gz"], 0)

    expected = {
        "AccX": np.rint(sim_y / G * 1000.0).astype(int),
        "AccY": np.rint(sim_x / G * 1000.0).astype(int),
        "AccZ": np.rint(sim_z / G * 1000.0).astype(int),
        "GyrX": np.rint(gy * RAD_TO_MDPS).astype(int),
        "GyrY": np.rint(gx * RAD_TO_MDPS).astype(int),
        "GyrZ": np.rint(gz * RAD_TO_MDPS).astype(int),
    }

    for c, exp in expected.items():
        got = pd.to_numeric(df[c], errors="coerce").fillna(0).to_numpy(int)
        good, msg = check_close_int(c, got, exp, 1)
        result[c + "_check"] = msg
        if not good:
            result["status"] = "FAIL"
            result["errors"].append(msg)

    # Unit/range sanity checks.
    for c in ["AccX", "AccY", "AccZ", "GyrX", "GyrY", "GyrZ"]:
        arr = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
        result[c + "_min"] = float(np.nanmin(arr))
        result[c + "_max"] = float(np.nanmax(arr))
        result[c + "_median_first_1s"] = float(np.nanmedian(arr[:min(100, len(arr))]))

    # Standing gravity should mainly be AccX because UniVr X is downward and sim Y is vertical.
    accx0 = abs(result["AccX_median_first_1s"])
    accy0 = abs(result["AccY_median_first_1s"])
    accz0 = abs(result["AccZ_median_first_1s"])
    if accx0 < 700:
        result["warnings"].append(f"First-second AccX gravity magnitude is low: {accx0}")
    if accx0 < accy0 or accx0 < accz0:
        result["warnings"].append("Gravity is not dominant on AccX in first second")

    # Label check against original labeled simulator CSV.
    pre = vec(src, ["PreImpact_Fall_Label"], 0)
    imp = vec(src, ["Impact_Label"], 0)
    pre_idx = np.where(pre > 0)[0]
    imp_idx = np.where(imp > 0)[0]

    expected_onset = int(pre_idx[0]) if len(pre_idx) else int(np.where(vec(src, ["Fall_Label", "fall_detected", "fall"], 0) > 0)[0][0])
    expected_impact = int(imp_idx[0]) if len(imp_idx) else int(np.nanargmax(vec(src, ["Accel_Raw_Magnitude_mps2", "accel_mag", "impact_magnitude"], 0)))

    result["expected_onset_frame"] = expected_onset
    result["expected_impact_frame"] = expected_impact

    return result

def main():
    VERIFY_DIR.mkdir(parents=True, exist_ok=True)

    summary = {
        "dataset": str(DATASET),
        "sensors_dir": str(SENSORS),
        "label_xlsx": str(LABEL_XLSX),
        "report_csv": str(REPORT_CSV),
        "overall_status": "OK",
        "checks": {},
    }

    # Structure checks.
    summary["checks"]["dataset_exists"] = ok() if DATASET.exists() else fail("Dataset folder missing")
    summary["checks"]["sensors_dir_exists"] = ok() if SENSORS.exists() else fail("Sensors folder missing")
    summary["checks"]["label_xlsx_exists"] = ok() if LABEL_XLSX.exists() else fail("Label XLSX missing")
    summary["checks"]["conversion_report_exists"] = ok() if REPORT_CSV.exists() else fail("Conversion report missing")

    sensor_files = sorted(SENSORS.glob("*.csv"))
    summary["sensor_file_count"] = len(sensor_files)

    if len(sensor_files) != 26:
        summary["overall_status"] = "FAIL"
        summary["checks"]["sensor_file_count"] = fail(f"Expected 26 sensor CSVs, found {len(sensor_files)}")
    else:
        summary["checks"]["sensor_file_count"] = ok("26 sensor CSVs found")

    if not LABEL_XLSX.exists() or not REPORT_CSV.exists():
        summary["overall_status"] = "FAIL"
        print(json.dumps(summary, indent=2))
        raise SystemExit(1)

    labels = pd.read_excel(LABEL_XLSX)
    report = pd.read_csv(REPORT_CSV)

    if list(labels.columns) != EXPECTED_LABEL_COLS:
        summary["overall_status"] = "FAIL"
        summary["checks"]["label_columns"] = fail(f"Wrong label columns: {list(labels.columns)}")
    else:
        summary["checks"]["label_columns"] = ok()

    if len(labels) != len(sensor_files):
        summary["overall_status"] = "FAIL"
        summary["checks"]["label_row_count"] = fail(f"Label rows {len(labels)} != sensor files {len(sensor_files)}")
    else:
        summary["checks"]["label_row_count"] = ok()

    # Per-file verification.
    detailed = []
    for _, row in report.iterrows():
        if str(row.get("status", "")).lower() != "converted":
            continue
        r = verify_one(row)
        detailed.append(r)
        if r["status"] != "OK":
            summary["overall_status"] = "FAIL"

    detailed_df = pd.DataFrame(detailed)
    detailed_csv = VERIFY_DIR / "verification_detailed.csv"
    detailed_json = VERIFY_DIR / "verification_detailed.json"
    summary_json = VERIFY_DIR / "verification_summary.json"

    detailed_df.to_csv(detailed_csv, index=False)
    detailed_json.write_text(json.dumps(detailed, indent=2), encoding="utf-8")
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 90)
    print("SIMULATED UNIVR DATASET VERIFICATION")
    print("=" * 90)
    print("Overall status:", summary["overall_status"])
    print("Sensor files  :", len(sensor_files))
    print("Label rows    :", len(labels))
    print("Detailed CSV  :", detailed_csv)
    print("Detailed JSON :", detailed_json)
    print("Summary JSON  :", summary_json)

    failed = [r for r in detailed if r["status"] != "OK"]
    warned = [r for r in detailed if r.get("warnings")]

    print("Failed files  :", len(failed))
    print("Warnings files:", len(warned))

    if failed:
        print("\nFAILED:")
        for r in failed[:20]:
            print(r["sensor_csv"])
            for e in r["errors"]:
                print("  -", e)

    if warned:
        print("\nWARNINGS:")
        for r in warned[:20]:
            print(r["sensor_csv"])
            for w in r["warnings"]:
                print("  -", w)

    print("=" * 90)

    if summary["overall_status"] != "OK":
        raise SystemExit(2)

if __name__ == "__main__":
    main()
