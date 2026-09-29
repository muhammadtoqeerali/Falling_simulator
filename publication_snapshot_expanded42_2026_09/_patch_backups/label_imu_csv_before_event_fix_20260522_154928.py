# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_NATIVE_HZ = 30.0
DEFAULT_OUTPUT_HZ = 100.0


# Known phase timings from your current logs.
# You can add more scenarios here later.
DEFAULT_PHASES = {
    # Backward fall while walking caused by slip
    34: [
        ("stand", 195),
        ("walk", 300),
        ("perturb", 60),
        ("react", 31),
        ("fall", 470),
    ],

    # Forward fall while climbing up ladder
    43: [
        ("hold", 52),
        ("climb", 280),
        ("release", 48),
        ("react", 18),
        ("fall", 281),
    ],
}


PREFALL_PHASES = {
    "perturb",
    "release",
    "slip",
    "trip",
    "push",
    "faint",
    "stumble",
}

NORMAL_PHASES = {
    "stand",
    "walk",
    "jog",
    "sit",
    "sitting",
    "hold",
    "climb",
    "step",
    "getup",
    "get_up",
}

REACTION_PHASES = {
    "react",
    "reaction",
}

FALL_PHASES = {
    "fall",
    "collapse",
    "settle",
    "impact",
    "postimpact",
    "post_impact",
}


def clean_col(x: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(x).strip().lower()).strip("_")


def infer_scenario_id(path: Path):
    text = str(path)
    m = re.search(r"scenario[_-]?(\d+)", text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def parse_phase_steps(text: str):
    """
    Example:
      stand:195,walk:300,perturb:60,react:31,fall:470
      hold:52,climb:280,release:48,react:18,fall:281
    """
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Bad phase part: {part}. Use name:steps")
        name, steps = part.split(":", 1)
        out.append((name.strip().lower(), int(steps.strip())))
    if not out:
        raise ValueError("No phase steps provided.")
    return out


def phase_boundaries(phase_steps, native_hz: float):
    """
    Returns list of:
      phase, start_step, end_step, start_time_s, end_time_s
    """
    rows = []
    cursor = 0
    for name, nsteps in phase_steps:
        start_step = cursor
        end_step = cursor + int(nsteps)
        rows.append({
            "phase": str(name).lower(),
            "start_step_native": start_step,
            "end_step_native": end_step,
            "start_time_s": start_step / native_hz,
            "end_time_s": end_step / native_hz,
            "duration_s": int(nsteps) / native_hz,
            "steps": int(nsteps),
        })
        cursor = end_step
    return rows


def find_time_column(df: pd.DataFrame):
    candidates = [
        "time_s", "time", "timestamp", "timestamp_s", "t",
        "TimeStamp(s)", "TimeStamp", "Time", "sim_time",
    ]
    cmap = {clean_col(c): c for c in df.columns}
    for c in candidates:
        key = clean_col(c)
        if key in cmap:
            return cmap[key]

    for c in df.columns:
        cc = clean_col(c)
        if "time" in cc or "timestamp" in cc:
            return c
    return None


def find_axis_columns(df: pd.DataFrame, sensor: str):
    """
    Finds accelerometer or gyroscope columns robustly.
    """
    cols = list(df.columns)

    if sensor == "acc":
        words = ["acc", "accel", "accelerometer", "linear_acc", "linear_acceleration"]
        short = "a"
    else:
        words = ["gyro", "gyr", "gyroscope", "angular_velocity", "omega"]
        short = "g"

    out = {"x": None, "y": None, "z": None}

    for axis in ["x", "y", "z"]:
        for c in cols:
            cc = clean_col(c)

            patterns = []
            for w in words:
                ww = clean_col(w)
                patterns += [
                    f"{ww}_{axis}",
                    f"{ww}{axis}",
                    f"{axis}_{ww}",
                ]
            patterns += [f"{short}{axis}", f"{short}_{axis}"]

            if cc in patterns:
                out[axis] = c
                break

    return out


def first_valid_csv_in_folder(folder: Path):
    csvs = sorted(folder.rglob("*.csv"))
    csvs = [p for p in csvs if "labeled" not in p.name.lower() and "enriched" not in p.name.lower()]

    if not csvs:
        return []

    good = []
    for p in csvs:
        try:
            with p.open("r", encoding="utf-8", errors="ignore", newline="") as f:
                header = next(csv.reader(f))
            h = [clean_col(x) for x in header]
            joined = " ".join(h)
            looks_imu = (
                ("acc" in joined or "accel" in joined or "accelerometer" in joined)
                and ("gyro" in joined or "gyr" in joined or "gyroscope" in joined or "angular_velocity" in joined)
            )
            if looks_imu:
                good.append(p)
        except Exception:
            continue

    return good


def build_time_vector(df: pd.DataFrame, output_hz: float):
    time_col = find_time_column(df)

    if time_col is not None:
        t = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(t).sum() > len(t) * 0.8:
            # normalize if time starts from non-zero
            t0 = np.nanmin(t)
            if math.isfinite(t0):
                t = t - t0
            return t, time_col

    t = np.arange(len(df), dtype=float) / float(output_hz)
    return t, None


def assign_phases(time_s: np.ndarray, boundaries):
    phase = np.array(["unknown"] * len(time_s), dtype=object)

    for row in boundaries:
        start = row["start_time_s"]
        end = row["end_time_s"]
        name = row["phase"]
        mask = (time_s >= start) & (time_s < end)
        phase[mask] = name

    if boundaries:
        last = boundaries[-1]
        phase[time_s >= last["end_time_s"]] = "after_" + last["phase"]

    return phase


def phase_id(name: str):
    table = {
        "unknown": -1,
        "stand": 0,
        "hold": 0,
        "walk": 1,
        "climb": 1,
        "step": 1,
        "jog": 1,
        "sit": 1,
        "perturb": 2,
        "release": 2,
        "trip": 2,
        "slip": 2,
        "react": 3,
        "reaction": 3,
        "fall": 4,
        "collapse": 4,
    }
    if name.startswith("after_"):
        return 5
    return table.get(name, -1)


def get_phase_start(boundaries, phase_names):
    for row in boundaries:
        if row["phase"] in phase_names:
            return row["start_time_s"]
    return math.nan


def infer_first_impact(df: pd.DataFrame, time_s: np.ndarray, fall_start_time_s: float):
    """
    Approximate first impact from acceleration magnitude after fall start.
    If there is a strong acceleration peak > 25 m/s², use first such point.
    Otherwise use max acceleration after fall start.
    """
    acc_cols = find_axis_columns(df, "acc")
    if not all(acc_cols.values()):
        return math.nan, None, "no_complete_accelerometer_columns"

    ax = pd.to_numeric(df[acc_cols["x"]], errors="coerce").to_numpy(dtype=float)
    ay = pd.to_numeric(df[acc_cols["y"]], errors="coerce").to_numpy(dtype=float)
    az = pd.to_numeric(df[acc_cols["z"]], errors="coerce").to_numpy(dtype=float)

    acc_mag = np.sqrt(ax * ax + ay * ay + az * az)
    valid = np.isfinite(acc_mag)

    if math.isfinite(fall_start_time_s):
        valid = valid & (time_s >= fall_start_time_s)

    idxs = np.where(valid & (acc_mag >= 25.0))[0]
    if len(idxs) > 0:
        i = int(idxs[0])
        return float(time_s[i]), i, "first_acc_magnitude_ge_25_mps2_after_fall_start"

    idxs = np.where(valid)[0]
    if len(idxs) > 0:
        local = idxs[np.nanargmax(acc_mag[idxs])]
        i = int(local)
        return float(time_s[i]), i, "max_acc_magnitude_after_fall_start"

    return math.nan, None, "could_not_infer_impact"


def add_standard_units(df: pd.DataFrame):
    """
    Keeps original columns and adds standard magnitude columns if possible.
    It does not change original data.
    """
    acc_cols = find_axis_columns(df, "acc")
    gyr_cols = find_axis_columns(df, "gyro")

    if all(acc_cols.values()):
        ax = pd.to_numeric(df[acc_cols["x"]], errors="coerce")
        ay = pd.to_numeric(df[acc_cols["y"]], errors="coerce")
        az = pd.to_numeric(df[acc_cols["z"]], errors="coerce")
        df["AccX_original"] = ax
        df["AccY_original"] = ay
        df["AccZ_original"] = az
        df["Acc_Magnitude_original"] = np.sqrt(ax * ax + ay * ay + az * az)

    if all(gyr_cols.values()):
        gx = pd.to_numeric(df[gyr_cols["x"]], errors="coerce")
        gy = pd.to_numeric(df[gyr_cols["y"]], errors="coerce")
        gz = pd.to_numeric(df[gyr_cols["z"]], errors="coerce")
        df["GyrX_original"] = gx
        df["GyrY_original"] = gy
        df["GyrZ_original"] = gz
        df["Gyr_Magnitude_original"] = np.sqrt(gx * gx + gy * gy + gz * gz)

    return {
        "acc_columns": acc_cols,
        "gyro_columns": gyr_cols,
    }



def read_sim_csv_flexible(csv_path: Path):
    """
    Robust reader for simulator CSV files that may contain title/metadata
    lines before the actual table header.
    """
    csv_path = Path(csv_path)

    with csv_path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    candidates = []

    for sep in [",", ";", "\t"]:
        for skip in range(min(len(lines), 250)):
            line = lines[skip].strip()
            if not line:
                continue

            # A useful data header normally has several separated columns.
            if line.count(sep) < 2:
                continue

            try:
                df_try = pd.read_csv(
                    csv_path,
                    skiprows=skip,
                    sep=sep,
                    engine="python",
                    on_bad_lines="skip",
                )
            except Exception:
                continue

            if df_try.empty or df_try.shape[1] < 3:
                continue

            cols_clean = [clean_col(c) for c in df_try.columns]
            joined = " ".join(cols_clean)

            score = df_try.shape[1]

            if any(("time" in c or "timestamp" in c or c == "t") for c in cols_clean):
                score += 20
            if any("frame" in c or "step" in c for c in cols_clean):
                score += 10

            acc_cols = find_axis_columns(df_try, "acc")
            gyr_cols = find_axis_columns(df_try, "gyro")

            if all(acc_cols.values()):
                score += 50
            if all(gyr_cols.values()):
                score += 50

            # Common simulator/export keywords.
            for key in ["acc", "accel", "gyro", "gyr", "imu", "qpos", "qvel", "root"]:
                if key in joined:
                    score += 5

            candidates.append((score, skip, sep, df_try))

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        score, skip, sep, df = candidates[0]
        print(f"[reader] flexible CSV reader: skiprows={skip}, sep={repr(sep)}, columns={df.shape[1]}, score={score}")
        return df

    # Final fallback: try pandas with skipped bad lines.
    print("[reader][warn] Could not detect table header; trying fallback reader.")
    return pd.read_csv(csv_path, engine="python", on_bad_lines="skip")

def label_one_csv(
    csv_path: Path,
    scenario_id: int,
    phase_steps,
    native_hz: float,
    output_hz: float,
    description: str,
):
    df = read_sim_csv_flexible(csv_path)
    if df.empty:
        raise RuntimeError(f"CSV is empty: {csv_path}")

    time_s, time_col = build_time_vector(df, output_hz)
    boundaries = phase_boundaries(phase_steps, native_hz)
    phase = assign_phases(time_s, boundaries)

    perturb_start_time_s = get_phase_start(boundaries, PREFALL_PHASES)
    reaction_start_time_s = get_phase_start(boundaries, REACTION_PHASES)
    fall_start_time_s = get_phase_start(boundaries, FALL_PHASES)

    impact_time_s, impact_idx, impact_source = infer_first_impact(df, time_s, fall_start_time_s)

    detected_cols = add_standard_units(df)

    df["Time_s_standard"] = time_s
    df["Frame_100Hz"] = np.rint(time_s * output_hz).astype(int)
    df["Frame_native_30Hz"] = np.rint(time_s * native_hz).astype(int)

    df["Scenario_ID"] = scenario_id
    df["Scenario_Description"] = description
    df["Sampling_Hz"] = output_hz
    df["Native_Simulation_Hz"] = native_hz

    df["Phase_Name"] = phase
    df["Phase_ID"] = [phase_id(str(x)) for x in phase]

    # Time metadata repeated in every row for easy ML/data filtering
    df["Perturbation_Start_Time_s"] = perturb_start_time_s
    df["Reaction_Start_Time_s"] = reaction_start_time_s
    df["Fall_Start_Time_s"] = fall_start_time_s
    df["First_Impact_Time_s"] = impact_time_s

    df["Perturbation_Start_Frame_100Hz"] = (
        int(round(perturb_start_time_s * output_hz)) if math.isfinite(perturb_start_time_s) else ""
    )
    df["Reaction_Start_Frame_100Hz"] = (
        int(round(reaction_start_time_s * output_hz)) if math.isfinite(reaction_start_time_s) else ""
    )
    df["Fall_Start_Frame_100Hz"] = (
        int(round(fall_start_time_s * output_hz)) if math.isfinite(fall_start_time_s) else ""
    )
    df["First_Impact_Frame_100Hz"] = (
        int(round(impact_time_s * output_hz)) if math.isfinite(impact_time_s) else ""
    )
    df["First_Impact_Source"] = impact_source

    # Simple labels
    df["Normal_Label"] = 0
    df["PreFall_Label"] = 0
    df["Reaction_Label"] = 0
    df["Fall_Label"] = 0
    df["Impact_Label"] = 0
    df["PostImpact_Label"] = 0

    for p in NORMAL_PHASES:
        df.loc[df["Phase_Name"] == p, "Normal_Label"] = 1

    for p in PREFALL_PHASES:
        df.loc[df["Phase_Name"] == p, "PreFall_Label"] = 1

    for p in REACTION_PHASES:
        df.loc[df["Phase_Name"] == p, "Reaction_Label"] = 1

    for p in FALL_PHASES:
        df.loc[df["Phase_Name"] == p, "Fall_Label"] = 1

    if impact_idx is not None:
        df.loc[impact_idx, "Impact_Label"] = 1
        df.loc[time_s >= impact_time_s, "PostImpact_Label"] = 1

    # Binary ML labels
    # 0 = non-fall/normal, 1 = fall event from perturbation/release onward
    if math.isfinite(perturb_start_time_s):
        df["Binary_Fall_Event_From_Perturbation"] = (time_s >= perturb_start_time_s).astype(int)
    else:
        df["Binary_Fall_Event_From_Perturbation"] = 0

    # 0 = before actual fall/collapse, 1 = from fall phase onward
    if math.isfinite(fall_start_time_s):
        df["Binary_Fall_Event_From_FallStart"] = (time_s >= fall_start_time_s).astype(int)
    else:
        df["Binary_Fall_Event_From_FallStart"] = 0

    # 3-class label:
    # 0 = normal activity
    # 1 = perturbation/release + reaction
    # 2 = fall/impact/post-impact
    label3 = np.zeros(len(df), dtype=int)

    if math.isfinite(perturb_start_time_s):
        label3[time_s >= perturb_start_time_s] = 1

    if math.isfinite(fall_start_time_s):
        label3[time_s >= fall_start_time_s] = 2

    if math.isfinite(impact_time_s):
        label3[time_s >= impact_time_s] = 2

    df["Label_3Class"] = label3

    # Human-readable activity label
    activity = np.array(["normal"] * len(df), dtype=object)

    if math.isfinite(perturb_start_time_s):
        activity[time_s >= perturb_start_time_s] = "pre_fall_perturbation"

    if math.isfinite(reaction_start_time_s):
        activity[time_s >= reaction_start_time_s] = "reaction"

    if math.isfinite(fall_start_time_s):
        activity[time_s >= fall_start_time_s] = "fall"

    if math.isfinite(impact_time_s):
        activity[time_s >= impact_time_s] = "post_impact"
        if impact_idx is not None:
            activity[impact_idx] = "first_impact"

    df["Activity_Label"] = activity

    out_csv = csv_path.with_name(csv_path.stem + "_labeled.csv")
    out_json = csv_path.with_name(csv_path.stem + "_labeled.metadata.json")

    df.to_csv(out_csv, index=False)

    meta = {
        "input_csv": str(csv_path),
        "output_csv": str(out_csv),
        "scenario_id": scenario_id,
        "description": description,
        "time_column_detected": time_col,
        "native_hz": native_hz,
        "output_hz": output_hz,
        "phase_steps": [{"phase": a, "steps": b} for a, b in phase_steps],
        "phase_boundaries": boundaries,
        "events": {
            "perturbation_start_time_s": perturb_start_time_s,
            "reaction_start_time_s": reaction_start_time_s,
            "fall_start_time_s": fall_start_time_s,
            "first_impact_time_s": impact_time_s,
            "first_impact_index_row": impact_idx,
            "first_impact_source": impact_source,
        },
        "detected_columns": detected_cols,
        "label_columns": [
            "Phase_Name",
            "Phase_ID",
            "Normal_Label",
            "PreFall_Label",
            "Reaction_Label",
            "Fall_Label",
            "Impact_Label",
            "PostImpact_Label",
            "Binary_Fall_Event_From_Perturbation",
            "Binary_Fall_Event_From_FallStart",
            "Label_3Class",
            "Activity_Label",
        ],
    }

    out_json.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    return out_csv, out_json, meta


def main():
    ap = argparse.ArgumentParser(description="Add labels to saved simulator IMU CSV files without modifying the simulator.")
    ap.add_argument("path", help="Path to a CSV file or an output folder containing IMU CSV files.")
    ap.add_argument("--scenario-id", type=int, default=None)
    ap.add_argument("--description", default="")
    ap.add_argument("--phase-steps", default=None,
                    help="Example: stand:195,walk:300,perturb:60,react:31,fall:470")
    ap.add_argument("--native-hz", type=float, default=DEFAULT_NATIVE_HZ)
    ap.add_argument("--output-hz", type=float, default=DEFAULT_OUTPUT_HZ)
    args = ap.parse_args()

    path = Path(args.path)

    if not path.exists():
        print(f"[error] Path does not exist: {path}")
        sys.exit(2)

    scenario_id = args.scenario_id or infer_scenario_id(path)
    if scenario_id is None:
        print("[error] Could not infer scenario ID. Please pass --scenario-id 34, etc.")
        sys.exit(2)

    if args.phase_steps:
        phase_steps = parse_phase_steps(args.phase_steps)
    else:
        phase_steps = DEFAULT_PHASES.get(scenario_id)

    if phase_steps is None:
        print(f"[error] No default phase timing is known for scenario {scenario_id}.")
        print("Please pass --phase-steps, for example:")
        print("  --phase-steps stand:195,walk:300,perturb:60,react:31,fall:470")
        sys.exit(2)

    if path.is_file():
        csvs = [path]
    else:
        csvs = first_valid_csv_in_folder(path)

    if not csvs:
        print(f"[error] No IMU CSV file found in: {path}")
        print("Check files with:")
        print(f"  find {path} -type f | sort")
        sys.exit(2)

    print("=" * 80)
    print("IMU LABEL POSTPROCESSOR")
    print(f"Scenario ID : {scenario_id}")
    print(f"Description : {args.description}")
    print(f"Native Hz   : {args.native_hz}")
    print(f"Output Hz   : {args.output_hz}")
    print("Phase steps : " + ", ".join([f"{a}:{b}" for a, b in phase_steps]))
    print(f"CSV files   : {len(csvs)}")
    print("=" * 80)

    for csv_path in csvs:
        print(f"\n[read]  {csv_path}")
        out_csv, out_json, meta = label_one_csv(
            csv_path=csv_path,
            scenario_id=scenario_id,
            phase_steps=phase_steps,
            native_hz=args.native_hz,
            output_hz=args.output_hz,
            description=args.description,
        )
        print(f"[save]  {out_csv}")
        print(f"[meta]  {out_json}")
        print("[events]")
        for k, v in meta["events"].items():
            print(f"  {k}: {v}")

    print("\nDone.")


if __name__ == "__main__":
    main()
