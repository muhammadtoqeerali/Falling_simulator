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


DEFAULT_PHASES = {
    34: [("stand", 195), ("walk", 300), ("perturb", 60), ("react", 31), ("fall", 470)],
    43: [("stand", 52), ("step", 280), ("perturb", 48), ("react", 18), ("fall", 281)],
}


PHASE_ALIASES = {
    "hold": "stand",
    "climb": "step",
    "release": "perturb",
    "reaction": "react",
    "collapse": "fall",
}


def clean_col(x: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(x).strip().lower()).strip("_")


def normalize_phase_name(name: str) -> str:
    name = str(name).strip().lower()
    return PHASE_ALIASES.get(name, name)


def infer_scenario_id(path: Path):
    m = re.search(r"scenario[_-]?(\d+)", str(path), re.IGNORECASE)
    return int(m.group(1)) if m else None


def parse_phase_steps(text: str):
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Bad phase step: {part}. Use name:steps")
        name, steps = part.split(":", 1)
        out.append((normalize_phase_name(name), int(steps)))
    if not out:
        raise ValueError("No phase steps provided.")
    return out


def read_sim_csv_flexible(csv_path: Path):
    """
    Reads simulator CSV files with comment metadata at the top.
    Returns dataframe and metadata-comment dictionary.
    """
    csv_path = Path(csv_path)

    comments = {}
    with csv_path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    for line in lines:
        s = line.strip()
        if not s.startswith("#"):
            continue
        body = s[1:].strip()
        if ":" in body:
            k, v = body.split(":", 1)
            comments[clean_col(k)] = v.strip()

    candidates = []
    for sep in [",", ";", "\t"]:
        for skip in range(min(len(lines), 300)):
            line = lines[skip].strip()
            if not line or line.startswith("#"):
                continue
            if line.count(sep) < 3:
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

            if df_try.empty or df_try.shape[1] < 5:
                continue

            cols = [clean_col(c) for c in df_try.columns]
            joined = " ".join(cols)
            score = df_try.shape[1]

            for key in ["timestamp", "accel_raw", "gyro", "fall_detected", "pelvis_height", "impact_magnitude"]:
                if key in joined:
                    score += 25

            candidates.append((score, skip, sep, df_try))

    if not candidates:
        raise RuntimeError(f"Could not find real CSV table header in {csv_path}")

    candidates.sort(key=lambda x: x[0], reverse=True)
    score, skip, sep, df = candidates[0]
    print(f"[reader] skiprows={skip}, sep={repr(sep)}, columns={df.shape[1]}, score={score}")
    return df, comments


def first_valid_csv_in_folder(folder: Path):
    exclude = [
        "contacts", "dynamics", "joints", "markers", "marker_quality",
        "segments", "labeled", "enriched"
    ]

    candidates = []
    for p in sorted(folder.rglob("*.csv")):
        name = p.name.lower()
        if any(x in name for x in exclude):
            continue
        candidates.append(p)

    return candidates


def phase_boundaries(phase_steps, native_hz: float):
    rows = []
    cursor = 0
    for name, nsteps in phase_steps:
        name = normalize_phase_name(name)
        start = cursor
        end = cursor + int(nsteps)
        rows.append({
            "phase": name,
            "start_step_native": start,
            "end_step_native": end,
            "start_time_s": start / native_hz,
            "end_time_s": end / native_hz,
            "duration_s": int(nsteps) / native_hz,
            "steps": int(nsteps),
        })
        cursor = end
    return rows


def build_time_vector(df: pd.DataFrame, output_hz: float):
    time_col = None
    for c in df.columns:
        cc = clean_col(c)
        if cc in ("timestamp", "time", "time_s", "t"):
            time_col = c
            break

    if time_col is not None:
        t = pd.to_numeric(df[time_col], errors="coerce").to_numpy(float)
        if np.isfinite(t).sum() > len(t) * 0.8:
            # Important: shift to zero so phase times from native steps match.
            return t - np.nanmin(t), time_col

    return np.arange(len(df), dtype=float) / float(output_hz), None


def assign_phases(time_s: np.ndarray, boundaries):
    phase = np.array(["unknown"] * len(time_s), dtype=object)
    for row in boundaries:
        mask = (time_s >= row["start_time_s"]) & (time_s < row["end_time_s"])
        phase[mask] = row["phase"]

    if boundaries:
        phase[time_s >= boundaries[-1]["end_time_s"]] = "after_" + boundaries[-1]["phase"]

    return phase


def get_phase_start(boundaries, phase_name: str):
    phase_name = normalize_phase_name(phase_name)
    for row in boundaries:
        if row["phase"] == phase_name:
            return row["start_time_s"]
    return math.nan


def get_phase_end(boundaries, phase_name: str):
    phase_name = normalize_phase_name(phase_name)
    for row in boundaries:
        if row["phase"] == phase_name:
            return row["end_time_s"]
    return math.nan


def safe_numeric(df, col):
    if col not in df.columns:
        return None
    return pd.to_numeric(df[col], errors="coerce").to_numpy(float)


def add_signal_magnitudes(df: pd.DataFrame):
    """
    Adds both gravity-compensated and raw acceleration magnitudes when available.
    The raw acceleration should be used for impact detection.
    """
    cols = {clean_col(c): c for c in df.columns}

    def add_mag(prefix, out_name):
        xs = cols.get(f"{prefix}_x")
        ys = cols.get(f"{prefix}_y")
        zs = cols.get(f"{prefix}_z")
        if xs and ys and zs:
            x = pd.to_numeric(df[xs], errors="coerce")
            y = pd.to_numeric(df[ys], errors="coerce")
            z = pd.to_numeric(df[zs], errors="coerce")
            df[out_name] = np.sqrt(x * x + y * y + z * z)
            return {"x": xs, "y": ys, "z": zs, "magnitude": out_name}
        return {"x": None, "y": None, "z": None, "magnitude": None}

    accel = add_mag("accel", "Accel_GravityComp_Magnitude_mps2")
    accel_raw = add_mag("accel_raw", "Accel_Raw_Magnitude_mps2")
    accel_true = add_mag("accel_true", "Accel_True_Magnitude_mps2")
    gyro = add_mag("gyro", "Gyro_Magnitude_rads")

    return {
        "accel_gravity_comp": accel,
        "accel_raw": accel_raw,
        "accel_true": accel_true,
        "gyro": gyro,
    }


def infer_fall_onset(df, time_s, perturb_start_time_s):
    """
    Prefer biomechanical onset from height drop.
    This matches scenario 43 where the validator reports onset_reason=height_drop_threshold.
    """
    h = safe_numeric(df, "pelvis_height")
    if h is not None and math.isfinite(perturb_start_time_s):
        pre = (time_s >= max(0.0, perturb_start_time_s - 2.0)) & (time_s < perturb_start_time_s)
        post = time_s >= perturb_start_time_s

        if np.any(pre) and np.any(post):
            h_ref = float(np.nanmax(h[pre]))
            # 12.5% drop from high posture, with a minimum 0.20 m drop.
            drop = max(0.20, 0.125 * max(h_ref, 1.0))
            threshold = h_ref - drop

            idxs = np.where(post & np.isfinite(h) & (h <= threshold))[0]
            if len(idxs) > 0:
                i = int(idxs[0])
                return float(time_s[i]), i, f"pelvis_height_drop_threshold_h_ref={h_ref:.3f}_threshold={threshold:.3f}"

    # fallback: first fall_detected
    if "fall_detected" in df.columns:
        fd = pd.to_numeric(df["fall_detected"], errors="coerce").fillna(0).to_numpy(int)
        idxs = np.where(fd == 1)[0]
        if len(idxs) > 0:
            i = int(idxs[0])
            return float(time_s[i]), i, "first_existing_fall_detected_frame"

    return math.nan, None, "could_not_infer_fall_onset"


def infer_impact(df, time_s, fall_onset_time_s):
    """
    Impact should be detected from accel_raw/impact_magnitude, not gravity-compensated accel.
    """
    if "Accel_Raw_Magnitude_mps2" in df.columns:
        raw = pd.to_numeric(df["Accel_Raw_Magnitude_mps2"], errors="coerce").to_numpy(float)
    elif "impact_magnitude" in df.columns:
        raw = pd.to_numeric(df["impact_magnitude"], errors="coerce").to_numpy(float)
    else:
        raw = None

    search = np.ones(len(df), dtype=bool)
    if math.isfinite(fall_onset_time_s):
        search = (time_s >= fall_onset_time_s) & (time_s <= fall_onset_time_s + 3.0)

    if raw is not None:
        idxs = np.where(search & np.isfinite(raw))[0]
        if len(idxs) > 0:
            # Main impact = peak raw/proper acceleration in early fall window.
            local = idxs[np.nanargmax(raw[idxs])]
            i = int(local)
            return float(time_s[i]), i, "peak_accel_raw_magnitude_within_3s_after_fall_onset"

    if "impact_force" in df.columns:
        force = pd.to_numeric(df["impact_force"], errors="coerce").to_numpy(float)
        idxs = np.where(search & np.isfinite(force) & (force > 50.0))[0]
        if len(idxs) > 0:
            i = int(idxs[0])
            return float(time_s[i]), i, "first_impact_force_gt_50N"

    return math.nan, None, "could_not_infer_impact"


def phase_id(name: str):
    name = str(name)
    table = {
        "unknown": -1,
        "stand": 0,
        "walk": 1,
        "step": 1,
        "climb": 1,
        "jog": 1,
        "perturb": 2,
        "release": 2,
        "react": 3,
        "fall": 4,
    }
    if name.startswith("after_"):
        return 5
    return table.get(name, -1)


def label_one_csv(csv_path, scenario_id, phase_steps, native_hz, output_hz, description):
    csv_path = Path(csv_path)
    df, comments = read_sim_csv_flexible(csv_path)

    time_s, time_col = build_time_vector(df, output_hz)
    signal_cols = add_signal_magnitudes(df)

    boundaries = phase_boundaries(phase_steps, native_hz)
    phase = assign_phases(time_s, boundaries)

    perturb_start_time_s = get_phase_start(boundaries, "perturb")
    reaction_start_time_s = get_phase_start(boundaries, "react")
    scripted_fall_phase_start_time_s = get_phase_start(boundaries, "fall")

    fall_onset_time_s, fall_onset_idx, fall_onset_source = infer_fall_onset(
        df, time_s, perturb_start_time_s
    )

    impact_time_s, impact_idx, impact_source = infer_impact(
        df, time_s, fall_onset_time_s
    )

    df["Time_s_standard"] = time_s
    df["Frame_100Hz"] = np.rint(time_s * output_hz).astype(int)
    df["Frame_native_30Hz"] = np.rint(time_s * native_hz).astype(int)

    df["Scenario_ID"] = scenario_id
    df["Scenario_Description"] = description
    df["Sampling_Hz"] = output_hz
    df["Native_Simulation_Hz"] = native_hz

    df["Phase_Name"] = phase
    df["Phase_ID"] = [phase_id(x) for x in phase]

    df["Perturbation_Start_Time_s"] = perturb_start_time_s
    df["Reaction_Start_Time_s"] = reaction_start_time_s
    df["Scripted_Fall_Phase_Start_Time_s"] = scripted_fall_phase_start_time_s

    df["Fall_Onset_Time_s"] = fall_onset_time_s
    df["Fall_Onset_Frame_100Hz"] = int(round(fall_onset_time_s * output_hz)) if math.isfinite(fall_onset_time_s) else ""
    df["Fall_Onset_Source"] = fall_onset_source

    df["Main_Impact_Time_s"] = impact_time_s
    df["Main_Impact_Frame_100Hz"] = int(round(impact_time_s * output_hz)) if math.isfinite(impact_time_s) else ""
    df["Main_Impact_Source"] = impact_source

    df["Perturbation_Start_Frame_100Hz"] = int(round(perturb_start_time_s * output_hz)) if math.isfinite(perturb_start_time_s) else ""
    df["Reaction_Start_Frame_100Hz"] = int(round(reaction_start_time_s * output_hz)) if math.isfinite(reaction_start_time_s) else ""
    df["Scripted_Fall_Phase_Start_Frame_100Hz"] = int(round(scripted_fall_phase_start_time_s * output_hz)) if math.isfinite(scripted_fall_phase_start_time_s) else ""

    # Phase-based labels
    df["Normal_Label"] = np.isin(phase, ["stand", "walk", "step", "climb"]).astype(int)
    df["Perturbation_Label"] = (phase == "perturb").astype(int)
    df["Reaction_Label"] = (phase == "react").astype(int)
    df["Scripted_Fall_Phase_Label"] = (phase == "fall").astype(int)

    # Event-based labels
    df["PreFall_Label"] = 0
    if math.isfinite(perturb_start_time_s) and math.isfinite(fall_onset_time_s):
        df.loc[(time_s >= perturb_start_time_s) & (time_s < fall_onset_time_s), "PreFall_Label"] = 1

    df["Fall_Label"] = 0
    if math.isfinite(fall_onset_time_s):
        df.loc[time_s >= fall_onset_time_s, "Fall_Label"] = 1

    df["PreImpact_Fall_Label"] = 0
    if math.isfinite(fall_onset_time_s) and math.isfinite(impact_time_s):
        df.loc[(time_s >= fall_onset_time_s) & (time_s < impact_time_s), "PreImpact_Fall_Label"] = 1

    df["Impact_Label"] = 0
    if impact_idx is not None:
        df.loc[impact_idx, "Impact_Label"] = 1

    df["PostImpact_Label"] = 0
    if math.isfinite(impact_time_s):
        df.loc[time_s > impact_time_s, "PostImpact_Label"] = 1

    if "fall_detected" in df.columns:
        df["Existing_FallDetected_Label"] = pd.to_numeric(df["fall_detected"], errors="coerce").fillna(0).astype(int)
    else:
        df["Existing_FallDetected_Label"] = 0

    # 3-class: 0 normal, 1 pre-fall/reaction, 2 fall/impact/post-impact
    label3 = np.zeros(len(df), dtype=int)
    if math.isfinite(perturb_start_time_s):
        label3[time_s >= perturb_start_time_s] = 1
    if math.isfinite(fall_onset_time_s):
        label3[time_s >= fall_onset_time_s] = 2
    df["Label_3Class"] = label3

    # 4-class: 0 normal, 1 perturbation/reaction before onset, 2 falling before impact, 3 post-impact
    label4 = np.zeros(len(df), dtype=int)
    if math.isfinite(perturb_start_time_s):
        label4[time_s >= perturb_start_time_s] = 1
    if math.isfinite(fall_onset_time_s):
        label4[time_s >= fall_onset_time_s] = 2
    if math.isfinite(impact_time_s):
        label4[time_s >= impact_time_s] = 3
    if impact_idx is not None:
        label4[impact_idx] = 3
    df["Label_4Class"] = label4

    activity = np.array(["normal"] * len(df), dtype=object)
    if math.isfinite(perturb_start_time_s):
        activity[time_s >= perturb_start_time_s] = "pre_fall_perturbation"
    if math.isfinite(reaction_start_time_s):
        activity[time_s >= reaction_start_time_s] = "reaction"
    if math.isfinite(fall_onset_time_s):
        activity[time_s >= fall_onset_time_s] = "falling_pre_impact"
    if math.isfinite(impact_time_s):
        activity[time_s > impact_time_s] = "post_impact"
    if impact_idx is not None:
        activity[impact_idx] = "main_impact"
    df["Activity_Label"] = activity

    out_csv = csv_path.with_name(csv_path.stem + "_labeled.csv")
    out_json = csv_path.with_name(csv_path.stem + "_labeled.metadata.json")

    df.to_csv(out_csv, index=False)

    meta = {
        "input_csv": str(csv_path),
        "output_csv": str(out_csv),
        "scenario_id": scenario_id,
        "description": description,
        "comments_from_original_csv": comments,
        "time_column_detected": time_col,
        "native_hz": native_hz,
        "output_hz": output_hz,
        "phase_steps": [{"phase": a, "steps": b} for a, b in phase_steps],
        "phase_boundaries": boundaries,
        "signal_columns": signal_cols,
        "events": {
            "perturbation_start_time_s": perturb_start_time_s,
            "reaction_start_time_s": reaction_start_time_s,
            "scripted_fall_phase_start_time_s": scripted_fall_phase_start_time_s,
            "fall_onset_time_s": fall_onset_time_s,
            "fall_onset_index_row": fall_onset_idx,
            "fall_onset_source": fall_onset_source,
            "main_impact_time_s": impact_time_s,
            "main_impact_index_row": impact_idx,
            "main_impact_source": impact_source,
        },
    }

    out_json.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return out_csv, out_json, meta


def main():
    ap = argparse.ArgumentParser(description="Add event-correct labels to saved simulator IMU CSV files.")
    ap.add_argument("path", help="Path to CSV file or output folder.")
    ap.add_argument("--scenario-id", type=int, default=None)
    ap.add_argument("--description", default="")
    ap.add_argument("--phase-steps", default=None, help="Example: stand:52,step:280,perturb:48,react:18,fall:281")
    ap.add_argument("--native-hz", type=float, default=DEFAULT_NATIVE_HZ)
    ap.add_argument("--output-hz", type=float, default=DEFAULT_OUTPUT_HZ)
    args = ap.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"[error] Path does not exist: {path}")
        sys.exit(2)

    scenario_id = args.scenario_id or infer_scenario_id(path)
    if scenario_id is None:
        print("[error] Could not infer scenario ID. Pass --scenario-id.")
        sys.exit(2)

    if args.phase_steps:
        phase_steps = parse_phase_steps(args.phase_steps)
    else:
        phase_steps = DEFAULT_PHASES.get(scenario_id)

    if phase_steps is None:
        print(f"[error] No default phase timing for scenario {scenario_id}. Use --phase-steps.")
        sys.exit(2)

    csvs = [path] if path.is_file() else first_valid_csv_in_folder(path)
    if not csvs:
        print(f"[error] No candidate IMU CSV found in: {path}")
        sys.exit(2)

    print("=" * 80)
    print("EVENT-CORRECT IMU LABEL POSTPROCESSOR")
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
