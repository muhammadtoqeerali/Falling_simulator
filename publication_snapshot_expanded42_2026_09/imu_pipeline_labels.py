# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_OUTPUT_HZ = 100.0
DEFAULT_NATIVE_HZ = 30.0


AUX_CSV_NAME_PARTS = [
    "contacts",
    "dynamics",
    "joints",
    "markers",
    "marker_quality",
    "segments",
    "labeled",
    "enriched",
    "backup",
    "quality",
]


def clean_col(x: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(x).strip().lower()).strip("_")


def parse_float(text: Any) -> float:
    try:
        if text is None:
            return math.nan
        m = re.search(r"[-+]?\d+(?:\.\d+)?", str(text))
        return float(m.group(0)) if m else math.nan
    except Exception:
        return math.nan


def read_sim_csv_flexible(csv_path: Path) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Reads simulator IMU CSV files that contain comment metadata lines before
    the real table header.
    """
    csv_path = Path(csv_path)

    with csv_path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    comments: Dict[str, str] = {}
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

            for key in [
                "timestamp",
                "accel_raw",
                "gyro",
                "fall_detected",
                "pelvis_height",
                "impact_magnitude",
                "sensor_pos",
                "jerk_mag",
            ]:
                if key in joined:
                    score += 25

            candidates.append((score, skip, sep, df_try))

    if not candidates:
        raise RuntimeError(f"Could not detect IMU CSV table header in {csv_path}")

    candidates.sort(key=lambda x: x[0], reverse=True)
    score, skip, sep, df = candidates[0]
    return df, comments


def find_main_imu_csvs(output_dir: Path) -> List[Path]:
    """
    Finds the main IMU CSV in a scenario output folder.
    Excludes biomechanics, marker, joint, contact, and already-labeled files.
    """
    output_dir = Path(output_dir)
    candidates = []

    for p in sorted(output_dir.glob("*.csv")):
        name = p.name.lower()
        if any(x in name for x in AUX_CSV_NAME_PARTS):
            continue

        try:
            df, _ = read_sim_csv_flexible(p)
        except Exception:
            continue

        cols = [clean_col(c) for c in df.columns]
        joined = " ".join(cols)

        looks_imu = (
            "accel" in joined
            and "gyro" in joined
            and ("timestamp" in joined or "time" in joined)
        )

        if looks_imu:
            candidates.append(p)

    return candidates


def parse_validation_events(output_dir: Path) -> Dict[str, Any]:
    """
    Reads *_validation.txt and extracts simulator-reported event timing.
    Preferred over estimating events from phase durations.
    """
    output_dir = Path(output_dir)
    validation_files = sorted(output_dir.glob("*_validation.txt"))

    events: Dict[str, Any] = {
        "perturbation_start_time_s": math.nan,
        "fall_onset_time_s": math.nan,
        "main_impact_time_s": math.nan,
        "settle_time_s": math.nan,
        "source": "not_found",
    }

    if not validation_files:
        return events

    txt_path = validation_files[-1]
    text = txt_path.read_text(encoding="utf-8", errors="ignore")

    patterns = {
        "perturbation_start_time_s": r"Perturb start\s*:\s*([0-9.]+)\s*s",
        "fall_onset_time_s": r"Fall onset\s*:\s*([0-9.]+)\s*s",
        "main_impact_time_s": r"Main impact\s*:\s*([0-9.]+)\s*s",
        "settle_time_s": r"Settle time\s*:\s*([0-9.]+)\s*s",
    }

    for key, pat in patterns.items():
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            events[key] = float(m.group(1))

    events["source"] = str(txt_path)
    return events


def build_time_vector(df: pd.DataFrame, output_hz: float) -> Tuple[np.ndarray, Optional[str]]:
    time_col = None
    for c in df.columns:
        cc = clean_col(c)
        if cc in ("timestamp", "time", "time_s", "t"):
            time_col = c
            break

    if time_col is not None:
        t = pd.to_numeric(df[time_col], errors="coerce").to_numpy(float)
        if np.isfinite(t).sum() > len(t) * 0.8:
            return t - np.nanmin(t), time_col

    return np.arange(len(df), dtype=float) / float(output_hz), None


def add_signal_magnitudes(df: pd.DataFrame) -> Dict[str, Any]:
    cols = {clean_col(c): c for c in df.columns}

    def add_mag(prefix: str, out_name: str):
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

    return {
        "accel_gravity_comp": add_mag("accel", "Accel_GravityComp_Magnitude_mps2"),
        "accel_raw": add_mag("accel_raw", "Accel_Raw_Magnitude_mps2"),
        "accel_true": add_mag("accel_true", "Accel_True_Magnitude_mps2"),
        "gyro": add_mag("gyro", "Gyro_Magnitude_rads"),
    }


def infer_fall_onset_from_height(df: pd.DataFrame, time_s: np.ndarray, perturb_t: float) -> Tuple[float, Optional[int], str]:
    if "pelvis_height" not in df.columns or not math.isfinite(perturb_t):
        return math.nan, None, "no_pelvis_height_or_no_perturb_time"

    h = pd.to_numeric(df["pelvis_height"], errors="coerce").to_numpy(float)

    pre = (time_s >= max(0.0, perturb_t - 2.0)) & (time_s < perturb_t)
    post = time_s >= perturb_t

    if not np.any(pre) or not np.any(post):
        return math.nan, None, "insufficient_pre_or_post_window"

    h_ref = float(np.nanmax(h[pre]))
    drop = max(0.20, 0.125 * max(h_ref, 1.0))
    threshold = h_ref - drop

    idxs = np.where(post & np.isfinite(h) & (h <= threshold))[0]
    if len(idxs) == 0:
        return math.nan, None, "no_height_drop_detected"

    i = int(idxs[0])
    return float(time_s[i]), i, f"pelvis_height_drop_threshold_h_ref={h_ref:.3f}_threshold={threshold:.3f}"


def infer_main_impact(df: pd.DataFrame, time_s: np.ndarray, fall_onset_t: float) -> Tuple[float, Optional[int], str]:
    """
    Uses raw/proper acceleration or impact_magnitude, not gravity-compensated acceleration.
    """
    signal = None
    signal_name = None

    if "Accel_Raw_Magnitude_mps2" in df.columns:
        signal = pd.to_numeric(df["Accel_Raw_Magnitude_mps2"], errors="coerce").to_numpy(float)
        signal_name = "Accel_Raw_Magnitude_mps2"
    elif "impact_magnitude" in df.columns:
        signal = pd.to_numeric(df["impact_magnitude"], errors="coerce").to_numpy(float)
        signal_name = "impact_magnitude"

    if signal is None:
        return math.nan, None, "no_accel_raw_or_impact_magnitude"

    search = np.ones(len(df), dtype=bool)

    if math.isfinite(fall_onset_t):
        search = (time_s >= fall_onset_t) & (time_s <= fall_onset_t + 3.0)

    idxs = np.where(search & np.isfinite(signal))[0]
    if len(idxs) == 0:
        return math.nan, None, "no_valid_impact_search_samples"

    local = idxs[np.nanargmax(signal[idxs])]
    i = int(local)

    return float(time_s[i]), i, f"peak_{signal_name}_within_3s_after_fall_onset"


def add_labels_to_dataframe(
    df: pd.DataFrame,
    events: Dict[str, Any],
    scenario_id: Optional[int],
    description: str,
    output_hz: float,
    native_hz: float,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = df.copy()

    time_s, time_col = build_time_vector(df, output_hz)
    signal_cols = add_signal_magnitudes(df)

    perturb_t = parse_float(events.get("perturbation_start_time_s"))
    fall_onset_t = parse_float(events.get("fall_onset_time_s"))
    impact_t = parse_float(events.get("main_impact_time_s"))
    settle_t = parse_float(events.get("settle_time_s"))

    fall_onset_idx = None
    fall_onset_source = "validation_txt"

    if not math.isfinite(fall_onset_t):
        fall_onset_t, fall_onset_idx, fall_onset_source = infer_fall_onset_from_height(df, time_s, perturb_t)
    else:
        fall_onset_idx = int(np.argmin(np.abs(time_s - fall_onset_t)))

    impact_idx = None
    impact_source = "validation_txt"

    if not math.isfinite(impact_t):
        impact_t, impact_idx, impact_source = infer_main_impact(df, time_s, fall_onset_t)
    else:
        impact_idx = int(np.argmin(np.abs(time_s - impact_t)))

    if not math.isfinite(settle_t) and math.isfinite(impact_t):
        settle_t = math.nan

    df["Time_s_standard"] = time_s
    df["Frame_100Hz"] = np.rint(time_s * output_hz).astype(int)
    df["Frame_native_30Hz"] = np.rint(time_s * native_hz).astype(int)

    df["Scenario_ID"] = "" if scenario_id is None else scenario_id
    df["Scenario_Description"] = description
    df["Sampling_Hz"] = output_hz
    df["Native_Simulation_Hz"] = native_hz

    df["Perturbation_Start_Time_s"] = perturb_t
    df["Fall_Onset_Time_s"] = fall_onset_t
    df["Main_Impact_Time_s"] = impact_t
    df["Settle_Time_s"] = settle_t

    df["Perturbation_Start_Frame_100Hz"] = int(round(perturb_t * output_hz)) if math.isfinite(perturb_t) else ""
    df["Fall_Onset_Frame_100Hz"] = int(round(fall_onset_t * output_hz)) if math.isfinite(fall_onset_t) else ""
    df["Main_Impact_Frame_100Hz"] = int(round(impact_t * output_hz)) if math.isfinite(impact_t) else ""
    df["Settle_Frame_100Hz"] = int(round(settle_t * output_hz)) if math.isfinite(settle_t) else ""

    df["Fall_Onset_Source"] = fall_onset_source
    df["Main_Impact_Source"] = impact_source

    activity = np.array(["normal"] * len(df), dtype=object)

    if math.isfinite(perturb_t):
        activity[time_s >= perturb_t] = "pre_fall_perturbation"

    if math.isfinite(fall_onset_t):
        activity[time_s >= fall_onset_t] = "falling_pre_impact"

    if math.isfinite(impact_t):
        activity[time_s > impact_t] = "post_impact"
        if impact_idx is not None:
            activity[impact_idx] = "main_impact"

    if math.isfinite(settle_t):
        activity[time_s >= settle_t] = "settled_post_fall"

    df["Activity_Label"] = activity

    df["Normal_Label"] = (activity == "normal").astype(int)
    df["PreFall_Label"] = (activity == "pre_fall_perturbation").astype(int)
    df["Fall_Label"] = np.isin(activity, ["falling_pre_impact", "main_impact", "post_impact", "settled_post_fall"]).astype(int)
    df["PreImpact_Fall_Label"] = (activity == "falling_pre_impact").astype(int)
    df["Impact_Label"] = (activity == "main_impact").astype(int)
    df["PostImpact_Label"] = np.isin(activity, ["post_impact", "settled_post_fall"]).astype(int)

    if "fall_detected" in df.columns:
        df["Existing_FallDetected_Label"] = pd.to_numeric(df["fall_detected"], errors="coerce").fillna(0).astype(int)
    else:
        df["Existing_FallDetected_Label"] = 0

    # 3-class: 0 = normal, 1 = perturbation/pre-fall, 2 = fall/post-impact
    label3 = np.zeros(len(df), dtype=int)

    if math.isfinite(perturb_t):
        label3[time_s >= perturb_t] = 1

    if math.isfinite(fall_onset_t):
        label3[time_s >= fall_onset_t] = 2

    df["Label_3Class"] = label3

    # 4-class: 0 = normal, 1 = pre-fall, 2 = falling before impact, 3 = impact/post-impact
    label4 = np.zeros(len(df), dtype=int)

    if math.isfinite(perturb_t):
        label4[time_s >= perturb_t] = 1

    if math.isfinite(fall_onset_t):
        label4[time_s >= fall_onset_t] = 2

    if math.isfinite(impact_t):
        label4[time_s >= impact_t] = 3

    if impact_idx is not None:
        label4[impact_idx] = 3

    df["Label_4Class"] = label4

    meta = {
        "time_column_detected": time_col,
        "signal_columns": signal_cols,
        "events": {
            "perturbation_start_time_s": perturb_t,
            "fall_onset_time_s": fall_onset_t,
            "fall_onset_index_row": fall_onset_idx,
            "fall_onset_source": fall_onset_source,
            "main_impact_time_s": impact_t,
            "main_impact_index_row": impact_idx,
            "main_impact_source": impact_source,
            "settle_time_s": settle_t,
        },
        "label_columns": [
            "Activity_Label",
            "Normal_Label",
            "PreFall_Label",
            "Fall_Label",
            "PreImpact_Fall_Label",
            "Impact_Label",
            "PostImpact_Label",
            "Existing_FallDetected_Label",
            "Label_3Class",
            "Label_4Class",
        ],
    }

    return df, meta


def already_labeled(csv_path: Path) -> bool:
    try:
        df, _ = read_sim_csv_flexible(csv_path)
        cols = set(df.columns)
        return "Label_4Class" in cols and "Activity_Label" in cols
    except Exception:
        return False


def label_imu_csv_inplace(
    csv_path: Path,
    output_dir: Path,
    scenario_id: Optional[int] = None,
    description: str = "",
    output_hz: float = DEFAULT_OUTPUT_HZ,
    native_hz: float = DEFAULT_NATIVE_HZ,
) -> Dict[str, Any]:
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)

    if already_labeled(csv_path):
        return {"csv": str(csv_path), "status": "already_labeled"}

    df, comments = read_sim_csv_flexible(csv_path)

    if not description:
        description = comments.get("fall_type", "")

    output_hz = parse_float(comments.get("sampling_rate_hz")) if comments.get("sampling_rate_hz") else output_hz
    native_hz = parse_float(comments.get("native_sampling_rate_hz")) if comments.get("native_sampling_rate_hz") else native_hz

    if not math.isfinite(output_hz) or output_hz <= 0:
        output_hz = DEFAULT_OUTPUT_HZ

    if not math.isfinite(native_hz) or native_hz <= 0:
        native_hz = DEFAULT_NATIVE_HZ

    events = parse_validation_events(output_dir)

    labeled_df, meta = add_labels_to_dataframe(
        df=df,
        events=events,
        scenario_id=scenario_id,
        description=description,
        output_hz=output_hz,
        native_hz=native_hz,
    )

    backup_csv = csv_path.with_name(csv_path.stem + ".unlabeled_backup.csv")

    if not backup_csv.exists():
        shutil.copy2(csv_path, backup_csv)

    labeled_df.to_csv(csv_path, index=False)

    meta_path = csv_path.with_name(csv_path.stem + ".label_metadata.json")
    payload = {
        "input_csv_overwritten_with_labels": str(csv_path),
        "backup_unlabeled_csv": str(backup_csv),
        "metadata_comments_from_original_csv": comments,
        "validation_event_source": events.get("source"),
        "scenario_id": scenario_id,
        "description": description,
        **meta,
    }
    meta_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    return {
        "csv": str(csv_path),
        "backup_csv": str(backup_csv),
        "metadata_json": str(meta_path),
        "status": "labeled_inplace",
        "events": meta["events"],
    }


def label_output_folder_inplace(
    output_dir: str | Path,
    scenario_id: Optional[int] = None,
    subject_params: Optional[Dict[str, Any]] = None,
    result: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    output_dir = Path(output_dir)
    result = result or {}

    description = (
        str(result.get("description", ""))
        or str(result.get("fall_type", ""))
    )

    csvs = find_main_imu_csvs(output_dir)

    outputs = []
    for csv_path in csvs:
        try:
            outputs.append(
                label_imu_csv_inplace(
                    csv_path=csv_path,
                    output_dir=output_dir,
                    scenario_id=scenario_id,
                    description=description,
                )
            )
        except Exception as exc:
            outputs.append({
                "csv": str(csv_path),
                "status": "error",
                "error": str(exc),
            })

    if not outputs:
        outputs.append({
            "status": "no_main_imu_csv_found",
            "output_dir": str(output_dir),
        })

    return outputs

# ---------------------------------------------------------------------
# PATCH v2: robust fallback labels when validation.txt timing is missing
# ---------------------------------------------------------------------

_FALLBACK_PHASE_STEPS_BY_SCENARIO = {
    34: [("stand", 195), ("walk", 300), ("perturb", 60), ("react", 31), ("fall", 470)],
    43: [("stand", 52), ("step", 280), ("perturb", 48), ("react", 18), ("fall", 281)],
}


def _scenario_int(x):
    try:
        if x in ("", None):
            return None
        return int(x)
    except Exception:
        return None


def _fallback_event_times_from_scenario(scenario_id, native_hz):
    sid = _scenario_int(scenario_id)
    phases = _FALLBACK_PHASE_STEPS_BY_SCENARIO.get(sid)
    out = {
        "perturbation_start_time_s": math.nan,
        "reaction_start_time_s": math.nan,
        "scripted_fall_phase_start_time_s": math.nan,
        "source": "none",
    }

    if not phases:
        return out

    cursor = 0
    for name, steps in phases:
        start_t = cursor / float(native_hz)
        if name == "perturb":
            out["perturbation_start_time_s"] = start_t
        elif name == "react":
            out["reaction_start_time_s"] = start_t
        elif name == "fall":
            out["scripted_fall_phase_start_time_s"] = start_t
        cursor += int(steps)

    out["source"] = f"fallback_phase_steps_scenario_{sid}"
    return out


def already_labeled(csv_path: Path) -> bool:
    """
    A CSV is considered correctly labeled only if the important event
    times are present. This allows automatic repair of earlier broken labels.
    """
    try:
        df, _ = read_sim_csv_flexible(csv_path)
        needed = ["Activity_Label", "Label_4Class", "Fall_Onset_Time_s", "Perturbation_Start_Time_s"]
        if not all(c in df.columns for c in needed):
            return False

        onset = pd.to_numeric(df["Fall_Onset_Time_s"], errors="coerce")
        perturb = pd.to_numeric(df["Perturbation_Start_Time_s"], errors="coerce")

        return onset.notna().any() and perturb.notna().any()
    except Exception:
        return False


def add_labels_to_dataframe(
    df: pd.DataFrame,
    events: Dict[str, Any],
    scenario_id: Optional[int],
    description: str,
    output_hz: float,
    native_hz: float,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Corrected labeler:
    - uses validation.txt event timing when available;
    - falls back to scenario phase timings when validation timing is missing;
    - infers fall onset from pelvis height drop;
    - detects main impact from accel_raw / impact_magnitude;
    - separates normal, pre-fall, falling, impact, and post-impact.
    """
    df = df.copy()

    time_s, time_col = build_time_vector(df, output_hz)
    signal_cols = add_signal_magnitudes(df)

    fallback = _fallback_event_times_from_scenario(scenario_id, native_hz)

    perturb_t = parse_float(events.get("perturbation_start_time_s"))
    if not math.isfinite(perturb_t):
        perturb_t = parse_float(fallback.get("perturbation_start_time_s"))

    reaction_t = parse_float(events.get("reaction_start_time_s"))
    if not math.isfinite(reaction_t):
        reaction_t = parse_float(fallback.get("reaction_start_time_s"))

    scripted_fall_t = parse_float(events.get("scripted_fall_phase_start_time_s"))
    if not math.isfinite(scripted_fall_t):
        scripted_fall_t = parse_float(fallback.get("scripted_fall_phase_start_time_s"))

    fall_onset_t = parse_float(events.get("fall_onset_time_s"))
    impact_t = parse_float(events.get("main_impact_time_s"))
    settle_t = parse_float(events.get("settle_time_s"))

    fall_onset_idx = None
    fall_onset_source = "validation_txt"

    if not math.isfinite(fall_onset_t):
        fall_onset_t, fall_onset_idx, fall_onset_source = infer_fall_onset_from_height(
            df, time_s, perturb_t
        )
    else:
        fall_onset_idx = int(np.argmin(np.abs(time_s - fall_onset_t)))

    impact_idx = None
    impact_source = "validation_txt"

    if not math.isfinite(impact_t):
        impact_t, impact_idx, impact_source = infer_main_impact(df, time_s, fall_onset_t)
    else:
        impact_idx = int(np.argmin(np.abs(time_s - impact_t)))

    df["Time_s_standard"] = time_s
    df["Frame_100Hz"] = np.rint(time_s * output_hz).astype(int)
    df["Frame_native_30Hz"] = np.rint(time_s * native_hz).astype(int)

    df["Scenario_ID"] = "" if scenario_id is None else scenario_id
    df["Scenario_Description"] = description
    df["Sampling_Hz"] = output_hz
    df["Native_Simulation_Hz"] = native_hz

    df["Perturbation_Start_Time_s"] = perturb_t
    df["Reaction_Start_Time_s"] = reaction_t
    df["Scripted_Fall_Phase_Start_Time_s"] = scripted_fall_t
    df["Fall_Onset_Time_s"] = fall_onset_t
    df["Main_Impact_Time_s"] = impact_t
    df["Settle_Time_s"] = settle_t

    df["Perturbation_Start_Frame_100Hz"] = int(round(perturb_t * output_hz)) if math.isfinite(perturb_t) else ""
    df["Reaction_Start_Frame_100Hz"] = int(round(reaction_t * output_hz)) if math.isfinite(reaction_t) else ""
    df["Scripted_Fall_Phase_Start_Frame_100Hz"] = int(round(scripted_fall_t * output_hz)) if math.isfinite(scripted_fall_t) else ""
    df["Fall_Onset_Frame_100Hz"] = int(round(fall_onset_t * output_hz)) if math.isfinite(fall_onset_t) else ""
    df["Main_Impact_Frame_100Hz"] = int(round(impact_t * output_hz)) if math.isfinite(impact_t) else ""
    df["Settle_Frame_100Hz"] = int(round(settle_t * output_hz)) if math.isfinite(settle_t) else ""

    df["Fall_Onset_Source"] = fall_onset_source
    df["Main_Impact_Source"] = impact_source

    activity = np.array(["normal"] * len(df), dtype=object)

    if math.isfinite(perturb_t):
        activity[time_s >= perturb_t] = "pre_fall_perturbation"

    if math.isfinite(reaction_t):
        activity[time_s >= reaction_t] = "reaction"

    if math.isfinite(fall_onset_t):
        activity[time_s >= fall_onset_t] = "falling_pre_impact"

    if math.isfinite(impact_t):
        activity[time_s > impact_t] = "post_impact"
        if impact_idx is not None:
            activity[impact_idx] = "main_impact"

    if math.isfinite(settle_t):
        activity[time_s >= settle_t] = "settled_post_fall"

    df["Activity_Label"] = activity

    df["Normal_Label"] = (activity == "normal").astype(int)
    df["PreFall_Label"] = np.isin(activity, ["pre_fall_perturbation", "reaction"]).astype(int)
    df["Reaction_Label"] = (activity == "reaction").astype(int)
    df["Fall_Label"] = np.isin(
        activity,
        ["falling_pre_impact", "main_impact", "post_impact", "settled_post_fall"],
    ).astype(int)
    df["PreImpact_Fall_Label"] = (activity == "falling_pre_impact").astype(int)
    df["Impact_Label"] = (activity == "main_impact").astype(int)
    df["PostImpact_Label"] = np.isin(activity, ["post_impact", "settled_post_fall"]).astype(int)

    if "fall_detected" in df.columns:
        df["Existing_FallDetected_Label"] = pd.to_numeric(
            df["fall_detected"], errors="coerce"
        ).fillna(0).astype(int)
    else:
        df["Existing_FallDetected_Label"] = 0

    label3 = np.zeros(len(df), dtype=int)
    if math.isfinite(perturb_t):
        label3[time_s >= perturb_t] = 1
    if math.isfinite(fall_onset_t):
        label3[time_s >= fall_onset_t] = 2
    df["Label_3Class"] = label3

    label4 = np.zeros(len(df), dtype=int)
    if math.isfinite(perturb_t):
        label4[time_s >= perturb_t] = 1
    if math.isfinite(fall_onset_t):
        label4[time_s >= fall_onset_t] = 2
    if math.isfinite(impact_t):
        label4[time_s >= impact_t] = 3
    if impact_idx is not None:
        label4[impact_idx] = 3
    df["Label_4Class"] = label4

    meta = {
        "time_column_detected": time_col,
        "signal_columns": signal_cols,
        "events": {
            "perturbation_start_time_s": perturb_t,
            "reaction_start_time_s": reaction_t,
            "scripted_fall_phase_start_time_s": scripted_fall_t,
            "fall_onset_time_s": fall_onset_t,
            "fall_onset_index_row": fall_onset_idx,
            "fall_onset_source": fall_onset_source,
            "main_impact_time_s": impact_t,
            "main_impact_index_row": impact_idx,
            "main_impact_source": impact_source,
            "settle_time_s": settle_t,
            "fallback_source": fallback.get("source"),
        },
        "label_columns": [
            "Activity_Label",
            "Normal_Label",
            "PreFall_Label",
            "Reaction_Label",
            "Fall_Label",
            "PreImpact_Fall_Label",
            "Impact_Label",
            "PostImpact_Label",
            "Existing_FallDetected_Label",
            "Label_3Class",
            "Label_4Class",
        ],
    }

    return df, meta

# ---------------------------------------------------------------------
# PATCH v3: generic fallback for tasks without COMMON FALL EVENT SUMMARY
# ---------------------------------------------------------------------

def _signal_raw_magnitude(df):
    if "Accel_Raw_Magnitude_mps2" in df.columns:
        return pd.to_numeric(df["Accel_Raw_Magnitude_mps2"], errors="coerce").to_numpy(float)

    cols = {clean_col(c): c for c in df.columns}
    xs, ys, zs = cols.get("accel_raw_x"), cols.get("accel_raw_y"), cols.get("accel_raw_z")
    if xs and ys and zs:
        x = pd.to_numeric(df[xs], errors="coerce").to_numpy(float)
        y = pd.to_numeric(df[ys], errors="coerce").to_numpy(float)
        z = pd.to_numeric(df[zs], errors="coerce").to_numpy(float)
        return np.sqrt(x*x + y*y + z*z)

    if "impact_magnitude" in df.columns:
        return pd.to_numeric(df["impact_magnitude"], errors="coerce").to_numpy(float)

    return None


def _infer_fall_onset_generic(df, time_s):
    """
    Generic fallback when validation.txt does not provide event times.
    Priority:
    1) first fall_detected frame
    2) pelvis height-drop event
    3) raw acceleration threshold event
    """
    if "fall_detected" in df.columns:
        fd = pd.to_numeric(df["fall_detected"], errors="coerce").fillna(0).to_numpy(int)
        idxs = np.where(fd == 1)[0]
        if len(idxs) > 0:
            i = int(idxs[0])
            return float(time_s[i]), i, "first_existing_fall_detected_frame"

    if "pelvis_height" in df.columns:
        h = pd.to_numeric(df["pelvis_height"], errors="coerce").to_numpy(float)
        early = (time_s >= 0.2) & (time_s <= min(3.0, np.nanmax(time_s)))
        search = time_s >= 1.0

        if np.any(early):
            h_ref = float(np.nanmedian(h[early]))
            threshold = h_ref - max(0.18, 0.18 * max(h_ref, 1.0))
            idxs = np.where(search & np.isfinite(h) & (h <= threshold))[0]
            if len(idxs) > 0:
                i = int(idxs[0])
                return float(time_s[i]), i, f"generic_pelvis_height_drop_h_ref={h_ref:.3f}_threshold={threshold:.3f}"

    raw = _signal_raw_magnitude(df)
    if raw is not None:
        base_mask = (time_s >= 0.2) & (time_s <= min(2.0, np.nanmax(time_s)))
        if np.any(base_mask):
            base = raw[base_mask]
            med = float(np.nanmedian(base))
            mad = float(np.nanmedian(np.abs(base - med)))
            threshold = max(18.0, med + 6.0 * max(mad, 0.5))
        else:
            threshold = 18.0

        idxs = np.where((time_s >= 1.0) & np.isfinite(raw) & (raw >= threshold))[0]
        if len(idxs) > 0:
            i = int(idxs[0])
            return float(time_s[i]), i, f"generic_raw_acc_threshold_{threshold:.2f}_mps2"

    return math.nan, None, "generic_fall_onset_not_found"


def _infer_perturbation_generic(df, time_s, fall_onset_t):
    """
    Generic pre-fall start estimate.
    If possible, use raw acceleration increase before fall onset.
    Otherwise use 0.30 s before fall onset.
    """
    if not math.isfinite(fall_onset_t):
        return math.nan, "no_fall_onset_for_perturbation"

    raw = _signal_raw_magnitude(df)

    if raw is not None:
        base_mask = (time_s >= 0.2) & (time_s <= min(2.0, np.nanmax(time_s)))
        win = (time_s >= max(0.0, fall_onset_t - 2.0)) & (time_s < fall_onset_t)

        if np.any(base_mask) and np.any(win):
            base = raw[base_mask]
            med = float(np.nanmedian(base))
            mad = float(np.nanmedian(np.abs(base - med)))
            threshold = max(14.0, med + 4.0 * max(mad, 0.5))

            idxs = np.where(win & np.isfinite(raw) & (raw >= threshold))[0]
            if len(idxs) > 0:
                i = int(idxs[0])
                return float(time_s[i]), f"generic_prefall_raw_acc_threshold_{threshold:.2f}_mps2"

    return max(0.0, fall_onset_t - 0.30), "generic_fall_onset_minus_0p30s"


def add_labels_to_dataframe(
    df: pd.DataFrame,
    events: Dict[str, Any],
    scenario_id: Optional[int],
    description: str,
    output_hz: float,
    native_hz: float,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Final robust labeler:
    - uses validation.txt if available;
    - falls back to scenario phase timing if known;
    - otherwise infers labels from fall_detected / height / accel_raw.
    """
    df = df.copy()

    time_s, time_col = build_time_vector(df, output_hz)
    signal_cols = add_signal_magnitudes(df)

    fallback = _fallback_event_times_from_scenario(scenario_id, native_hz)

    perturb_t = parse_float(events.get("perturbation_start_time_s"))
    if not math.isfinite(perturb_t):
        perturb_t = parse_float(fallback.get("perturbation_start_time_s"))

    reaction_t = parse_float(events.get("reaction_start_time_s"))
    if not math.isfinite(reaction_t):
        reaction_t = parse_float(fallback.get("reaction_start_time_s"))

    scripted_fall_t = parse_float(events.get("scripted_fall_phase_start_time_s"))
    if not math.isfinite(scripted_fall_t):
        scripted_fall_t = parse_float(fallback.get("scripted_fall_phase_start_time_s"))

    fall_onset_t = parse_float(events.get("fall_onset_time_s"))
    impact_t = parse_float(events.get("main_impact_time_s"))
    settle_t = parse_float(events.get("settle_time_s"))

    fall_onset_idx = None
    fall_onset_source = "validation_txt"

    if not math.isfinite(fall_onset_t):
        if math.isfinite(perturb_t):
            fall_onset_t, fall_onset_idx, fall_onset_source = infer_fall_onset_from_height(
                df, time_s, perturb_t
            )

        if not math.isfinite(fall_onset_t):
            fall_onset_t, fall_onset_idx, fall_onset_source = _infer_fall_onset_generic(df, time_s)
    else:
        fall_onset_idx = int(np.argmin(np.abs(time_s - fall_onset_t)))

    perturb_source = "validation_or_phase_fallback"
    if not math.isfinite(perturb_t):
        perturb_t, perturb_source = _infer_perturbation_generic(df, time_s, fall_onset_t)

    impact_idx = None
    impact_source = "validation_txt"

    if not math.isfinite(impact_t):
        impact_t, impact_idx, impact_source = infer_main_impact(df, time_s, fall_onset_t)
    else:
        impact_idx = int(np.argmin(np.abs(time_s - impact_t)))

    df["Time_s_standard"] = time_s
    df["Frame_100Hz"] = np.rint(time_s * output_hz).astype(int)
    df["Frame_native_30Hz"] = np.rint(time_s * native_hz).astype(int)

    df["Scenario_ID"] = "" if scenario_id is None else scenario_id
    df["Scenario_Description"] = description
    df["Sampling_Hz"] = output_hz
    df["Native_Simulation_Hz"] = native_hz

    df["Perturbation_Start_Time_s"] = perturb_t
    df["Reaction_Start_Time_s"] = reaction_t
    df["Scripted_Fall_Phase_Start_Time_s"] = scripted_fall_t
    df["Fall_Onset_Time_s"] = fall_onset_t
    df["Main_Impact_Time_s"] = impact_t
    df["Settle_Time_s"] = settle_t

    df["Perturbation_Start_Frame_100Hz"] = int(round(perturb_t * output_hz)) if math.isfinite(perturb_t) else ""
    df["Reaction_Start_Frame_100Hz"] = int(round(reaction_t * output_hz)) if math.isfinite(reaction_t) else ""
    df["Scripted_Fall_Phase_Start_Frame_100Hz"] = int(round(scripted_fall_t * output_hz)) if math.isfinite(scripted_fall_t) else ""
    df["Fall_Onset_Frame_100Hz"] = int(round(fall_onset_t * output_hz)) if math.isfinite(fall_onset_t) else ""
    df["Main_Impact_Frame_100Hz"] = int(round(impact_t * output_hz)) if math.isfinite(impact_t) else ""
    df["Settle_Frame_100Hz"] = int(round(settle_t * output_hz)) if math.isfinite(settle_t) else ""

    df["Perturbation_Start_Source"] = perturb_source
    df["Fall_Onset_Source"] = fall_onset_source
    df["Main_Impact_Source"] = impact_source

    activity = np.array(["normal"] * len(df), dtype=object)

    if math.isfinite(perturb_t):
        activity[time_s >= perturb_t] = "pre_fall_perturbation"

    if math.isfinite(reaction_t):
        activity[time_s >= reaction_t] = "reaction"

    if math.isfinite(fall_onset_t):
        activity[time_s >= fall_onset_t] = "falling_pre_impact"

    if math.isfinite(impact_t):
        activity[time_s > impact_t] = "post_impact"
        if impact_idx is not None:
            activity[impact_idx] = "main_impact"

    if math.isfinite(settle_t):
        activity[time_s >= settle_t] = "settled_post_fall"

    df["Activity_Label"] = activity

    df["Normal_Label"] = (activity == "normal").astype(int)
    df["PreFall_Label"] = np.isin(activity, ["pre_fall_perturbation", "reaction"]).astype(int)
    df["Reaction_Label"] = (activity == "reaction").astype(int)
    df["Fall_Label"] = np.isin(
        activity,
        ["falling_pre_impact", "main_impact", "post_impact", "settled_post_fall"],
    ).astype(int)
    df["PreImpact_Fall_Label"] = (activity == "falling_pre_impact").astype(int)
    df["Impact_Label"] = (activity == "main_impact").astype(int)
    df["PostImpact_Label"] = np.isin(activity, ["post_impact", "settled_post_fall"]).astype(int)

    if "fall_detected" in df.columns:
        df["Existing_FallDetected_Label"] = pd.to_numeric(
            df["fall_detected"], errors="coerce"
        ).fillna(0).astype(int)
    else:
        df["Existing_FallDetected_Label"] = 0

    label3 = np.zeros(len(df), dtype=int)
    if math.isfinite(perturb_t):
        label3[time_s >= perturb_t] = 1
    if math.isfinite(fall_onset_t):
        label3[time_s >= fall_onset_t] = 2
    df["Label_3Class"] = label3

    label4 = np.zeros(len(df), dtype=int)
    if math.isfinite(perturb_t):
        label4[time_s >= perturb_t] = 1
    if math.isfinite(fall_onset_t):
        label4[time_s >= fall_onset_t] = 2
    if math.isfinite(impact_t):
        label4[time_s >= impact_t] = 3
    if impact_idx is not None:
        label4[impact_idx] = 3
    df["Label_4Class"] = label4

    meta = {
        "time_column_detected": time_col,
        "signal_columns": signal_cols,
        "events": {
            "perturbation_start_time_s": perturb_t,
            "perturbation_start_source": perturb_source,
            "reaction_start_time_s": reaction_t,
            "scripted_fall_phase_start_time_s": scripted_fall_t,
            "fall_onset_time_s": fall_onset_t,
            "fall_onset_index_row": fall_onset_idx,
            "fall_onset_source": fall_onset_source,
            "main_impact_time_s": impact_t,
            "main_impact_index_row": impact_idx,
            "main_impact_source": impact_source,
            "settle_time_s": settle_t,
            "fallback_source": fallback.get("source"),
        },
        "label_columns": [
            "Activity_Label",
            "Normal_Label",
            "PreFall_Label",
            "Reaction_Label",
            "Fall_Label",
            "PreImpact_Fall_Label",
            "Impact_Label",
            "PostImpact_Label",
            "Existing_FallDetected_Label",
            "Label_3Class",
            "Label_4Class",
        ],
    }

    return df, meta

# ---------------------------------------------------------------------
# PATCH v4: force COMMON FALL EVENT SUMMARY from validation TXT
# ---------------------------------------------------------------------

def parse_validation_events(output_dir: Path) -> Dict[str, Any]:
    """
    Robustly reads scenario validation report and extracts COMMON FALL EVENT SUMMARY.
    Searches inside the output folder recursively and also checks likely nearby files.
    """
    output_dir = Path(output_dir)

    candidates = []
    candidates.extend(sorted(output_dir.glob("*_validation.txt")))
    candidates.extend(sorted(output_dir.rglob("*_validation.txt")))

    # Sometimes legacy scripts save relative to cwd first, so also search project outputs.
    try:
        candidates.extend(sorted(Path("outputs").rglob("*_validation.txt")))
    except Exception:
        pass

    # Deduplicate while preserving files.
    seen = set()
    unique = []
    for p in candidates:
        try:
            rp = str(p.resolve())
        except Exception:
            rp = str(p)
        if rp not in seen:
            unique.append(p)
            seen.add(rp)

    events = {
        "perturbation_start_time_s": math.nan,
        "fall_onset_time_s": math.nan,
        "main_impact_time_s": math.nan,
        "settle_time_s": math.nan,
        "source": "not_found",
    }

    if not unique:
        return events

    # Prefer validation files inside the selected output directory.
    def score_file(p):
        s = 0
        text_name = str(p)
        if str(output_dir) in text_name:
            s += 100
        try:
            s += int(p.stat().st_mtime)
        except Exception:
            pass
        return s

    unique = sorted(unique, key=score_file, reverse=True)

    patterns = {
        "perturbation_start_time_s": r"Perturb\s+start\s*:\s*([0-9.]+)\s*s",
        "fall_onset_time_s": r"Fall\s+onset\s*:\s*([0-9.]+)\s*s",
        "main_impact_time_s": r"Main\s+impact\s*:\s*([0-9.]+)\s*s",
        "settle_time_s": r"Settle\s+time\s*:\s*([0-9.]+)\s*s",
    }

    for txt_path in unique:
        try:
            text = txt_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        if "COMMON FALL EVENT SUMMARY" not in text:
            continue

        found_any = False
        tmp = dict(events)

        for key, pat in patterns.items():
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                tmp[key] = float(m.group(1))
                found_any = True

        if found_any:
            tmp["source"] = str(txt_path)
            return tmp

    return events

# ---------------------------------------------------------------------
# PATCH v5: accept event_overrides from captured simulator run log
# ---------------------------------------------------------------------

def _merge_event_overrides(events: Dict[str, Any], event_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    out = dict(events or {})
    if not event_overrides:
        return out

    keymap = {
        "perturbation_start_time_s": "perturbation_start_time_s",
        "perturb_start_time_s": "perturbation_start_time_s",
        "fall_onset_time_s": "fall_onset_time_s",
        "main_impact_time_s": "main_impact_time_s",
        "impact_time_s": "main_impact_time_s",
        "settle_time_s": "settle_time_s",
    }

    for src, dst in keymap.items():
        val = event_overrides.get(src)
        try:
            f = float(val)
            if math.isfinite(f):
                out[dst] = f
        except Exception:
            pass

    if event_overrides:
        out["source"] = event_overrides.get("source", "run_log_event_override")

    return out


def label_imu_csv_inplace(
    csv_path: Path,
    output_dir: Path,
    scenario_id: Optional[int] = None,
    description: str = "",
    output_hz: float = DEFAULT_OUTPUT_HZ,
    native_hz: float = DEFAULT_NATIVE_HZ,
    event_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    In-place labeler.
    If event_overrides are provided from captured run log, they are treated as
    source-of-truth and the CSV is relabeled even if it already contains labels.
    """
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)

    if not event_overrides and already_labeled(csv_path):
        return {"csv": str(csv_path), "status": "already_labeled"}

    df, comments = read_sim_csv_flexible(csv_path)

    if not description:
        description = comments.get("fall_type", "")

    output_hz_from_csv = parse_float(comments.get("sampling_rate_hz")) if comments.get("sampling_rate_hz") else output_hz
    native_hz_from_csv = parse_float(comments.get("native_sampling_rate_hz")) if comments.get("native_sampling_rate_hz") else native_hz

    if math.isfinite(output_hz_from_csv) and output_hz_from_csv > 0:
        output_hz = output_hz_from_csv

    if math.isfinite(native_hz_from_csv) and native_hz_from_csv > 0:
        native_hz = native_hz_from_csv

    events = parse_validation_events(output_dir)
    events = _merge_event_overrides(events, event_overrides)

    labeled_df, meta = add_labels_to_dataframe(
        df=df,
        events=events,
        scenario_id=scenario_id,
        description=description,
        output_hz=output_hz,
        native_hz=native_hz,
    )

    backup_csv = csv_path.with_name(csv_path.stem + ".unlabeled_backup.csv")
    if not backup_csv.exists():
        shutil.copy2(csv_path, backup_csv)

    labeled_df.to_csv(csv_path, index=False)

    meta_path = csv_path.with_name(csv_path.stem + ".label_metadata.json")
    payload = {
        "input_csv_overwritten_with_labels": str(csv_path),
        "backup_unlabeled_csv": str(backup_csv),
        "metadata_comments_from_original_csv": comments,
        "validation_event_source": events.get("source"),
        "event_overrides_used": event_overrides or {},
        "scenario_id": scenario_id,
        "description": description,
        **meta,
    }
    meta_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    return {
        "csv": str(csv_path),
        "backup_csv": str(backup_csv),
        "metadata_json": str(meta_path),
        "status": "labeled_inplace",
        "events": meta["events"],
    }


def label_output_folder_inplace(
    output_dir: str | Path,
    scenario_id: Optional[int] = None,
    subject_params: Optional[Dict[str, Any]] = None,
    result: Optional[Dict[str, Any]] = None,
    event_overrides: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Labels the main IMU CSV in an output folder.
    event_overrides should come from the captured simulator COMMON FALL EVENT SUMMARY.
    """
    output_dir = Path(output_dir)
    result = result or {}

    description = (
        str(result.get("description", ""))
        or str(result.get("fall_type", ""))
    )

    csvs = find_main_imu_csvs(output_dir)

    outputs = []
    for csv_path in csvs:
        try:
            outputs.append(
                label_imu_csv_inplace(
                    csv_path=csv_path,
                    output_dir=output_dir,
                    scenario_id=scenario_id,
                    description=description,
                    event_overrides=event_overrides,
                )
            )
        except Exception as exc:
            outputs.append({
                "csv": str(csv_path),
                "status": "error",
                "error": str(exc),
            })

    if not outputs:
        outputs.append({
            "status": "no_main_imu_csv_found",
            "output_dir": str(output_dir),
        })

    return outputs
