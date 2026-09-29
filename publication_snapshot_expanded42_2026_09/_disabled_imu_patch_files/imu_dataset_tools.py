# -*- coding: utf-8 -*-
from __future__ import annotations

import json, math, re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import pandas as pd
except Exception as exc:
    pd = None
    _PANDAS_IMPORT_ERROR = exc
else:
    _PANDAS_IMPORT_ERROR = None

G0 = 9.80665
DEFAULT_FS_HZ = 100.0
AXIS_CONVENTION_ID = "MUJOCO_WORLD_Z_UP_TORSO_L1L2_PROXY_V1"


def _as_float(x: Any, default: float = math.nan) -> float:
    try:
        if x is None:
            return default
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _as_int(x: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if x is None:
            return default
        return int(round(float(x)))
    except Exception:
        return default


def _clean_col(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower()).strip("_")


def _find_time_column(columns: Iterable[str]) -> Optional[str]:
    cols = list(columns)
    priority = [
        "time_s", "time", "timestamp", "timestamp_s", "t",
        "TimeStamp(s)", "TimeStamp", "Time", "sim_time",
        "simulation_time", "seconds",
    ]
    cleaned = {_clean_col(c): c for c in cols}
    for p in priority:
        key = _clean_col(p)
        if key in cleaned:
            return cleaned[key]
    for c in cols:
        cc = _clean_col(c)
        if "time" in cc or "timestamp" in cc:
            return c
    return None


def _find_axis_columns(columns: Iterable[str], sensor: str) -> Dict[str, Optional[str]]:
    cols = list(columns)
    out = {"x": None, "y": None, "z": None}

    if sensor == "acc":
        sensor_words = r"(acc|accel|accelerometer|linear_acc|linear_acceleration|a)"
        single_letter = "a"
    else:
        sensor_words = r"(gyro|gyr|gyroscope|angular_velocity|omega|w)"
        single_letter = "g"

    for axis in ("x", "y", "z"):
        regexes = [
            rf"^{sensor_words}_{axis}$",
            rf"^{sensor_words}{axis}$",
            rf"^{axis}_{sensor_words}$",
            rf"^{single_letter}{axis}$",
            rf"^{single_letter}_{axis}$",
        ]
        for c in cols:
            cc = _clean_col(c)
            if any(re.match(rx, cc) for rx in regexes):
                out[axis] = c
                break
    return out


def _has_all_axes(axis_cols: Dict[str, Optional[str]]) -> bool:
    return all(axis_cols.get(a) is not None for a in ("x", "y", "z"))


def _numeric_series(df, col: str):
    return pd.to_numeric(df[col], errors="coerce")


def _infer_acc_unit(df, acc_cols: Dict[str, Optional[str]]) -> Tuple[str, float]:
    if not _has_all_axes(acc_cols):
        return "unknown", 1.0
    vals = np.vstack([
        _numeric_series(df, acc_cols[a]).to_numpy(dtype=float)
        for a in ("x", "y", "z")
    ]).T
    vals = vals[np.all(np.isfinite(vals), axis=1)]
    if len(vals) == 0:
        return "unknown", 1.0
    med_norm = float(np.nanmedian(np.linalg.norm(vals[: min(300, len(vals))], axis=1)))
    if 5.0 <= med_norm <= 20.0:
        return "m/s^2", 1.0
    if 0.4 <= med_norm <= 2.5:
        return "g", G0
    return "unknown", 1.0


def _infer_gyro_unit(df, gyro_cols: Dict[str, Optional[str]]) -> Tuple[str, float]:
    if not _has_all_axes(gyro_cols):
        return "unknown", 1.0
    vals = np.vstack([
        _numeric_series(df, gyro_cols[a]).to_numpy(dtype=float)
        for a in ("x", "y", "z")
    ]).T
    vals = vals[np.all(np.isfinite(vals), axis=1)]
    if len(vals) == 0:
        return "unknown", 1.0
    p95 = float(np.nanpercentile(np.abs(vals), 95))
    if p95 > 20.0:
        return "deg/s", math.pi / 180.0
    return "rad/s", 1.0


def _walk_nested(obj: Any, prefix: str = ""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            yield key, v
            yield from _walk_nested(v, key)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            key = f"{prefix}[{i}]"
            yield key, v
            yield from _walk_nested(v, key)


def _find_event_value(event_summary: Any, include_terms: Tuple[str, ...], value_terms: Tuple[str, ...]):
    for key, val in _walk_nested(event_summary):
        k = key.lower()
        if any(t in k for t in include_terms) and any(vt in k for vt in value_terms):
            f = _as_float(val)
            if math.isfinite(f):
                return f, key
    return math.nan, ""


def _extract_event_times(event_summary: Any, explicit: Optional[Dict[str, Any]] = None):
    explicit = explicit or {}

    first_impact_time, impact_time_key = _find_event_value(
        event_summary,
        include_terms=("first_impact", "impact", "floor_contact", "contact", "hit"),
        value_terms=("time", "timestamp", "time_s", "t_s"),
    )
    first_impact_frame, impact_frame_key = _find_event_value(
        event_summary,
        include_terms=("first_impact", "impact", "floor_contact", "contact", "hit"),
        value_terms=("frame", "index", "idx", "step"),
    )

    fall_time, fall_time_key = _find_event_value(
        event_summary,
        include_terms=("fall_start", "fall_onset", "onset"),
        value_terms=("time", "timestamp", "time_s", "t_s"),
    )
    fall_frame, fall_frame_key = _find_event_value(
        event_summary,
        include_terms=("fall_start", "fall_onset", "onset"),
        value_terms=("frame", "index", "idx", "step"),
    )

    for k in ("fall_start_time_s", "fall_start_time", "programmed_fall_start_time_s"):
        if math.isfinite(_as_float(explicit.get(k))):
            fall_time = _as_float(explicit.get(k))
            fall_time_key = f"explicit.{k}"
            break

    for k in ("fall_start_frame", "fall_start_index", "programmed_fall_start_frame"):
        if explicit.get(k) is not None:
            fall_frame = _as_float(explicit.get(k))
            fall_frame_key = f"explicit.{k}"
            break

    return {
        "fall_start_time_s": fall_time,
        "fall_start_frame": _as_int(fall_frame),
        "fall_start_source": fall_time_key or fall_frame_key,
        "first_impact_time_s": first_impact_time,
        "first_impact_frame": _as_int(first_impact_frame),
        "first_impact_source": impact_time_key or impact_frame_key,
    }


def _phase_for_times(time_s: np.ndarray, phase_log: Optional[List[Dict[str, Any]]]):
    if not phase_log:
        return ["unknown"] * len(time_s)

    rows = []
    for r in phase_log:
        t = _as_float(r.get("time_s", r.get("time", r.get("t"))))
        ph = str(r.get("phase", "unknown"))
        if math.isfinite(t):
            rows.append((t, ph))

    if not rows:
        return ["unknown"] * len(time_s)

    rows.sort(key=lambda x: x[0])
    pts = np.array([r[0] for r in rows], dtype=float)
    phases = [r[1] for r in rows]
    idx = np.searchsorted(pts, time_s, side="right") - 1
    idx = np.clip(idx, 0, len(phases) - 1)
    return [phases[int(i)] for i in idx]


def _derive_fallback_impact_time(df, time_s: np.ndarray, acc_cols_std: Dict[str, str], fall_start_time_s: float):
    if not all(k in df.columns for k in acc_cols_std.values()):
        return math.nan, ""

    acc = np.vstack([
        pd.to_numeric(df[acc_cols_std[a]], errors="coerce").to_numpy(float)
        for a in ("x", "y", "z")
    ]).T
    mag = np.linalg.norm(acc, axis=1)

    start_mask = np.ones(len(df), dtype=bool)
    if math.isfinite(fall_start_time_s):
        start_mask = time_s >= fall_start_time_s

    candidate_idx = np.where(start_mask & np.isfinite(mag) & (mag >= 25.0))[0]
    if len(candidate_idx) > 0:
        return float(time_s[int(candidate_idx[0])]), "fallback.acc_magnitude_ge_25mps2"

    post_idx = np.where(start_mask & np.isfinite(mag))[0]
    if len(post_idx) > 0:
        j = int(post_idx[np.nanargmax(mag[post_idx])])
        return float(time_s[j]), "fallback.max_acc_magnitude_post_fall"

    return math.nan, ""


def export_enriched_imu_dataset(
    original_csv: str | Path,
    output_csv: str | Path | None = None,
    metadata: Optional[Dict[str, Any]] = None,
    event_summary: Any = None,
    phase_log: Optional[List[Dict[str, Any]]] = None,
    sampling_hz: float = DEFAULT_FS_HZ,
    native_dt: Optional[float] = None,
):
    if pd is None:
        raise RuntimeError(f"pandas is required, but import failed: {_PANDAS_IMPORT_ERROR}")

    original_csv = Path(original_csv)
    if output_csv is None:
        output_csv = original_csv.with_name(original_csv.stem + "_enriched.csv")
    output_csv = Path(output_csv)
    metadata = dict(metadata or {})

    df = pd.read_csv(original_csv)
    if df.empty:
        raise ValueError(f"Cannot enrich empty CSV: {original_csv}")

    fs = float(metadata.get("sampling_hz", sampling_hz) or DEFAULT_FS_HZ)
    dt = _as_float(native_dt, default=math.nan)

    time_col = _find_time_column(df.columns)
    if time_col is not None:
        time_s = pd.to_numeric(df[time_col], errors="coerce").to_numpy(float)
        if not np.all(np.isfinite(time_s)):
            time_s = np.arange(len(df), dtype=float) / fs
    else:
        time_s = np.arange(len(df), dtype=float) / fs

    if math.isfinite(dt) and dt > 0:
        frame_index = np.rint(time_s / dt).astype(int)
    else:
        frame_index = np.arange(len(df), dtype=int)

    acc_cols = _find_axis_columns(df.columns, "acc")
    gyro_cols = _find_axis_columns(df.columns, "gyro")

    acc_unit, acc_factor = _infer_acc_unit(df, acc_cols)
    gyro_unit, gyro_factor = _infer_gyro_unit(df, gyro_cols)

    std_acc_cols = {}
    if _has_all_axes(acc_cols):
        for axis, out_name in zip(("x", "y", "z"), ("AccX_mps2", "AccY_mps2", "AccZ_mps2")):
            df[out_name] = pd.to_numeric(df[acc_cols[axis]], errors="coerce") * acc_factor
            std_acc_cols[axis] = out_name
        df["Acc_Magnitude_mps2"] = np.sqrt(
            df["AccX_mps2"] ** 2 + df["AccY_mps2"] ** 2 + df["AccZ_mps2"] ** 2
        )

    if _has_all_axes(gyro_cols):
        for axis, out_name in zip(("x", "y", "z"), ("GyrX_rads", "GyrY_rads", "GyrZ_rads")):
            df[out_name] = pd.to_numeric(df[gyro_cols[axis]], errors="coerce") * gyro_factor
        df["Gyr_Magnitude_rads"] = np.sqrt(
            df["GyrX_rads"] ** 2 + df["GyrY_rads"] ** 2 + df["GyrZ_rads"] ** 2
        )

    explicit_events = {
        "fall_start_index": metadata.get("fall_start_index"),
        "fall_start_frame": metadata.get("fall_start_frame"),
        "fall_start_time_s": metadata.get("fall_start_time_s"),
    }
    events = _extract_event_times(event_summary, explicit=explicit_events)

    if not math.isfinite(events["fall_start_time_s"]) and events["fall_start_frame"] is not None:
        events["fall_start_time_s"] = float(events["fall_start_frame"]) * (
            dt if math.isfinite(dt) and dt > 0 else 1.0 / fs
        )

    if not math.isfinite(events["first_impact_time_s"]) and events["first_impact_frame"] is not None:
        events["first_impact_time_s"] = float(events["first_impact_frame"]) * (
            dt if math.isfinite(dt) and dt > 0 else 1.0 / fs
        )

    if not math.isfinite(events["first_impact_time_s"]):
        fallback_t, fallback_src = _derive_fallback_impact_time(
            df, time_s, std_acc_cols, events["fall_start_time_s"]
        )
        if math.isfinite(fallback_t):
            events["first_impact_time_s"] = fallback_t
            events["first_impact_frame"] = int(round(fallback_t / (
                dt if math.isfinite(dt) and dt > 0 else 1.0 / fs
            )))
            events["first_impact_source"] = fallback_src

    perturb_start_time_s = _as_float(metadata.get("perturb_start_time_s"))
    react_start_time_s = _as_float(metadata.get("react_start_time_s"))
    fall_start_time_s = _as_float(events["fall_start_time_s"])
    impact_time_s = _as_float(events["first_impact_time_s"])

    df["Time_s_standard"] = time_s
    df["SimFrame"] = frame_index
    df["Scenario_ID"] = metadata.get("scenario_id", "")
    df["Scenario_Description"] = metadata.get("scenario_description", metadata.get("fall_type", ""))
    df["Subject_ID"] = metadata.get("subject_id", "")
    df["Subject_Age"] = metadata.get("age", "")
    df["Subject_Height_m"] = metadata.get("height", "")
    df["Subject_Sex"] = metadata.get("sex", "")
    df["Subject_BodyMass_kg"] = metadata.get("weight", metadata.get("body_mass_kg", ""))
    df["IMU_Location"] = metadata.get("imu_location", "lower_back_L1_L2_proxy")
    df["IMU_Proxy_Body"] = metadata.get("imu_proxy_body", "Torso")
    df["Sampling_Hz"] = fs
    df["Axis_Convention_ID"] = AXIS_CONVENTION_ID

    df["Input_Acceleration_Unit_Inferred"] = acc_unit
    df["Input_Gyroscope_Unit_Inferred"] = gyro_unit
    df["Standard_Acceleration_Unit"] = "m/s^2"
    df["Standard_Gyroscope_Unit"] = "rad/s"

    phases = _phase_for_times(time_s, phase_log)
    phase_map = {"unknown": -1, "stand": 0, "walk": 1, "perturb": 2, "react": 3, "fall": 4, "rest": 5}
    df["Phase"] = phases
    df["Phase_ID"] = [phase_map.get(str(p).lower(), -1) for p in phases]

    df["Perturbation_Start_Time_s"] = perturb_start_time_s
    df["Reaction_Start_Time_s"] = react_start_time_s
    df["Fall_Start_Time_s"] = fall_start_time_s
    df["Fall_Start_Frame"] = events["fall_start_frame"] if events["fall_start_frame"] is not None else ""
    df["First_Impact_Time_s"] = impact_time_s
    df["First_Impact_Frame"] = events["first_impact_frame"] if events["first_impact_frame"] is not None else ""
    df["First_Impact_Source"] = events["first_impact_source"]

    df["PrePerturbation_Label"] = 0
    df["Perturbation_Label"] = 0
    df["Reaction_Label"] = 0
    df["FallPhase_Label"] = 0
    df["Impact_Label"] = 0
    df["PostImpact_Label"] = 0

    if math.isfinite(perturb_start_time_s):
        df.loc[time_s < perturb_start_time_s, "PrePerturbation_Label"] = 1
        if math.isfinite(react_start_time_s):
            df.loc[(time_s >= perturb_start_time_s) & (time_s < react_start_time_s), "Perturbation_Label"] = 1
        elif math.isfinite(fall_start_time_s):
            df.loc[(time_s >= perturb_start_time_s) & (time_s < fall_start_time_s), "Perturbation_Label"] = 1

    if math.isfinite(react_start_time_s) and math.isfinite(fall_start_time_s):
        df.loc[(time_s >= react_start_time_s) & (time_s < fall_start_time_s), "Reaction_Label"] = 1

    if math.isfinite(fall_start_time_s):
        df.loc[time_s >= fall_start_time_s, "FallPhase_Label"] = 1

    if math.isfinite(impact_time_s):
        impact_idx = int(np.argmin(np.abs(time_s - impact_time_s)))
        df.loc[impact_idx, "Impact_Label"] = 1
        df.loc[time_s >= impact_time_s, "PostImpact_Label"] = 1

    activity = np.array(["normal_or_unknown"] * len(df), dtype=object)
    if math.isfinite(perturb_start_time_s):
        activity[time_s < perturb_start_time_s] = "normal_pre_perturbation"
        activity[time_s >= perturb_start_time_s] = "perturbation_or_fall_event"
    if math.isfinite(react_start_time_s):
        activity[time_s >= react_start_time_s] = "reaction_delay"
    if math.isfinite(fall_start_time_s):
        activity[time_s >= fall_start_time_s] = "fall_phase"
    if math.isfinite(impact_time_s):
        activity[time_s >= impact_time_s] = "post_impact"
        impact_idx = int(np.argmin(np.abs(time_s - impact_time_s)))
        activity[impact_idx] = "first_impact"

    df["Activity_Label"] = activity
    df["Fall_Event_Label_From_Perturbation"] = (
        (time_s >= perturb_start_time_s).astype(int) if math.isfinite(perturb_start_time_s) else 0
    )
    df["Fall_Event_Label_From_Collapse"] = (
        (time_s >= fall_start_time_s).astype(int) if math.isfinite(fall_start_time_s) else 0
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    sidecar = {
        "original_csv": str(original_csv),
        "enriched_csv": str(output_csv),
        "axis_convention_id": AXIS_CONVENTION_ID,
        "sampling_hz": fs,
        "native_dt": None if not math.isfinite(dt) else dt,
        "detected_columns": {"time": time_col, "acc": acc_cols, "gyro": gyro_cols},
        "unit_inference": {
            "input_acceleration_unit": acc_unit,
            "input_gyroscope_unit": gyro_unit,
            "standard_acceleration_unit": "m/s^2",
            "standard_gyroscope_unit": "rad/s",
        },
        "events": events,
        "metadata": metadata,
        "event_summary_original": event_summary,
    }
    meta_path = output_csv.with_suffix(".metadata.json")
    meta_path.write_text(json.dumps(sidecar, indent=2, default=str), encoding="utf-8")

    return {
        "filename": str(output_csv),
        "metadata_json": str(meta_path),
        "axis_convention_id": AXIS_CONVENTION_ID,
        "events": events,
        "detected_columns": sidecar["detected_columns"],
    }


def save_axis_convention_report(
    output_prefix: str | Path,
    mj_model: Any = None,
    mj_data: Any = None,
    imu: Any = None,
    metadata: Optional[Dict[str, Any]] = None,
    mount_info: Optional[Dict[str, Any]] = None,
):
    metadata = dict(metadata or {})
    mount_info = dict(mount_info or {})
    output_prefix = Path(output_prefix)
    path = output_prefix.with_name(output_prefix.name + "_axis_convention.json")

    sensor_body = (
        mount_info.get("sensor_body")
        or metadata.get("imu_proxy_body")
        or getattr(imu, "sensor_body", None)
        or getattr(imu, "sensor_body_name", None)
        or "Torso"
    )

    body_local_axes_in_world = None
    note = "Axis matrix could not be extracted; run verify_imu_axis.py on the generated CSV."

    try:
        import mujoco
        body_id = None
        if isinstance(sensor_body, str) and mj_model is not None:
            body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, sensor_body)
        elif isinstance(sensor_body, int):
            body_id = sensor_body

        if body_id is not None and body_id >= 0 and mj_data is not None:
            R = np.asarray(mj_data.xmat[body_id], dtype=float).reshape(3, 3)
            body_local_axes_in_world = {
                "+body_or_sensor_X_axis_in_world_XYZ": R[:, 0].round(8).tolist(),
                "+body_or_sensor_Y_axis_in_world_XYZ": R[:, 1].round(8).tolist(),
                "+body_or_sensor_Z_axis_in_world_XYZ": R[:, 2].round(8).tolist(),
            }
            note = (
                "MuJoCo body-local axis directions at report time. "
                "If IMU export is body-frame, AccX/GyrX follow +body X etc. "
                "If IMU export is world-frame, columns follow MuJoCo world XYZ. "
                "Run verify_imu_axis.py to confirm gravity axis and sign."
            )
    except Exception as exc:
        note = f"Axis matrix extraction failed: {exc}. Run verify_imu_axis.py on the generated CSV."

    report = {
        "axis_convention_id": AXIS_CONVENTION_ID,
        "imu_location": metadata.get("imu_location", "lower_back_L1_L2_proxy"),
        "imu_proxy_body": sensor_body,
        "mujoco_world_frame": {
            "+X": "MuJoCo world X; normally forward/backward depending initial avatar heading",
            "+Y": "MuJoCo world Y; normally lateral depending initial avatar heading",
            "+Z": "MuJoCo world vertical up",
        },
        "body_local_axes_in_world": body_local_axes_in_world,
        "units_standard": {
            "acceleration": "m/s^2",
            "gyroscope": "rad/s",
            "angles_if_present": "degrees unless explicitly named otherwise",
        },
        "recommended_dataset_standard": {
            "AccX": "confirm after calibration",
            "AccY": "confirm after calibration",
            "AccZ": "vertical axis should show approximately +9.81 or -9.81 m/s^2 during quiet standing",
            "GyrX_GyrY_GyrZ": "roll/pitch/yaw signs must be verified using controlled motions",
        },
        "note": note,
        "metadata": metadata,
        "mount_info": mount_info,
    }

    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return str(path)


def _csv_header(path: Path):
    try:
        import csv
        with path.open("r", newline="", encoding="utf-8", errors="ignore") as f:
            return next(csv.reader(f))
    except Exception:
        return []


def is_likely_imu_csv(path: str | Path) -> bool:
    path = Path(path)
    if "enriched" in path.stem.lower():
        return False
    cols = _csv_header(path)
    if not cols:
        return False
    acc = _find_axis_columns(cols, "acc")
    gyro = _find_axis_columns(cols, "gyro")
    return _has_all_axes(acc) and _has_all_axes(gyro)


def enrich_output_folder(
    output_dir: str | Path,
    scenario_id: int,
    subject_params: Optional[Dict[str, Any]] = None,
    result: Optional[Dict[str, Any]] = None,
):
    output_dir = Path(output_dir)
    subject_params = dict(subject_params or {})
    result = dict(result or {})
    event_summary = result.get("event_summary", result.get("events", {}))

    metadata = {
        "scenario_id": scenario_id,
        "scenario_description": result.get("description", result.get("fall_type", "")),
        "age": subject_params.get("age"),
        "height": subject_params.get("height"),
        "sex": subject_params.get("sex"),
        "weight": subject_params.get("weight", result.get("body_mass_kg")),
        "body_mass_kg": result.get("body_mass_kg"),
        "sampling_hz": result.get("sampling_hz", DEFAULT_FS_HZ),
        "imu_location": "lower_back_L1_L2_proxy",
        "imu_proxy_body": "Torso",
        "fall_start_time_s": result.get("fall_start_time_s"),
        "fall_start_index": result.get("fall_start_index"),
        "fall_start_frame": result.get("fall_start_frame"),
    }

    outputs = []
    for csv_path in sorted(output_dir.rglob("*.csv")):
        if "enriched" in csv_path.stem.lower():
            continue
        if not is_likely_imu_csv(csv_path):
            continue
        out_csv = csv_path.with_name(csv_path.stem + "_enriched.csv")
        try:
            outputs.append(
                export_enriched_imu_dataset(
                    csv_path,
                    out_csv,
                    metadata=metadata,
                    event_summary=event_summary,
                )
            )
        except Exception as exc:
            outputs.append({"filename": str(csv_path), "error": str(exc)})
    return outputs


def verify_axis_from_csv(csv_path: str | Path, n_standing_samples: int = 200):
    if pd is None:
        raise RuntimeError(f"pandas is required: {_PANDAS_IMPORT_ERROR}")

    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    acc_cols = _find_axis_columns(df.columns, "acc")
    gyro_cols = _find_axis_columns(df.columns, "gyro")

    result = {
        "csv": str(csv_path),
        "axis_convention_id": AXIS_CONVENTION_ID,
        "acc_columns": acc_cols,
        "gyro_columns": gyro_cols,
        "message": "",
    }

    if not _has_all_axes(acc_cols):
        result["message"] = "Could not find complete accelerometer X/Y/Z columns."
        return result

    acc_unit, acc_factor = _infer_acc_unit(df, acc_cols)
    acc = np.vstack([
        pd.to_numeric(df[acc_cols[a]], errors="coerce").to_numpy(float) * acc_factor
        for a in ("x", "y", "z")
    ]).T

    sample = acc[: min(n_standing_samples, len(acc))]
    med = np.nanmedian(sample, axis=0)
    abs_med = np.abs(med)
    gravity_axis = ["X", "Y", "Z"][int(np.nanargmax(abs_med))]
    gravity_sign = "+" if med[int(np.nanargmax(abs_med))] >= 0 else "-"
    mag = float(np.linalg.norm(med))

    result.update({
        "input_acc_unit_inferred": acc_unit,
        "standing_median_acc_mps2": {
            "X": float(med[0]),
            "Y": float(med[1]),
            "Z": float(med[2]),
        },
        "standing_median_acc_norm_mps2": mag,
        "dominant_gravity_axis": gravity_axis,
        "dominant_gravity_sign": gravity_sign,
        "interpretation": (
            f"During the first samples, gravity is mainly on {gravity_sign}{gravity_axis}. "
            f"The norm is {mag:.3f} m/s^2. For quiet standing, this should be close to 9.81 m/s^2."
        ),
    })
    return result
