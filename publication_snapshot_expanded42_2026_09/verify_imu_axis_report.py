# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ACC_COLS = ["accel_raw_x", "accel_raw_y", "accel_raw_z"]
GYRO_COLS = ["gyro_x", "gyro_y", "gyro_z"]
POS_COLS = ["sensor_pos_x", "sensor_pos_y", "sensor_pos_z"]


def axis_name(i):
    return ["X", "Y", "Z"][int(i)]


def signed_axis(values):
    arr = np.asarray(values, dtype=float)
    idx = int(np.nanargmax(np.abs(arr)))
    sign = "+" if arr[idx] >= 0 else "-"
    return sign + axis_name(idx), float(arr[idx])


def load_csv(path):
    return pd.read_csv(path)


def summarize_axis(csv_path, stand_end_s=1.5, fall_onset_s=None, impact_s=None):
    csv_path = Path(csv_path)
    df = load_csv(csv_path)

    if "Time_s_standard" in df.columns:
        t = pd.to_numeric(df["Time_s_standard"], errors="coerce").to_numpy(float)
    elif "timestamp" in df.columns:
        t = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(float)
        t = t - np.nanmin(t)
    else:
        t = np.arange(len(df), dtype=float) / 100.0

    report = {
        "csv": str(csv_path),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "units": {
            "acceleration": "m/s^2",
            "gyroscope": "rad/s",
            "sensor_position": "m",
            "time": "s",
        },
    }

    # Use event times from the labeled CSV if available.
    if fall_onset_s is None and "Fall_Onset_Time_s" in df.columns:
        fall_onset_s = pd.to_numeric(df["Fall_Onset_Time_s"], errors="coerce").dropna()
        fall_onset_s = float(fall_onset_s.iloc[0]) if len(fall_onset_s) else None

    if impact_s is None and "Main_Impact_Time_s" in df.columns:
        impact_s = pd.to_numeric(df["Main_Impact_Time_s"], errors="coerce").dropna()
        impact_s = float(impact_s.iloc[0]) if len(impact_s) else None

    # Standing window: avoid first few samples, use stable early part.
    stand_mask = (t >= 0.2) & (t <= stand_end_s)

    if not np.any(stand_mask):
        stand_mask = t <= stand_end_s

    if all(c in df.columns for c in ACC_COLS):
        acc_stand = df.loc[stand_mask, ACC_COLS].apply(pd.to_numeric, errors="coerce")
        acc_med = acc_stand.median().to_numpy(float)
        acc_mean = acc_stand.mean().to_numpy(float)
        acc_norm = float(np.linalg.norm(acc_med))

        dominant_gravity_axis, dominant_gravity_value = signed_axis(acc_med)

        report["standing_raw_acceleration"] = {
            "window_s": [float(np.nanmin(t[stand_mask])), float(np.nanmax(t[stand_mask]))],
            "median_xyz_mps2": {
                "x": float(acc_med[0]),
                "y": float(acc_med[1]),
                "z": float(acc_med[2]),
            },
            "mean_xyz_mps2": {
                "x": float(acc_mean[0]),
                "y": float(acc_mean[1]),
                "z": float(acc_mean[2]),
            },
            "median_norm_mps2": acc_norm,
            "dominant_gravity_axis": dominant_gravity_axis,
            "dominant_gravity_axis_value_mps2": dominant_gravity_value,
        }

    if all(c in df.columns for c in GYRO_COLS):
        if fall_onset_s is not None and impact_s is not None:
            fall_mask = (t >= fall_onset_s) & (t <= impact_s)
        elif fall_onset_s is not None:
            fall_mask = (t >= fall_onset_s) & (t <= fall_onset_s + 2.0)
        else:
            fall_mask = np.ones(len(df), dtype=bool)

        gyro_fall = df.loc[fall_mask, GYRO_COLS].apply(pd.to_numeric, errors="coerce")
        gyro_abs_peak = gyro_fall.abs().max().to_numpy(float)
        gyro_peak_signed = []
        for c in GYRO_COLS:
            s = pd.to_numeric(df.loc[fall_mask, c], errors="coerce")
            if len(s.dropna()) == 0:
                gyro_peak_signed.append(np.nan)
            else:
                idx = s.abs().idxmax()
                gyro_peak_signed.append(float(s.loc[idx]))

        dominant_gyro_axis, dominant_gyro_value = signed_axis(gyro_peak_signed)

        report["fall_rotation_gyro"] = {
            "window_s": [float(fall_onset_s) if fall_onset_s is not None else None,
                         float(impact_s) if impact_s is not None else None],
            "peak_abs_xyz_rads": {
                "x": float(gyro_abs_peak[0]),
                "y": float(gyro_abs_peak[1]),
                "z": float(gyro_abs_peak[2]),
            },
            "peak_signed_xyz_rads": {
                "x": float(gyro_peak_signed[0]),
                "y": float(gyro_peak_signed[1]),
                "z": float(gyro_peak_signed[2]),
            },
            "dominant_rotation_axis": dominant_gyro_axis,
            "dominant_rotation_axis_value_rads": dominant_gyro_value,
        }

    if all(c in df.columns for c in POS_COLS):
        pos = df[POS_COLS].apply(pd.to_numeric, errors="coerce")
        displacement = pos.iloc[-1].to_numpy(float) - pos.iloc[0].to_numpy(float)
        dominant_translation_axis, dominant_translation_value = signed_axis(displacement)

        report["world_sensor_position_change"] = {
            "start_xyz_m": {
                "x": float(pos.iloc[0, 0]),
                "y": float(pos.iloc[0, 1]),
                "z": float(pos.iloc[0, 2]),
            },
            "end_xyz_m": {
                "x": float(pos.iloc[-1, 0]),
                "y": float(pos.iloc[-1, 1]),
                "z": float(pos.iloc[-1, 2]),
            },
            "displacement_xyz_m": {
                "x": float(displacement[0]),
                "y": float(displacement[1]),
                "z": float(displacement[2]),
            },
            "dominant_world_translation_axis": dominant_translation_axis,
            "dominant_world_translation_value_m": dominant_translation_value,
        }

    report["recommended_current_interpretation"] = {
        "accel_raw_y": "main vertical/proper-acceleration axis during quiet standing",
        "gyro_x": "dominant sagittal forward-fall rotation axis in scenario 43",
        "sensor_pos_z": "MuJoCo/world vertical height, not the same as accel_raw_z",
        "important_note": "Before comparing to a real IMU dataset, map the real dataset axes to this simulator convention or rotate this simulator output to the real dataset convention.",
    }

    return report


def write_report(csv_path, report):
    csv_path = Path(csv_path)
    json_path = csv_path.with_name(csv_path.stem + "_axis_report.json")
    txt_path = csv_path.with_name(csv_path.stem + "_axis_report.txt")

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = []
    lines.append("IMU AXIS VERIFICATION REPORT")
    lines.append("=" * 60)
    lines.append(f"CSV: {report['csv']}")
    lines.append(f"Rows: {report['rows']}")
    lines.append("")
    lines.append("Units:")
    for k, v in report["units"].items():
        lines.append(f"  {k}: {v}")

    if "standing_raw_acceleration" in report:
        s = report["standing_raw_acceleration"]
        lines.append("")
        lines.append("Standing raw acceleration:")
        lines.append(f"  median xyz [m/s^2]: {s['median_xyz_mps2']}")
        lines.append(f"  norm [m/s^2]      : {s['median_norm_mps2']:.4f}")
        lines.append(f"  dominant gravity  : {s['dominant_gravity_axis']}")

    if "fall_rotation_gyro" in report:
        g = report["fall_rotation_gyro"]
        lines.append("")
        lines.append("Fall rotation gyro:")
        lines.append(f"  peak abs xyz [rad/s]   : {g['peak_abs_xyz_rads']}")
        lines.append(f"  peak signed xyz [rad/s]: {g['peak_signed_xyz_rads']}")
        lines.append(f"  dominant rotation axis : {g['dominant_rotation_axis']}")

    if "world_sensor_position_change" in report:
        w = report["world_sensor_position_change"]
        lines.append("")
        lines.append("World sensor position change:")
        lines.append(f"  displacement xyz [m]: {w['displacement_xyz_m']}")
        lines.append(f"  dominant world translation: {w['dominant_world_translation_axis']}")

    lines.append("")
    lines.append("Recommended interpretation:")
    for k, v in report["recommended_current_interpretation"].items():
        lines.append(f"  {k}: {v}")

    txt_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, txt_path


def main():
    ap = argparse.ArgumentParser(description="Verify IMU axis direction from a labeled simulator CSV.")
    ap.add_argument("csv", help="Path to main labeled IMU CSV")
    ap.add_argument("--stand-end-s", type=float, default=1.5)
    args = ap.parse_args()

    report = summarize_axis(args.csv, stand_end_s=args.stand_end_s)
    json_path, txt_path = write_report(args.csv, report)

    print(json.dumps(report, indent=2))
    print(f"\nSaved JSON: {json_path}")
    print(f"Saved TXT : {txt_path}")


if __name__ == "__main__":
    main()
