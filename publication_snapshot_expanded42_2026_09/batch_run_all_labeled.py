# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_TASKS = [
    20, 21, 22, 23, 24,
    250, 25, 26, 27, 28, 29, 290, 291,
    30, 31, 32, 33, 34,
    37, 38, 39, 40, 41, 42, 43, 44,
]

EXCLUDE_MAIN_CSV_TOKENS = [
    "backup",
    "unlabeled",
    "contacts",
    "dynamics",
    "joints",
    "markers",
    "marker_quality",
    "segments",
    "opensim",
    "external",
    "summary",
    "batch",
]


def finite(x):
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def to_float(x, default=math.nan):
    try:
        y = float(x)
        return y if math.isfinite(y) else default
    except Exception:
        return default


def json_safe(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        x = float(obj)
        return x if math.isfinite(x) else None
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return str(obj)


def parse_task_list(text: str):
    text = str(text).strip().lower()

    if text in {"all", "default"}:
        return DEFAULT_TASKS

    tasks = []

    for part in text.split(","):
        part = part.strip()

        if not part:
            continue

        if "-" in part:
            a, b = part.split("-", 1)
            tasks.extend(range(int(a), int(b) + 1))
        else:
            tasks.append(int(part))

    return tasks


def scenario_id_from_folder(folder: Path):
    m = re.search(r"scenario(\d+)_", folder.name)
    return int(m.group(1)) if m else None


def parse_latest_output_folder(log_text: str):
    matches = re.findall(r"Output folder\s*->\s*(.+)", log_text)

    if not matches:
        return None

    raw = matches[-1].strip()
    raw = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", raw).strip()

    p = Path(raw)

    if p.exists():
        return p

    if "outputs/" in raw:
        tail = raw.split("outputs/", 1)[1].strip()
        p2 = Path("outputs") / tail

        if p2.exists():
            return p2

    return p


def parse_native_hz(log_text: str):
    patterns = [
        r"native frames\s*=\s*\d+\s+at\s+([0-9.]+)\s*Hz",
        r"native_hz\s*=\s*([0-9.]+)",
        r"Native dt / Hz\s*:\s*[0-9.]+\s*s\s*/\s*([0-9.]+)\s*Hz",
    ]

    for pat in patterns:
        m = re.search(pat, log_text, re.IGNORECASE)

        if m:
            hz = to_float(m.group(1))

            if finite(hz) and hz > 0:
                return hz

    return 30.0


def parse_common_fall_event_summary(log_text: str):
    blocks = re.split(r"COMMON FALL EVENT SUMMARY", log_text)

    if len(blocks) < 2:
        return {}

    block = blocks[-1]

    def grab(pattern):
        m = re.search(pattern, block, re.IGNORECASE)
        return to_float(m.group(1)) if m else math.nan

    events = {
        "perturbation_start_time_s": grab(r"Perturb\s+start\s*:\s*([0-9.]+)\s*s"),
        "fall_onset_time_s": grab(r"Fall\s+onset\s*:\s*([0-9.]+)\s*s"),
        "main_impact_time_s": grab(r"Main\s+impact\s*:\s*([0-9.]+)\s*s"),
        "settle_time_s": grab(r"Settle\s+time\s*:\s*([0-9.]+)\s*s"),
        "source": "captured_batch_stdout_common_fall_event_summary",
    }

    return {k: v for k, v in events.items() if k == "source" or finite(v)}


def parse_legacy_phase_event_summary(log_text: str):
    native_hz = parse_native_hz(log_text)
    events = {}

    phase_match = re.search(
        r"Phases:\s*(.+?)total\s*=\s*(\d+)",
        log_text,
        re.IGNORECASE,
    )

    phases = []

    if phase_match:
        phase_text = phase_match.group(0)

        for name, steps in re.findall(r"([A-Za-z_]+)\s*=\s*(\d+)", phase_text):
            if name.lower() == "total":
                continue
            phases.append((name.lower(), int(steps)))

    if phases:
        cumulative = 0
        phase_starts = {}

        for name, nsteps in phases:
            phase_starts[name] = cumulative / native_hz
            cumulative += nsteps

        for key in ["perturb", "release", "collapse"]:
            if key in phase_starts:
                events["perturbation_start_time_s"] = phase_starts[key]
                break

        if "perturbation_start_time_s" not in events and "fall" in phase_starts:
            events["perturbation_start_time_s"] = max(0.0, phase_starts["fall"] - 0.5)

    first_fall_step = None

    for line in log_text.splitlines():
        if "FALL!" not in line:
            continue

        m = re.match(r"\s*(\d+)\s+", line)

        if m:
            first_fall_step = int(m.group(1))
            break

    if first_fall_step is not None:
        events["fall_onset_time_s"] = first_fall_step / native_hz

    dur_match = re.search(r"Fall duration\s*:\s*([0-9.]+)\s*s", log_text, re.IGNORECASE)

    if dur_match and "fall_onset_time_s" in events:
        events["settle_time_s"] = events["fall_onset_time_s"] + to_float(dur_match.group(1))

    if events:
        events["source"] = "captured_batch_legacy_phase_and_first_fall_fallback"

    return events


def parse_event_summary(log_text: str):
    events = parse_common_fall_event_summary(log_text)

    if events:
        return events

    return parse_legacy_phase_event_summary(log_text)


def first_nonempty_group(match):
    if not match:
        return ""

    for g in match.groups():
        if g is not None and str(g).strip():
            return str(g).strip()

    return ""


def parse_validation_quality(log_text: str):
    def grab_float(pattern):
        m = re.search(pattern, log_text, re.IGNORECASE)
        val = first_nonempty_group(m)

        if not val:
            return ""

        return to_float(val, "")

    def grab_text(pattern):
        m = re.search(pattern, log_text, re.IGNORECASE)
        return first_nonempty_group(m)

    return {
        "overall_confidence_score": grab_float(
            r"Overall Confidence Score\s*:\s*([0-9.]+)%|Score:\s*([0-9.]+)%"
        ),
        "classification": grab_text(
            r"Classification\s*:\s*([A-Z_]+)|\(([A-Z_]+)\)"
        ),
        "sisfall_compliant": grab_text(
            r"SISFall compliant\s*:\s*(True|False)|SISFall:\s*(True|False)"
        ),
        "kfall_style_compliant": grab_text(
            r"KFall-style compliant\s*:\s*(True|False)"
        ),
        "task_validity": grab_text(
            r"Task validity\s*:\s*(PASS|FAIL)"
        ),
        "authenticity_score": grab_float(
            r"Authenticity score\s*:\s*([0-9.]+)%"
        ),
    }


def extract_output_file_paths_from_log(log_text: str):
    paths = set()
    suffixes = r"(?:csv|txt|trc|mot|json|png|xml)"

    for m in re.finditer(r"([A-Za-z0-9_./:-]+\.%s)" % suffixes, log_text):
        raw = m.group(1).strip()
        raw = raw.lstrip("?:->")
        raw = raw.strip()

        if not raw or raw.endswith(".py"):
            continue

        paths.add(raw)

    return sorted(paths)


def materialize_relative_outputs(log_text: str, output_dir: Path, project_dir: Path):
    output_dir = Path(output_dir)
    project_dir = Path(project_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    moved = []
    raw_paths = extract_output_file_paths_from_log(log_text)

    for raw in raw_paths:
        p = Path(raw)
        src = p if p.is_absolute() else project_dir / p

        if not src.exists() or not src.is_file():
            continue

        try:
            if src.resolve().parent == output_dir.resolve():
                continue
        except Exception:
            pass

        dest = output_dir / src.name

        try:
            if dest.exists():
                if dest.stat().st_size == src.stat().st_size:
                    try:
                        src.unlink()
                    except Exception:
                        pass
                continue

            shutil.move(str(src), str(dest))
            moved.append(str(dest))

        except Exception as exc:
            print(f"[warn] Could not move {src} -> {dest}: {exc}")

    return moved


def find_main_imu_csv(output_dir: Path):
    output_dir = Path(output_dir)
    csvs = sorted(output_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime)

    candidates = []

    for p in csvs:
        name = p.name.lower()

        if any(tok in name for tok in EXCLUDE_MAIN_CSV_TOKENS):
            continue

        candidates.append(p)

    if not candidates:
        return None

    return candidates[-1]


def wait_for_main_csv(output_dir: Path, timeout_s=20.0):
    t0 = time.time()
    last_size = None
    stable = 0

    while time.time() - t0 < timeout_s:
        p = find_main_imu_csv(output_dir)

        if p and p.exists():
            size = p.stat().st_size

            if size == last_size and size > 0:
                stable += 1
            else:
                stable = 0

            last_size = size

            if stable >= 2:
                return p

        time.sleep(0.5)

    return find_main_imu_csv(output_dir)


def read_comments(csv_path: Path):
    comments = {}

    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("#"):
                break

            text = line[1:].strip()

            if ":" in text:
                k, v = text.split(":", 1)
                comments[k.strip()] = v.strip()

    return comments


def clean_columns(df: pd.DataFrame):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def col_lookup(df: pd.DataFrame):
    return {str(c).strip().lower(): c for c in df.columns}


def get_col(df: pd.DataFrame, names):
    lookup = col_lookup(df)

    for name in names:
        key = str(name).strip().lower()

        if key in lookup:
            return lookup[key]

    return None


def get_time_vector(df: pd.DataFrame):
    c = get_col(df, ["Time_s_standard"])

    if c:
        t = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
        return t

    c = get_col(df, ["t", "time_s"])

    if c:
        t = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
        return t

    c = get_col(df, ["timestamp"])

    if c:
        t_raw = pd.to_numeric(df[c], errors="coerce").to_numpy(float)

        if np.nanmin(t_raw) > 100:
            t_raw = t_raw - np.nanmin(t_raw)

        return t_raw

    return np.arange(len(df), dtype=float) / 100.0


def get_accel_magnitude(df: pd.DataFrame):
    c = get_col(df, ["Accel_Raw_Magnitude_mps2", "accel_mag", "impact_magnitude"])

    if c:
        return pd.to_numeric(df[c], errors="coerce").fillna(0).to_numpy(float)

    xyz_sets = [
        ["accel_raw_x", "accel_raw_y", "accel_raw_z"],
        ["ax", "ay", "az"],
        ["accel_x", "accel_y", "accel_z"],
    ]

    for names in xyz_sets:
        cols = [get_col(df, [n]) for n in names]

        if all(cols):
            arrs = [
                pd.to_numeric(df[c], errors="coerce").fillna(0).to_numpy(float)
                for c in cols
            ]
            return np.sqrt(arrs[0] ** 2 + arrs[1] ** 2 + arrs[2] ** 2)

    return np.zeros(len(df), dtype=float)


def get_impact_score(df: pd.DataFrame):
    """
    Return a usable impact score.

    Important for legacy tasks:
    some files contain impact_n but it is all zeros.
    In that case we must ignore it and use acceleration magnitude instead.
    """
    priority = [
        ["impact_magnitude"],
        ["impact_force"],
        ["impact_n"],
        ["Accel_Raw_Magnitude_mps2"],
        ["accel_mag"],
    ]

    for names in priority:
        c = get_col(df, names)

        if not c:
            continue

        arr = np.abs(
            pd.to_numeric(df[c], errors="coerce")
            .fillna(0)
            .to_numpy(float)
        )

        # Ignore unusable impact columns: all zero, constant, or no real peak.
        if len(arr) == 0:
            continue

        if np.nanmax(arr) <= 1e-9:
            continue

        if np.nanstd(arr) <= 1e-9:
            continue

        return arr

    # Final fallback from accel axes / accel_mag.
    arr = get_accel_magnitude(df)

    if np.nanmax(arr) <= 1e-9 or np.nanstd(arr) <= 1e-9:
        raise RuntimeError("Could not compute a usable impact score")

    return arr


def nearest_index(t, value):
    if not finite(value):
        return None

    return int(np.nanargmin(np.abs(t - float(value))))


def infer_impact_time(df: pd.DataFrame, t, fall_onset_t, settle_t):
    score = get_impact_score(df)

    if not finite(fall_onset_t):
        idx = int(np.nanargmax(score))
        return float(t[idx]), idx

    end_t = fall_onset_t + 3.0

    if finite(settle_t):
        end_t = max(settle_t + 0.5, fall_onset_t + 0.3)

    mask = (t >= fall_onset_t) & (t <= end_t)

    if not np.any(mask):
        idx = int(np.nanargmax(score))
        return float(t[idx]), idx

    idxs = np.where(mask)[0]
    local = int(np.nanargmax(score[idxs]))
    idx = int(idxs[local])

    return float(t[idx]), idx


def label_csv_direct(
    csv_path: Path,
    output_dir: Path,
    scenario_id=None,
    event_overrides=None,
    subject_params=None,
    log_quality=None,
):
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    event_overrides = event_overrides or {}
    subject_params = subject_params or {}
    log_quality = log_quality or {}

    comments = read_comments(csv_path)

    df = pd.read_csv(csv_path, comment="#")
    df = clean_columns(df)

    if df.empty:
        raise RuntimeError(f"Empty CSV: {csv_path}")

    t = get_time_vector(df)
    accel_mag = get_accel_magnitude(df)

    df["Time_s_standard"] = t
    df["Frame_100Hz"] = np.rint(t * 100.0).astype(int)
    df["Frame_native_30Hz"] = np.rint(t * 30.0).astype(int)
    df["Accel_Raw_Magnitude_mps2"] = accel_mag

    perturb_t = to_float(event_overrides.get("perturbation_start_time_s"))
    fall_t = to_float(event_overrides.get("fall_onset_time_s"))
    impact_t = to_float(event_overrides.get("main_impact_time_s"))
    settle_t = to_float(event_overrides.get("settle_time_s"))

    fall_col = get_col(df, ["fall_detected", "fall"])

    if not finite(fall_t) and fall_col:
        fall_arr = pd.to_numeric(df[fall_col], errors="coerce").fillna(0).to_numpy(float)
        idxs = np.where(fall_arr > 0)[0]

        if len(idxs):
            fall_t = float(t[int(idxs[0])])

    if not finite(fall_t):
        fall_t = float(t[int(np.nanargmax(accel_mag))])

    if not finite(perturb_t):
        perturb_t = max(0.0, fall_t - 0.5)

    if finite(impact_t):
        impact_idx = nearest_index(t, impact_t)
    else:
        impact_t, impact_idx = infer_impact_time(df, t, fall_t, settle_t)

    if impact_idx is None:
        impact_idx = int(np.nanargmax(accel_mag))
        impact_t = float(t[impact_idx])

    activity = np.array(["normal"] * len(df), dtype=object)

    activity[t >= perturb_t] = "pre_fall_perturbation"
    activity[t >= fall_t] = "falling_pre_impact"
    activity[t > impact_t] = "post_impact"

    if finite(settle_t):
        activity[t >= settle_t] = "settled_post_fall"

    activity[impact_idx] = "main_impact"

    df["Perturbation_Start_Time_s"] = perturb_t
    df["Reaction_Start_Time_s"] = math.nan
    df["Scripted_Fall_Phase_Start_Time_s"] = math.nan
    df["Fall_Onset_Time_s"] = fall_t
    df["Main_Impact_Time_s"] = impact_t
    df["Settle_Time_s"] = settle_t if finite(settle_t) else math.nan

    df["Perturbation_Start_Frame_100Hz"] = int(round(perturb_t * 100.0))
    df["Reaction_Start_Frame_100Hz"] = -1
    df["Scripted_Fall_Phase_Start_Frame_100Hz"] = -1
    df["Fall_Onset_Frame_100Hz"] = int(round(fall_t * 100.0))
    df["Main_Impact_Frame_100Hz"] = int(round(impact_t * 100.0))
    df["Settle_Frame_100Hz"] = int(round(settle_t * 100.0)) if finite(settle_t) else -1

    src = event_overrides.get("source", "direct_signal_fallback")
    df["Perturbation_Start_Source"] = src
    df["Fall_Onset_Source"] = src
    df["Main_Impact_Source"] = src if finite(event_overrides.get("main_impact_time_s", math.nan)) else "direct_peak_impact_score"

    df["Activity_Label"] = activity
    df["Normal_Label"] = (activity == "normal").astype(int)
    df["PreFall_Label"] = (activity == "pre_fall_perturbation").astype(int)
    df["Reaction_Label"] = 0
    df["Fall_Label"] = np.isin(
        activity,
        ["falling_pre_impact", "main_impact", "post_impact", "settled_post_fall"],
    ).astype(int)
    df["PreImpact_Fall_Label"] = (activity == "falling_pre_impact").astype(int)
    df["Impact_Label"] = 0
    df.loc[impact_idx, "Impact_Label"] = 1
    df["PostImpact_Label"] = np.isin(
        activity,
        ["post_impact", "settled_post_fall"],
    ).astype(int)

    if fall_col:
        df["Existing_FallDetected_Label"] = (
            pd.to_numeric(df[fall_col], errors="coerce").fillna(0).astype(int)
        )
    else:
        df["Existing_FallDetected_Label"] = 0

    label3 = np.zeros(len(df), dtype=int)
    label3[t >= perturb_t] = 1
    label3[t >= fall_t] = 2
    df["Label_3Class"] = label3

    label4 = np.zeros(len(df), dtype=int)
    label4[t >= perturb_t] = 1
    label4[t >= fall_t] = 2
    label4[t >= impact_t] = 3
    label4[impact_idx] = 3
    df["Label_4Class"] = label4

    backup_csv = csv_path.with_name(csv_path.stem + ".unlabeled_backup.csv")

    if not backup_csv.exists():
        shutil.copy2(csv_path, backup_csv)

    df.to_csv(csv_path, index=False)

    metadata_json = csv_path.with_name(csv_path.stem + ".label_metadata.json")

    events = {
        "perturbation_start_time_s": perturb_t,
        "fall_onset_time_s": fall_t,
        "fall_onset_index_row": nearest_index(t, fall_t),
        "main_impact_time_s": impact_t,
        "main_impact_index_row": impact_idx,
        "settle_time_s": settle_t if finite(settle_t) else None,
        "source": src,
    }

    payload = {
        "status": "labeled_inplace",
        "csv": str(csv_path),
        "backup_csv": str(backup_csv),
        "metadata_json": str(metadata_json),
        "scenario_id": scenario_id,
        "subject_params": subject_params,
        "metadata_comments_from_original_csv": comments,
        "event_overrides_used": event_overrides,
        "validation_quality_from_log": log_quality,
        "events": events,
        "label_counts": {
            "Activity_Label": df["Activity_Label"].value_counts(dropna=False).to_dict(),
            "Label_3Class": df["Label_3Class"].value_counts(dropna=False).to_dict(),
            "Label_4Class": df["Label_4Class"].value_counts(dropna=False).to_dict(),
        },
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
    }

    metadata_json.write_text(
        json.dumps(payload, indent=2, default=json_safe),
        encoding="utf-8",
    )

    return {
        "csv": str(csv_path),
        "backup_csv": str(backup_csv),
        "metadata_json": str(metadata_json),
        "status": "labeled_inplace",
        "events": events,
    }


def verify_labeled_csv(csv_path: Path):
    df = pd.read_csv(csv_path)

    required = [
        "Activity_Label",
        "Label_3Class",
        "Label_4Class",
        "PreFall_Label",
        "Fall_Label",
        "PreImpact_Fall_Label",
        "Impact_Label",
        "PostImpact_Label",
        "Existing_FallDetected_Label",
        "Fall_Onset_Time_s",
        "Main_Impact_Time_s",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise RuntimeError(f"CSV not labeled. Missing columns: {missing}")

    impact_count = int(pd.to_numeric(df["Impact_Label"], errors="coerce").fillna(0).sum())

    if impact_count != 1:
        raise RuntimeError(f"Impact_Label must be exactly 1 frame, found {impact_count}")

    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "impact_label_count": impact_count,
        "fall_onset_time_s": float(pd.to_numeric(df["Fall_Onset_Time_s"], errors="coerce").dropna().iloc[0]),
        "main_impact_time_s": float(pd.to_numeric(df["Main_Impact_Time_s"], errors="coerce").dropna().iloc[0]),
    }


def run_one_task(task_id, age, height, sex, weight, log_path, display):
    env = os.environ.copy()

    if display:
        env["DISPLAY"] = display

    input_text = f"{task_id}\n{age}\n{height}\n{sex}\n{weight}\n"

    print("\n" + "=" * 90)
    print(f"RUNNING SCENARIO {task_id}")
    print("=" * 90)
    print(f"Subject: age={age}, height={height}, sex={sex}, weight={weight}")
    print(f"Log: {log_path}")
    print("=" * 90 + "\n")

    proc = subprocess.Popen(
        [sys.executable, "fall_dispatcher.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    assert proc.stdin is not None
    assert proc.stdout is not None

    proc.stdin.write(input_text)
    proc.stdin.flush()
    proc.stdin.close()

    chunks = []

    with log_path.open("w", encoding="utf-8", errors="ignore") as f:
        for line in proc.stdout:
            print(line, end="")
            f.write(line)
            f.flush()
            chunks.append(line)

    code = proc.wait()
    return code, "".join(chunks)


def write_summary(rows, batch_dir):
    if not rows:
        return

    csv_path = batch_dir / "batch_summary.csv"
    json_path = batch_dir / "batch_summary.json"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path.write_text(
        json.dumps(rows, indent=2, default=json_safe),
        encoding="utf-8",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--age", required=True)
    ap.add_argument("--height", required=True)
    ap.add_argument("--sex", required=True, choices=["male", "female"])
    ap.add_argument("--weight", required=True)
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--display", default=os.environ.get("DISPLAY", ":2"))
    ap.add_argument("--continue-on-error", action="store_true")
    args = ap.parse_args()

    tasks = parse_task_list(args.tasks)

    project_dir = Path.cwd()
    batch_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = Path("outputs") / "batch_runs" / f"batch_{batch_stamp}"
    log_dir = batch_dir / "logs"

    batch_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    print("\n" + "#" * 90)
    print("BATCH FALL SIMULATION + DIRECT SAFE LABELING")
    print("#" * 90)
    print(f"Tasks: {tasks}")
    print(f"Subject: age={args.age}, height={args.height}, sex={args.sex}, weight={args.weight}")
    print(f"Batch folder: {batch_dir}")
    print("#" * 90 + "\n")

    for i, task_id in enumerate(tasks, start=1):
        log_path = log_dir / f"scenario{task_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        row = {
            "task_id": task_id,
            "status": "",
            "output_folder": "",
            "main_csv": "",
            "backup_csv": "",
            "metadata_json": "",
            "label_status": "",
            "rows": "",
            "columns": "",
            "impact_label_count": "",
            "perturbation_start_time_s": "",
            "fall_onset_time_s": "",
            "main_impact_time_s": "",
            "settle_time_s": "",
            "event_source": "",
            "overall_confidence_score": "",
            "classification": "",
            "sisfall_compliant": "",
            "kfall_style_compliant": "",
            "task_validity": "",
            "authenticity_score": "",
            "log_path": str(log_path),
            "error": "",
        }

        try:
            code, log_text = run_one_task(
                task_id=task_id,
                age=args.age,
                height=args.height,
                sex=args.sex,
                weight=args.weight,
                log_path=log_path,
                display=args.display,
            )

            if code != 0:
                raise RuntimeError(f"fall_dispatcher.py exited with code {code}")

            out_dir = parse_latest_output_folder(log_text)

            if out_dir is None:
                raise RuntimeError("Could not detect output folder from log")

            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            row["output_folder"] = str(out_dir)

            moved = materialize_relative_outputs(log_text, out_dir, project_dir)

            main_csv = wait_for_main_csv(out_dir)

            if main_csv is None or not main_csv.exists():
                raise RuntimeError(f"No main IMU CSV found in {out_dir}")

            events = parse_event_summary(log_text)
            quality = parse_validation_quality(log_text)

            print("\n" + "-" * 90)
            print(f"POST-RUN DIRECT LABELING FOR SCENARIO {task_id}")
            print(f"Output folder : {out_dir}")
            print(f"Main IMU CSV  : {main_csv}")
            print(f"Moved files   : {len(moved)}")
            print(f"Event summary : {events if events else 'NOT FOUND - direct signal fallback'}")
            print("-" * 90)

            result = label_csv_direct(
                csv_path=main_csv,
                output_dir=out_dir,
                scenario_id=scenario_id_from_folder(out_dir) or task_id,
                event_overrides=events,
                subject_params={
                    "age": args.age,
                    "height": args.height,
                    "sex": args.sex,
                    "weight": args.weight,
                },
                log_quality=quality,
            )

            verify = verify_labeled_csv(Path(result["csv"]))

            row.update({
                "status": "completed_saved_labeled_verified",
                "main_csv": result["csv"],
                "backup_csv": result["backup_csv"],
                "metadata_json": result["metadata_json"],
                "label_status": result["status"],
                "rows": verify["rows"],
                "columns": verify["columns"],
                "impact_label_count": verify["impact_label_count"],
                "perturbation_start_time_s": result["events"].get("perturbation_start_time_s", ""),
                "fall_onset_time_s": verify["fall_onset_time_s"],
                "main_impact_time_s": verify["main_impact_time_s"],
                "settle_time_s": result["events"].get("settle_time_s", ""),
                "event_source": result["events"].get("source", ""),
                **quality,
            })

            print("\nSAVE/LABEL VERIFICATION PASSED")
            print(json.dumps(verify, indent=2))
            print("Backup CSV    :", result["backup_csv"])
            print("Metadata JSON :", result["metadata_json"])

        except Exception as exc:
            row["status"] = "failed"
            row["error"] = str(exc)
            print("\n" + "!" * 90)
            print(f"FAILED SCENARIO {task_id}: {exc}")
            print("!" * 90 + "\n")

            rows.append(row)
            write_summary(rows, batch_dir)

            if not args.continue_on_error:
                print("Batch stopped because this scenario was not safely saved/labeled.")
                break

            continue

        rows.append(row)
        write_summary(rows, batch_dir)

        print("\n" + "#" * 90)
        print(f"FINISHED {i}/{len(tasks)} SCENARIOS")
        print(f"Summary CSV : {batch_dir / 'batch_summary.csv'}")
        print(f"Summary JSON: {batch_dir / 'batch_summary.json'}")
        print("#" * 90 + "\n")

    print("\n" + "=" * 90)
    print("BATCH ENDED")
    print(f"Summary folder: {batch_dir}")
    print(f"Summary CSV   : {batch_dir / 'batch_summary.csv'}")
    print(f"Summary JSON  : {batch_dir / 'batch_summary.json'}")
    print("=" * 90)


if __name__ == "__main__":
    main()
