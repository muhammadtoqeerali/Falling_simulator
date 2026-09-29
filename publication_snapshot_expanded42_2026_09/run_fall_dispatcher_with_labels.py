# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from imu_pipeline_labels import label_output_folder_inplace


def scenario_id_from_folder(folder: Path):
    m = re.search(r"scenario(\d+)_", folder.name)
    return int(m.group(1)) if m else None


def parse_latest_output_folder(log_text: str):
    matches = re.findall(r"Output folder\s*->\s*(.+)", log_text)
    if not matches:
        return None

    raw = matches[-1].strip()
    # remove terminal escape/control junk if present
    raw = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", raw).strip()
    p = Path(raw)

    if p.exists():
        return p

    # Sometimes log has absolute path but we are in project dir; fallback to outputs basename.
    if "outputs/" in raw:
        tail = raw.split("outputs/", 1)[1].strip()
        p2 = Path("outputs") / tail
        if p2.exists():
            return p2

    return p


def parse_common_fall_event_summary(log_text: str):
    """
    Extracts the final COMMON FALL EVENT SUMMARY from simulator stdout.
    This is the preferred source for labels.
    """
    blocks = re.split(r"COMMON FALL EVENT SUMMARY", log_text)
    if len(blocks) < 2:
        return {}

    block = blocks[-1]

    def grab(pattern):
        m = re.search(pattern, block, re.IGNORECASE)
        if not m:
            return None
        try:
            return float(m.group(1))
        except Exception:
            return None

    events = {
        "perturbation_start_time_s": grab(r"Perturb\s+start\s*:\s*([0-9.]+)\s*s"),
        "fall_onset_time_s": grab(r"Fall\s+onset\s*:\s*([0-9.]+)\s*s"),
        "main_impact_time_s": grab(r"Main\s+impact\s*:\s*([0-9.]+)\s*s"),
        "settle_time_s": grab(r"Settle\s+time\s*:\s*([0-9.]+)\s*s"),
        "source": "captured_dispatcher_stdout_common_fall_event_summary",
    }

    return {k: v for k, v in events.items() if v is not None}


def run_dispatcher_with_terminal_log(log_path: Path):
    """
    Runs fall_dispatcher.py interactively while saving the full terminal output.
    Uses `script` when available because it preserves prompts/input behavior.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    python_exe = sys.executable
    cmd_str = f"{python_exe} fall_dispatcher.py"

    if shutil.which("script"):
        # util-linux script syntax on Ubuntu/Linux:
        cmd = ["script", "-q", "-f", "-c", cmd_str, str(log_path)]
        return subprocess.call(cmd, env=os.environ.copy())

    # Fallback: no capture, still run normally.
    print("[warn] `script` command not found; running without full terminal capture.")
    return subprocess.call([python_exe, "fall_dispatcher.py"], env=os.environ.copy())


def main():
    outputs = Path("outputs")
    outputs.mkdir(exist_ok=True)

    before = set(outputs.glob("scenario*_*"))

    logs_dir = Path("outputs") / "_run_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / ("dispatcher_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".log")

    print("=" * 80)
    print("RUNNING FALL DISPATCHER WITH AUTOMATIC LABELING")
    print(f"Terminal log: {log_path}")
    print("=" * 80)

    code = run_dispatcher_with_terminal_log(log_path)

    try:
        log_text = log_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        log_text = ""

    event_overrides = parse_common_fall_event_summary(log_text)
    out_dir = parse_latest_output_folder(log_text)

    after = set(outputs.glob("scenario*_*"))
    new_dirs = sorted(after - before, key=lambda p: p.stat().st_mtime)

    if out_dir is not None and out_dir.exists():
        target_dirs = [out_dir]
    elif new_dirs:
        target_dirs = new_dirs
    else:
        all_dirs = sorted(outputs.glob("scenario*_*"), key=lambda p: p.stat().st_mtime)
        target_dirs = all_dirs[-1:] if all_dirs else []

    print("\n" + "=" * 80)
    print("POST-RUN IMU LABELING")
    print("=" * 80)
    print(f"Parsed event summary: {event_overrides if event_overrides else 'NOT FOUND - will use fallback'}")

    all_results = []

    for folder in target_dirs:
        folder = Path(folder)
        sid = scenario_id_from_folder(folder)
        print(f"[label] folder={folder} scenario_id={sid}")

        result = label_output_folder_inplace(
            output_dir=folder,
            scenario_id=sid,
            subject_params={},
            result={},
            event_overrides=event_overrides if event_overrides else None,
        )

        print(result)
        all_results.append({"folder": str(folder), "scenario_id": sid, "result": result})

    summary_path = log_path.with_suffix(".label_summary.json")
    summary_path.write_text(json.dumps({
        "dispatcher_exit_code": code,
        "log_path": str(log_path),
        "event_overrides": event_overrides,
        "label_results": all_results,
    }, indent=2, default=str), encoding="utf-8")

    print(f"Label summary: {summary_path}")
    print("=" * 80)

    return code


if __name__ == "__main__":
    raise SystemExit(main())
