# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import builtins
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Prefer headless rendering/backend behavior before MuJoCo is imported deeply.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


class _DummyViewer:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def sync(self):
        return None

    def close(self):
        return None

    def is_running(self):
        return True


def enable_headless_viewer_patch():
    """
    Prevent mujoco.viewer.launch_passive from opening a GLFW window.
    This is useful on SSH / no-DISPLAY machines.
    """
    try:
        import mujoco.viewer

        def _launch_passive_no_window(*args, **kwargs):
            print("[headless] mujoco.viewer.launch_passive disabled; using DummyViewer")
            return _DummyViewer()

        mujoco.viewer.launch_passive = _launch_passive_no_window
        print("[headless] MuJoCo viewer monkeypatch active")
    except Exception as exc:
        print(f"[headless][warn] Could not patch mujoco.viewer: {exc}")


def disable_blocking_input():
    """
    Some legacy scripts call input('Press Enter to exit ...') after an error.
    In batch mode we do not want the terminal to block.
    """
    def _fake_input(prompt=""):
        if prompt:
            print(prompt)
        return ""

    builtins.input = _fake_input


from fall_dispatcher import SCENARIO_CATALOGUE, load_scenario_module


def parse_scenarios(text: str):
    valid = sorted(SCENARIO_CATALOGUE.keys())
    if text.lower() == "all":
        return valid

    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend([x for x in valid if int(a) <= x <= int(b)])
        else:
            out.append(int(part))

    seen = set()
    clean = []
    for sid in out:
        if sid in valid and sid not in seen:
            clean.append(sid)
            seen.add(sid)
        elif sid not in valid:
            print(f"[warn] scenario {sid} is not in SCENARIO_CATALOGUE; skipping")
    return clean


def main():
    ap = argparse.ArgumentParser(description="Batch-generate IMU datasets for fall scenarios.")
    ap.add_argument("--scenarios", default="all", help="all, comma list like 20,21,34, or range like 20-34")
    ap.add_argument("--age", type=int, default=75)
    ap.add_argument("--height", type=float, default=1.65)
    ap.add_argument("--sex", choices=["male", "female"], default="male")
    ap.add_argument("--weight", default="auto", help="auto or body mass in kg")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--viewer", action="store_true", help="Open the MuJoCo viewer. Default is headless/no viewer.")
    args = ap.parse_args()

    if not args.viewer:
        enable_headless_viewer_patch()
        disable_blocking_input()

    weight = None if str(args.weight).lower() in ("auto", "none", "") else float(args.weight)
    subject = {"age": args.age, "height": args.height, "sex": args.sex, "weight": weight}

    scenario_ids = parse_scenarios(args.scenarios)
    if not scenario_ids:
        print("[error] No valid scenarios selected.")
        sys.exit(1)

    print("=" * 80)
    print("BATCH FALL IMU DATASET GENERATION")
    print(f"Subject: {subject}")
    print(f"Scenarios: {scenario_ids}")
    print(f"Trials per scenario: {args.trials}")
    print(f"Viewer: {'ON' if args.viewer else 'OFF / headless'}")
    print("=" * 80)

    results = []
    for trial in range(1, args.trials + 1):
        for sid in scenario_ids:
            desc, cat = SCENARIO_CATALOGUE[sid]
            print(f"\n[batch] Trial {trial}/{args.trials} | scenario {sid}: {desc}")
            mod = load_scenario_module(sid)
            if mod is None or not hasattr(mod, "run"):
                print(f"[batch][skip] scenario {sid}: module missing or has no run(subject_params)")
                results.append({"scenario_id": sid, "trial": trial, "status": "skipped_missing_module"})
                continue

            try:
                result = mod.run(dict(subject))
                results.append({"scenario_id": sid, "trial": trial, "status": "ok", "result": result})
            except KeyboardInterrupt:
                print("\n[batch] interrupted by user")
                raise
            except Exception as exc:
                print(f"[batch][error] scenario {sid}: {exc}")
                results.append({"scenario_id": sid, "trial": trial, "status": "error", "error": str(exc)})

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    manifest = out_dir / ("batch_manifest_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".json")
    manifest.write_text(json.dumps({
        "subject": subject,
        "scenario_ids": scenario_ids,
        "trials": args.trials,
        "created_at": datetime.now().isoformat(),
        "results": results,
    }, indent=2, default=str), encoding="utf-8")

    print(f"\n[batch] Manifest written: {manifest}")
    print("[batch] Enriched CSV files can be listed with:")
    print("        find outputs -name '*enriched*.csv' | sort")


if __name__ == "__main__":
    main()
