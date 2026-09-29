# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path

from imu_dataset_tools import (
    export_enriched_imu_dataset,
    is_likely_imu_csv,
    verify_axis_from_csv,
)


def make_phase_log(stand, walk, perturb, react, fall, native_hz):
    t0 = 0.0
    t_stand_end = stand / native_hz
    t_walk_end = (stand + walk) / native_hz
    t_perturb_end = (stand + walk + perturb) / native_hz
    t_react_end = (stand + walk + perturb + react) / native_hz
    t_fall_end = (stand + walk + perturb + react + fall) / native_hz

    return [
        {"time_s": t0, "phase": "stand"},
        {"time_s": t_stand_end, "phase": "walk"},
        {"time_s": t_walk_end, "phase": "perturb"},
        {"time_s": t_perturb_end, "phase": "react"},
        {"time_s": t_react_end, "phase": "fall"},
        {"time_s": t_fall_end, "phase": "rest"},
    ]


def find_imu_csvs(folder: Path):
    csvs = []
    for p in sorted(folder.rglob("*.csv")):
        name = p.name.lower()
        if "enriched" in name:
            continue
        if is_likely_imu_csv(p):
            csvs.append(p)
    return csvs


def main():
    ap = argparse.ArgumentParser(
        description="Create labelled/enriched IMU CSV files from completed simulator outputs."
    )
    ap.add_argument("output_folder", help="Completed scenario output folder containing the original IMU CSV")
    ap.add_argument("--scenario-id", type=int, required=True)
    ap.add_argument("--description", default="")
    ap.add_argument("--age", type=int, default=75)
    ap.add_argument("--height", type=float, default=1.65)
    ap.add_argument("--sex", default="male")
    ap.add_argument("--weight", type=float, default=70.8)
    ap.add_argument("--native-hz", type=float, default=30.0)
    ap.add_argument("--output-hz", type=float, default=100.0)

    # Defaults are from your scenario-34 log:
    # stand=195, walk=300, perturb=60, react=31, fall=470 at 30 Hz.
    ap.add_argument("--stand-steps", type=int, default=195)
    ap.add_argument("--walk-steps", type=int, default=300)
    ap.add_argument("--perturb-steps", type=int, default=60)
    ap.add_argument("--react-steps", type=int, default=31)
    ap.add_argument("--fall-steps", type=int, default=470)

    args = ap.parse_args()

    folder = Path(args.output_folder)
    if not folder.exists() or not folder.is_dir():
        raise SystemExit(f"[error] Output folder not found: {folder}")

    csvs = find_imu_csvs(folder)
    if not csvs:
        print(f"[error] No original IMU CSV found inside: {folder}")
        print("This means the simulator has not completed successfully yet, or the IMU CSV has a different format/name.")
        print("First run the scenario normally until it finishes and saves CSV files.")
        raise SystemExit(2)

    phase_log = make_phase_log(
        args.stand_steps,
        args.walk_steps,
        args.perturb_steps,
        args.react_steps,
        args.fall_steps,
        args.native_hz,
    )

    perturb_start_time_s = (args.stand_steps + args.walk_steps) / args.native_hz
    react_start_time_s = (args.stand_steps + args.walk_steps + args.perturb_steps) / args.native_hz
    fall_start_time_s = (
        args.stand_steps + args.walk_steps + args.perturb_steps + args.react_steps
    ) / args.native_hz

    metadata = {
        "scenario_id": args.scenario_id,
        "scenario_description": args.description,
        "age": args.age,
        "height": args.height,
        "sex": args.sex,
        "weight": args.weight,
        "body_mass_kg": args.weight,
        "sampling_hz": args.output_hz,
        "native_hz": args.native_hz,
        "native_dt": 1.0 / args.native_hz,
        "imu_location": "lower_back_L1_L2_proxy",
        "imu_proxy_body": "Torso",
        "perturb_start_time_s": perturb_start_time_s,
        "react_start_time_s": react_start_time_s,
        "fall_start_time_s": fall_start_time_s,
        "fall_start_index": int(round(fall_start_time_s * args.native_hz)),
    }

    print("=" * 80)
    print("POST-PROCESS IMU CSV")
    print(f"Folder       : {folder}")
    print(f"IMU CSV files: {len(csvs)}")
    print(f"Perturb start: {perturb_start_time_s:.3f} s")
    print(f"React start  : {react_start_time_s:.3f} s")
    print(f"Fall start   : {fall_start_time_s:.3f} s")
    print("=" * 80)

    enriched_files = []
    for csv in csvs:
        out = csv.with_name(csv.stem + "_enriched.csv")
        print(f"[process] {csv}")
        result = export_enriched_imu_dataset(
            original_csv=csv,
            output_csv=out,
            metadata=metadata,
            event_summary={},
            phase_log=phase_log,
            sampling_hz=args.output_hz,
            native_dt=1.0 / args.native_hz,
        )
        enriched_files.append(Path(result["filename"]))
        print(f"[saved]   {result['filename']}")
        print(f"[meta]    {result['metadata_json']}")

    print("\nAXIS CHECK")
    for f in enriched_files:
        axis = verify_axis_from_csv(f)
        print(f"\n{f}")
        print(axis.get("interpretation", axis))


if __name__ == "__main__":
    main()
