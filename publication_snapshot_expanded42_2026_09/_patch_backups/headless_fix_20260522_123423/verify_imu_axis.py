# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse, json
from pathlib import Path
from imu_dataset_tools import verify_axis_from_csv


def main():
    ap = argparse.ArgumentParser(description="Verify IMU axis direction/sign from a generated CSV.")
    ap.add_argument("csv", help="Path to original or enriched IMU CSV")
    ap.add_argument("--n", type=int, default=200, help="Number of first samples used as standing/calibration window")
    args = ap.parse_args()

    result = verify_axis_from_csv(args.csv, n_standing_samples=args.n)
    print(json.dumps(result, indent=2))

    out = Path(args.csv).with_suffix(".axis_check.json")
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n[axis-check] Saved: {out}")


if __name__ == "__main__":
    main()
