# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from imu_dataset_tools import verify_axis_from_csv


def main():
    ap = argparse.ArgumentParser(description="Verify IMU axis direction/sign from a generated CSV.")
    ap.add_argument("csv", help="Path to original or enriched IMU CSV")
    ap.add_argument("--n", type=int, default=200, help="Number of first samples used as standing/calibration window")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not args.csv.strip() or not csv_path.exists() or csv_path.is_dir():
        print("[axis-check][error] No valid CSV file was provided.")
        print("Use:")
        print("  CSV_FILE=$(find outputs -name '*enriched*.csv' | sort | tail -1)")
        print("  echo \"$CSV_FILE\"")
        print("  python3 verify_imu_axis.py \"$CSV_FILE\"")
        sys.exit(2)

    result = verify_axis_from_csv(csv_path, n_standing_samples=args.n)
    print(json.dumps(result, indent=2))

    out = csv_path.with_suffix(".axis_check.json")
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n[axis-check] Saved: {out}")


if __name__ == "__main__":
    main()
