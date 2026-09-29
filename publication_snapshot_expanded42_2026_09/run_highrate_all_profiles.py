from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time

import pandas as pd


ROOT = Path("/mnt/hdd16T/ToqeerHomeBackup/mujoco_project")

REPORT = (
    ROOT
    / "outputs"
    / "simulated_dataset_all_subjects"
    / "conversion_reports"
    / "conversion_report_all_subjects.csv"
)

CAMPAIGN = (
    ROOT
    / "outputs"
    / "_highrate_overnight"
    / "campaign_highrate_truth_v1"
)

EXPECTED_AGES = {
    25, 28, 32, 33, 35, 37, 38,
    41, 46, 55, 60, 62, 65,
}


def token_float(v):
    return float(
        str(v).strip().replace("p", ".")
    )


def load_profiles():
    if not REPORT.exists():
        raise RuntimeError(
            f"Previous conversion report not found:\n{REPORT}"
        )

    df = pd.read_csv(REPORT)

    required = [
        "age",
        "height_token",
        "sex",
        "weight_token",
    ]

    missing = [
        x for x in required
        if x not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing columns in conversion report: {missing}"
        )

    rows = []

    for _, r in df[required].dropna().iterrows():
        rows.append(
            (
                int(r["age"]),
                token_float(r["height_token"]),
                str(r["sex"]).strip().lower(),
                token_float(r["weight_token"]),
            )
        )

    profiles = sorted(set(rows))

    if len(profiles) != 13:
        raise RuntimeError(
            f"Expected 13 unique anthropometric profiles, "
            f"found {len(profiles)}:\n{profiles}"
        )

    ages = {x[0] for x in profiles}

    if ages != EXPECTED_AGES:
        raise RuntimeError(
            "Age/profile set differs from previous dataset.\n"
            f"Expected: {sorted(EXPECTED_AGES)}\n"
            f"Found:    {sorted(ages)}"
        )

    for age in sorted(ages):
        same_age = [
            x for x in profiles
            if x[0] == age
        ]

        if len(same_age) != 1:
            raise RuntimeError(
                f"Age {age} has multiple parameter profiles: "
                f"{same_age}"
            )

    return profiles


def profile_key(age, height, sex, weight):
    h = str(height).replace(".", "p")
    w = str(weight).replace(".", "p")

    return (
        f"age{age}_h{h}_"
        f"sex_{sex}_w{w}"
    )


def count_highrate():
    return len(
        list(
            (ROOT / "outputs").rglob(
                "*_highrate_truth.csv"
            )
        )
    )


def compile_preflight():
    files = [
        ROOT / "fall_core.py",
        ROOT / "backward_fall_walking_best.py",
        ROOT / "high_rate_imu_sidecar.py",
        ROOT / "highrate_runtime_bridge.py",
    ]

    files += sorted(
        ROOT.glob("scenario*_legacy.py")
    )

    cmd = [
        str(ROOT / ".venv/bin/python"),
        "-m",
        "py_compile",
    ] + [str(x) for x in files]

    print(
        "\nRunning compile preflight...",
        flush=True,
    )

    rc = subprocess.run(
        cmd,
        cwd=ROOT,
    ).returncode

    if rc != 0:
        raise RuntimeError(
            f"Compile preflight failed: rc={rc}"
        )

    print("Compile preflight OK", flush=True)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--preview",
        action="store_true",
    )

    args = parser.parse_args()

    profiles = load_profiles()

    print("=" * 90)
    print("HIGH-RATE FULL PROFILE CAMPAIGN")
    print("=" * 90)

    print(
        f"Previous profile report:\n{REPORT}"
    )

    print("\nProfiles to run:")
    print(
        " age   height   sex      weight"
    )
    print(
        " ----  ------   -------  ------"
    )

    for age, height, sex, weight in profiles:
        print(
            f" {age:>4}   "
            f"{height:>6.2f}   "
            f"{sex:<7}  "
            f"{weight:>6.1f}"
        )

    sexes = {}

    for _, _, sex, _ in profiles:
        sexes[sex] = sexes.get(sex, 0) + 1

    print(
        "\nSex distribution in ORIGINAL profiles:",
        sexes,
    )

    print(
        "Profile count:",
        len(profiles),
    )

    if args.preview:
        print("\nPROFILE_PREVIEW_OK")
        return

    CAMPAIGN.mkdir(
        parents=True,
        exist_ok=True,
    )

    profile_csv = (
        CAMPAIGN / "profiles_used.csv"
    )

    pd.DataFrame(
        profiles,
        columns=[
            "age",
            "height",
            "sex",
            "weight",
        ],
    ).to_csv(
        profile_csv,
        index=False,
    )

    compile_preflight()

    before_all = count_highrate()

    print(
        "\nHigh-rate files before campaign:",
        before_all,
    )

    status = []

    python = str(
        ROOT / ".venv/bin/python"
    )

    for index, (
        age,
        height,
        sex,
        weight,
    ) in enumerate(profiles, 1):

        key = profile_key(
            age,
            height,
            sex,
            weight,
        )

        done = (
            CAMPAIGN
            / f"{key}.done"
        )

        failed = (
            CAMPAIGN
            / f"{key}.failed"
        )

        running = (
            CAMPAIGN
            / f"{key}.running"
        )

        print("\n" + "=" * 90)
        print(
            f"PROFILE {index}/{len(profiles)}"
        )
        print("=" * 90)

        print(
            f"age={age} "
            f"height={height:.2f} "
            f"sex={sex} "
            f"weight={weight:.1f}"
        )

        if done.exists():
            print(
                "Already completed in this campaign; SKIP."
            )

            status.append({
                "age": age,
                "height": height,
                "sex": sex,
                "weight": weight,
                "status": "SKIPPED_DONE",
                "rc": 0,
            })

            continue

        failed.unlink(
            missing_ok=True
        )

        running.write_text(
            f"started={time.strftime('%Y-%m-%d %H:%M:%S')}\n",
            encoding="utf-8",
        )

        before = count_highrate()

        cmd = [
            python,
            "-u",
            "batch_run_all_labeled.py",
            "--age",
            str(age),
            "--height",
            str(height),
            "--sex",
            str(sex),
            "--weight",
            str(weight),
            "--continue-on-error",
            "--display",
            os.environ.get("DISPLAY", ":99"),
        ]

        print(
            "\nCOMMAND:",
            " ".join(cmd),
            flush=True,
        )

        rc = subprocess.run(
            cmd,
            cwd=ROOT,
            env=os.environ.copy(),
        ).returncode

        after = count_highrate()

        running.unlink(
            missing_ok=True
        )

        generated = (
            after - before
        )

        row = {
            "age": age,
            "height": height,
            "sex": sex,
            "weight": weight,
            "rc": rc,
            "new_highrate_files": generated,
        }

        if rc == 0:
            done.write_text(
                json.dumps(
                    row,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            row["status"] = "DONE"

            print(
                f"\nPROFILE COMPLETE: "
                f"age={age}, "
                f"new high-rate files={generated}"
            )

        else:
            failed.write_text(
                json.dumps(
                    row,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            row["status"] = "FAILED"

            print(
                f"\nPROFILE FAILED: "
                f"age={age}, rc={rc}; "
                "continuing to next profile."
            )

        status.append(row)

        pd.DataFrame(
            status
        ).to_csv(
            CAMPAIGN
            / "campaign_status.csv",
            index=False,
        )

    after_all = count_highrate()

    completed = sum(
        1
        for x in status
        if x["status"]
        in ("DONE", "SKIPPED_DONE")
    )

    failed_n = sum(
        1
        for x in status
        if x["status"] == "FAILED"
    )

    summary = {
        "profiles_total": len(profiles),
        "profiles_completed": completed,
        "profiles_failed": failed_n,
        "highrate_before": before_all,
        "highrate_after": after_all,
        "new_highrate_files": (
            after_all - before_all
        ),
        "sex_distribution": sexes,
    }

    (
        CAMPAIGN / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 90)
    print("CAMPAIGN SUMMARY")
    print("=" * 90)

    for k, v in summary.items():
        print(
            f"{k}: {v}"
        )

    print(
        "\nSummary:",
        CAMPAIGN / "summary.json",
    )

    print(
        "Status:",
        CAMPAIGN / "campaign_status.csv",
    )

    if failed_n:
        print(
            "\nCAMPAIGN_FINISHED_WITH_FAILURES"
        )
        sys.exit(1)

    print(
        "\nCAMPAIGN_COMPLETE_OK"
    )


if __name__ == "__main__":
    main()
