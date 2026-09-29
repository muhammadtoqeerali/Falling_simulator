from pathlib import Path
import argparse
import subprocess
import sys


ROOT = Path(
    "/mnt/hdd16T/ToqeerHomeBackup/mujoco_project"
)

SOURCE = ROOT / "validate_simulator_against_real_v2.py"


ap = argparse.ArgumentParser()

ap.add_argument(
    "--sim-dataset",
    required=True,
)

ap.add_argument(
    "--out-name",
    required=True,
)

args = ap.parse_args()


dataset = ROOT / "outputs" / args.sim_dataset

report = (
    dataset
    / "conversion_reports"
    / "conversion_report_all_subjects.csv"
)

out = (
    ROOT
    / "outputs/validation_v2"
    / args.out_name
)

generated = (
    ROOT
    / "outputs/validation_v2"
    / f"_generated_{args.out_name}.py"
)


print("=" * 100)
print("PUBLICATION-AWARE V2 VARIANT")
print("=" * 100)
print("dataset :", dataset)
print("report  :", report)
print("output  :", out)


if not SOURCE.exists():
    raise SystemExit(
        f"Missing source validator:\n{SOURCE}"
    )

if not report.exists():
    raise SystemExit(
        f"Missing simulated report:\n{report}"
    )

if out.exists():
    raise SystemExit(
        "STOP: result directory already exists:\n"
        f"{out}\n"
        "Nothing was overwritten."
    )


src = SOURCE.read_text(
    encoding="utf-8"
)


# ============================================================
# 1. Simulated dataset path
# ============================================================

old_dataset = '"outputs/simulated_dataset_all_subjects/"'
new_dataset = f'"outputs/{args.sim_dataset}/"'

n = src.count(old_dataset)

if n != 1:
    raise SystemExit(
        "Expected one simulated dataset path, "
        f"found {n}"
    )

src = src.replace(
    old_dataset,
    new_dataset,
    1,
)


# ============================================================
# 2. Output directory
# ============================================================

old_out = '"outputs/validation_v2/results_core"'
new_out = f'"outputs/validation_v2/{args.out_name}"'

n = src.count(old_out)

if n != 1:
    raise SystemExit(
        "Expected one V2 output path, "
        f"found {n}"
    )

src = src.replace(
    old_out,
    new_out,
    1,
)


# ============================================================
# 3. Protocol-review tasks
# ============================================================

old_protocol = "PROTOCOL_REVIEW = {41, 42}"
new_protocol = "PROTOCOL_REVIEW = {39, 40, 41, 42}"

n = src.count(old_protocol)

if n != 1:
    raise SystemExit(
        "Expected one PROTOCOL_REVIEW definition, "
        f"found {n}"
    )

src = src.replace(
    old_protocol,
    new_protocol,
    1,
)


# ============================================================
# 4. Patch ONLY simulated impact input.
# ============================================================

sim_marker = (
    "# ============================================================\n"
    "# LOAD SIMULATED INVENTORY\n"
    "# ============================================================"
)

if sim_marker not in src:
    raise SystemExit(
        "Could not locate simulated inventory section"
    )


before_sim, sim_and_after = src.split(
    sim_marker,
    1,
)


old_impact = "        impact=r.fall_impact_frame,"

new_impact = (
    "        impact=(\n"
    "            r.fall_impact_frame_secondary\n"
    "            if hasattr(r, \"fall_impact_frame_secondary\")\n"
    "            else r.fall_impact_frame\n"
    "        ),"
)


# Real section should retain its physical impact.
real_count = before_sim.count(
    old_impact
)

# Simulated section should contain exactly one occurrence.
sim_count = sim_and_after.count(
    old_impact
)

print(
    "real impact argument occurrences:",
    real_count,
)

print(
    "sim impact argument occurrences :",
    sim_count,
)


if real_count != 1:
    raise SystemExit(
        "Expected exactly one physical impact argument "
        f"before simulated section; found {real_count}"
    )

if sim_count != 1:
    raise SystemExit(
        "Expected exactly one simulated impact argument "
        f"in simulated section; found {sim_count}"
    )


sim_and_after = sim_and_after.replace(
    old_impact,
    new_impact,
    1,
)

src = (
    before_sim
    + sim_marker
    + sim_and_after
)


# ============================================================
# 5. Remove fall_duration from task discriminability.
# ============================================================

disc_marker = (
    "# ============================================================\n"
    "# MATCHED-TASK VS WRONG-TASK DISCRIMINABILITY"
)

if disc_marker not in src:
    raise SystemExit(
        "Could not locate discriminability section"
    )

insert = (
    "# Publication-QC: task discriminability uses only "
    "onset-centered inertial features.\n"
    "DISCRIM_FEATURES = [\n"
    "    x for x in DISCRIM_FEATURES\n"
    "    if x != \"fall_duration_s\"\n"
    "]\n\n"
)

src = src.replace(
    disc_marker,
    insert + disc_marker,
    1,
)


# ============================================================
# Static safety checks
# ============================================================

# Physical loader must still use its original impact.
if before_sim.count(
    "impact=r.fall_impact_frame,"
) != 1:
    raise SystemExit(
        "Physical impact loader changed unexpectedly"
    )

# Generated source must contain the QC field.
if (
    "r.fall_impact_frame_secondary"
    not in src
):
    raise SystemExit(
        "Secondary impact gating was not inserted"
    )


generated.parent.mkdir(
    parents=True,
    exist_ok=True,
)

generated.write_text(
    src,
    encoding="utf-8",
)


print()
print(
    "Generated validator:",
    generated,
)

print("Running...")
print()


proc = subprocess.run(
    [
        sys.executable,
        str(generated),
    ],
    cwd=str(ROOT),
)


if proc.returncode != 0:
    raise SystemExit(
        f"V2 failed with return code {proc.returncode}"
    )


tables = out / "tables"

required = [
    "features_all_trials_v2.csv",
    "dataset_qc_summary.csv",
    "task_feature_summary.csv",
    "task_distribution_metrics.csv",
    "task_median_waveform_similarity.csv",
]

missing = [
    name
    for name in required
    if not (
        tables / name
    ).exists()
]

if missing:
    raise SystemExit(
        "V2 completed but required tables are missing:\n"
        + "\n".join(missing)
    )


print()
print("=" * 100)
print("PUBLICATION_V2_VARIANT_OK")
print("=" * 100)
print("Dataset :", args.sim_dataset)
print("Results :", out)
