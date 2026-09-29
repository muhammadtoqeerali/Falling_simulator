#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT="/mnt/hdd16T/ToqeerHomeBackup/mujoco_project"

# Expanded42 publication batch does not require rendered videos.
# This disables only MuJoCo native video capture; simulation physics,
# IMU/truth generation, event detection and CSV exports remain active.
export MUJOCO_DISABLE_NATIVE_VIDEO=1

SIM_PY="$PROJECT/.venv/bin/python"
CNN_PY="/mnt/hdd16T/ToqeerHomeBackup/toqeer/protechto_env/bin/python"

PROTECHTO="/mnt/hdd16T/ToqeerHomeBackup/toqeer/Protechto-master"

STATE="$PROJECT/outputs/expanded42_v2_master_pipeline"
STAGES="$STATE/stages"

mkdir -p "$STAGES"

cd "$PROJECT"

timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

say() {
    echo
    echo "===================================================================================================="
    echo "[$(timestamp)] $*"
    echo "===================================================================================================="
}

mark() {
    local name="$1"
    printf '%s\n' "$(timestamp)" > "$STAGES/${name}.COMPLETE"
}

done_stage() {
    test -f "$STAGES/$1.COMPLETE"
}

run_stage() {
    local name="$1"
    shift

    if done_stage "$name"; then
        say "SKIP COMPLETE STAGE: $name"
        return
    fi

    say "START STAGE: $name"

    "$@"

    mark "$name"

    say "COMPLETE STAGE: $name"
}


preflight() {
    say "EXPANDED42 V2 MASTER PREFLIGHT"

    test -x "$SIM_PY"
    test -x "$CNN_PY"

    required=(
        run_rne_expanded_42_profile_campaign_v2.py
        build_expanded42_canonical_event_manifest_v2.py
        build_phase2_corrected_event_dataset_expanded42_v2.py
        prepare_protechto_exact_synthetic_cache_expanded42_v2.py
        run_protechto_exact_full_campaign_expanded42_v2.py
        collect_protechto_exact_results_expanded42_v2.py
        audit_expanded42_final_dataflow_v2.py
    )

    for f in "${required[@]}"; do
        if [ ! -f "$PROJECT/$f" ]; then
            echo "MISSING REQUIRED FILE: $PROJECT/$f"
            exit 1
        fi
    done

    echo
    echo "Python syntax:"
    "$SIM_PY" -m py_compile \
        run_rne_expanded_42_profile_campaign_v2.py \
        build_expanded42_canonical_event_manifest_v2.py \
        build_phase2_corrected_event_dataset_expanded42_v2.py \
        prepare_protechto_exact_synthetic_cache_expanded42_v2.py \
        run_protechto_exact_full_campaign_expanded42_v2.py \
        collect_protechto_exact_results_expanded42_v2.py \
        audit_expanded42_final_dataflow_v2.py

    echo "  PASS"

    echo
    echo "Simulator environment:"
    "$SIM_PY" - <<'PY'
import sys
import numpy
import pandas
import scipy
import torch
import mujoco

print("python :", sys.executable)
print("numpy  :", numpy.__version__)
print("pandas :", pandas.__version__)
print("scipy  :", scipy.__version__)
print("torch  :", torch.__version__)
print("mujoco :", mujoco.__version__)
print("SIM ENV IMPORT GATE: PASS")
PY

    echo
    echo "Protechto/CNN environment:"
    "$CNN_PY" - <<'PY'
import sys
import numpy
import pandas
import sklearn
import joblib
import torch
import lightning

print("python    :", sys.executable)
print("numpy     :", numpy.__version__)
print("pandas    :", pandas.__version__)
print("sklearn   :", sklearn.__version__)
print("joblib    :", joblib.__version__)
print("torch     :", torch.__version__)
print("lightning :", lightning.__version__)
print("CUDA      :", torch.cuda.is_available())
print("GPU count :", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")

if torch.cuda.device_count() < 2:
    raise SystemExit(
        "V2 campaign inherits GPU=1 and therefore requires at least two visible GPUs"
    )

for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

print("CNN ENVIRONMENT / GPU GATE: PASS")
PY

    echo
    echo "Frozen source hashes:"

    "$SIM_PY" - <<'PY'
from pathlib import Path
import hashlib

root = Path("/mnt/hdd16T/ToqeerHomeBackup/mujoco_project")

expected = {
    "publication_snapshot_2026_09/dataset_generation/run_rne_corrected_396_campaign.py":
        "001bff4bd97b527396c27ad81f0e8584214a22d82b09b132566286dd7d21a978",

    "publication_snapshot_2026_09/dataset_generation/build_phase2_corrected_event_dataset.py":
        "9476d48164765911ea887b17396f733d9ee95a3bada3fe86234e1777a908df49",

    "publication_snapshot_2026_09/dataset_generation/prepare_protechto_exact_synthetic_cache.py":
        "cc6d5b4d34a086ef96cbde60b032a8d1828a94438eb2d5d65ad57152c805b648",

    "publication_snapshot_2026_09/cnn_evaluation/campaign/run_protechto_exact_full_campaign.py":
        "9e1ddb586374a9af879df4395fbc66895b06287bfe3a0f985e96765c2850b5c9",

    "publication_snapshot_2026_09/cnn_evaluation/campaign/collect_protechto_exact_results.py":
        "d28d973bf4ec31a858028af60dad0701fbe553006f72ac9bdd178bc7933cbd2d",
}

def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

for rel, exp in expected.items():
    p = root / rel
    if not p.exists():
        raise SystemExit(f"Missing frozen source: {p}")

    got = sha(p)
    print(rel)
    print(" expected:", exp)
    print(" actual  :", got)

    if got != exp:
        raise SystemExit(f"Frozen source hash mismatch: {rel}")

print("FROZEN SOURCE HASH GATE: PASS")
PY

    echo
    echo "Current V2 source hashes:"
    sha256sum \
        run_rne_expanded_42_profile_campaign_v2.py \
        build_expanded42_canonical_event_manifest_v2.py \
        build_phase2_corrected_event_dataset_expanded42_v2.py \
        prepare_protechto_exact_synthetic_cache_expanded42_v2.py \
        run_protechto_exact_full_campaign_expanded42_v2.py \
        collect_protechto_exact_results_expanded42_v2.py \
        audit_expanded42_final_dataflow_v2.py \
        | tee "$STATE/V2_SOURCE_SHA256.txt"

    echo
    echo "Storage:"
    df -h "$PROJECT" || true

    echo
    echo "GPU snapshot:"
    nvidia-smi \
        --query-gpu=index,name,utilization.gpu,memory.used,memory.total \
        --format=csv,noheader \
        || true

    echo
    echo "PREFLIGHT: PASS"
}


validate_generation() {
    "$SIM_PY" - <<'PY'
from pathlib import Path
import pandas as pd

root = Path(
    "/mnt/hdd16T/ToqeerHomeBackup/mujoco_project/"
    "outputs/_highrate_overnight/"
    "campaign_highrate_truth_v4_expanded42_new20"
)

progress = root / "campaign_progress.csv"
manifest = root / "campaign_run_manifest.csv"

if not progress.exists():
    raise SystemExit(f"Missing: {progress}")

if not manifest.exists():
    raise SystemExit(f"Missing: {manifest}")

p = pd.read_csv(progress)
m = pd.read_csv(manifest)

print("Progress rows :", len(p))
print("Manifest rows :", len(m))
print()
print("Statuses:")
print(p["status"].value_counts(dropna=False).to_string())

if len(p) != 360:
    raise SystemExit(
        f"Expected 360 progress rows, got {len(p)}"
    )

if len(m) != 360:
    raise SystemExit(
        f"Expected 360 run-manifest rows, got {len(m)}"
    )

if p.duplicated(
    ["profile_id", "scenario_id"]
).any():
    raise SystemExit(
        "Duplicate profile/task rows in progress"
    )

if m.duplicated(
    ["profile_id", "scenario_id"]
).any():
    raise SystemExit(
        "Duplicate profile/task rows in manifest"
    )

if not p["status"].astype(str).eq("completed").all():
    raise SystemExit(
        "Not every simulation is status=completed. "
        "Rerunning the master pipeline will retry failed rows."
    )

truth = (
    p["truth_qc_ok"]
    .astype(str)
    .str.lower()
    .isin(["true", "1", "yes"])
)

if not truth.all():
    raise SystemExit(
        "Not every simulation passed high-rate truth QC."
    )

profiles = sorted(
    p["profile_id"]
    .astype(str)
    .unique()
)

expected_profiles = [
    f"P{i:03d}"
    for i in range(36, 56)
]

if profiles != expected_profiles:
    raise SystemExit(
        f"Unexpected new profile IDs: {profiles}"
    )

tasks = sorted(
    pd.to_numeric(
        p["scenario_id"],
        errors="raise",
    )
    .astype(int)
    .unique()
    .tolist()
)

expected_tasks = [
    20,21,22,23,24,
    28,29,30,31,32,33,34,
    37,38,39,40,41,42,
]

if tasks != expected_tasks:
    raise SystemExit(
        f"Unexpected task set: {tasks}"
    )

per_profile = (
    p.groupby("profile_id")
    .size()
)

if not (per_profile == 18).all():
    raise SystemExit(
        f"Not every profile has 18 tasks:\n{per_profile}"
    )

print()
print("360-RUN GENERATION GATE: PASS")
PY
}


stage_generation() {
    "$SIM_PY" -u \
        "$PROJECT/run_rne_expanded_42_profile_campaign_v2.py" \
        --retry-failed

    validate_generation
}


stage_event_manifest() {
    "$SIM_PY" -u \
        "$PROJECT/build_expanded42_canonical_event_manifest_v2.py"

    test -f \
        "$PROJECT/outputs/phase2_publication_gate_expanded42_v2/final_synthetic_event_policy_v2/COMPLETE"
}


stage_dataset() {
    "$SIM_PY" -u \
        "$PROJECT/build_phase2_corrected_event_dataset_expanded42_v2.py"

    test -f \
        "$PROJECT/outputs/phase2_corrected_event_dataset_expanded42_v2/synthetic_X.npy"

    test -f \
        "$PROJECT/outputs/phase2_corrected_event_dataset_expanded42_v2/synthetic_y.npy"
}


stage_cache() {
    "$CNN_PY" -u \
        "$PROJECT/prepare_protechto_exact_synthetic_cache_expanded42_v2.py"

    test -f \
        "$PROJECT/outputs/protechto_exact_synthetic_cache_expanded42_v2/COMPLETE"
}


stage_cnn_campaign() {
    "$CNN_PY" -u \
        "$PROJECT/run_protechto_exact_full_campaign_expanded42_v2.py"

    test -f \
        "$PROJECT/outputs/protechto_exact_full_campaign_expanded42_v2/FINAL_CAMPAIGN_COMPLETE"
}


stage_collect() {
    "$CNN_PY" -u \
        "$PROJECT/collect_protechto_exact_results_expanded42_v2.py"

    test -f \
        "$PROJECT/outputs/protechto_exact_full_campaign_expanded42_v2/final_results_posthoc/POOLED_ALL_RESULTS.csv"
}


stage_audit() {
    "$CNN_PY" -u \
        "$PROJECT/audit_expanded42_final_dataflow_v2.py"

    test -f \
        "$PROJECT/outputs/final_dataset_split_audit_expanded42_v2/FINAL_EXPANDED42_DATAFLOW_LEAKAGE_EVIDENCE_RECORD.txt"
}


preflight

if [ "${1:-}" = "--preflight-only" ]; then
    say "PREFLIGHT-ONLY REQUEST COMPLETE"
    exit 0
fi


run_stage \
    01_old396_event_policy_regression \
    "$SIM_PY" -u \
    "$PROJECT/build_expanded42_canonical_event_manifest_v2.py" \
    --validate-old-only


run_stage \
    02_generate_new360 \
    stage_generation


run_stage \
    03_build_expanded756_event_manifest \
    stage_event_manifest


run_stage \
    04_build_expanded_event_dataset \
    stage_dataset


run_stage \
    05_prepare_expanded_protechto_cache \
    stage_cache


run_stage \
    06_run_six_condition_cnn_campaign \
    stage_cnn_campaign


run_stage \
    07_collect_final_results \
    stage_collect


run_stage \
    08_final_dataflow_leakage_audit \
    stage_audit


say "EXPANDED42 V2 FULL PIPELINE COMPLETE"

echo
echo "Final results:"
echo "  $PROJECT/outputs/protechto_exact_full_campaign_expanded42_v2/final_results_posthoc"

echo
echo "Final audit:"
echo "  $PROJECT/outputs/final_dataset_split_audit_expanded42_v2"

echo
echo "Final evidence record:"
echo "  $PROJECT/outputs/final_dataset_split_audit_expanded42_v2/FINAL_EXPANDED42_DATAFLOW_LEAKAGE_EVIDENCE_RECORD.txt"
