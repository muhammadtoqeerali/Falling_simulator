#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT = Path("/mnt/hdd16T/ToqeerHomeBackup/mujoco_project")
PROTECHTO = Path("/mnt/hdd16T/ToqeerHomeBackup/toqeer/Protechto-master")
PROTECHTO_PY = Path("/mnt/hdd16T/ToqeerHomeBackup/toqeer/protechto_env/bin/python")

CAMPAIGN = PROJECT / "outputs/protechto_exact_full_campaign_v1"
POST = CAMPAIGN / "final_results_posthoc"

SYNTH_DATASET = PROJECT / "outputs/phase2_corrected_event_dataset_v1"
SYNTH_CACHE = PROJECT / "outputs/protechto_exact_synthetic_cache_v1"

PUB_GATE = PROJECT / "outputs/phase2_publication_gate_audit_20260826"
EVENT_MANIFEST = (
    PUB_GATE
    / "final_synthetic_event_policy_v2"
    / "canonical_synthetic_event_manifest_v2.csv"
)

CORRECTED396 = (
    PROJECT
    / "outputs/_highrate_overnight"
    / "campaign_highrate_truth_v3_rne_corrected396"
)

PREFLIGHT_AUDIT = PROJECT / "outputs/protechto_clean_lightweight_pipeline_audit.txt"

PAPER_PHYSICAL = [
    "EXP01_REAL_ONLY",
    "EXP04_MIX20",
    "EXP05_MIX50",
    "EXP06_MIX70",
    "EXP07_MIX100",
]

PAPER_SEGMENT = [
    "EXP01_REAL_ONLY",
    "EXP03_SIM_ONLY_FULL",
    "EXP04_MIX20",
    "EXP05_MIX50",
    "EXP06_MIX70",
    "EXP07_MIX100",
]

EXCLUDED_DIAGNOSTIC = "EXP02_SIM_FALL_SUBSTITUTION"

DISPLAY = {
    "EXP01_REAL_ONLY": "REAL_ONLY",
    "EXP02_SIM_FALL_SUBSTITUTION": "REAL_ACTIVITY + SIM_FALLING",
    "EXP03_SIM_ONLY_FULL": "SIM_ACTIVITY + SIM_FALLING",
    "EXP04_MIX20": "REAL + SIM20",
    "EXP05_MIX50": "REAL + SIM50",
    "EXP06_MIX70": "REAL + SIM70",
    "EXP07_MIX100": "REAL + SIM100",
}


def die(msg: str) -> None:
    raise RuntimeError(msg)


def sha256_file(path: Path, chunk: int = 4 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_tree(src: Path, dst: Path, ignore_names=()) -> None:
    if not src.exists():
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(
        src,
        dst,
        ignore=shutil.ignore_patterns(*ignore_names) if ignore_names else None,
    )


def copy_source_tree_filtered(src: Path, dst: Path) -> int:
    allowed = {
        ".py", ".sh", ".json", ".yaml", ".yml", ".toml",
        ".md", ".txt", ".xml", ".csv",
    }
    n = 0
    if not src.exists():
        return n
    for p in src.rglob("*"):
        if not p.is_file():
            continue
        if any(part.startswith(".venv") for part in p.parts):
            continue
        if "outputs" in p.parts or "__pycache__" in p.parts:
            continue
        if p.suffix.lower() not in allowed:
            continue
        rel = p.relative_to(src)
        copy_file(p, dst / rel)
        n += 1
    return n


def safe_cmd(cmd: list[str]) -> str:
    try:
        r = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        return r.stdout
    except Exception as e:
        return f"COMMAND FAILED: {cmd!r}\n{e!r}\n"


def percent(x) -> float:
    return 100.0 * float(x)


def last_bytes(path: Path, nbytes: int = 2_000_000) -> bytes:
    if not path.exists():
        return b""
    size = path.stat().st_size
    with path.open("rb") as f:
        if size > nbytes:
            f.seek(size - nbytes)
        return f.read()


def require(path: Path, label: str) -> None:
    if not path.exists():
        die(f"Missing required {label}: {path}")



def dataframe_to_markdown(df, floatfmt=".2f"):
    # Small dependency-free markdown table formatter.
    cols = [str(c) for c in df.columns]
    rows = []

    for _, r in df.iterrows():
        vals = []
        for c in df.columns:
            v = r[c]
            if isinstance(v, float):
                if v != v:
                    vals.append("N/A")
                else:
                    if floatfmt.startswith(".") and floatfmt.endswith("f"):
                        vals.append(format(v, floatfmt))
                    else:
                        vals.append(str(v))
            else:
                vals.append(str(v))
        rows.append(vals)

    def esc(x):
        return str(x).replace("|", r"\|").replace("\n", " ")

    lines = []
    lines.append("| " + " | ".join(esc(x) for x in cols) + " |")
    lines.append("| " + " | ".join("---" for _ in cols) + " |")
    for row in rows:
        lines.append("| " + " | ".join(esc(x) for x in row) + " |")
    return "\n".join(lines)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def plot_grouped_metrics(df: pd.DataFrame, conditions: list[str], metrics: list[tuple[str, str]], title: str, ylabel: str, outbase: Path):
    sub = df[df["condition"].isin(conditions)].copy()
    order = {c: i for i, c in enumerate(conditions)}
    sub["_order"] = sub["condition"].map(order)
    sub = sub.sort_values("_order")

    x = np.arange(len(sub))
    width = 0.8 / len(metrics)

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    for i, (col, label) in enumerate(metrics):
        vals = 100.0 * sub[col].astype(float).to_numpy()
        ax.bar(x + (i - (len(metrics)-1)/2) * width, vals, width, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[c] for c in sub["condition"]], rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    for ext in ("png", "pdf", "svg"):
        fig.savefig(outbase.with_suffix("." + ext), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(tn, fp, fn, tp, title: str, outbase: Path):
    cm = np.array([[tn, fp], [fn, tp]], dtype=float)
    fig, ax = plt.subplots(figsize=(5.2, 4.7))
    im = ax.imshow(cm)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Activity", "Falling"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Activity", "Falling"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    vmax = cm.max() if cm.size else 1.0
    for i in range(2):
        for j in range(2):
            val = int(cm[i, j])
            ax.text(j, i, f"{val:,}", ha="center", va="center")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(outbase.with_suffix("." + ext), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output-dir",
        default=str(PROJECT / "outputs/paper_evidence_archive_v1"),
        help="staging directory",
    )
    ap.add_argument(
        "--zip-path",
        default=str(PROJECT / "outputs/paper_evidence_archive_v1.zip"),
        help="final ZIP path",
    )
    ap.add_argument(
        "--include-checkpoints",
        action="store_true",
        help="include all 35 checkpoint binaries; can make ZIP very large",
    )
    args = ap.parse_args()

    stage = Path(args.output_dir)
    zip_path = Path(args.zip_path)

    print("=" * 116)
    print("BUILD PAPER EVIDENCE ARCHIVE")
    print("=" * 116)

    require(CAMPAIGN / "RUN_ID", "campaign RUN_ID")
    require(POST / "POOLED_ALL_RESULTS.csv", "pooled results")
    require(POST / "ALL_FOLD_METRICS.csv", "fold metrics")
    require(EVENT_MANIFEST, "canonical synthetic event manifest")

    run_id = (CAMPAIGN / "RUN_ID").read_text().strip()
    result_root = PROTECHTO / "results/CNN/300ms" / run_id
    checkpoint_root = PROTECHTO / "checkpoints/CNN/300ms" / run_id

    require(result_root, "Protechto result root")
    require(checkpoint_root, "checkpoint root")

    markers = sorted(CAMPAIGN.glob("conditions/*/folds/fold_*.COMPLETE"))
    if len(markers) != 35:
        die(f"Expected 35 completed folds, found {len(markers)}")

    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    d00 = stage / "00_README_AND_MANIFESTS"
    d01 = stage / "01_METHOD_PROTECHTO"
    d02 = stage / "02_METHOD_SIMULATOR"
    d03 = stage / "03_CAMPAIGN_RESULTS"
    d04 = stage / "04_PAPER_TABLES"
    d05 = stage / "05_FIGURES_AND_PLOT_DATA"
    d06 = stage / "06_VALIDATION_AND_AUDITS"
    d07 = stage / "07_ENVIRONMENT_AND_REPRODUCIBILITY"
    d08 = stage / "08_EXCLUDED_DIAGNOSTICS"

    for d in (d00,d01,d02,d03,d04,d05,d06,d07,d08):
        d.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------------------------
    # README + policy
    # ----------------------------------------------------------------------------------
    readme = f"""# Paper Evidence Archive

Run ID: `{run_id}`

This archive is the paper-writing evidence package for the corrected MuJoCo fall simulator
and the exact lightweight Protechto CNN campaign.

## Primary manuscript conditions

Segment-level primary conditions:
- REAL_ONLY
- SIM_ACTIVITY + SIM_FALLING (held-out simulated trials; internal synthetic-domain diagnostic)
- REAL + SIM20
- REAL + SIM50
- REAL + SIM70
- REAL + SIM100

Event-level physical-test primary conditions:
- REAL_ONLY
- REAL + SIM20
- REAL + SIM50
- REAL + SIM70
- REAL + SIM100

For the SIM_ONLY event diagnostic, the archive reports only fall-event sensitivity
(detected fall trajectories / all held-out fall trajectories). Precision, specificity,
balanced accuracy, and F1 are deliberately not used for this diagnostic because the
synthetic event test contains no independent Activity-only trajectories.

## Excluded diagnostic

`EXP02_SIM_FALL_SUBSTITUTION` is retained under `08_EXCLUDED_DIAGNOSTICS` for transparency
and reproducibility, but is excluded from the primary manuscript tables because it tests a
different domain-substitution question (real Activity combined with synthetic Falling),
not the intended real+synthetic augmentation setting.

## Primary metric convention

Pooled five-fold metrics are the primary descriptive results because the physical folds
have unequal numbers of windows. Five-fold mean ± SD is retained as fold-variability
support.

## Integrity

`FILE_MANIFEST_SHA256.csv` contains size and SHA256 for every staged file before ZIP creation.
`CHECKPOINT_MANIFEST.csv` records all checkpoint paths, sizes, and SHA256 hashes. By default
checkpoint binaries are not copied into this paper-writing ZIP because they are large.
Use `--include-checkpoints` to include them.
"""
    write_text(d00 / "README.md", readme)

    policy = """# Reporting Policy

1. Segment-level table: report full Activity/Falling precision, recall, F1 for all primary
   conditions, including the synthetic-only internal-domain diagnostic.
2. Event-level physical table: report Fall precision, Fall recall/detection, and Fall F1 for
   REAL_ONLY and MIX20/50/70/100.
3. Synthetic-only event diagnostic: report only detected/total and sensitivity.
4. Do not report SIM_ONLY event precision=100% or F1=72.69% as primary manuscript metrics;
   the negative event class is absent, so those quantities are not comparable to the physical
   event tests.
5. EXP02 substitution is archived but not used in primary tables.
6. No manual 150-ms trim is part of the accepted pipeline.
"""
    write_text(d00 / "REPORTING_POLICY.md", policy)

    # ----------------------------------------------------------------------------------
    # Exact method source snapshot: canonical Protechto
    # ----------------------------------------------------------------------------------
    protechto_files = [
        "train.py",
        "test.py",
        "constants.py",
        "config.json",
        "requirements.txt",
        "dataloaders/KFoldDataloader.py",
        "dataloaders/helper.py",
        "simulation/SimulationKFoldDataloader.py",
        "simulation/helper.py",
        "simulation/testing_simulation_func.py",
        "models/CNN.py",
        "models/wrappers/Predictor.py",
        "models/wrappers/Simulator.py",
        "models/modules/IMUNormalizer.py",
        "models/configs/CNN_config.py",
        "preprocessing/windowing.py",
        "preprocessing/helper.py",
    ]
    for rel in protechto_files:
        src = PROTECHTO / rel
        if src.exists():
            copy_file(src, d01 / "canonical_source" / rel)

    exec_mirror = PROJECT / "outputs/protechto_clean_execution_mirror_v1"
    if (exec_mirror / "train.py").exists():
        copy_file(exec_mirror / "train.py", d01 / "execution_mirror" / "train.py")

    campaign_scripts = [
        "prepare_protechto_execution_mirror.py",
        "preflight_protechto_clean_real_only.py",
        "prepare_protechto_exact_synthetic_cache.py",
        "preflight_protechto_exact_full_campaign.py",
        "run_protechto_exact_full_campaign.py",
        "launch_protechto_exact_full_campaign.sh",
        "collect_protechto_exact_results.py",
    ]
    for name in campaign_scripts:
        p = PROJECT / name
        if p.exists():
            copy_file(p, d01 / "campaign_scripts" / name)

    # ----------------------------------------------------------------------------------
    # Simulator method/source snapshot + key synthetic artifacts
    # ----------------------------------------------------------------------------------
    for p in PROJECT.iterdir():
        if p.is_file() and p.suffix.lower() in {".py",".sh",".json",".yaml",".yml",".toml",".md"}:
            copy_file(p, d02 / "project_root_source" / p.name)

    for dirname in ("scripts", "src", "simulator"):
        src = PROJECT / dirname
        if src.exists() and src.is_dir():
            copy_source_tree_filtered(src, d02 / "source_trees" / dirname)

    copy_file(EVENT_MANIFEST, d02 / "synthetic_event_policy" / EVENT_MANIFEST.name)

    for name in ("synthetic_X.npy", "synthetic_y.npy", "synthetic_metadata.csv"):
        p = SYNTH_DATASET / name
        if p.exists():
            copy_file(p, d02 / "synthetic_dataset" / name)

    if SYNTH_CACHE.exists():
        copy_tree(
            SYNTH_CACHE,
            d02 / "protechto_exact_synthetic_cache",
            ignore_names=("__pycache__", "*.pyc"),
        )

    # Publication-gate outputs are important validation evidence.
    # Copy all modest-sized textual/tabular/figure files, and modest-sized npy/npz.
    allowed_val = {
        ".csv",".json",".txt",".md",".yaml",".yml",".pdf",".png",".svg",".tex",".npy",".npz"
    }
    max_val_size = 100 * 1024 * 1024
    if PUB_GATE.exists():
        for p in PUB_GATE.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in allowed_val:
                continue
            if p.stat().st_size > max_val_size:
                continue
            rel = p.relative_to(PUB_GATE)
            copy_file(p, d06 / "simulator_publication_gate" / rel)

    # Corrected396: retain manifest/audit/summary/report/policy metadata without copying all raw trials.
    if CORRECTED396.exists():
        pat = re.compile(r"(manifest|audit|summary|validation|metric|report|policy|event|metadata)", re.I)
        for p in CORRECTED396.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in {".csv",".json",".txt",".md",".yaml",".yml",".pdf",".png",".svg",".tex"}:
                continue
            if not pat.search(p.name):
                continue
            if p.stat().st_size > max_val_size:
                continue
            rel = p.relative_to(CORRECTED396)
            copy_file(p, d06 / "corrected396_metadata" / rel)

    # ----------------------------------------------------------------------------------
    # Campaign result directories
    # ----------------------------------------------------------------------------------
    for cond in PAPER_SEGMENT:
        src = result_root / cond
        if src.exists():
            copy_tree(src, d03 / "conditions" / cond)

    # Preserve EXP02 separately, never silently delete it.
    exp02 = result_root / EXCLUDED_DIAGNOSTIC
    if exp02.exists():
        copy_tree(exp02, d08 / EXCLUDED_DIAGNOSTIC)
        write_text(
            d08 / "README.md",
            """# Excluded diagnostic

EXP02_SIM_FALL_SUBSTITUTION is preserved here for full transparency.
It is excluded from the primary paper tables because it tests synthetic Falling as a
replacement for real Falling while retaining real Activity, which is a distinct
cross-domain substitution experiment rather than the intended augmentation experiment.
The archive does not alter or delete its results.
""",
        )

    # Copy campaign result summaries and completion markers.
    for p in POST.glob("*"):
        if p.is_file():
            copy_file(p, d03 / "posthoc_full" / p.name)

    for p in markers:
        rel = p.relative_to(CAMPAIGN)
        copy_file(p, d06 / "completion_markers" / rel)

    for name in ("RUN_ID","POSTHOC_AGGREGATION_COMPLETE","FINAL_CAMPAIGN_COMPLETE","LATEST_PID","LATEST_LOG"):
        p = CAMPAIGN / name
        if p.exists():
            copy_file(p, d06 / "campaign_state" / name)

    # ----------------------------------------------------------------------------------
    # Validation / audit material
    # ----------------------------------------------------------------------------------
    if PREFLIGHT_AUDIT.exists():
        copy_file(PREFLIGHT_AUDIT, d06 / PREFLIGHT_AUDIT.name)

    terminal_log = CAMPAIGN / "final_results_posthoc_terminal.txt"
    if terminal_log.exists():
        copy_file(terminal_log, d06 / terminal_log.name)

    latest_log_file = CAMPAIGN / "LATEST_LOG"
    if latest_log_file.exists():
        log_path = Path(latest_log_file.read_text().strip())
        if log_path.exists():
            info = {
                "path": str(log_path),
                "size_bytes": log_path.stat().st_size,
                "mtime_ns": log_path.stat().st_mtime_ns,
            }
            write_text(d06 / "full_training_log_reference.json", json.dumps(info, indent=2))
            (d06 / "full_training_log_tail.txt").write_bytes(last_bytes(log_path))

    fix_note = """# Original aggregation exception and post-hoc recovery

All 35 training/evaluation folds completed successfully. The original campaign process
then stopped in the final aggregation helper because it attempted to parse the textual
event-stats summary row `task=Overall` as a numeric task ID.

No model training or inference was rerun. The accepted post-hoc collector filters
non-numeric report-only rows and recomputes the final pooled and fold-level summaries
from the already-completed result artifacts.
"""
    write_text(d06 / "POSTHOC_AGGREGATION_RECOVERY.md", fix_note)

    # ----------------------------------------------------------------------------------
    # Paper tables
    # ----------------------------------------------------------------------------------
    pooled = pd.read_csv(POST / "POOLED_ALL_RESULTS.csv")
    folds = pd.read_csv(POST / "ALL_FOLD_METRICS.csv")
    seg_mean = pd.read_csv(POST / "SEGMENT_LEVEL_MEAN_STD.csv")
    event_mean = pd.read_csv(POST / "EVENT_LEVEL_MEAN_STD.csv")

    seg = pooled[(pooled["level"] == "Segment") & pooled["condition"].isin(PAPER_SEGMENT)].copy()
    seg["_order"] = seg["condition"].map({c:i for i,c in enumerate(PAPER_SEGMENT)})
    seg = seg.sort_values("_order")

    seg_paper = pd.DataFrame({
        "Condition": seg["condition"].map(DISPLAY),
        "Evaluation": seg["evaluation_domain"],
        "N_windows": seg["n"].astype(int),
        "Overall_Accuracy_percent": seg["overall_accuracy"].map(percent),
        "Balanced_Accuracy_percent": seg["balanced_accuracy"].map(percent),
        "Activity_Precision_percent": seg["activity_precision"].map(percent),
        "Activity_Recall_percent": seg["activity_recall"].map(percent),
        "Activity_F1_percent": seg["activity_f1"].map(percent),
        "Fall_Precision_percent": seg["fall_precision"].map(percent),
        "Fall_Recall_percent": seg["fall_recall"].map(percent),
        "Fall_F1_percent": seg["fall_f1"].map(percent),
        "TN": seg["tn"].astype(int),
        "FP": seg["fp"].astype(int),
        "FN": seg["fn"].astype(int),
        "TP": seg["tp"].astype(int),
    })
    seg_paper.to_csv(d04 / "SEGMENT_LEVEL_PRIMARY_POOLED.csv", index=False)

    seg_mean_filtered = seg_mean[seg_mean["Condition"].isin([DISPLAY[c] for c in PAPER_SEGMENT])].copy()
    order_disp = {DISPLAY[c]:i for i,c in enumerate(PAPER_SEGMENT)}
    seg_mean_filtered["_order"] = seg_mean_filtered["Condition"].map(order_disp)
    seg_mean_filtered = seg_mean_filtered.sort_values("_order").drop(columns="_order")
    seg_mean_filtered.to_csv(d04 / "SEGMENT_LEVEL_PRIMARY_MEAN_STD.csv", index=False)

    ev = pooled[(pooled["level"] == "Event") & pooled["condition"].isin(PAPER_PHYSICAL)].copy()
    ev["_order"] = ev["condition"].map({c:i for i,c in enumerate(PAPER_PHYSICAL)})
    ev = ev.sort_values("_order")

    event_paper = pd.DataFrame({
        "Condition": ev["condition"].map(DISPLAY),
        "N_trials_events": ev["n"].astype(int),
        "Fall_Precision_percent": ev["fall_precision"].map(percent),
        "Fall_Recall_Detection_percent": ev["fall_recall"].map(percent),
        "Fall_F1_percent": ev["fall_f1"].map(percent),
    })
    event_paper.to_csv(d04 / "EVENT_LEVEL_PHYSICAL_PRIMARY_POOLED.csv", index=False)

    event_full_support = pd.DataFrame({
        "Condition": ev["condition"].map(DISPLAY),
        "N_trials_events": ev["n"].astype(int),
        "Overall_Accuracy_percent": ev["overall_accuracy"].map(percent),
        "Balanced_Accuracy_percent": ev["balanced_accuracy"].map(percent),
        "Activity_Precision_percent": ev["activity_precision"].map(percent),
        "Activity_Recall_Specificity_percent": ev["activity_recall"].map(percent),
        "Activity_F1_percent": ev["activity_f1"].map(percent),
        "Fall_Precision_percent": ev["fall_precision"].map(percent),
        "Fall_Recall_Detection_percent": ev["fall_recall"].map(percent),
        "Fall_F1_percent": ev["fall_f1"].map(percent),
        "TN": ev["tn"].astype(int),
        "FP": ev["fp"].astype(int),
        "FN": ev["fn"].astype(int),
        "TP": ev["tp"].astype(int),
    })
    event_full_support.to_csv(d04 / "EVENT_LEVEL_PHYSICAL_SUPPORTING_FULL_METRICS.csv", index=False)

    event_mean_filtered = event_mean[event_mean["Condition"].isin([DISPLAY[c] for c in PAPER_PHYSICAL])].copy()
    event_mean_filtered["_order"] = event_mean_filtered["Condition"].map({DISPLAY[c]:i for i,c in enumerate(PAPER_PHYSICAL)})
    event_mean_filtered = event_mean_filtered.sort_values("_order").drop(columns="_order")
    event_mean_filtered.to_csv(d04 / "EVENT_LEVEL_PHYSICAL_MEAN_STD.csv", index=False)

    sim_event = pooled[(pooled["level"] == "Event") & (pooled["condition"] == "EXP03_SIM_ONLY_FULL")].iloc[0]
    sim_diag = pd.DataFrame([{
        "Condition": DISPLAY["EXP03_SIM_ONLY_FULL"],
        "Evaluation": sim_event["evaluation_domain"],
        "Held_out_fall_trials": int(sim_event["tp"] + sim_event["fn"]),
        "Detected_fall_trials": int(sim_event["tp"]),
        "Missed_fall_trials": int(sim_event["fn"]),
        "Fall_event_sensitivity_percent": percent(sim_event["fall_recall"]),
        "Reporting_note": "Precision/specificity/F1 intentionally omitted because no independent Activity-only event trials are present.",
    }])
    sim_diag.to_csv(d04 / "EVENT_LEVEL_SIM_ONLY_DIAGNOSTIC.csv", index=False)

    # A single combined paper summary markdown.
    md = []
    md.append("# Paper-ready Results Summary\n")
    md.append("## Segment level — primary pooled results\n")
    md.append(dataframe_to_markdown(seg_paper, floatfmt=".2f"))
    md.append("\n\n## Event level — physical held-out primary results\n")
    md.append(dataframe_to_markdown(event_paper, floatfmt=".2f"))
    md.append("\n\n## Synthetic-only event diagnostic\n")
    md.append(dataframe_to_markdown(sim_diag, floatfmt=".2f"))
    md.append("\n")
    write_text(d04 / "PAPER_READY_RESULTS.md", "\n".join(md))

    # ----------------------------------------------------------------------------------
    # Uniform classification reports derived from pooled confusion counts
    # ----------------------------------------------------------------------------------
    reports = []
    for _, r in seg.iterrows():
        reports.extend([
            {
                "Condition": DISPLAY[r["condition"]],
                "Level": "Segment",
                "Class": "Activity",
                "Precision_percent": percent(r["activity_precision"]),
                "Recall_percent": percent(r["activity_recall"]),
                "F1_percent": percent(r["activity_f1"]),
                "Support": int(r["support_activity"]),
            },
            {
                "Condition": DISPLAY[r["condition"]],
                "Level": "Segment",
                "Class": "Falling",
                "Precision_percent": percent(r["fall_precision"]),
                "Recall_percent": percent(r["fall_recall"]),
                "F1_percent": percent(r["fall_f1"]),
                "Support": int(r["support_fall"]),
            },
        ])
    pd.DataFrame(reports).to_csv(d04 / "SEGMENT_CLASSIFICATION_REPORT_POOLED.csv", index=False)

    ereports = []
    for _, r in ev.iterrows():
        ereports.extend([
            {
                "Condition": DISPLAY[r["condition"]],
                "Level": "Event",
                "Class": "Activity",
                "Precision_percent": percent(r["activity_precision"]),
                "Recall_percent": percent(r["activity_recall"]),
                "F1_percent": percent(r["activity_f1"]),
                "Support": int(r["support_activity"]),
            },
            {
                "Condition": DISPLAY[r["condition"]],
                "Level": "Event",
                "Class": "Falling",
                "Precision_percent": percent(r["fall_precision"]),
                "Recall_percent": percent(r["fall_recall"]),
                "F1_percent": percent(r["fall_f1"]),
                "Support": int(r["support_fall"]),
            },
        ])
    pd.DataFrame(ereports).to_csv(d04 / "EVENT_CLASSIFICATION_REPORT_PHYSICAL_POOLED.csv", index=False)

    # ----------------------------------------------------------------------------------
    # Plot data + publication figures
    # ----------------------------------------------------------------------------------
    seg_paper.to_csv(d05 / "plot_data_segment_primary.csv", index=False)
    event_paper.to_csv(d05 / "plot_data_event_physical_primary.csv", index=False)
    sim_diag.to_csv(d05 / "plot_data_sim_only_event_detection.csv", index=False)

    plot_grouped_metrics(
        seg,
        PAPER_SEGMENT,
        [
            ("fall_precision", "Precision"),
            ("fall_recall", "Recall"),
            ("fall_f1", "F1"),
        ],
        "Segment-level Falling performance",
        "Performance (%)",
        d05 / "fig_segment_falling_metrics",
    )

    plot_grouped_metrics(
        ev,
        PAPER_PHYSICAL,
        [
            ("fall_precision", "Precision"),
            ("fall_recall", "Recall"),
            ("fall_f1", "F1"),
        ],
        "Event-level fall performance on held-out physical trials",
        "Performance (%)",
        d05 / "fig_event_physical_fall_metrics",
    )

    # Augmentation deltas relative to REAL_ONLY
    base_seg = float(seg.loc[seg["condition"]=="EXP01_REAL_ONLY", "fall_f1"].iloc[0])
    base_ev = float(ev.loc[ev["condition"]=="EXP01_REAL_ONLY", "fall_f1"].iloc[0])
    mix_names = ["EXP04_MIX20","EXP05_MIX50","EXP06_MIX70","EXP07_MIX100"]
    doses = np.array([20,50,70,100], dtype=float)
    seg_d = np.array([
        100.0 * (float(seg.loc[seg["condition"]==c, "fall_f1"].iloc[0]) - base_seg)
        for c in mix_names
    ])
    ev_d = np.array([
        100.0 * (float(ev.loc[ev["condition"]==c, "fall_f1"].iloc[0]) - base_ev)
        for c in mix_names
    ])
    delta_df = pd.DataFrame({
        "Synthetic_addition_percent_of_real_fall_windows": doses.astype(int),
        "Segment_Fall_F1_delta_pp_vs_REAL_ONLY": seg_d,
        "Event_Fall_F1_delta_pp_vs_REAL_ONLY": ev_d,
    })
    delta_df.to_csv(d05 / "plot_data_augmentation_f1_delta.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.plot(doses, seg_d, marker="o", label="Segment Fall F1")
    ax.plot(doses, ev_d, marker="s", label="Event Fall F1")
    ax.axhline(0.0, linewidth=1)
    ax.set_xlabel("Synthetic Falling addition (% of real Falling windows)")
    ax.set_ylabel("F1 change vs REAL_ONLY (percentage points)")
    ax.set_title("Effect of synthetic augmentation relative to REAL_ONLY")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    for ext in ("png","pdf","svg"):
        fig.savefig((d05 / "fig_augmentation_f1_delta").with_suffix("." + ext), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # SIM-only event detection figure
    detected = int(sim_event["tp"])
    missed = int(sim_event["fn"])
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.bar(["Detected", "Missed"], [detected, missed])
    ax.set_ylabel("Held-out simulated fall trials")
    ax.set_title(f"Synthetic-only event detection: {detected}/{detected+missed} ({percent(sim_event['fall_recall']):.2f}%)")
    for i, v in enumerate([detected, missed]):
        ax.text(i, v, str(v), ha="center", va="bottom")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    for ext in ("png","pdf","svg"):
        fig.savefig((d05 / "fig_sim_only_event_detection").with_suffix("." + ext), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Confusion matrices for paper-facing segment conditions
    for _, r in seg.iterrows():
        label = r["condition"]
        plot_confusion(
            int(r["tn"]), int(r["fp"]), int(r["fn"]), int(r["tp"]),
            f"Segment confusion matrix — {DISPLAY[label]}",
            d05 / f"cm_segment_{label}",
        )

    # Event confusion matrices only for the physically comparable paper conditions
    for _, r in ev.iterrows():
        label = r["condition"]
        plot_confusion(
            int(r["tn"]), int(r["fp"]), int(r["fn"]), int(r["tp"]),
            f"Event confusion matrix — {DISPLAY[label]}",
            d05 / f"cm_event_{label}",
        )

    # Fold-wise plot data retained for later statistical plots.
    folds_primary = folds[
        folds["condition"].isin(set(PAPER_SEGMENT + PAPER_PHYSICAL))
    ].copy()
    folds_primary.to_csv(d05 / "plot_data_all_primary_fold_metrics.csv", index=False)

    # ----------------------------------------------------------------------------------
    # Environment + exact hashes
    # ----------------------------------------------------------------------------------
    env_lines = []
    env_lines.append("SYSTEM\n")
    env_lines.append(platform.platform() + "\n")
    env_lines.append(platform.python_version() + "\n\n")
    env_lines.append("NVIDIA-SMI\n")
    env_lines.append(safe_cmd(["nvidia-smi", "-L"]))
    env_lines.append("\nPROTECHTO PYTHON VERSION\n")
    env_lines.append(safe_cmd([str(PROTECHTO_PY), "--version"]))
    env_lines.append("\nPROTECHTO PIP FREEZE\n")
    env_lines.append(safe_cmd([str(PROTECHTO_PY), "-m", "pip", "freeze"]))
    mujoco_py = PROJECT / ".venv/bin/python"
    if mujoco_py.exists():
        env_lines.append("\nMUJOCO PROJECT PYTHON VERSION\n")
        env_lines.append(safe_cmd([str(mujoco_py), "--version"]))
        env_lines.append("\nMUJOCO PROJECT PIP FREEZE\n")
        env_lines.append(safe_cmd([str(mujoco_py), "-m", "pip", "freeze"]))
    write_text(d07 / "ENVIRONMENT.txt", "".join(env_lines))

    # Source hash table
    source_rows = []
    for root_name, root_path in [
        ("canonical_protechto", d01 / "canonical_source"),
        ("campaign_scripts", d01 / "campaign_scripts"),
        ("simulator_source", d02 / "project_root_source"),
    ]:
        if not root_path.exists():
            continue
        for p in root_path.rglob("*"):
            if p.is_file():
                source_rows.append({
                    "group": root_name,
                    "path": str(p.relative_to(stage)),
                    "size_bytes": p.stat().st_size,
                    "sha256": sha256_file(p),
                })
    pd.DataFrame(source_rows).to_csv(d07 / "SOURCE_HASHES.csv", index=False)

    # Checkpoint manifest; include binaries only if requested.
    ck_rows = []
    for p in sorted(checkpoint_root.rglob("*.ckpt")):
        ck_rows.append({
            "condition": p.parent.name,
            "checkpoint_name": p.name,
            "original_path": str(p),
            "size_bytes": p.stat().st_size,
            "sha256": sha256_file(p),
        })
        if args.include_checkpoints:
            rel = p.relative_to(checkpoint_root)
            copy_file(p, d07 / "checkpoints" / rel)
    ck_df = pd.DataFrame(ck_rows)
    ck_df.to_csv(d07 / "CHECKPOINT_MANIFEST.csv", index=False)

    # Campaign structure audit
    audit = {
        "run_id": run_id,
        "completed_fold_markers": len(markers),
        "expected_fold_markers": 35,
        "paper_segment_conditions": PAPER_SEGMENT,
        "paper_event_physical_conditions": PAPER_PHYSICAL,
        "excluded_diagnostic": EXCLUDED_DIAGNOSTIC,
        "canonical_event_manifest": str(EVENT_MANIFEST),
        "canonical_event_manifest_sha256": sha256_file(EVENT_MANIFEST),
        "synthetic_dataset_present": SYNTH_DATASET.exists(),
        "synthetic_cache_present": SYNTH_CACHE.exists(),
        "checkpoints_found": len(ck_df),
        "checkpoint_binaries_included": bool(args.include_checkpoints),
    }
    write_text(d07 / "CAMPAIGN_AUDIT.json", json.dumps(audit, indent=2))

    # ----------------------------------------------------------------------------------
    # Final archive-wide manifest
    # ----------------------------------------------------------------------------------
    manifest_rows = []
    for p in sorted(stage.rglob("*")):
        if not p.is_file():
            continue
        manifest_rows.append({
            "path": str(p.relative_to(stage)),
            "size_bytes": p.stat().st_size,
            "sha256": sha256_file(p),
        })

    man_df = pd.DataFrame(manifest_rows)
    man_df.to_csv(d00 / "FILE_MANIFEST_SHA256.csv", index=False)

    total_files = len(man_df)
    total_bytes = int(man_df["size_bytes"].sum()) if total_files else 0

    summary = {
        "run_id": run_id,
        "staging_dir": str(stage),
        "files": total_files,
        "staged_bytes": total_bytes,
        "completed_folds": 35,
        "primary_segment_conditions": PAPER_SEGMENT,
        "primary_event_physical_conditions": PAPER_PHYSICAL,
        "sim_only_event_reporting": "sensitivity only",
        "excluded_diagnostic": EXCLUDED_DIAGNOSTIC,
    }
    write_text(d00 / "BUNDLE_SUMMARY.json", json.dumps(summary, indent=2))

    # ----------------------------------------------------------------------------------
    # ZIP
    # ----------------------------------------------------------------------------------
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()

    print()
    print("Creating ZIP:", zip_path)
    with zipfile.ZipFile(
        zip_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True,
    ) as zf:
        for p in sorted(stage.rglob("*")):
            if p.is_file():
                arc = Path(stage.name) / p.relative_to(stage)
                zf.write(p, arcname=str(arc))

    zip_sha = sha256_file(zip_path)

    final_note = f"""PAPER EVIDENCE ARCHIVE COMPLETE

Run ID: {run_id}
Completed folds: 35
Staged files: {total_files}
Staged bytes: {total_bytes}
ZIP: {zip_path}
ZIP bytes: {zip_path.stat().st_size}
ZIP SHA256: {zip_sha}

Primary segment conditions:
{os.linesep.join('  - ' + DISPLAY[c] for c in PAPER_SEGMENT)}

Primary physical event conditions:
{os.linesep.join('  - ' + DISPLAY[c] for c in PAPER_PHYSICAL)}

SIM-only event reporting:
  sensitivity only ({int(sim_event['tp'])}/{int(sim_event['tp']+sim_event['fn'])}
  = {percent(sim_event['fall_recall']):.2f}%)

Excluded primary-table diagnostic:
  {DISPLAY[EXCLUDED_DIAGNOSTIC]}

No training or inference was performed by this archive builder.
"""
    write_text(stage / "ARCHIVE_COMPLETE.txt", final_note)

    # Re-open ZIP to add the final note, summary manifest changed only after zip creation otherwise.
    # We keep ARCHIVE_COMPLETE outside the ZIP creation manifest on purpose to avoid recursive hashing.
    with zipfile.ZipFile(zip_path, mode="a", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        zf.write(
            stage / "ARCHIVE_COMPLETE.txt",
            arcname=str(Path(stage.name) / "ARCHIVE_COMPLETE.txt"),
        )

    zip_sha = sha256_file(zip_path)

    print()
    print("=" * 116)
    print("PAPER EVIDENCE ARCHIVE GATE: PASS")
    print("=" * 116)
    print("Run ID            :", run_id)
    print("Completed folds    :", 35)
    print("Staged files       :", total_files)
    print("ZIP                :", zip_path)
    print("ZIP size bytes     :", zip_path.stat().st_size)
    print("ZIP SHA256         :", zip_sha)
    print("Checkpoints copied :", bool(args.include_checkpoints))
    print()
    print("NO TRAINING OR INFERENCE WAS RUN.")


if __name__ == "__main__":
    main()
