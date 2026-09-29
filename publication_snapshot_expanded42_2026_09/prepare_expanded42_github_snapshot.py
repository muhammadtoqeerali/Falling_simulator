#!/usr/bin/env python3
"""
Prepare a compact, reproducible GitHub snapshot of the final expanded-42 pipeline.

Run on the workstation:
    python3 prepare_expanded42_github_snapshot.py \
      --dest /tmp/Falling_simulator_expanded42/publication_snapshot_expanded42_2026_09

The script:
  * copies implementation/code/config files from mujoco_project
  * excludes raw data, videos, checkpoints, virtual envs, caches, MuJoCo menagerie, old snapshots
  * includes the final expanded-42 runner and compact final result/audit CSVs
  * includes lightweight expanded-42 output metadata outside per-run raw folders
  * creates MANIFEST.csv with SHA-256 checksums
"""
from pathlib import Path
from datetime import datetime
import argparse, csv, hashlib, shutil, sys

DEFAULT_ROOT = Path("/mnt/hdd16T/ToqeerHomeBackup/mujoco_project")

CODE_EXT = {
    ".py", ".sh", ".bash", ".zsh", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".json", ".txt", ".md", ".csv", ".tsv", ".ipynb"
}

EXCLUDE_DIRS = {
    ".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".idea", ".vscode", "venv", ".venv", "env", "site-packages", "node_modules",
    "mujoco_menagerie", "checkpoints", "videos",
    "_chatgpt_recent_code_inventory_20260929",
    "publication_snapshot_2026_09",
    "publication_snapshot_expanded42_2026_09",
}

# Output roots containing the authoritative final results that are useful for audit.
FINAL_OUTPUT_ROOTS = [
    "outputs/protechto_exact_full_campaign_expanded42_v2",
    "outputs/final_dataset_split_audit_expanded42_v2",
]

# Final simulator campaign metadata root. We keep only compact top-level metadata,
# not per-run raw output trees.
FINAL_CAMPAIGN_ROOT = "outputs/_highrate_overnight/campaign_highrate_truth_v4_expanded42_new20"

OUTPUT_KEYWORDS = (
    "result", "metric", "summary", "manifest", "audit", "profile", "task",
    "split", "reuse", "integrity", "pooled", "fold", "comparison", "event",
    "canonical", "eligib", "count", "inventory", "report"
)

MAX_CODE_BYTES = 8 * 1024 * 1024
MAX_OUTPUT_BYTES = 8 * 1024 * 1024


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def excluded(rel: Path) -> bool:
    return any(part in EXCLUDE_DIRS for part in rel.parts[:-1])


def copy_one(src: Path, root: Path, dest: Path, records, category: str):
    rel = src.relative_to(root)
    out = dest / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, out)
    st = src.stat()
    records.append({
        "category": category,
        "source_relative_path": str(rel),
        "snapshot_relative_path": str(rel),
        "size_bytes": st.st_size,
        "modified": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
        "sha256": sha256(src),
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--dest", type=Path, required=True)
    args = ap.parse_args()

    root = args.root.resolve()
    dest = args.dest.resolve()

    if not root.exists():
        print(f"ERROR: project root does not exist: {root}", file=sys.stderr)
        return 2

    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    records = []
    copied = set()

    # 1) Current implementation snapshot: all manageable code/config/text files
    # outside generated/raw trees.
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if excluded(rel):
            continue
        # Generated output trees are handled separately below.
        if rel.parts and rel.parts[0] == "outputs":
            continue
        # Avoid raw datasets if present under the project root.
        lower_parts = {x.lower() for x in rel.parts[:-1]}
        if lower_parts & {"data", "dataset", "datasets", "raw_data", "raw", "recordings"}:
            continue
        if p.suffix.lower() not in CODE_EXT:
            continue
        try:
            if p.stat().st_size > MAX_CODE_BYTES:
                continue
        except OSError:
            continue
        copy_one(p, root, dest, records, "implementation")
        copied.add(str(rel))

    # 2) Authoritative final CNN result/audit directories: compact text/CSV/JSON only.
    for relroot in FINAL_OUTPUT_ROOTS:
        base = root / relroot
        if not base.exists():
            print(f"WARNING: final output root not found: {base}")
            continue
        for p in base.rglob("*"):
            if not p.is_file():
                continue
            rel = p.relative_to(root)
            if p.suffix.lower() not in CODE_EXT:
                continue
            try:
                if p.stat().st_size > MAX_OUTPUT_BYTES:
                    continue
            except OSError:
                continue
            copy_one(p, root, dest, records, "final_result_or_audit")
            copied.add(str(rel))

    # 3) Expanded-42 simulator campaign: keep compact top-level metadata and code,
    # but explicitly skip runs/ to avoid hundreds of per-run raw manifests/data.
    camp = root / FINAL_CAMPAIGN_ROOT
    if camp.exists():
        for p in camp.rglob("*"):
            if not p.is_file():
                continue
            rel_camp = p.relative_to(camp)
            if rel_camp.parts and rel_camp.parts[0].lower() == "runs":
                continue
            if p.suffix.lower() not in CODE_EXT:
                continue
            try:
                if p.stat().st_size > MAX_OUTPUT_BYTES:
                    continue
            except OSError:
                continue
            name = p.name.lower()
            if p.suffix.lower() in {".py", ".sh", ".bash", ".zsh"} or any(k in name for k in OUTPUT_KEYWORDS):
                rel = p.relative_to(root)
                if str(rel) not in copied:
                    copy_one(p, root, dest, records, "expanded42_campaign_metadata")
                    copied.add(str(rel))
    else:
        print(f"WARNING: expanded-42 campaign root not found: {camp}")

    # 4) Write a snapshot README with the exact provenance.
    readme = dest / "README_SNAPSHOT.md"
    readme.write_text(
        "# Expanded-42 final implementation snapshot\n\n"
        "Source workstation project:\n"
        f"`{root}`\n\n"
        "Purpose: compact code/config/result snapshot for auditing the final 42-profile, "
        "18-task, 756-simulation pipeline and its downstream CNN evaluation.\n\n"
        "Authoritative final result roots included where present:\n"
        "- `outputs/protechto_exact_full_campaign_expanded42_v2/`\n"
        "- `outputs/final_dataset_split_audit_expanded42_v2/`\n\n"
        "Expanded simulator campaign metadata root:\n"
        "- `outputs/_highrate_overnight/campaign_highrate_truth_v4_expanded42_new20/`\n\n"
        "Per-run raw campaign outputs, raw physical datasets, checkpoints, videos, "
        "virtual environments, and MuJoCo menagerie assets are intentionally excluded.\n",
        encoding="utf-8",
    )

    # 5) Manifest.
    manifest = dest / "MANIFEST.csv"
    records.sort(key=lambda r: r["source_relative_path"])
    with manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "category", "source_relative_path", "snapshot_relative_path",
                "size_bytes", "modified", "sha256"
            ],
        )
        w.writeheader()
        w.writerows(records)

    print(f"SNAPSHOT: {dest}")
    print(f"FILES COPIED: {len(records)}")
    print(f"MANIFEST: {manifest}")
    print("\nKey final files found:")
    important = [
        "run_protechto_exact_full_campaign_expanded42_v2.py",
        "outputs/protechto_exact_full_campaign_expanded42_v2/final_results_posthoc/POOLED_ALL_RESULTS.csv",
        "outputs/protechto_exact_full_campaign_expanded42_v2/final_results_posthoc/FOLD_MEAN_STD.csv",
        "outputs/final_dataset_split_audit_expanded42_v2/integrity_checks.csv",
        "outputs/final_dataset_split_audit_expanded42_v2/mixed_fold_data_usage.csv",
    ]
    for rel in important:
        print(("  OK  " if (dest / rel).exists() else "  MISS") + "  " + rel)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
