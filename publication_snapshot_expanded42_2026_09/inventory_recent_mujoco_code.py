#!/usr/bin/env python3
from pathlib import Path
from datetime import datetime, timezone
import hashlib, csv, os, sys

ROOT = Path('/mnt/hdd16T/ToqeerHomeBackup/mujoco_project').resolve()
SINCE = datetime(2026, 9, 20, 0, 0, 0, tzinfo=timezone.utc).timestamp()

# Code/config/text extensions useful for reconstructing the latest pipeline.
KEEP_EXT = {
    '.py', '.sh', '.bash', '.zsh', '.yaml', '.yml', '.toml', '.json', '.ini', '.cfg',
    '.txt', '.md', '.csv', '.tsv', '.ipynb'
}

# Exclude large/raw/generated/runtime trees. We want implementation + small manifests, not data dumps.
EXCLUDE_PARTS = {
    '.git', '__pycache__', '.pytest_cache', '.mypy_cache', '.ruff_cache', '.idea', '.vscode',
    'venv', '.venv', 'env', 'site-packages', 'node_modules',
    'checkpoints', 'videos'
}

# Large output/data roots are skipped by default, except lightweight metadata/manifests explicitly allowed below.
HEAVY_PARTS = {'outputs', 'output', 'results', 'raw_data', 'data', 'datasets'}
ALLOW_IN_HEAVY_NAMES = {
    'manifest.csv', 'manifest.json', 'run_manifest.json', 'summary.csv', 'summary.json',
    'metrics.csv', 'metrics.json', 'config.json', 'config.yaml', 'config.yml',
    'profile_summary.csv', 'task_summary.csv', 'pooled_all_results.csv', 'fold_mean_std.csv',
    'all_fold_metrics.csv', 'dataset_global_comparison.csv', 'physical_fold_split_counts.csv',
    'simulator_only_fold_counts.csv', 'mixed_fold_data_usage.csv', 'mixed_reuse_summary.csv',
    'integrity_checks.csv'
}

MAX_BYTES = 5 * 1024 * 1024  # 5 MB per file for inventory candidate list


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

rows = []
if not ROOT.exists():
    print(f'ERROR: root does not exist: {ROOT}', file=sys.stderr)
    sys.exit(2)

for p in ROOT.rglob('*'):
    try:
        if not p.is_file():
            continue
        rel = p.relative_to(ROOT)
        parts_lower = {x.lower() for x in rel.parts[:-1]}
        if parts_lower & EXCLUDE_PARTS:
            continue
        st = p.stat()
        if st.st_mtime < SINCE:
            continue
        if p.suffix.lower() not in KEEP_EXT:
            continue
        if st.st_size > MAX_BYTES:
            continue
        in_heavy = bool(parts_lower & HEAVY_PARTS)
        if in_heavy and p.name.lower() not in ALLOW_IN_HEAVY_NAMES:
            # Keep Python/shell implementation files even if someone stored them under an output tree.
            if p.suffix.lower() not in {'.py', '.sh', '.bash', '.zsh'}:
                continue
        rows.append({
            'relative_path': str(rel),
            'modified_local': datetime.fromtimestamp(st.st_mtime).isoformat(timespec='seconds'),
            'size_bytes': st.st_size,
            'suffix': p.suffix.lower(),
            'sha256': sha256(p),
        })
    except (PermissionError, FileNotFoundError, OSError):
        continue

rows.sort(key=lambda r: (r['modified_local'], r['relative_path']))

outdir = ROOT / '_chatgpt_recent_code_inventory_20260929'
outdir.mkdir(exist_ok=True)
csv_path = outdir / 'recent_code_since_2026-09-20.csv'
txt_path = outdir / 'recent_code_since_2026-09-20.txt'

with csv_path.open('w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=['relative_path','modified_local','size_bytes','suffix','sha256'])
    w.writeheader(); w.writerows(rows)

with txt_path.open('w', encoding='utf-8') as f:
    f.write(f'ROOT: {ROOT}\n')
    f.write('SINCE: 2026-09-20 00:00:00 UTC\n')
    f.write(f'CANDIDATE FILES: {len(rows)}\n\n')
    for r in rows:
        f.write(f"{r['modified_local']}\t{r['size_bytes']:>9}\t{r['relative_path']}\n")

print(f'CANDIDATE FILES: {len(rows)}')
print(f'CSV: {csv_path}')
print(f'TXT: {txt_path}')
print('\nMost recent 80 candidates:\n')
for r in rows[-80:]:
    print(f"{r['modified_local']}  {r['size_bytes']:>9}  {r['relative_path']}")
