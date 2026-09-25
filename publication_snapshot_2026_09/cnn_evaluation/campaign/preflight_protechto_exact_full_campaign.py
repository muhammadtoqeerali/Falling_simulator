#!/usr/bin/env python3
from pathlib import Path
import importlib
import json
import subprocess
import sys
import numpy as np
import pandas as pd

PROJECT = Path('/mnt/hdd16T/ToqeerHomeBackup/mujoco_project')
ROOT = Path('/mnt/hdd16T/ToqeerHomeBackup/toqeer/Protechto-master')
EXEC = PROJECT / 'outputs/protechto_clean_execution_mirror_v1'
PYTHON = Path('/mnt/hdd16T/ToqeerHomeBackup/toqeer/protechto_env/bin/python')
CACHE = PROJECT / 'outputs/protechto_exact_synthetic_cache_v1'
BASE_PREFLIGHT = PROJECT / 'preflight_protechto_clean_real_only.py'
PREP = PROJECT / 'prepare_protechto_exact_synthetic_cache.py'
RUNNER = PROJECT / 'run_protechto_exact_full_campaign.py'

CONDITIONS = [
    'EXP01_REAL_ONLY',
    'EXP02_SIM_FALL_SUBSTITUTION',
    'EXP03_SIM_ONLY_FULL',
    'EXP04_MIX20',
    'EXP05_MIX50',
    'EXP06_MIX70',
    'EXP07_MIX100',
]


def run(cmd, cwd=None):
    print('+', ' '.join(map(str, cmd)), flush=True)
    rc = subprocess.run(list(map(str, cmd)), cwd=cwd).returncode
    if rc != 0:
        raise RuntimeError(f'Command failed with status {rc}: {cmd}')


def main():
    print('=' * 110)
    print('PROTECHTO EXACT FULL CAMPAIGN PREFLIGHT')
    print('=' * 110)
    for p in [ROOT, EXEC, PYTHON, BASE_PREFLIGHT, PREP, RUNNER]:
        if not p.exists():
            raise RuntimeError(f'Missing required path: {p}')

    print('\n1) Re-run frozen physical pipeline preflight')
    run([sys.executable, '-u', BASE_PREFLIGHT], cwd=PROJECT)

    print('\n2) Verify exact Protechto execution environment')
    code = """
import torch, lightning, torchmetrics, sklearn, numpy, scipy, pandas, ray, matplotlib, seaborn, pdfkit
assert torch.cuda.is_available()
assert torch.cuda.device_count() >= 2
print('Torch', torch.__version__)
print('Lightning', lightning.__version__)
print('Ray', ray.__version__)
print('GPU1', torch.cuda.get_device_name(1))
print('PROTECHTO ENVIRONMENT GATE: PASS')
"""
    run([PYTHON, '-c', code], cwd=ROOT)

    print('\n3) Prepare/verify untrimmed synthetic cache')
    run([PYTHON, '-u', PREP], cwd=ROOT)

    X = np.load(CACHE / 'synthetic_X_protechto_rawunits.npy', mmap_mode='r')
    y = np.load(CACHE / 'synthetic_y01.npy')
    meta = pd.read_csv(CACHE / 'synthetic_metadata_resolved.csv')
    prov = json.loads((CACHE / 'provenance.json').read_text())
    assert tuple(X.shape) == (20080, 30, 9)
    assert len(y) == len(meta) == len(X)
    assert int((y == 0).sum()) == 18824
    assert int((y == 1).sum()) == 1256
    assert prov['physical_temporal_trim_applied'] is False
    assert prov['synthetic_temporal_trim_applied'] is False
    assert meta['trial_uid_resolved'].nunique() >= 5
    print('Synthetic cache shape/count/trial gate: PASS')

    print('\n4) Campaign source safety scan')
    src = RUNNER.read_text(encoding='utf-8', errors='replace')
    forbidden = ['end_fall_frame - 15', 'physical_segments_300ms_50ov_preimpact150']
    for token in forbidden:
        if token in src:
            raise RuntimeError(f'Forbidden obsolete temporal-alignment token in campaign runner: {token}')
    for c in CONDITIONS:
        if c not in src:
            raise RuntimeError(f'Missing campaign condition in runner: {c}')
    print('No manual physical/synthetic 150-ms trimming in runner: PASS')
    print('All seven campaign conditions present: PASS')

    print('\n5) Compile campaign code')
    run([PYTHON, '-m', 'py_compile', PREP, RUNNER], cwd=ROOT)

    print('\n' + '=' * 110)
    print('PROTECHTO EXACT FULL CAMPAIGN PRELAUNCH GATE: PASS')
    print('=' * 110)

if __name__ == '__main__':
    main()
