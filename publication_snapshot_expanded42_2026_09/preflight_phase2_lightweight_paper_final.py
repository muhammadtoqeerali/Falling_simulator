#!/usr/bin/env python3
import ast
import importlib.util
import re
from pathlib import Path
import numpy as np
import pandas as pd
import torch

PROJECT = Path('/mnt/hdd16T/ToqeerHomeBackup/mujoco_project')
RUNNER = PROJECT / 'run_phase2_lightweight_paper_final_fold.py'

if not RUNNER.exists():
    raise SystemExit(f'Missing runner: {RUNNER}')

text = RUNNER.read_text()
forbidden = [
    r'end_fall_frame\s*-\s*15',
    r'impact(?:_frame|_index|_idx)?\s*-\s*15',
    r'drop_last.*fall',
    r'trim.*150',
]
# The metadata key below documents that no second trim occurs; do not flag that literal.
scan = text.replace("'no_additional_150ms_trimming': True", "")
for pat in forbidden:
    if re.search(pat, scan, flags=re.IGNORECASE):
        raise SystemExit(f'Forbidden second 150-ms trimming pattern found: {pat}')

spec = importlib.util.spec_from_file_location('final_runner', str(RUNNER))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

required = [
    mod.PHYSICAL_ROOT,
    mod.SPLIT_CSV,
    mod.SYN_X,
    mod.SYN_META,
    mod.DRAW_ROOT,
    mod.ORIG / 'models/CNN.py',
    mod.ORIG / 'models/modules/IMUNormalizer.py',
    mod.EVENT_SRC / 'simulation/LearningCurveMetrics.py',
    mod.EVENT_SRC / 'simulation/helper.py',
]
for p in required:
    if not Path(p).exists():
        raise SystemExit(f'Missing required input: {p}')

syn_pool = np.load(mod.SYN_X, mmap_mode='r')
if tuple(syn_pool.shape) != (946, 30, 9):
    raise SystemExit(f'Unexpected aligned synthetic pool: {syn_pool.shape}')
syn_meta = pd.read_csv(mod.SYN_META)
if len(syn_meta) != 946:
    raise SystemExit(f'Unexpected synthetic metadata rows: {len(syn_meta)}')

print('=' * 112)
print('FINAL CAMPAIGN PRELAUNCH AUDIT')
print('=' * 112)
print('Physical root :', mod.PHYSICAL_ROOT)
print('Synthetic pool:', mod.SYN_X, tuple(syn_pool.shape))
print('Output root   :', mod.DEFAULT_OUT)
print('Second -150ms trim in runner: NONE')
print()

for fold in range(5):
    roles = mod.subjects_for_fold(fold)
    all_s = roles['train'] + roles['val'] + roles['test']
    if len(all_s) != 71 or len(set(all_s)) != 71:
        raise SystemExit(f'Fold {fold}: expected 71 unique eligible subjects, got {len(set(all_s))}')
    counts = {}
    for role in ['train', 'val', 'test']:
        counts[role] = mod.count_physical_labels(roles[role])
        mod.assert_expected_counts(fold, role, counts[role])
    real_falling = counts['train'][2]
    print(f'fold={fold} subjects train/val/test={len(roles["train"])}/{len(roles["val"])}/{len(roles["test"])}')
    print(f'  train={counts["train"][:3]} val={counts["val"][:3]} test={counts["test"][:3]}')
    for exp in mod.EXPERIMENTS[1:]:
        _, draw, info = mod.load_synthetic_draws(exp, fold, real_falling, load_x=False)
        idx = info['indices']
        expected = mod.expected_draws(exp, real_falling)
        if len(idx) != expected:
            raise SystemExit(f'{exp} fold {fold}: draw count mismatch')
        unique = len(np.unique(idx))
        if exp == 'EXP03_MIX20' and unique != len(idx):
            raise SystemExit(f'{exp} fold {fold}: MIX20 must be fully unique')
        if len(idx) > 946 and unique != 946:
            raise SystemExit(f'{exp} fold {fold}: expected full 946-window pool before reuse, unique={unique}')
        print(f'  {exp:<29} draws={len(idx):5d} unique={unique:3d} file={Path(info["draw_file"]).name}')

CNN = mod.load_original_cnn()
model = CNN(n_features=9, n_classes=2, config=mod.CNN_CONFIG)
with torch.no_grad():
    out = model(torch.zeros((2, 30, 9), dtype=torch.float32))
if tuple(out.shape) != (2, 2):
    raise SystemExit(f'Original CNN dummy forward failed: {tuple(out.shape)}')

cfun = mod.load_function_from_source(mod.EVENT_SRC / 'simulation/LearningCurveMetrics.py', 'cnn_predictions_from_probabilities')
efun = mod.load_function_from_source(mod.EVENT_SRC / 'simulation/helper.py', 'is_simulation_passed_threshold')
pred = np.asarray(cfun(np.array([[0.90,0.10],[0.20,0.80],[0.55,0.45]]), prediction_bias=mod.PREDICTION_BIAS))
if pred.shape != (3,):
    raise SystemExit('Historical probability-to-label function failed preflight')
if not bool(efun(np.array([0,1,1]), np.array([0,1,1]), threshold=mod.EVENT_THRESHOLD)):
    raise SystemExit('Historical threshold=2 function failed preflight')

print()
print('Original CNN SHA256       :', mod.sha256_file(mod.ORIG / 'models/CNN.py'))
print('Original normalizer SHA256:', mod.sha256_file(mod.ORIG / 'models/modules/IMUNormalizer.py'))
print('CNN config                 :', mod.CNN_CONFIG)
print('Optimizer                  : AdamW')
print('LR / WD                    :', mod.LR, mod.WD)
print('Batch / epochs / patience  :', mod.BATCH_SIZE, mod.MAX_EPOCHS, mod.PATIENCE)
print('Prediction bias / threshold:', mod.PREDICTION_BIAS, mod.EVENT_THRESHOLD)
print()
print('FINAL CAMPAIGN PRELAUNCH GATE: PASS')
