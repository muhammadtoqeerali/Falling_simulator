#!/usr/bin/env python3
import argparse
import ast
import gc
import hashlib
import importlib.util
import json
import os
import random
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset

PROJECT = Path('/mnt/hdd16T/ToqeerHomeBackup/mujoco_project')
INPUT = PROJECT / 'outputs/phase2_lightweight_paper_aligned_inputs_v1'
PHYSICAL_ROOT = INPUT / 'physical_segments_300ms_50ov_preimpact150'
SPLIT_CSV = INPUT / 'split_manifests/physical_subject_kfold_assignments.csv'
SYN_X = INPUT / 'synthetic_falling_preimpact150_X.npy'
SYN_META = INPUT / 'synthetic_falling_preimpact150_metadata.csv'
DRAW_ROOT = INPUT / 'synthetic_draw_manifests'
ORIG = Path('/mnt/hdd16T/ToqeerHomeBackup/toqeer/uniVr-dataset/UniVR_Protechto_dataset/Protechto_Code')
EVENT_SRC = Path('/mnt/hdd16T/ToqeerHomeBackup/toqeer/Protechto_master')
DEFAULT_OUT = PROJECT / 'outputs/phase2_lightweight_paper_final_campaign_v1'

EXPERIMENTS = [
    'EXP01_REAL_ONLY',
    'EXP02_SIM_FALL_SUBSTITUTION',
    'EXP03_MIX20',
    'EXP04_MIX50',
    'EXP05_MIX70',
    'EXP06_MIX100',
]
DRAW_KEY = {
    'EXP02_SIM_FALL_SUBSTITUTION': 'SIM_FALL_SUBSTITUTION',
    'EXP03_MIX20': 'MIX20',
    'EXP04_MIX50': 'MIX50',
    'EXP05_MIX70': 'MIX70',
    'EXP06_MIX100': 'MIX100',
}
RATIO = {
    'EXP03_MIX20': 0.20,
    'EXP04_MIX50': 0.50,
    'EXP05_MIX70': 0.70,
    'EXP06_MIX100': 1.00,
}
CNN_CONFIG = {
    'conv_1_dim': 64,
    'conv_1_filter': 4,
    'conv_pool': 2,
    'conv_dropout': 0.4,
    'fc': 128,
    'fc_dropout': 0.4,
}
LR = 0.001078759046949597
WD = 4.864667834921314e-05
BATCH_SIZE = 64
MAX_EPOCHS = 100
PATIENCE = 20
PREDICTION_BIAS = 0.65
EVENT_THRESHOLD = 2
SPLIT_SEED = 42

def model_seed(fold):
    # Same deterministic initialization/shuffle seed across conditions within a fold.
    return int(420100 + int(fold) * 1000)

EXPECTED = {
    0: {'train': (592436, 587731, 4705), 'val': (250316, 249208, 1108), 'test': (450206, 449131, 1075)},
    1: {'train': (664203, 659620, 4583), 'val': (333597, 332600, 997), 'test': (295158, 293850, 1308)},
    2: {'train': (581980, 577557, 4423), 'val': (510214, 509100, 1114), 'test': (200764, 199413, 1351)},
    3: {'train': (896391, 892225, 4166), 'val': (334208, 333164, 1044), 'test': (62359, 60681, 1678)},
    4: {'train': (521217, 516800, 4417), 'val': (487270, 486275, 995), 'test': (284471, 282995, 1476)},
}


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot load module {name} from {path}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_original_cnn():
    if str(ORIG) not in sys.path:
        sys.path.insert(0, str(ORIG))
    models_pkg = types.ModuleType('models')
    models_pkg.__path__ = [str(ORIG / 'models')]
    sys.modules['models'] = models_pkg
    modules_pkg = types.ModuleType('models.modules')
    modules_pkg.__path__ = [str(ORIG / 'models/modules')]
    sys.modules['models.modules'] = modules_pkg
    load_module('models.helper', ORIG / 'models/helper.py')
    load_module('models.modules.IMUNormalizer', ORIG / 'models/modules/IMUNormalizer.py')
    cnn_mod = load_module('models.CNN', ORIG / 'models/CNN.py')
    return cnn_mod.CNN


def load_function_from_source(path, function_name):
    tree = ast.parse(Path(path).read_text())
    node = next(
        (n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == function_name),
        None,
    )
    if node is None:
        raise RuntimeError(f'{function_name} not found in {path}')
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    ns = {'np': np}
    exec(compile(module, str(path), 'exec'), ns, ns)
    return ns[function_name]


def encode_labels(y):
    a = np.asarray(y)
    if np.issubdtype(a.dtype, np.number):
        return (a.astype(np.float64) > 0).astype(np.int64)
    s = np.char.lower(np.char.strip(a.astype(str)))
    known = np.isin(s, ['activity', 'falling', '0', '1'])
    if not np.all(known):
        bad = np.unique(s[~known])[:20]
        raise RuntimeError(f'Unknown labels: {bad.tolist()}')
    return np.isin(s, ['falling', '1']).astype(np.int64)


def split_columns(df):
    fold_col = next((c for c in ['fold', 'outer_fold'] if c in df.columns), None)
    subject_col = next((c for c in ['subject', 'subject_id', 'group_id'] if c in df.columns), None)
    role_col = next((c for c in ['role', 'split', 'set'] if c in df.columns), None)
    if not all([fold_col, subject_col, role_col]):
        raise RuntimeError(f'Cannot resolve split columns from {df.columns.tolist()}')
    return fold_col, subject_col, role_col


def subjects_for_fold(fold):
    df = pd.read_csv(SPLIT_CSV, dtype=str)
    fold_col, subject_col, role_col = split_columns(df)
    rows = df[pd.to_numeric(df[fold_col], errors='raise').astype(int) == int(fold)].copy()
    if rows.empty:
        raise RuntimeError(f'No split rows for fold {fold}')
    roles = {}
    for role in ['train', 'val', 'test']:
        vals = rows[rows[role_col].str.lower() == role][subject_col].astype(str).tolist()
        if not vals:
            raise RuntimeError(f'No {role} subjects for fold {fold}')
        roles[role] = vals
    all_s = roles['train'] + roles['val'] + roles['test']
    if len(all_s) != len(set(all_s)):
        raise RuntimeError(f'Subject overlap in fold {fold}')
    return roles


def iter_trial_dirs(subjects):
    for subject in subjects:
        sp = PHYSICAL_ROOT / str(subject)
        if not sp.is_dir():
            raise RuntimeError(f'Missing subject directory: {sp}')
        for task in sorted([p for p in sp.iterdir() if p.is_dir() and not p.name.startswith('.')], key=lambda p: p.name):
            for trial in sorted([p for p in task.iterdir() if p.is_dir() and not p.name.startswith('.')], key=lambda p: p.name):
                yield str(subject), task.name, trial.name, trial


def count_physical_labels(subjects):
    n0 = n1 = 0
    n_trials = 0
    for _, _, _, td in iter_trial_dirs(subjects):
        yp = td / 'labels.npy'
        xp = td / 'segments.npy'
        if not yp.exists() or not xp.exists():
            raise RuntimeError(f'Missing segments/labels under {td}')
        y = encode_labels(np.load(yp, allow_pickle=False))
        n0 += int((y == 0).sum())
        n1 += int((y == 1).sum())
        n_trials += 1
    return n0 + n1, n0, n1, n_trials


def assert_expected_counts(fold, role, counts):
    observed = tuple(int(x) for x in counts[:3])
    expected = EXPECTED[int(fold)][role]
    if observed != expected:
        raise RuntimeError(f'{role} count mismatch fold={fold}: observed={observed}, expected={expected}')


def load_physical(subjects, activity_only=False, return_ranges=False):
    xs, ys = [], []
    ranges = []
    cursor = 0
    for subject, task, trial, td in iter_trial_dirs(subjects):
        x = np.load(td / 'segments.npy', allow_pickle=False)
        y = encode_labels(np.load(td / 'labels.npy', allow_pickle=False))
        if x.ndim != 3 or x.shape[1:] != (30, 9) or len(x) != len(y):
            raise RuntimeError(f'Bad trial shape at {td}: X={x.shape}, y={y.shape}')
        if activity_only:
            mask = (y == 0)
            if not np.any(mask):
                continue
            x = x[mask]
            y = y[mask]
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        xs.append(x)
        ys.append(y)
        if return_ranges:
            start = cursor
            cursor += len(y)
            ranges.append({
                'subject_id': subject,
                'task_id': task,
                'trial_id': trial,
                'trial_dir': str(td),
                'start_index': start,
                'end_index_exclusive': cursor,
                'n_windows': len(y),
                'true_event': int(np.any(y == 1)),
            })
    if not xs:
        raise RuntimeError('No physical windows loaded')
    X = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    return X, y, pd.DataFrame(ranges) if return_ranges else None


def find_draw_file(fold, key):
    exact = DRAW_ROOT / f'fold_{fold}_{key}_synthetic_draws.csv'
    if exact.exists():
        return exact
    matches = sorted(DRAW_ROOT.glob(f'*fold*{fold}*{key}*synthetic_draws.csv'))
    if len(matches) == 1:
        return matches[0]
    matches = sorted(DRAW_ROOT.glob(f'*fold*{fold}*{key}*.csv'))
    if len(matches) == 1:
        return matches[0]
    raise RuntimeError(f'Cannot uniquely resolve draw manifest for fold={fold}, key={key}; matches={matches}')


def find_draw_index_column(df, n_pool):
    priority = [
        'pool_index', 'synthetic_pool_index', 'aligned_pool_index',
        'synthetic_index', 'aligned_index', 'pool_idx', 'synthetic_idx',
        'window_index', 'source_window_index',
    ]
    for c in priority:
        if c in df.columns:
            v = pd.to_numeric(df[c], errors='coerce')
            if v.notna().all() and np.all(np.equal(v, np.floor(v))) and int(v.min()) >= 0 and int(v.max()) < n_pool:
                return c
    candidates = []
    excluded = {'draw_index', 'draw_order', 'fold', 'task_id', 'trial_id', 'source_row_start', 'source_row_end'}
    for c in df.columns:
        lc = c.lower()
        if c in excluded or not any(t in lc for t in ['index', 'idx', 'pool']):
            continue
        v = pd.to_numeric(df[c], errors='coerce')
        if not v.notna().all() or not np.all(np.equal(v, np.floor(v))):
            continue
        if int(v.min()) < 0 or int(v.max()) >= n_pool:
            continue
        score = (4 if 'pool' in lc else 0) + (3 if 'synthetic' in lc else 0) + (2 if 'aligned' in lc else 0) + (1 if 'window' in lc else 0)
        candidates.append((score, c))
    if not candidates:
        raise RuntimeError(f'Cannot resolve synthetic pool index column from {df.columns.tolist()}')
    candidates.sort(reverse=True)
    return candidates[0][1]


def expected_draws(experiment, real_falling):
    if experiment == 'EXP01_REAL_ONLY':
        return 0
    if experiment == 'EXP02_SIM_FALL_SUBSTITUTION':
        return int(real_falling)
    return int(np.floor(real_falling * RATIO[experiment] + 0.5))


def load_synthetic_draws(experiment, fold, real_falling, load_x=True):
    if experiment == 'EXP01_REAL_ONLY':
        return None, None, None
    n_pool = int(np.load(SYN_X, mmap_mode='r').shape[0])
    if n_pool != 946:
        raise RuntimeError(f'Expected aligned synthetic pool=946, found {n_pool}')
    draw_file = find_draw_file(fold, DRAW_KEY[experiment])
    draw = pd.read_csv(draw_file)
    if len(draw) != expected_draws(experiment, real_falling):
        raise RuntimeError(f'Draw count mismatch {experiment} fold={fold}: {len(draw)} vs {expected_draws(experiment, real_falling)}')
    idx_col = find_draw_index_column(draw, n_pool)
    idx = pd.to_numeric(draw[idx_col], errors='raise').astype(int).to_numpy()
    if idx.min(initial=0) < 0 or idx.max(initial=0) >= n_pool:
        raise RuntimeError('Synthetic draw index out of range')
    if not load_x:
        return None, draw, {'draw_file': str(draw_file), 'index_column': idx_col, 'indices': idx}
    pool = np.load(SYN_X, mmap_mode='r')
    x = np.array(pool[idx], dtype=np.float32, copy=True)
    if x.ndim != 3 or x.shape[1:] != (30, 9):
        raise RuntimeError(f'Bad synthetic shape: {x.shape}')
    x[:, :, 0:3] /= np.float32(0.00980665)
    x[:, :, 3:6] *= np.float32(1000.0)
    y = np.ones(len(x), dtype=np.int64)
    return x, y, {'draw_file': str(draw_file), 'index_column': idx_col, 'indices': idx, 'draw_df': draw}


def make_tensor_dataset(X, y):
    return TensorDataset(torch.from_numpy(X), torch.from_numpy(y))


def binary_metrics(y_true, y_pred, prob_fall=None):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]
    out = {
        'n': int(len(y_true)),
        'n_activity': int((y_true == 0).sum()),
        'n_falling': int((y_true == 1).sum()),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'precision_fall': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall_fall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1_fall': float(f1_score(y_true, y_pred, zero_division=0)),
        'specificity': float(tn / (tn + fp)) if (tn + fp) else float('nan'),
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
    }
    if prob_fall is not None and len(np.unique(y_true)) == 2:
        out['roc_auc'] = float(roc_auc_score(y_true, prob_fall))
        out['average_precision'] = float(average_precision_score(y_true, prob_fall))
    return out


def train_model(model, train_loader, val_loader, device, outdir):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    criterion = nn.CrossEntropyLoss()
    best_loss = float('inf')
    best_epoch = 0
    wait = 0
    history = []
    best_path = outdir / 'best_model.pth'
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        n_seen = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            bs = int(yb.shape[0])
            loss_sum += float(loss.item()) * bs
            n_seen += bs
        train_loss = loss_sum / max(n_seen, 1)
        model.eval()
        val_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                logits = model(xb)
                loss = criterion(logits, yb)
                bs = int(yb.shape[0])
                val_sum += float(loss.item()) * bs
                val_n += bs
        val_loss = val_sum / max(val_n, 1)
        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss})
        print(f'epoch={epoch:03d} train_loss={train_loss:.8f} val_loss={val_loss:.8f}', flush=True)
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            wait = 0
            torch.save(model.state_dict(), best_path)
        else:
            wait += 1
        if wait >= PATIENCE:
            print(f'EARLY STOP epoch={epoch} best_epoch={best_epoch}', flush=True)
            break
    state = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    pd.DataFrame(history).to_csv(outdir / 'training_history.csv', index=False)
    return best_epoch, best_loss


def predict(model, loader, device):
    model.eval()
    true_all, pred_all, prob_all = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32, copy=False)
            true_all.append(yb.numpy().astype(np.int64, copy=False))
            pred_all.append(np.argmax(probs, axis=1).astype(np.int64))
            prob_all.append(probs)
    return np.concatenate(true_all), np.concatenate(pred_all), np.concatenate(prob_all)


def event_evaluation(test_ranges, y_true, probs, cnn_predictions_from_probabilities, is_simulation_passed_threshold):
    rows = []
    for r in test_ranges.to_dict('records'):
        a = int(r['start_index'])
        b = int(r['end_index_exclusive'])
        yt = np.asarray(y_true[a:b], dtype=np.int64)
        pp = np.asarray(probs[a:b], dtype=np.float64)
        yp = np.asarray(cnn_predictions_from_probabilities(pp, prediction_bias=PREDICTION_BIAS), dtype=np.int64)
        passed = bool(is_simulation_passed_threshold(yt, yp, threshold=EVENT_THRESHOLD))
        true_event = int(np.any(yt == 1))
        pred_event = int(true_event if passed else 1 - true_event)
        rows.append({
            **r,
            'predicted_event': pred_event,
            'passed_historical_rule': int(passed),
            'n_true_falling_windows': int((yt == 1).sum()),
            'n_predicted_falling_windows': int((yp == 1).sum()),
            'max_prob_fall': float(pp[:, 1].max()) if len(pp) else float('nan'),
        })
    ev = pd.DataFrame(rows)
    metrics = binary_metrics(ev['true_event'].to_numpy(), ev['predicted_event'].to_numpy())
    return ev, metrics


def dry_run(experiment, fold):
    for p in [PHYSICAL_ROOT, SPLIT_CSV, SYN_X, SYN_META, DRAW_ROOT, ORIG / 'models/CNN.py', ORIG / 'models/modules/IMUNormalizer.py', EVENT_SRC / 'simulation/LearningCurveMetrics.py', EVENT_SRC / 'simulation/helper.py']:
        if not Path(p).exists():
            raise RuntimeError(f'Missing required input: {p}')
    roles = subjects_for_fold(fold)
    print(f'fold={fold} subjects train={len(roles["train"])} val={len(roles["val"])} test={len(roles["test"])}')
    counts = {}
    for role in ['train', 'val', 'test']:
        counts[role] = count_physical_labels(roles[role])
        assert_expected_counts(fold, role, counts[role])
        print(f'{role}: windows={counts[role][0]} activity={counts[role][1]} falling={counts[role][2]} trials={counts[role][3]}')
    real_falling = counts['train'][2]
    _, draw, info = load_synthetic_draws(experiment, fold, real_falling, load_x=False)
    if draw is not None:
        idx = info['indices']
        print(f'synthetic: draws={len(idx)} unique_windows={len(np.unique(idx))} manifest={info["draw_file"]} index_col={info["index_column"]}')
    else:
        print('synthetic: draws=0')
    CNN = load_original_cnn()
    model = CNN(n_features=9, n_classes=2, config=CNN_CONFIG)
    with torch.no_grad():
        out = model(torch.zeros((2, 30, 9), dtype=torch.float32))
    if tuple(out.shape) != (2, 2):
        raise RuntimeError(f'Original CNN dummy forward failed: {tuple(out.shape)}')
    cfun = load_function_from_source(EVENT_SRC / 'simulation/LearningCurveMetrics.py', 'cnn_predictions_from_probabilities')
    efun = load_function_from_source(EVENT_SRC / 'simulation/helper.py', 'is_simulation_passed_threshold')
    p = cfun(np.array([[0.9, 0.1], [0.2, 0.8], [0.55, 0.45]]), prediction_bias=PREDICTION_BIAS)
    if len(p) != 3:
        raise RuntimeError('Historical prediction function preflight failed')
    _ = efun(np.array([0, 1, 1]), np.array([0, 1, 1]), threshold=EVENT_THRESHOLD)
    print('original CNN sha256       :', sha256_file(ORIG / 'models/CNN.py'))
    print('original normalizer sha256:', sha256_file(ORIG / 'models/modules/IMUNormalizer.py'))
    print('prediction_bias           :', PREDICTION_BIAS)
    print('event threshold           :', EVENT_THRESHOLD)
    print('optimizer                 : AdamW')
    print('lr / wd                   :', LR, WD)
    print('batch / epochs / patience :', BATCH_SIZE, MAX_EPOCHS, PATIENCE)
    print('FINAL FOLD DRY-RUN GATE: PASS')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--experiment', required=True, choices=EXPERIMENTS)
    ap.add_argument('--fold', required=True, type=int, choices=range(5))
    ap.add_argument('--output-root', default=str(DEFAULT_OUT))
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    experiment = args.experiment
    fold = int(args.fold)
    if args.dry_run:
        dry_run(experiment, fold)
        return

    seed = model_seed(fold)
    set_seed(seed)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the full campaign run')
    device = torch.device('cuda:0')
    outdir = Path(args.output_root) / experiment / f'fold_{fold}'
    outdir.mkdir(parents=True, exist_ok=True)
    complete = outdir / 'COMPLETE'
    if complete.exists():
        print(f'SKIP COMPLETE: {complete}')
        return

    t0 = time.time()
    roles = subjects_for_fold(fold)
    role_counts = {}
    for role in ['train', 'val', 'test']:
        role_counts[role] = count_physical_labels(roles[role])
        assert_expected_counts(fold, role, role_counts[role])
    real_falling = int(role_counts['train'][2])

    print('=' * 100)
    print(f'{experiment} | FOLD {fold}')
    print('=' * 100)
    print('Physical root:', PHYSICAL_ROOT)
    print('NO additional temporal trimming is performed by this runner.')
    print('Train physical counts:', role_counts['train'][:3])
    print('Val physical counts  :', role_counts['val'][:3])
    print('Test physical counts :', role_counts['test'][:3])

    activity_only = (experiment == 'EXP02_SIM_FALL_SUBSTITUTION')
    train_X, train_y, _ = load_physical(roles['train'], activity_only=activity_only, return_ranges=False)
    val_X, val_y, _ = load_physical(roles['val'], activity_only=False, return_ranges=False)
    test_X, test_y, test_ranges = load_physical(roles['test'], activity_only=False, return_ranges=True)

    syn_X, syn_y, syn_info = load_synthetic_draws(experiment, fold, real_falling, load_x=True)

    train_parts = [make_tensor_dataset(train_X, train_y)]
    synthetic_draws = 0
    if syn_X is not None:
        train_parts.append(make_tensor_dataset(syn_X, syn_y))
        synthetic_draws = int(len(syn_y))
    train_ds = train_parts[0] if len(train_parts) == 1 else ConcatDataset(train_parts)
    val_ds = make_tensor_dataset(val_X, val_y)
    test_ds = make_tensor_dataset(test_X, test_y)

    expected_train_activity = int(role_counts['train'][1])
    expected_train_real_fall = 0 if activity_only else real_falling
    expected_train_total = expected_train_activity + expected_train_real_fall + synthetic_draws
    if len(train_ds) != expected_train_total:
        raise RuntimeError(f'Training composition mismatch: {len(train_ds)} vs {expected_train_total}')

    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=g, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    CNN = load_original_cnn()
    model = CNN(n_features=9, n_classes=2, config=CNN_CONFIG).to(device)
    best_epoch, best_val_loss = train_model(model, train_loader, val_loader, device, outdir)

    y_true, y_pred, probs = predict(model, test_loader, device)
    seg_metrics = binary_metrics(y_true, y_pred, probs[:, 1])
    seg_metrics.update({'experiment': experiment, 'fold': fold, 'best_epoch': int(best_epoch), 'best_val_loss': float(best_val_loss)})

    cnn_predictions_from_probabilities = load_function_from_source(EVENT_SRC / 'simulation/LearningCurveMetrics.py', 'cnn_predictions_from_probabilities')
    is_simulation_passed_threshold = load_function_from_source(EVENT_SRC / 'simulation/helper.py', 'is_simulation_passed_threshold')
    ev_df, ev_metrics = event_evaluation(test_ranges, y_true, probs, cnn_predictions_from_probabilities, is_simulation_passed_threshold)
    ev_metrics.update({'experiment': experiment, 'fold': fold, 'prediction_bias': PREDICTION_BIAS, 'threshold': EVENT_THRESHOLD})

    with (outdir / 'segment_metrics.json').open('w') as f:
        json.dump(seg_metrics, f, indent=2)
    with (outdir / 'event_metrics.json').open('w') as f:
        json.dump(ev_metrics, f, indent=2)
    pd.DataFrame([seg_metrics]).to_csv(outdir / 'segment_metrics.csv', index=False)
    pd.DataFrame([ev_metrics]).to_csv(outdir / 'event_metrics.csv', index=False)
    test_ranges.to_csv(outdir / 'test_trial_ranges.csv', index=False)
    ev_df.to_csv(outdir / 'event_predictions.csv', index=False)
    np.savez_compressed(outdir / 'segment_predictions.npz', true_label=y_true, predicted_label=y_pred, probabilities=probs)
    pd.DataFrame([[seg_metrics['tn'], seg_metrics['fp']], [seg_metrics['fn'], seg_metrics['tp']]], index=['true_activity', 'true_falling'], columns=['pred_activity', 'pred_falling']).to_csv(outdir / 'segment_confusion_matrix.csv')
    pd.DataFrame([[ev_metrics['tn'], ev_metrics['fp']], [ev_metrics['fn'], ev_metrics['tp']]], index=['true_activity_event', 'true_fall_event'], columns=['pred_activity_event', 'pred_fall_event']).to_csv(outdir / 'event_confusion_matrix.csv')

    run_cfg = {
        'experiment': experiment,
        'fold': fold,
        'model_seed': seed,
        'split_seed': SPLIT_SEED,
        'physical_root': str(PHYSICAL_ROOT),
        'split_manifest': str(SPLIT_CSV),
        'synthetic_pool': str(SYN_X),
        'synthetic_draw_manifest': None if syn_info is None else syn_info['draw_file'],
        'no_additional_150ms_trimming': True,
        'synthetic_conversion_at_model_boundary': {'acc_mps2_to_mg': '/0.00980665', 'gyro_dps_to_mdps': '*1000'},
        'physical_units_at_model_boundary': 'stored Protechto raw mg/mdps',
        'cnn_config': CNN_CONFIG,
        'optimizer': 'AdamW',
        'lr': LR,
        'weight_decay': WD,
        'criterion': 'CrossEntropyLoss unweighted',
        'batch_size': BATCH_SIZE,
        'max_epochs': MAX_EPOCHS,
        'early_stopping_patience': PATIENCE,
        'prediction_bias': PREDICTION_BIAS,
        'event_threshold': EVENT_THRESHOLD,
        'train_real_activity': expected_train_activity,
        'train_real_falling': expected_train_real_fall,
        'train_synthetic_falling': synthetic_draws,
        'val_windows': int(len(val_ds)),
        'test_windows': int(len(test_ds)),
        'train_subjects': roles['train'],
        'val_subjects': roles['val'],
        'test_subjects': roles['test'],
        'source_hashes': {
            'CNN.py': sha256_file(ORIG / 'models/CNN.py'),
            'IMUNormalizer.py': sha256_file(ORIG / 'models/modules/IMUNormalizer.py'),
            'LearningCurveMetrics.py': sha256_file(EVENT_SRC / 'simulation/LearningCurveMetrics.py'),
            'simulation/helper.py': sha256_file(EVENT_SRC / 'simulation/helper.py'),
        },
        'elapsed_seconds': float(time.time() - t0),
    }
    with (outdir / 'run_config.json').open('w') as f:
        json.dump(run_cfg, f, indent=2)

    complete.write_text('PASS\n')
    print('SEGMENT RESULT:', json.dumps(seg_metrics, indent=2), flush=True)
    print('EVENT RESULT:', json.dumps(ev_metrics, indent=2), flush=True)
    print(f'{experiment} | FOLD {fold} COMPLETE', flush=True)


if __name__ == '__main__':
    main()
