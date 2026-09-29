#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import gc
import json
import os
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import lightning as pl
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

PROJECT = Path('/mnt/hdd16T/ToqeerHomeBackup/mujoco_project')
ROOT = Path('/mnt/hdd16T/ToqeerHomeBackup/toqeer/Protechto-master')
EXEC = PROJECT / 'outputs/protechto_clean_execution_mirror_v1'
CACHE = PROJECT / 'outputs/protechto_exact_synthetic_cache_v1'
PHYSICAL_ROOT = ROOT / 'data/dataset/segments/300ms_50ov_npseg_filt_binary'
DATASET_REL = 'dataset/segments/300ms_50ov_npseg_filt_binary'
DEFAULT_OUT = PROJECT / 'outputs/protechto_exact_full_campaign_v1'
GPU = 1
BATCH = 64
RATIOS = {
    'EXP04_MIX20': 0.20,
    'EXP05_MIX50': 0.50,
    'EXP06_MIX70': 0.70,
    'EXP07_MIX100': 1.00,
}
PHYSICAL_CONDITIONS = [
    'EXP01_REAL_ONLY',
    'EXP02_SIM_FALL_SUBSTITUTION',
    'EXP04_MIX20',
    'EXP05_MIX50',
    'EXP06_MIX70',
    'EXP07_MIX100',
]
ALL_CONDITIONS = [
    'EXP01_REAL_ONLY',
    'EXP02_SIM_FALL_SUBSTITUTION',
    'EXP03_SIM_ONLY_FULL',
    'EXP04_MIX20',
    'EXP05_MIX50',
    'EXP06_MIX70',
    'EXP07_MIX100',
]

os.chdir(ROOT)
sys.path.insert(0, str(EXEC))

import constants as const
from dataloaders.KFoldDataloader import KFoldDataloader
from dataloaders.helper import dataloader
from models.CNN import CNN
from models.configs.CNN_config import best_config
from models.wrappers.Predictor import Predictor
from models.wrappers.Simulator import Simulator
from simulation.helper import is_simulation_passed_threshold

pl.seed_everything = pl.seed_everything  # explicit no-op reference; no seed is set by this campaign.
torch.set_float32_matmul_precision('high')


def round_half_up(x: float) -> int:
    return int(np.floor(float(x) + 0.5))


def metrics_from_arrays(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    acc = (tn + tp) / max(1, tn + fp + fn + tp)
    bal = 0.5 * (tn / max(1, tn + fp) + tp / max(1, tp + fn))
    return {
        'n': int(len(y_true)), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        'accuracy': float(acc), 'balanced_accuracy': float(bal),
        'precision_activity': float(p[0]), 'recall_activity': float(r[0]), 'f1_activity': float(f[0]),
        'precision_fall': float(p[1]), 'recall_fall': float(r[1]), 'f1_fall': float(f[1]),
        'support_activity': int(s[0]), 'support_fall': int(s[1]),
        'specificity': float(tn / max(1, tn + fp)),
    }


def event_metrics_from_counts(tn, fp, fn, tp):
    yt = np.concatenate([np.zeros(tn + fp, dtype=np.int64), np.ones(fn + tp, dtype=np.int64)])
    yp = np.concatenate([
        np.zeros(tn, dtype=np.int64), np.ones(fp, dtype=np.int64),
        np.zeros(fn, dtype=np.int64), np.ones(tp, dtype=np.int64)
    ])
    return metrics_from_arrays(yt, yp)


def trial_balanced_order(meta_fall: pd.DataFrame, seed: int, needed: int):
    # Unique-first, trial-balanced. Reuse only after every unique synthetic Falling window has been used once.
    groups = {}
    for uid, g in meta_fall.groupby('trial_uid_resolved', sort=True):
        groups[str(uid)] = list(map(int, g['synthetic_row'].tolist()))
    rng = np.random.default_rng(seed)
    for uid in groups:
        rng.shuffle(groups[uid])
    trial_order = list(groups)
    rng.shuffle(trial_order)

    unique = []
    pos = {u: 0 for u in trial_order}
    while len(unique) < sum(len(v) for v in groups.values()):
        progressed = False
        for u in trial_order:
            i = pos[u]
            if i < len(groups[u]):
                unique.append(groups[u][i])
                pos[u] += 1
                progressed = True
        if not progressed:
            break
    if len(unique) != len(meta_fall):
        raise RuntimeError('Synthetic unique-first order construction failed')

    rows = []
    cycle = 0
    while len(rows) < needed:
        seq = unique.copy()
        if cycle > 0:
            # deterministic reshuffle only between complete reuse cycles
            rr = np.random.default_rng(seed + 100003 * cycle)
            rr.shuffle(seq)
        for idx in seq:
            rows.append((idx, cycle))
            if len(rows) >= needed:
                break
        cycle += 1
    return rows


def write_draw_manifest(out_root, condition, fold, meta, selected):
    d = out_root / 'synthetic_draw_manifests'
    d.mkdir(parents=True, exist_ok=True)
    p = d / f'{condition}_fold{fold}.csv'
    rows = []
    meta_idx = meta.set_index('synthetic_row', drop=False)
    for pos, (idx, cycle) in enumerate(selected):
        r = meta_idx.loc[int(idx)]
        rows.append({
            'draw_position': pos,
            'synthetic_row': int(idx),
            'reuse_cycle': int(cycle),
            'trial_uid_resolved': str(r['trial_uid_resolved']),
            'label01': int(r['label01']),
        })
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def load_synthetic_cache():
    X = np.load(CACHE / 'synthetic_X_protechto_rawunits.npy', mmap_mode='r')
    y = np.load(CACHE / 'synthetic_y01.npy').astype(np.int64)
    meta = pd.read_csv(CACHE / 'synthetic_metadata_resolved.csv')
    if len(X) != len(y) or len(X) != len(meta):
        raise RuntimeError('Synthetic cache length mismatch')
    return X, y, meta


def clean_torch():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def make_trainer(ckpt_dir: Path, fold1: int):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename=f'best-checkpoint-fold_{fold1}',
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
    )
    early_stopping = EarlyStopping(monitor='val_loss', patience=20)
    logger = TensorBoardLogger(str(ROOT / 'lightning_logs'), name='UniVrFall')
    trainer = Trainer(
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping],
        max_epochs=100,
        accelerator='gpu',
        devices=[GPU],
        enable_progress_bar=True,
    )
    return trainer, checkpoint_callback


def train_physical_condition(condition, run_id, out_root, Xs, ys, meta):
    print('\n' + '=' * 120)
    print('START PHYSICAL-TEST CONDITION:', condition)
    print('=' * 120)
    condition_marker = out_root / 'conditions' / condition / 'COMPLETE'
    if condition_marker.exists():
        print('SKIP COMPLETE:', condition)
        return

    ckpt_dir = ROOT / 'checkpoints/CNN/300ms' / run_id / condition
    fold_state = out_root / 'conditions' / condition / 'folds'
    fold_state.mkdir(parents=True, exist_ok=True)

    synth_fall_rows = np.where(ys == 1)[0]
    meta_fall = meta[meta['synthetic_row'].isin(synth_fall_rows)].copy()

    dm = KFoldDataloader(root_directory=str(PHYSICAL_ROOT), k=5, batch_size=BATCH)

    for fold0 in range(5):
        fold1 = fold0 + 1
        complete = fold_state / f'fold_{fold1}.COMPLETE'
        if complete.exists() and (ckpt_dir / f'best-checkpoint-fold_{fold1}.ckpt').exists():
            print(f'{condition} fold {fold1}: resume skip COMPLETE')
            continue

        print('\n' + '-' * 100)
        print(f'{condition} | FOLD {fold1}')
        print('-' * 100)
        clean_torch()
        dm.construct(fold_index=fold0, verbose=True)

        if condition != 'EXP01_REAL_ONLY':
            x_train_t, y_train_t = dm.train_dl.dataset.tensors
            x_train = x_train_t.cpu().numpy()
            y_train = y_train_t.cpu().numpy().astype(np.int64)
            n_real_fall = int((y_train == 1).sum())
            n_real_activity = int((y_train == 0).sum())

            if condition == 'EXP02_SIM_FALL_SUBSTITUTION':
                target = n_real_fall
            else:
                target = round_half_up(n_real_fall * RATIOS[condition])

            selected = trial_balanced_order(meta_fall, seed=420100 + fold0 * 1000, needed=target)
            write_draw_manifest(out_root, condition, fold1, meta, selected)
            idx = np.array([r[0] for r in selected], dtype=np.int64)
            x_syn = np.asarray(Xs[idx], dtype=np.float32)
            y_syn = np.ones(len(idx), dtype=np.int64)

            if condition == 'EXP02_SIM_FALL_SUBSTITUTION':
                keep = (y_train == 0)
                x_new = np.concatenate([x_train[keep], x_syn], axis=0)
                y_new = np.concatenate([y_train[keep], y_syn], axis=0)
            else:
                x_new = np.concatenate([x_train, x_syn], axis=0)
                y_new = np.concatenate([y_train, y_syn], axis=0)

            dm.train_dl = dataloader(x_new, y_new, batch_size=BATCH, shuffle=True)
            print('Real train Activity/Falling:', n_real_activity, n_real_fall)
            print('Synthetic Falling requested:', target)
            print('Synthetic unique rows used   :', len(np.unique(idx)))
            print('Final train Activity/Falling :', int((y_new == 0).sum()), int((y_new == 1).sum()))
        else:
            y_train = dm.train_dl.dataset.tensors[1].cpu().numpy().astype(np.int64)
            print('REAL_ONLY train Activity/Falling:', int((y_train == 0).sum()), int((y_train == 1).sum()))

        cfg = dict(best_config)
        model = Predictor(
            CNN,
            n_features=dm.input_shape[1],
            n_classes=dm.label_encoder.classes_.shape[0],
            classes=dm.label_encoder.classes_,
            experiment=f'300ms/{run_id}/{condition}',
            fold=fold1,
            cfg=cfg,
        )
        trainer, cb = make_trainer(ckpt_dir, fold1)
        trainer.fit(model, dm)
        # Preserve original train.py behavior: evaluate the current model after training.
        trainer.test(model, dataloaders=dm.test_dl)
        best = Path(cb.best_model_path)
        expected = ckpt_dir / f'best-checkpoint-fold_{fold1}.ckpt'
        if best.resolve() != expected.resolve():
            if not best.exists():
                raise RuntimeError(f'ModelCheckpoint did not produce a best checkpoint for {condition} fold {fold1}')
            # Lightning can append a version suffix only on collision; normalize within this isolated campaign directory.
            if expected.exists():
                expected.unlink()
            best.replace(expected)
        if not expected.exists():
            raise RuntimeError(f'Missing expected checkpoint: {expected}')
        complete.write_text('COMPLETE\n')
        print(f'{condition} | FOLD {fold1} COMPLETE')
        del trainer, model
        clean_torch()

    # Final evaluation is the untouched clean Protechto test.py pipeline on physical sessions.
    ckpt_rel = f'checkpoints/CNN/300ms/{run_id}/{condition}'
    cmd = [
        sys.executable, '-u', str(EXEC / 'test.py'),
        '-m', 'CNN', '-c', ckpt_rel, '-t', 'k-fold', '-d', DATASET_REL, '-g', str(GPU)
    ]
    print('+', ' '.join(cmd))
    rc = subprocess.run(cmd, cwd=ROOT).returncode
    if rc != 0:
        raise RuntimeError(f'Original Protechto test.py failed for {condition}, status={rc}')

    result_dir = ROOT / 'results/CNN/300ms' / run_id / condition
    for p in [
        result_dir / 'CNN_confusion_matrix_GLOBAL.csv',
        result_dir / 'CNN_EVENT_confusion_matrix_GLOBAL.csv',
        result_dir / 'event_stats__bias_0.65.csv',
    ]:
        if not p.exists():
            raise RuntimeError(f'Missing final original-Protechto result: {p}')
    condition_marker.parent.mkdir(parents=True, exist_ok=True)
    condition_marker.write_text('COMPLETE\n')
    print('PHYSICAL-TEST CONDITION COMPLETE:', condition)


class SimDataModule(pl.LightningDataModule):
    def __init__(self, X, y, meta, fold0):
        super().__init__()
        self.X = X
        self.y = y
        self.meta = meta
        self.fold0 = fold0
        self.label_encoder = LabelEncoder().fit(['Activity', 'Falling'])
        self.input_shape = (30, 9)
        self.train_dl = self.validation_dl = self.test_dl = None
        self.train_uids = self.val_uids = self.test_uids = None

    def construct(self, verbose=True):
        uids = np.array(sorted(self.meta['trial_uid_resolved'].astype(str).unique()))
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        train_idx, test_idx = list(kf.split(uids))[self.fold0]
        train_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=42)
        self.train_uids = uids[train_idx]
        self.val_uids = uids[val_idx]
        self.test_uids = uids[test_idx]

        def arr(which):
            mask = self.meta['trial_uid_resolved'].astype(str).isin(which).to_numpy()
            return np.asarray(self.X[mask], dtype=np.float32), self.y[mask].astype(np.int64)

        xtr, ytr = arr(self.train_uids)
        xva, yva = arr(self.val_uids)
        xte, yte = arr(self.test_uids)
        self.train_dl = dataloader(xtr, ytr, batch_size=BATCH, shuffle=True)
        self.validation_dl = dataloader(xva, yva, batch_size=BATCH, shuffle=False)
        self.test_dl = dataloader(xte, yte, batch_size=BATCH, shuffle=False)
        if verbose:
            print('SIM train trials/windows/classes:', len(self.train_uids), len(ytr), np.bincount(ytr, minlength=2).tolist())
            print('SIM val   trials/windows/classes:', len(self.val_uids), len(yva), np.bincount(yva, minlength=2).tolist())
            print('SIM test  trials/windows/classes:', len(self.test_uids), len(yte), np.bincount(yte, minlength=2).tolist())

    def train_dataloader(self): return self.train_dl
    def val_dataloader(self): return self.validation_dl
    def test_dataloader(self): return self.test_dl


def evaluate_sim_fold(ckpt, dm, Xs, ys, meta, out_dir, fold1):
    device = torch.device(f'cuda:{GPU}')
    sim = Simulator(
        CNN, n_features=9, n_classes=2,
        classes=np.array(['Activity', 'Falling']),
        best_checkpoint_path=str(ckpt), fold=fold1, cfg=dict(best_config)
    ).to(device)
    sim.eval()

    fold_meta = meta[meta['trial_uid_resolved'].astype(str).isin(dm.test_uids)].copy()
    seg_true, seg_pred = [], []
    event_rows = []
    for uid, g in fold_meta.groupby('trial_uid_resolved', sort=True):
        g = g.sort_values(['order_resolved', 'synthetic_row'])
        idx = g['synthetic_row'].to_numpy(dtype=np.int64)
        x = torch.tensor(np.asarray(Xs[idx], dtype=np.float32), device=device)
        yt = ys[idx].astype(np.int64)
        with torch.no_grad():
            logits = sim.model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        yp = Simulator.get_output(probs, prediction_bias=float(best_config['prediction_bias']))
        seg_true.append(yt)
        seg_pred.append(yp)
        passed = bool(is_simulation_passed_threshold(yt, yp, threshold=2))
        true_event = int(np.any(yt == 1))
        pred_event = true_event if passed else 1 - true_event
        event_rows.append({
            'trial_uid': str(uid), 'true_event': true_event, 'pred_event': int(pred_event),
            'passed': int(passed), 'n_windows': int(len(idx)),
            'n_activity_windows': int((yt == 0).sum()), 'n_falling_windows': int((yt == 1).sum()),
        })
    yt = np.concatenate(seg_true)
    yp = np.concatenate(seg_pred)
    sm = metrics_from_arrays(yt, yp)
    evdf = pd.DataFrame(event_rows)
    em = metrics_from_arrays(evdf['true_event'].to_numpy(), evdf['pred_event'].to_numpy())
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / 'segment_predictions.npz', y_true=yt, y_pred=yp)
    evdf.to_csv(out_dir / 'event_predictions.csv', index=False)
    (out_dir / 'segment_metrics.json').write_text(json.dumps(sm, indent=2, sort_keys=True))
    (out_dir / 'event_metrics.json').write_text(json.dumps(em, indent=2, sort_keys=True))
    return sm, em


def train_sim_only(run_id, out_root, Xs, ys, meta):
    condition = 'EXP03_SIM_ONLY_FULL'
    marker = out_root / 'conditions' / condition / 'COMPLETE'
    if marker.exists():
        print('SKIP COMPLETE:', condition)
        return
    print('\n' + '=' * 120)
    print('START SIMULATION-ONLY CONDITION:', condition)
    print('=' * 120)
    ckpt_dir = ROOT / 'checkpoints/CNN/300ms' / run_id / condition
    state = out_root / 'conditions' / condition / 'folds'
    state.mkdir(parents=True, exist_ok=True)

    for fold0 in range(5):
        fold1 = fold0 + 1
        fold_marker = state / f'fold_{fold1}.COMPLETE'
        ckpt = ckpt_dir / f'best-checkpoint-fold_{fold1}.ckpt'
        eval_dir = out_root / 'conditions' / condition / f'fold_{fold1}'
        if fold_marker.exists() and ckpt.exists() and (eval_dir / 'segment_metrics.json').exists():
            print(f'{condition} fold {fold1}: resume skip COMPLETE')
            continue
        clean_torch()
        dm = SimDataModule(Xs, ys, meta, fold0)
        dm.construct(verbose=True)
        model = Predictor(
            CNN, n_features=9, n_classes=2,
            classes=dm.label_encoder.classes_,
            experiment=f'300ms/{run_id}/{condition}', fold=fold1, cfg=dict(best_config)
        )
        trainer, cb = make_trainer(ckpt_dir, fold1)
        trainer.fit(model, dm)
        trainer.test(model, dataloaders=dm.test_dl)
        best = Path(cb.best_model_path)
        if best.resolve() != ckpt.resolve():
            if ckpt.exists(): ckpt.unlink()
            best.replace(ckpt)
        evaluate_sim_fold(ckpt, dm, Xs, ys, meta, eval_dir, fold1)
        fold_marker.write_text('COMPLETE\n')
        del trainer, model, dm
        clean_torch()
        print(f'{condition} | FOLD {fold1} COMPLETE')

    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text('COMPLETE\n')
    print('SIMULATION-ONLY CONDITION COMPLETE')


def physical_fold_event_metrics(result_dir: Path, fold1: int):
    p = result_dir / str(fold1) / 'event_stats__bias_0.65.csv'
    df = pd.read_csv(p)
    tn = fp = fn = tp = 0
    for r in df.itertuples():
        task = int(float(r.task))
        passed = int(r.passed_simulations)
        missed = int(r.missed_simulations)
        if task in const.FALL_TASKS:
            tp += passed; fn += missed
        else:
            tn += passed; fp += missed
    return event_metrics_from_counts(tn, fp, fn, tp)


def aggregate(run_id, out_root):
    rows = []
    for condition in PHYSICAL_CONDITIONS:
        result_dir = ROOT / 'results/CNN/300ms' / run_id / condition
        for fold1 in range(1, 6):
            y_true = np.load(result_dir / str(fold1) / 'y_true.npy').astype(np.int64)
            y_pred = np.load(result_dir / str(fold1) / 'y_pred.npy').astype(np.int64)
            sm = metrics_from_arrays(y_true, y_pred)
            em = physical_fold_event_metrics(result_dir, fold1)
            for level, m in [('segment', sm), ('event', em)]:
                rows.append({'condition': condition, 'fold': fold1, 'level': level, **m})

    sim_cond = 'EXP03_SIM_ONLY_FULL'
    for fold1 in range(1, 6):
        d = out_root / 'conditions' / sim_cond / f'fold_{fold1}'
        for level in ['segment', 'event']:
            m = json.loads((d / f'{level}_metrics.json').read_text())
            rows.append({'condition': sim_cond, 'fold': fold1, 'level': level, **m})

    df = pd.DataFrame(rows)
    df.to_csv(out_root / 'all_fold_metrics.csv', index=False)
    metric_cols = [
        'accuracy', 'balanced_accuracy', 'precision_fall', 'recall_fall', 'f1_fall',
        'specificity', 'precision_activity', 'recall_activity', 'f1_activity'
    ]
    summary = df.groupby(['condition', 'level'])[metric_cols].agg(['mean', 'std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(out_root / 'summary_mean_std.csv', index=False)

    print('\n' + '=' * 120)
    print('FINAL CAMPAIGN SUMMARY')
    print('=' * 120)
    show = summary[['condition', 'level', 'precision_fall_mean', 'recall_fall_mean', 'f1_fall_mean', 'specificity_mean']]
    print(show.to_string(index=False, float_format=lambda x: f'{x:.6f}'))
    print('=' * 120)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-root', default=str(DEFAULT_OUT))
    args = ap.parse_args()
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    run_id_file = out_root / 'RUN_ID'
    if run_id_file.exists():
        run_id = run_id_file.read_text().strip()
    else:
        run_id = time.strftime('PROTECHTO_EXACT_FULL_%Y%m%d_%H%M%S')
        run_id_file.write_text(run_id + '\n')
    print('RUN_ID:', run_id)
    print('Canonical physical root:', PHYSICAL_ROOT)
    print('Synthetic cache:', CACHE)
    print('GPU:', GPU)
    print('NO manual physical temporal trimming.')
    print('NO manual synthetic temporal trimming.')

    Xs, ys, meta = load_synthetic_cache()

    # Condition-major campaign. Physical validation/test are untouched for all physical-test conditions.
    train_physical_condition('EXP01_REAL_ONLY', run_id, out_root, Xs, ys, meta)
    train_physical_condition('EXP02_SIM_FALL_SUBSTITUTION', run_id, out_root, Xs, ys, meta)
    train_sim_only(run_id, out_root, Xs, ys, meta)
    train_physical_condition('EXP04_MIX20', run_id, out_root, Xs, ys, meta)
    train_physical_condition('EXP05_MIX50', run_id, out_root, Xs, ys, meta)
    train_physical_condition('EXP06_MIX70', run_id, out_root, Xs, ys, meta)
    train_physical_condition('EXP07_MIX100', run_id, out_root, Xs, ys, meta)

    aggregate(run_id, out_root)
    (out_root / 'FINAL_CAMPAIGN_COMPLETE').write_text('COMPLETE\n')
    print('PROTECHTO EXACT FULL CAMPAIGN ARTIFACT GATE: PASS')
    print('PROTECHTO EXACT FULL CAMPAIGN COMPLETE')

if __name__ == '__main__':
    main()
