#!/usr/bin/env python3
"""
Synthetic-only TRAINING with REAL physical VALIDATION and REAL physical TEST.

This is a new cross-domain transfer experiment built from the verified final
campaign:
    run_protechto_exact_full_campaign.py

What stays the same:
  - same frozen synthetic cache
  - same canonical Protechto CNN
  - same normalization / 300-ms input / 5-Hz preprocessing already in data
  - same AdamW hyperparameters
  - same batch size = 64
  - same max epochs = 100
  - same early stopping patience = 20
  - same prediction bias = 0.65
  - same 2-consecutive-window event rule
  - same 5-fold physical subject split (KFold seed 42)
  - same physical inner validation split (80/20, random_state 42)
  - same untouched Protechto physical test pipeline

What changes:
  - MODEL TRAINING uses ONLY the complete synthetic cache:
        18,824 Activity + 1,256 Falling = 20,080 windows
  - VALIDATION is REAL physical data for the corresponding physical fold
  - TEST is REAL held-out physical data for the corresponding physical fold
  - NO physical training windows are used for gradient updates

Important interpretation:
  This is NOT a strict zero-shot synthetic->real experiment because real
  validation data are used for early stopping/model selection. It is a
  "synthetic-only training with real-domain validation and test" experiment.

Default output:
  /mnt/hdd16T/ToqeerHomeBackup/mujoco_project/
      outputs/protechto_sim_train_real_val_test_v1
"""

from __future__ import annotations

from pathlib import Path
import argparse
import gc
import json
import os
import subprocess
import sys
import time


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
DEFAULT_OUT = PROJECT / 'outputs/protechto_sim_train_real_val_test_v1'

CONDITION = 'EXP08_SIM_TRAIN_REAL_VAL_TEST'
GPU = 1
BATCH = 64

os.chdir(ROOT)
sys.path.insert(0, str(EXEC))

import constants as const
from dataloaders.helper import load, dataloader
from models.CNN import CNN
from models.configs.CNN_config import best_config
from models.wrappers.Predictor import Predictor

torch.set_float32_matmul_precision('high')


def clean_torch():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def metrics_from_arrays(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )

    acc = (tn + tp) / max(1, tn + fp + fn + tp)
    bal = 0.5 * (
        tn / max(1, tn + fp)
        + tp / max(1, tp + fn)
    )

    return {
        'n': int(len(y_true)),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'accuracy': float(acc),
        'balanced_accuracy': float(bal),
        'precision_activity': float(p[0]),
        'recall_activity': float(r[0]),
        'f1_activity': float(f[0]),
        'precision_fall': float(p[1]),
        'recall_fall': float(r[1]),
        'f1_fall': float(f[1]),
        'support_activity': int(s[0]),
        'support_fall': int(s[1]),
        'specificity': float(tn / max(1, tn + fp)),
    }


def metrics_from_counts(tn, fp, fn, tp):
    yt = np.concatenate([
        np.zeros(tn + fp, dtype=np.int64),
        np.ones(fn + tp, dtype=np.int64),
    ])
    yp = np.concatenate([
        np.zeros(tn, dtype=np.int64),
        np.ones(fp, dtype=np.int64),
        np.zeros(fn, dtype=np.int64),
        np.ones(tp, dtype=np.int64),
    ])
    return metrics_from_arrays(yt, yp)


def load_synthetic_cache():
    X = np.load(
        CACHE / 'synthetic_X_protechto_rawunits.npy',
        mmap_mode='r',
    )
    y = np.load(
        CACHE / 'synthetic_y01.npy'
    ).astype(np.int64)

    meta = pd.read_csv(
        CACHE / 'synthetic_metadata_resolved.csv'
    )

    if len(X) != len(y) or len(X) != len(meta):
        raise RuntimeError('Synthetic cache length mismatch')

    counts = np.bincount(y, minlength=2).tolist()
    if counts != [18824, 1256]:
        raise RuntimeError(
            f'Expected synthetic Activity/Falling [18824, 1256], got {counts}'
        )

    if int(meta['trial_uid_resolved'].astype(str).nunique()) != 310:
        raise RuntimeError('Expected exactly 310 resolved synthetic trajectories')

    return X, y, meta


class SimTrainRealValTestDataModule(pl.LightningDataModule):
    """
    Training: ALL synthetic windows.
    Validation/test: exact subject-independent physical fold partitions.
    """

    def __init__(self, Xs, ys, meta, fold0):
        super().__init__()
        self.Xs = Xs
        self.ys = ys
        self.meta = meta
        self.fold0 = fold0

        self.label_encoder = LabelEncoder().fit(
            ['Activity', 'Falling']
        )
        self.input_shape = (30, 9)

        self.train_dl = None
        self.validation_dl = None
        self.test_dl = None

        self.physical_train_subjects = None
        self.physical_val_subjects = None
        self.physical_test_subjects = None

        self.val_counts = None
        self.test_counts = None

    def construct(self, verbose=True):
        subjects = np.array([
            s for s in sorted(os.listdir(PHYSICAL_ROOT))
            if s not in const.DATA_AUGMENTATION_SUBJECTS
        ])

        if len(subjects) != 71:
            raise RuntimeError(
                f'Expected 71 eligible physical subject identities, got {len(subjects)}'
            )

        kf = KFold(
            n_splits=5,
            shuffle=True,
            random_state=42,
        )

        trainval_idx, test_idx = list(
            kf.split(subjects)
        )[self.fold0]

        train_idx, val_idx = train_test_split(
            trainval_idx,
            test_size=0.2,
            random_state=42,
        )

        self.physical_train_subjects = subjects[train_idx]
        self.physical_val_subjects = subjects[val_idx]
        self.physical_test_subjects = subjects[test_idx]

        # --------------------------------------------------------------
        # SYNTHETIC-ONLY TRAINING
        # --------------------------------------------------------------
        x_train = np.asarray(
            self.Xs,
            dtype=np.float32,
        )
        y_train = np.asarray(
            self.ys,
            dtype=np.int64,
        )

        # --------------------------------------------------------------
        # REAL PHYSICAL VALIDATION / TEST
        # Same physical fold construction as canonical KFoldDataloader.
        # --------------------------------------------------------------
        x_val, y_val_text = load(
            self.physical_val_subjects,
            str(PHYSICAL_ROOT),
        )
        x_test, y_test_text = load(
            self.physical_test_subjects,
            str(PHYSICAL_ROOT),
        )

        y_val = self.label_encoder.transform(
            y_val_text
        ).astype(np.int64)

        y_test = self.label_encoder.transform(
            y_test_text
        ).astype(np.int64)

        self.train_dl = dataloader(
            x_train,
            y_train,
            batch_size=BATCH,
            shuffle=True,
        )

        self.validation_dl = dataloader(
            x_val,
            y_val,
            batch_size=BATCH,
            shuffle=False,
        )

        self.test_dl = dataloader(
            x_test,
            y_test,
            batch_size=BATCH,
            shuffle=False,
        )

        self.val_counts = np.bincount(
            y_val,
            minlength=2,
        ).astype(int).tolist()

        self.test_counts = np.bincount(
            y_test,
            minlength=2,
        ).astype(int).tolist()

        if verbose:
            print()
            print('SYNTHETIC-ONLY TRAINING:')
            print(
                '  trajectories:',
                int(self.meta['trial_uid_resolved'].astype(str).nunique())
            )
            print(
                '  windows/classes:',
                len(y_train),
                np.bincount(y_train, minlength=2).tolist(),
            )
            print(
                '  physical training windows used for gradients: 0'
            )
            print()
            print('REAL VALIDATION:')
            print(
                '  subjects:',
                list(map(str, self.physical_val_subjects))
            )
            print(
                '  windows/classes:',
                len(y_val),
                self.val_counts,
            )
            print()
            print('REAL TEST:')
            print(
                '  subjects:',
                list(map(str, self.physical_test_subjects))
            )
            print(
                '  windows/classes:',
                len(y_test),
                self.test_counts,
            )
            print()

    def train_dataloader(self):
        return self.train_dl

    def val_dataloader(self):
        return self.validation_dl

    def test_dataloader(self):
        return self.test_dl


def make_trainer(ckpt_dir: Path, fold1: int):
    ckpt_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename=f'best-checkpoint-fold_{fold1}',
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,
    )

    logger = TensorBoardLogger(
        str(ROOT / 'lightning_logs'),
        name='UniVrFall',
    )

    trainer = Trainer(
        logger=logger,
        callbacks=[
            checkpoint_callback,
            early_stopping,
        ],
        max_epochs=100,
        accelerator='gpu',
        devices=[GPU],
        enable_progress_bar=True,
    )

    return trainer, checkpoint_callback


def physical_fold_event_metrics(result_dir: Path, fold1: int):
    p = (
        result_dir
        / str(fold1)
        / 'event_stats__bias_0.65.csv'
    )

    if not p.exists():
        raise RuntimeError(
            f'Missing event result: {p}'
        )

    df = pd.read_csv(p)

    # Remove report-only rows such as "Overall".
    task_num = pd.to_numeric(
        df['task'],
        errors='coerce',
    )
    df = df.loc[
        task_num.notna()
    ].copy()
    df['task_num'] = (
        task_num.loc[task_num.notna()]
        .astype(int)
    )

    tn = fp = fn = tp = 0

    for r in df.itertuples():
        task = int(r.task_num)
        passed = int(r.passed_simulations)
        missed = int(r.missed_simulations)

        if task in const.FALL_TASKS:
            tp += passed
            fn += missed
        else:
            tn += passed
            fp += missed

    return metrics_from_counts(
        tn, fp, fn, tp
    )


def train_campaign(run_id, out_root, Xs, ys, meta):
    condition_dir = (
        out_root
        / 'conditions'
        / CONDITION
    )

    condition_marker = (
        condition_dir
        / 'COMPLETE'
    )

    if condition_marker.exists():
        print('SKIP COMPLETE:', CONDITION)
        return

    ckpt_dir = (
        ROOT
        / 'checkpoints/CNN/300ms'
        / run_id
        / CONDITION
    )

    fold_state = (
        condition_dir
        / 'folds'
    )
    fold_state.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_rows = []

    for fold0 in range(5):
        fold1 = fold0 + 1
        complete = (
            fold_state
            / f'fold_{fold1}.COMPLETE'
        )
        ckpt = (
            ckpt_dir
            / f'best-checkpoint-fold_{fold1}.ckpt'
        )

        if (
            complete.exists()
            and ckpt.exists()
        ):
            print(
                f'{CONDITION} fold {fold1}: '
                'resume skip COMPLETE'
            )
            continue

        print()
        print('=' * 120)
        print(
            f'{CONDITION} | FOLD {fold1}'
        )
        print('=' * 120)

        clean_torch()

        dm = SimTrainRealValTestDataModule(
            Xs,
            ys,
            meta,
            fold0,
        )
        dm.construct(
            verbose=True
        )

        cfg = dict(best_config)

        model = Predictor(
            CNN,
            n_features=9,
            n_classes=2,
            classes=dm.label_encoder.classes_,
            experiment=f'300ms/{run_id}/{CONDITION}',
            fold=fold1,
            cfg=cfg,
        )

        trainer, cb = make_trainer(
            ckpt_dir,
            fold1,
        )

        trainer.fit(
            model,
            dm,
        )

        # Preserve the original final-campaign behavior.
        trainer.test(
            model,
            dataloaders=dm.test_dl,
        )

        best = Path(
            cb.best_model_path
        )

        if not best.exists():
            raise RuntimeError(
                f'No best checkpoint produced for fold {fold1}'
            )

        if best.resolve() != ckpt.resolve():
            if ckpt.exists():
                ckpt.unlink()
            best.replace(ckpt)

        if not ckpt.exists():
            raise RuntimeError(
                f'Missing normalized checkpoint: {ckpt}'
            )

        best_score = (
            float(cb.best_model_score.detach().cpu())
            if cb.best_model_score is not None
            else np.nan
        )

        manifest_rows.append({
            'fold': fold1,
            'condition': CONDITION,

            'train_domain': 'SIMULATED',
            'train_windows': int(len(ys)),
            'train_activity_windows': int((ys == 0).sum()),
            'train_falling_windows': int((ys == 1).sum()),
            'train_trajectories': int(
                meta['trial_uid_resolved']
                .astype(str)
                .nunique()
            ),

            'physical_train_subjects_not_used_for_gradients':
                len(dm.physical_train_subjects),
            'physical_train_subject_ids_not_used':
                ';'.join(
                    map(str, dm.physical_train_subjects)
                ),
            'physical_train_windows_used_for_gradients': 0,

            'validation_domain': 'PHYSICAL',
            'validation_subjects':
                len(dm.physical_val_subjects),
            'validation_subject_ids':
                ';'.join(
                    map(str, dm.physical_val_subjects)
                ),
            'validation_activity_windows':
                int(dm.val_counts[0]),
            'validation_falling_windows':
                int(dm.val_counts[1]),

            'test_domain': 'PHYSICAL',
            'test_subjects':
                len(dm.physical_test_subjects),
            'test_subject_ids':
                ';'.join(
                    map(str, dm.physical_test_subjects)
                ),
            'test_activity_windows':
                int(dm.test_counts[0]),
            'test_falling_windows':
                int(dm.test_counts[1]),

            'best_val_loss':
                best_score,
            'trainer_current_epoch':
                int(trainer.current_epoch),
            'checkpoint':
                str(ckpt),
        })

        pd.DataFrame(
            manifest_rows
        ).to_csv(
            out_root
            / 'fold_dataflow_manifest.csv',
            index=False,
        )

        complete.write_text(
            'COMPLETE\n'
        )

        del trainer, model, dm
        clean_torch()

        print(
            f'{CONDITION} | FOLD {fold1} COMPLETE'
        )

    # --------------------------------------------------------------
    # Final untouched physical evaluation using canonical test.py.
    # --------------------------------------------------------------
    ckpt_rel = (
        f'checkpoints/CNN/300ms/'
        f'{run_id}/{CONDITION}'
    )

    cmd = [
        sys.executable,
        '-u',
        str(EXEC / 'test.py'),
        '-m', 'CNN',
        '-c', ckpt_rel,
        '-t', 'k-fold',
        '-d', DATASET_REL,
        '-g', str(GPU),
    ]

    print()
    print('FINAL PHYSICAL TEST:')
    print('+', ' '.join(cmd))

    rc = subprocess.run(
        cmd,
        cwd=ROOT,
    ).returncode

    if rc != 0:
        raise RuntimeError(
            f'Canonical Protechto test.py failed, '
            f'status={rc}'
        )

    result_dir = (
        ROOT
        / 'results/CNN/300ms'
        / run_id
        / CONDITION
    )

    required = [
        result_dir
        / 'CNN_confusion_matrix_GLOBAL.csv',

        result_dir
        / 'CNN_EVENT_confusion_matrix_GLOBAL.csv',

        result_dir
        / 'event_stats__bias_0.65.csv',
    ]

    for p in required:
        if not p.exists():
            raise RuntimeError(
                f'Missing final physical result artifact: {p}'
            )

    condition_marker.write_text(
        'COMPLETE\n'
    )


def aggregate(run_id, out_root):
    result_dir = (
        ROOT
        / 'results/CNN/300ms'
        / run_id
        / CONDITION
    )

    rows = []

    segment_true = []
    segment_pred = []

    etn = efp = efn = etp = 0

    for fold1 in range(1, 6):
        d = result_dir / str(fold1)

        y_true = np.load(
            d / 'y_true.npy'
        ).astype(np.int64)

        y_pred = np.load(
            d / 'y_pred.npy'
        ).astype(np.int64)

        sm = metrics_from_arrays(
            y_true,
            y_pred,
        )

        rows.append({
            'condition': CONDITION,
            'fold': fold1,
            'level': 'Segment',
            'evaluation_domain': 'Held-out physical',
            **sm,
        })

        segment_true.append(
            y_true
        )
        segment_pred.append(
            y_pred
        )

        em = physical_fold_event_metrics(
            result_dir,
            fold1,
        )

        rows.append({
            'condition': CONDITION,
            'fold': fold1,
            'level': 'Event',
            'evaluation_domain': 'Held-out physical',
            **em,
        })

        etn += em['tn']
        efp += em['fp']
        efn += em['fn']
        etp += em['tp']

    fold_df = pd.DataFrame(
        rows
    )
    fold_df.to_csv(
        out_root
        / 'all_fold_metrics.csv',
        index=False,
    )

    pooled_segment = metrics_from_arrays(
        np.concatenate(segment_true),
        np.concatenate(segment_pred),
    )

    pooled_event = metrics_from_counts(
        etn, efp, efn, etp
    )

    pooled = pd.DataFrame([
        {
            'condition': CONDITION,
            'level': 'Segment',
            'evaluation_domain': 'Held-out physical',
            **pooled_segment,
        },
        {
            'condition': CONDITION,
            'level': 'Event',
            'evaluation_domain': 'Held-out physical',
            **pooled_event,
        },
    ])

    pooled.to_csv(
        out_root
        / 'pooled_results.csv',
        index=False,
    )

    metrics = [
        'accuracy',
        'balanced_accuracy',
        'precision_fall',
        'recall_fall',
        'f1_fall',
        'specificity',
    ]

    summary_rows = []

    for level, g in fold_df.groupby(
        'level',
        sort=False,
    ):
        row = {
            'condition': CONDITION,
            'level': level,
        }

        for m in metrics:
            row[f'{m}_mean'] = float(
                g[m].mean()
            )
            row[f'{m}_std'] = float(
                g[m].std(ddof=1)
            )

        summary_rows.append(
            row
        )

    pd.DataFrame(
        summary_rows
    ).to_csv(
        out_root
        / 'fold_mean_std.csv',
        index=False,
    )

    print()
    print('=' * 120)
    print(
        'SIM-TRAIN -> REAL-VAL/TEST '
        'POOLED RESULTS'
    )
    print('=' * 120)

    show_cols = [
        'level',
        'n',
        'tn',
        'fp',
        'fn',
        'tp',
        'balanced_accuracy',
        'precision_fall',
        'recall_fall',
        'f1_fall',
    ]

    print(
        pooled[show_cols].to_string(
            index=False,
            float_format=lambda x: f'{x:.6f}',
        )
    )

    print('=' * 120)


def main():
    global GPU

    ap = argparse.ArgumentParser()

    ap.add_argument(
        '--output-root',
        default=str(DEFAULT_OUT),
    )

    ap.add_argument(
        '--gpu',
        type=int,
        default=GPU,
        help=(
            'GPU index. For exact parity with the '
            'final campaign use GPU 1.'
        ),
    )

    args = ap.parse_args()

    GPU = int(args.gpu)

    out_root = Path(
        args.output_root
    )
    out_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Isolated run ID so existing paper campaign is never overwritten.
    run_id_file = (
        out_root
        / 'RUN_ID'
    )

    if run_id_file.exists():
        run_id = (
            run_id_file
            .read_text()
            .strip()
        )
    else:
        run_id = (
            'SIMTRAIN_REALVALTEST_'
            + time.strftime(
                '%Y%m%d_%H%M%S'
            )
        )
        run_id_file.write_text(
            run_id + '\n'
        )

    print('=' * 120)
    print(
        'SYNTHETIC-ONLY TRAINING / '
        'REAL VALIDATION / REAL TEST'
    )
    print('=' * 120)
    print('RUN_ID:', run_id)
    print('Synthetic cache:', CACHE)
    print('Physical root :', PHYSICAL_ROOT)
    print('Output root   :', out_root)
    print('GPU           :', GPU)
    print()
    print('TRAIN      = simulated Activity + simulated Falling only')
    print('VALIDATION = held-out physical subjects')
    print('TEST       = held-out physical subjects')
    print('PHYSICAL TRAIN WINDOWS USED FOR GRADIENTS = 0')
    print()
    print(
        'NOTE: real validation is used for early stopping, '
        'so this is not strict zero-shot domain transfer.'
    )
    print('=' * 120)

    Xs, ys, meta = load_synthetic_cache()

    train_campaign(
        run_id,
        out_root,
        Xs,
        ys,
        meta,
    )

    aggregate(
        run_id,
        out_root,
    )

    provenance = {
        'condition': CONDITION,
        'run_id': run_id,
        'training_domain': 'synthetic only',
        'validation_domain': 'physical',
        'test_domain': 'physical',
        'synthetic_train_windows': int(len(ys)),
        'synthetic_train_activity_windows': int((ys == 0).sum()),
        'synthetic_train_falling_windows': int((ys == 1).sum()),
        'synthetic_train_trajectories': int(
            meta['trial_uid_resolved']
            .astype(str)
            .nunique()
        ),
        'physical_training_windows_used_for_gradients': 0,
        'physical_split':
            '5-fold subject KFold shuffle=True random_state=42; '
            'inner validation split test_size=0.2 random_state=42',
        'model_selection':
            'real physical validation loss',
        'batch_size': BATCH,
        'max_epochs': 100,
        'early_stopping_patience': 20,
        'prediction_bias': float(
            best_config['prediction_bias']
        ),
        'important_interpretation':
            'synthetic-only parameter training with real-domain '
            'validation and held-out real-domain test; not strict zero-shot',
    }

    (
        out_root
        / 'EXPERIMENT_PROVENANCE.json'
    ).write_text(
        json.dumps(
            provenance,
            indent=2,
        )
    )

    (
        out_root
        / 'FINAL_EXPERIMENT_COMPLETE'
    ).write_text(
        'COMPLETE\n'
    )

    print()
    print(
        'SIM-TRAIN / REAL-VAL / REAL-TEST '
        'EXPERIMENT COMPLETE'
    )
    print('Results:', out_root)


if __name__ == '__main__':
    main()
