#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import pandas as pd

PROJECT = Path('/mnt/hdd16T/ToqeerHomeBackup/mujoco_project')
SRC = PROJECT / 'outputs/phase2_corrected_event_dataset_expanded42_v2'
OUT = PROJECT / 'outputs/protechto_exact_synthetic_cache_expanded42_v2'
X_PATH = SRC / 'synthetic_X.npy'
Y_PATH = SRC / 'synthetic_y.npy'
# Expanded42 V2 deliberately does not guess the new number of eligible
# trajectories/windows. Only the canonical window geometry and both-class
# requirement are fixed here.
EXPECTED_WINDOW_SHAPE = (30, 9)

TRIAL_SINGLE = [
    'source_trial_uid', 'trial_uid', 'session_uid', 'source_trial_id',
    'source_trial', 'session_id', 'trial_key', 'trajectory_uid'
]
TRIAL_GROUPS = [
    ['scenario_id', 'profile_id', 'trial'],
    ['scenario', 'profile', 'trial'],
    ['task_id', 'profile_id', 'trial'],
    ['task_id', 'profile', 'trial'],
    ['task', 'profile', 'trial'],
    ['scenario_id', 'profile_id'],
    ['scenario', 'profile'],
    ['task_id', 'profile_id'],
    ['task_id', 'profile'],
    ['task', 'profile'],
]
ORDER_COLS = [
    'window_index', 'window_idx', 'window_start_sample', 'start_sample',
    'start_idx', 'window_start', 'start_time', 'time_s', 'timestamp'
]


def labels01(y):
    if np.issubdtype(y.dtype, np.number):
        z = y.astype(np.int64).reshape(-1)
        if not np.all(np.isin(z, [0, 1])):
            raise RuntimeError(f'Unexpected numeric synthetic labels: {np.unique(z)}')
        return z
    s = np.asarray(y).astype(str).reshape(-1)
    out = np.full(len(s), -1, dtype=np.int64)
    sl = np.char.lower(np.char.strip(s))
    out[np.isin(sl, ['activity', '0', 'false'])] = 0
    out[np.isin(sl, ['falling', 'fall', '1', 'true'])] = 1
    if np.any(out < 0):
        raise RuntimeError(f'Unexpected string synthetic labels: {np.unique(s[out < 0])[:20]}')
    return out


def choose_metadata(n):
    candidates = []
    for p in sorted(SRC.rglob('*.csv')):
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if len(df) == n:
            score = 0
            name = p.name.lower()
            if 'metadata' in name: score += 10
            if 'window' in name: score += 5
            if 'synthetic' in name: score += 3
            if 'manifest' in name: score += 1
            candidates.append((score, p, df))
    if not candidates:
        raise RuntimeError(
            f'No CSV with exactly {n} rows was found under {SRC}. '
            'SIM_ONLY_FULL requires row-aligned synthetic metadata.'
        )
    candidates.sort(key=lambda x: (-x[0], str(x[1])))
    print('Row-aligned metadata candidates:')
    for score, p, df in candidates:
        print(f'  score={score:2d} rows={len(df):5d} cols={len(df.columns):3d} {p}')
    return candidates[0][1], candidates[0][2]


def resolve_trial_uid(df):
    for c in TRIAL_SINGLE:
        if c in df.columns and df[c].notna().all():
            return df[c].astype(str), [c]
    for cols in TRIAL_GROUPS:
        if all(c in df.columns for c in cols):
            vals = df[cols].astype(str).agg('|'.join, axis=1)
            return vals, cols
    raise RuntimeError(
        'Could not resolve a source-trial identifier from synthetic metadata. '
        f'Available columns: {list(df.columns)}'
    )


def main():
    print('=' * 108)
    print('PREPARE PROTECHTO-EXACT SYNTHETIC CACHE')
    print('=' * 108)
    if not X_PATH.exists() or not Y_PATH.exists():
        raise RuntimeError(f'Missing frozen synthetic dataset: {X_PATH} / {Y_PATH}')

    X = np.load(X_PATH, mmap_mode='r')
    y_raw = np.load(Y_PATH, allow_pickle=True)
    y = labels01(y_raw)

    print('Synthetic X:', X.shape, X.dtype)
    print('Synthetic y:', y.shape, y.dtype)
    if X.ndim != 3 or tuple(X.shape[1:]) != EXPECTED_WINDOW_SHAPE:
        raise RuntimeError(
            f'Expected synthetic shape (N, 30, 9), got {tuple(X.shape)}'
        )
    if len(y) != len(X):
        raise RuntimeError('Synthetic X/y length mismatch')

    counts = {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
    print('Label counts:', counts)
    if set(counts) != {0, 1}:
        raise RuntimeError(
            f'Expanded42 cache must contain Activity and Falling, got {counts}'
        )

    if counts.get(0, 0) <= 0 or counts.get(1, 0) <= 0:
        raise RuntimeError(
            f'Expanded42 cache contains an empty class: {counts}'
        )

    meta_path, meta = choose_metadata(len(X))
    trial_uid, trial_cols = resolve_trial_uid(meta)
    meta = meta.copy()
    meta.insert(0, 'synthetic_row', np.arange(len(meta), dtype=np.int64))
    meta['label01'] = y
    meta['trial_uid_resolved'] = trial_uid.astype(str)

    order_col = next((c for c in ORDER_COLS if c in meta.columns), None)
    if order_col is None:
        meta['order_resolved'] = meta.groupby('trial_uid_resolved', sort=False).cumcount()
        order_source = 'row_order_within_trial'
    else:
        order_vals = pd.to_numeric(meta[order_col], errors='coerce')
        if order_vals.isna().any():
            meta['order_resolved'] = meta.groupby('trial_uid_resolved', sort=False).cumcount()
            order_source = f'row_order_within_trial (because {order_col} was nonnumeric)'
        else:
            meta['order_resolved'] = order_vals
            order_source = order_col

    n_trials = int(meta['trial_uid_resolved'].nunique())
    both = meta.groupby('trial_uid_resolved')['label01'].agg(lambda s: set(map(int, s)))
    n_both = int(sum(v == {0, 1} for v in both))
    n_fall_trials = int(sum(1 in v for v in both))
    n_activity_only = int(sum(v == {0} for v in both))

    if "profile_id" not in meta.columns:
        raise RuntimeError(
            "Expanded42 synthetic metadata is missing profile_id"
        )

    if "task_id" not in meta.columns:
        raise RuntimeError(
            "Expanded42 synthetic metadata is missing task_id"
        )

    if meta["profile_id"].astype(str).nunique() != 42:
        raise RuntimeError(
            "Expanded42 synthetic cache metadata must cover 42 profiles"
        )

    if pd.to_numeric(
        meta["task_id"],
        errors="raise",
    ).astype(int).nunique() != 18:
        raise RuntimeError(
            "Expanded42 synthetic cache metadata must cover 18 tasks"
        )

    print('Metadata selected:', meta_path)
    print('Trial id source :', trial_cols)
    print('Order source    :', order_source)
    print('Resolved trials :', n_trials)
    print('Trials with fall:', n_fall_trials)
    print('Trials with both Activity/Falling:', n_both)
    print('Activity-only trials:', n_activity_only)

    if n_trials < 5 or n_fall_trials < 5:
        raise RuntimeError('Insufficient resolved simulated trials for 5-fold SIM_ONLY_FULL')

    OUT.mkdir(parents=True, exist_ok=True)
    x_out = OUT / 'synthetic_X_protechto_rawunits.npy'
    y_out = OUT / 'synthetic_y01.npy'
    m_out = OUT / 'synthetic_metadata_resolved.csv'

    # Existing frozen synthetic windows are already 300 ms / 50% overlap / 5-Hz filtered.
    # Only unit adaptation is done here to match the original Protechto IMUNormalizer input:
    # acceleration m/s^2 -> mg; gyro deg/s -> mdps. Euler channels are preserved but CNN discards them.
    converted = np.empty(X.shape, dtype=np.float32)
    chunk = 4096
    for start in range(0, len(X), chunk):
        end = min(len(X), start + chunk)
        a = np.asarray(X[start:end], dtype=np.float32)
        converted[start:end, :, 0:3] = a[:, :, 0:3] / np.float32(0.00980665)
        converted[start:end, :, 3:6] = a[:, :, 3:6] * np.float32(1000.0)
        converted[start:end, :, 6:9] = a[:, :, 6:9]

    np.save(x_out, converted)
    np.save(y_out, y.astype(np.int64))
    meta.to_csv(m_out, index=False)

    provenance = {
        'source_root': str(SRC),
        'source_X': str(X_PATH),
        'source_y': str(Y_PATH),
        'source_metadata': str(meta_path),
        'shape': list(map(int, X.shape)),
        'label_counts': counts,
        'trial_id_columns': trial_cols,
        'order_source': order_source,
        'resolved_trials': n_trials,
        'trials_with_fall': n_fall_trials,
        'trials_with_both_classes': n_both,
        'activity_only_trials': n_activity_only,
        'physical_temporal_trim_applied': False,
        'synthetic_temporal_trim_applied': False,
        'unit_conversion': {
            'acc_m_s2_to_mg': 'divide by 0.00980665',
            'gyro_deg_s_to_mdps': 'multiply by 1000',
            'euler': 'unchanged; discarded by CNN normalizer'
        }
    }
    (OUT / 'provenance.json').write_text(json.dumps(provenance, indent=2, sort_keys=True))
    (OUT / 'COMPLETE').write_text('PROTECHTO-EXACT SYNTHETIC CACHE COMPLETE\n')
    print('Cache:', OUT)
    print('PROTECHTO-EXACT SYNTHETIC CACHE GATE: PASS')

if __name__ == '__main__':
    main()
