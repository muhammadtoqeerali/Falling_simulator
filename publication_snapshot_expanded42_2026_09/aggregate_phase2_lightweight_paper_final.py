#!/usr/bin/env python3
import json
from pathlib import Path
import pandas as pd

ROOT = Path('/mnt/hdd16T/ToqeerHomeBackup/mujoco_project/outputs/phase2_lightweight_paper_final_campaign_v1')
EXPS = [
    'EXP01_REAL_ONLY',
    'EXP02_SIM_FALL_SUBSTITUTION',
    'EXP03_MIX20',
    'EXP04_MIX50',
    'EXP05_MIX70',
    'EXP06_MIX100',
]
required = [
    'COMPLETE', 'best_model.pth', 'training_history.csv', 'run_config.json',
    'segment_metrics.json', 'event_metrics.json', 'segment_predictions.npz',
    'test_trial_ranges.csv', 'event_predictions.csv',
]
seg_rows, ev_rows, missing = [], [], []
for exp in EXPS:
    for fold in range(5):
        d = ROOT / exp / f'fold_{fold}'
        for name in required:
            if not (d / name).exists():
                missing.append(str(d / name))
        if (d / 'segment_metrics.json').exists():
            seg_rows.append(json.loads((d / 'segment_metrics.json').read_text()))
        if (d / 'event_metrics.json').exists():
            ev_rows.append(json.loads((d / 'event_metrics.json').read_text()))

if missing:
    print('MISSING ARTIFACTS:')
    print('\n'.join(missing[:100]))
    raise SystemExit(1)
if len(seg_rows) != 30 or len(ev_rows) != 30:
    raise SystemExit(f'Expected 30 segment + 30 event rows, got {len(seg_rows)} and {len(ev_rows)}')

seg = pd.DataFrame(seg_rows).sort_values(['experiment', 'fold']).reset_index(drop=True)
ev = pd.DataFrame(ev_rows).sort_values(['experiment', 'fold']).reset_index(drop=True)
seg.to_csv(ROOT / 'all_folds_segment_metrics.csv', index=False)
ev.to_csv(ROOT / 'all_folds_event_metrics.csv', index=False)

seg_metrics = ['accuracy','balanced_accuracy','precision_fall','recall_fall','f1_fall','specificity','roc_auc','average_precision']
ev_metrics = ['accuracy','balanced_accuracy','precision_fall','recall_fall','f1_fall','specificity']

def summarize(df, metrics):
    rows = []
    for exp, g in df.groupby('experiment', sort=False):
        row = {'experiment': exp}
        for m in metrics:
            row[f'{m}_mean'] = float(g[m].mean())
            row[f'{m}_std'] = float(g[m].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)

seg_sum = summarize(seg, seg_metrics)
ev_sum = summarize(ev, ev_metrics)
seg_sum.to_csv(ROOT / 'summary_segment_mean_std.csv', index=False)
ev_sum.to_csv(ROOT / 'summary_event_mean_std.csv', index=False)
(ROOT / 'FINAL_CAMPAIGN_COMPLETE').write_text('PASS\n')

print('=' * 110)
print('FINAL LIGHTWEIGHT-PAPER CAMPAIGN: SEGMENT LEVEL')
print('=' * 110)
print(seg_sum.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
print()
print('=' * 110)
print('FINAL LIGHTWEIGHT-PAPER CAMPAIGN: EVENT LEVEL')
print('=' * 110)
print(ev_sum.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
print()
print('FINAL LIGHTWEIGHT PAPER CAMPAIGN ARTIFACT GATE: PASS')
print('FINAL LIGHTWEIGHT PAPER CAMPAIGN NUMERICAL GATE: PASS')
