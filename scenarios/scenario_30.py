# -*- coding: utf-8 -*-
"""
scenarios/scenario_30.py
Thin wrapper for Scenario 30: forward fall while walking caused by a trip.

Runtime contract:
  stand -> walk -> swing-foot catch / trip -> unsupported forward fall -> prone settle.
No chair, no slip-friction trick, no fainting full-tone collapse, and no active
hand-dampen support are used.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import scenario30_legacy as legacy30

SCENARIO_ID = 30
DESCRIPTION = 'Forward fall while walking caused by a trip'


def _height_slug(height: float) -> str:
    return f"{float(height):.2f}".replace('.', 'p')


def _weight_slug(weight) -> str:
    if weight is None:
        return 'wauto'
    return 'w' + f"{float(weight):.1f}".replace('.', 'p')


def _build_output_dir(subject: dict) -> Path:
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    age = int(subject['age'])
    height = float(subject['height'])
    sex = str(subject.get('sex', 'male')).lower()
    weight = subject.get('weight')
    folder = f"scenario30_age{age}_h{_height_slug(height)}_sex_{sex}_{_weight_slug(weight)}_{ts}"
    out_dir = _ROOT / 'outputs' / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _write_manifest(out_dir: Path, subject: dict, result: dict) -> Path:
    files = sorted([p.name for p in out_dir.iterdir() if p.is_file()])
    manifest = {
        'scenario_id': SCENARIO_ID,
        'description': DESCRIPTION,
        'task_contract': {
            'initial_motion': 'quiet stand, then straight walking exposure',
            'trigger': 'leading swing foot catches/blocks briefly; COM continues forward',
            'fall_direction': 'forward/prone',
            'support_policy': 'arms passive; no hand-dampen assist',
            'settle': 'post-impact prone rest with anti-slip stabilizer',
        },
        'subject': {
            'age': int(subject['age']),
            'height': float(subject['height']),
            'sex': str(subject.get('sex', 'male')).lower(),
            'weight': subject.get('weight'),
        },
        'result': result,
        'files': files,
    }
    manifest_path = out_dir / 'run_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest_path


def run(subject_params: dict | None = None):
    subject = dict(subject_params or {'age': 75, 'height': 1.65, 'sex': 'male', 'weight': None})
    print("\n  [Scenario 30] Walking trip runtime: stand -> 4-5 s walk -> swing-foot catch -> forward fall -> prone settle.")
    out_dir = _build_output_dir(subject)
    print(f"  [Scenario 30] Output folder -> {out_dir}")

    cwd = Path.cwd()
    try:
        os.chdir(out_dir)
        result = legacy30.run_with_subject(subject)
    finally:
        os.chdir(cwd)

    manifest = _write_manifest(out_dir, subject, dict(result or {}))
    print(f"\n  [Scenario 30] Run manifest -> {manifest}")
    print(f"  [Scenario 30] Output folder -> {out_dir}")
    payload = {'scenario_id': SCENARIO_ID, 'output_dir': str(out_dir), 'manifest': str(manifest)}
    if isinstance(result, dict):
        payload.update(result)
    return payload


if __name__ == '__main__':
    run({'age': 75, 'height': 1.65, 'sex': 'male', 'weight': None})
