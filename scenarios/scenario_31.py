# -*- coding: utf-8 -*-
"""
scenarios/scenario_31.py
Thin wrapper for Scenario 31: forward fall while jogging caused by a trip.

Runtime contract:
  stand -> accelerate/jog -> swing-foot catch / trip -> unsupported fast forward fall -> prone settle.
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

import scenario31_legacy as legacy31

SCENARIO_ID = 31
DESCRIPTION = 'Forward fall while jogging caused by a trip'


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
    folder = f"scenario31_age{age}_h{_height_slug(height)}_sex_{sex}_{_weight_slug(weight)}_{ts}"
    out_dir = _ROOT / 'outputs' / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _write_manifest(out_dir: Path, subject: dict, result: dict) -> Path:
    files = sorted([p.name for p in out_dir.iterdir() if p.is_file()])
    manifest = {
        'scenario_id': SCENARIO_ID,
        'description': DESCRIPTION,
        'task_contract': {
            'initial_motion': 'quiet stand, then straight jogging exposure',
            'trigger': 'leading swing foot catches/blocks briefly at jogging speed; COM continues forward',
            'fall_direction': 'forward/prone',
            'support_policy': 'weak late arm reflex; no hand-dampen assist',
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
    subject = dict(subject_params or {'age': 32, 'height': 1.75, 'sex': 'male', 'weight': 80.0})
    print("\n  [Scenario 31] Jogging trip runtime: stand -> accelerate/jog -> swing-foot catch -> fast forward fall -> prone settle.")
    out_dir = _build_output_dir(subject)
    print(f"  [Scenario 31] Output folder -> {out_dir}")

    cwd = Path.cwd()
    try:
        os.chdir(out_dir)
        result = legacy31.run_with_subject(subject)
    finally:
        os.chdir(cwd)

    manifest = _write_manifest(out_dir, subject, dict(result or {}))
    print(f"\n  [Scenario 31] Run manifest -> {manifest}")
    print(f"  [Scenario 31] Output folder -> {out_dir}")
    payload = {'scenario_id': SCENARIO_ID, 'output_dir': str(out_dir), 'manifest': str(manifest)}
    if isinstance(result, dict):
        payload.update(result)
    return payload


if __name__ == '__main__':
    run({'age': 32, 'height': 1.75, 'sex': 'male', 'weight': 80.0})
