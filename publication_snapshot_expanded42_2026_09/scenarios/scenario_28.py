# -*- coding: utf-8 -*-
"""
scenarios/scenario_28.py
Thin wrapper for Scenario 28: vertical/forward fall while walking, caused by fainting.

This runtime uses the shared walking/export infrastructure, but the fall trigger is
a dedicated syncope sequence: short standing stabilization, 4-5 s walking, gradual
tone loss, then a mostly vertical + forward fall with a small lateral drift.  No
chair, no slip impulse, and no gait-window perturbation are used.
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

import scenario28_legacy as legacy28


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
    folder = f"scenario28_age{age}_h{_height_slug(height)}_sex_{sex}_{_weight_slug(weight)}_{ts}"
    out_dir = _ROOT / 'outputs' / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _write_manifest(out_dir: Path, subject: dict, result: dict) -> Path:
    files = sorted([p.name for p in out_dir.iterdir() if p.is_file()])
    manifest = {
        'scenario_id': 28,
        'description': 'Vertical/forward fall while walking, caused by fainting',
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
    subject = dict(subject_params or {})
    print("\n  [Scenario 28] Walking fainting runtime v2: 4-5 s walk -> syncope tone loss -> vertical/forward/lateral fall.")
    out_dir = _build_output_dir(subject)
    print(f"  [Scenario 28] Output folder -> {out_dir}")

    cwd = Path.cwd()
    try:
        os.chdir(out_dir)
        result = legacy28.run_with_subject(subject)
    finally:
        os.chdir(cwd)

    manifest = _write_manifest(out_dir, subject, dict(result or {}))
    print(f"\n  [Scenario 28] Run manifest -> {manifest}")
    print(f"  [Scenario 28] Output folder -> {out_dir}")
    payload = {'scenario_id': 28, 'output_dir': str(out_dir), 'manifest': str(manifest)}
    if isinstance(result, dict):
        payload.update(result)
    return payload


if __name__ == '__main__':
    run({'age': 75, 'height': 1.65, 'sex': 'male', 'weight': None})
