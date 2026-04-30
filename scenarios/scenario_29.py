# -*- coding: utf-8 -*-
"""
scenarios/scenario_29.py
Thin wrapper for Scenario 29: fall while walking, hands used to dampen.

This runtime uses the walking/export backbone, but implements a task-specific
forward fall with active hand reach and hand-first impact damping. No chair,
no slip, and no fainting-style full tone loss are used.
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

import scenario29_legacy as legacy29


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
    folder = f"scenario29_age{age}_h{_height_slug(height)}_sex_{sex}_{_weight_slug(weight)}_{ts}"
    out_dir = _ROOT / 'outputs' / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _write_manifest(out_dir: Path, subject: dict, result: dict) -> Path:
    files = sorted([p.name for p in out_dir.iterdir() if p.is_file()])
    manifest = {
        'scenario_id': 29,
        'description': 'Fall while walking, hands used to dampen',
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
    print("\n  [Scenario 29] Walking forward fall runtime: 4-5 s walk -> forward fall -> hands dampen impact.")
    out_dir = _build_output_dir(subject)
    print(f"  [Scenario 29] Output folder -> {out_dir}")

    cwd = Path.cwd()
    try:
        os.chdir(out_dir)
        result = legacy29.run_with_subject(subject)
    finally:
        os.chdir(cwd)

    manifest = _write_manifest(out_dir, subject, dict(result or {}))
    print(f"\n  [Scenario 29] Run manifest -> {manifest}")
    print(f"  [Scenario 29] Output folder -> {out_dir}")
    payload = {'scenario_id': 29, 'output_dir': str(out_dir), 'manifest': str(manifest)}
    if isinstance(result, dict):
        payload.update(result)
    return payload


if __name__ == '__main__':
    run({'age': 75, 'height': 1.65, 'sex': 'male', 'weight': None})
