# -*- coding: utf-8 -*-
"""
scenarios/scenario_22.py
Thin wrapper for Scenario 22: lateral fall when trying to sit down.
Keeps the standard centralized output-folder + manifest structure while
leaving motion generation to scenario22_legacy.py.
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

import scenario22_legacy as legacy22


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
    folder = f"scenario22_age{age}_h{_height_slug(height)}_sex_{sex}_{_weight_slug(weight)}_{ts}"
    out_dir = _ROOT / 'outputs' / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _write_manifest(out_dir: Path, subject: dict, result: dict) -> Path:
    files = sorted([p.name for p in out_dir.iterdir() if p.is_file()])
    manifest = {
        'scenario_id': 22,
        'description': 'Lateral fall when trying to sit down',
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
    print("\n  [Scenario 22] Standard wrapper + task-specific motion runtime.")
    out_dir = _build_output_dir(subject)
    print(f"  [Scenario 22] Output folder -> {out_dir}")

    cwd = Path.cwd()
    try:
        os.chdir(out_dir)
        result = legacy22.run(subject)
    finally:
        os.chdir(cwd)

    manifest = _write_manifest(out_dir, subject, dict(result or {}))
    print(f"\n  [Scenario 22] Run manifest -> {manifest}")
    print(f"  [Scenario 22] Output folder -> {out_dir}")
    return result


if __name__ == '__main__':
    run({'age': 75, 'height': 1.65, 'sex': 'male', 'weight': None})
