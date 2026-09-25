# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path


def _san(v: str) -> str:
    return str(v).replace('.', 'p').replace('-', 'm')


def make_output_dir(project_root: Path, scenario_id: int, subject_params: dict) -> Path:
    out_root = project_root / 'outputs'
    out_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    age = int(subject_params['age'])
    height = float(subject_params['height'])
    sex = str(subject_params.get('sex', 'male')).lower()
    weight = subject_params.get('weight')
    w_txt = 'auto' if weight is None else _san(f'{float(weight):.1f}')
    folder = out_root / f"scenario{scenario_id}_age{age}_h{_san(f'{height:.2f}')}_sex_{sex}_w{w_txt}_{ts}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


@contextmanager
def standardized_run_context(project_root: Path, output_dir: Path, subject_params: dict):
    old_cwd = Path.cwd()
    old_env = {k: os.environ.get(k) for k in (
        'FALL_DEFAULT_AGE', 'FALL_DEFAULT_HEIGHT', 'FALL_DEFAULT_SEX', 'FALL_DEFAULT_WEIGHT', 'FALL_OUTPUT_DIR'
    )}
    try:
        os.chdir(output_dir)
        os.environ['FALL_DEFAULT_AGE'] = str(int(subject_params['age']))
        os.environ['FALL_DEFAULT_HEIGHT'] = str(float(subject_params['height']))
        os.environ['FALL_DEFAULT_SEX'] = str(subject_params.get('sex', 'male')).lower()
        weight = subject_params.get('weight')
        os.environ['FALL_DEFAULT_WEIGHT'] = '' if weight is None else str(float(weight))
        os.environ['FALL_OUTPUT_DIR'] = str(output_dir)
        py = os.environ.get('PYTHONPATH', '')
        proj = str(project_root)
        os.environ['PYTHONPATH'] = proj if not py else proj + os.pathsep + py
        yield
    finally:
        os.chdir(old_cwd)
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def write_run_manifest(output_dir: Path, scenario_id: int, subject_params: dict, result: dict | None = None):
    payload = {
        'scenario_id': int(scenario_id),
        'subject': {
            'age': int(subject_params['age']),
            'height': float(subject_params['height']),
            'sex': str(subject_params.get('sex', 'male')).lower(),
            'weight': None if subject_params.get('weight') is None else float(subject_params.get('weight')),
        },
        'output_dir': str(output_dir),
        'result': result or {},
        'created_at': datetime.now().isoformat(),
    }
    manifest = output_dir / 'run_manifest.json'
    try:
        from imu_pipeline_labels import label_output_folder_inplace
        payload["imu_labeling"] = label_output_folder_inplace(
            output_dir=output_dir,
            scenario_id=scenario_id,
            subject_params=subject_params,
            result=result or {},
        )
    except Exception as exc:
        payload["imu_labeling_error"] = str(exc)

    manifest.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return manifest
