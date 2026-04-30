# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

from legacy_runtime_common import make_output_dir, standardized_run_context, write_run_manifest

SCENARIO_ID = 34


def run(subject_params: dict | None = None) -> dict:
    if subject_params is None:
        subject_params = {'age': 75, 'height': 1.65, 'sex': 'male', 'weight': None}

    project_root = Path(__file__).resolve().parents[1]
    output_dir = make_output_dir(project_root, SCENARIO_ID, subject_params)

    print("  [Scenario 34] Standard wrapper + original task-34 motion runtime.")
    print(f"  [Scenario 34] Output folder -> {output_dir}\n")

    with standardized_run_context(project_root, output_dir, subject_params):
        import backward_fall_walking_best as task34
        result = task34.run_with_subject(subject_params)

    manifest = write_run_manifest(output_dir, SCENARIO_ID, subject_params, result if isinstance(result, dict) else {})
    print(f"\n  [Scenario 34] Run manifest -> {manifest}")
    print(f"  [Scenario 34] Output folder -> {output_dir}")
    payload = {'scenario_id': SCENARIO_ID, 'output_dir': str(output_dir), 'manifest': str(manifest)}
    if isinstance(result, dict):
        payload.update(result)
    return payload


if __name__ == '__main__':
    run()
