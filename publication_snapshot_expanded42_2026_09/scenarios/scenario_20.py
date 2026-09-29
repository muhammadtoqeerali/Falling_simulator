# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

from legacy_runtime_common import make_output_dir, standardized_run_context, write_run_manifest

SCENARIO_ID = 20


def run(subject_params: dict | None = None) -> dict:
    if subject_params is None:
        subject_params = {'age': 75, 'height': 1.65, 'sex': 'male', 'weight': None}

    project_root = Path(__file__).resolve().parents[1]
    output_dir = make_output_dir(project_root, SCENARIO_ID, subject_params)

    print("  [Scenario 20] Standard wrapper + task-specific motion runtime.")
    print(f"  [Scenario 20] Output folder -> {output_dir}\n")

    with standardized_run_context(project_root, output_dir, subject_params):
        import scenario20_legacy as legacy20
        result = legacy20.run(subject_params)

    manifest = write_run_manifest(output_dir, SCENARIO_ID, subject_params, result)
    print(f"\n  [Scenario 20] Run manifest -> {manifest}")
    print(f"  [Scenario 20] Output folder -> {output_dir}")
    return {'scenario_id': SCENARIO_ID, 'output_dir': str(output_dir), 'manifest': str(manifest), **(result or {})}


if __name__ == '__main__':
    run()
