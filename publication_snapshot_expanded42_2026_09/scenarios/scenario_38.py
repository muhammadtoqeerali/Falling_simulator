# -*- coding: utf-8 -*-
"""
scenarios/scenario_38.py
Thin wrapper for Scenario 38: quick backward fall while moving back.

Runtime contract:
  stand -> fast/clear backward-moving replay gait -> backward loss of balance ->
  supine grounded settle.

This is the non-legacy scenario entrypoint. Keep `scenario38_legacy.py` in the
project root, and place this file at `scenarios/scenario_38.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from legacy_runtime_common import make_output_dir, standardized_run_context, write_run_manifest

SCENARIO_ID = 38
DESCRIPTION = "Quick backward fall while moving back"


def run(subject_params: dict | None = None) -> dict:
    """Run Scenario 38 with standardized output handling.

    Defaults match the validated Task 38 quick backward-fall setup:
    32-year-old male, 1.75 m, 80 kg, reverse-replayed forward gait at the
    full/quick target speed, then backward fall and grounded supine settle.
    """
    if subject_params is None:
        subject_params = {"age": 32, "height": 1.75, "sex": "male", "weight": 80.0}

    subject_params = dict(subject_params)
    project_root = Path(__file__).resolve().parents[1]
    output_dir = make_output_dir(project_root, SCENARIO_ID, subject_params)

    print(
        "  [Scenario 38] Quick backward-fall runtime: stand -> clear fast backward-moving gait "
        "-> backward loss of balance -> grounded supine settle."
    )
    print(f"  [Scenario 38] Output folder -> {output_dir}\n")

    with standardized_run_context(project_root, output_dir, subject_params):
        import scenario38_legacy as legacy38
        result = legacy38.run_with_subject(subject_params)

    manifest_payload = result if isinstance(result, dict) else {}
    manifest_payload.setdefault("description", DESCRIPTION)
    manifest_payload.setdefault(
        "task_contract",
        {
            "initial_motion": "quiet stand, then quick backward-moving gait using reverse replay of clean forward locomotion",
            "trigger": "posterior COM loss / backward-loss perturbation after the replay handoff",
            "fall_direction": "backward/supine",
            "support_policy": "protective reflex monitored; no artificial sitting recovery",
            "settle": "grounded supine post-fall rest",
        },
    )

    manifest = write_run_manifest(output_dir, SCENARIO_ID, subject_params, manifest_payload)
    print(f"\n  [Scenario 38] Run manifest -> {manifest}")
    print(f"  [Scenario 38] Output folder -> {output_dir}")

    payload = {"scenario_id": SCENARIO_ID, "output_dir": str(output_dir), "manifest": str(manifest)}
    if isinstance(result, dict):
        payload.update(result)
    return payload


if __name__ == "__main__":
    run()
