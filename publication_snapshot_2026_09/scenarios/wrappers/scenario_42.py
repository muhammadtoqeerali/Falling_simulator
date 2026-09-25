# -*- coding: utf-8 -*-
"""
scenarios/scenario_42.py
Thin wrapper for Scenario 42: Backward fall while climbing down the ladder.

Runtime contract:
  physical ladder with real MuJoCo rung/rail contact -> stable upper-rung hold
  -> 2 visible downward ladder steps -> mid-ladder backward release using the
  validated Scenario-40/Task-38 posterior-push backward fall and supine settle.

Place this file at `scenarios/scenario_42.py` and keep `scenario42_legacy.py`
in the project root.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from legacy_runtime_common import make_output_dir, standardized_run_context, write_run_manifest

SCENARIO_ID = 42
DESCRIPTION = "Backward fall while climbing down the ladder"


def run(subject_params: dict | None = None) -> dict:
    """Run Scenario 42 with standardized output handling."""
    if subject_params is None:
        subject_params = {"age": 75, "height": 1.65, "sex": "male", "weight": None}

    subject_params = dict(subject_params)
    project_root = Path(__file__).resolve().parents[1]
    output_dir = make_output_dir(project_root, SCENARIO_ID, subject_params)

    print(
        "  [Scenario 42] Ladder-descend backward-fall runtime: physical ladder "
        "-> upper-rung hold -> 2 downward climbing steps -> mid-ladder backward "
        "fall using Scenario-40 posterior fall/post-fall logic -> protective supine settle."
    )
    print(f"  [Scenario 42] Output folder -> {output_dir}\n")

    with standardized_run_context(project_root, output_dir, subject_params):
        import scenario42_legacy as legacy42
        result = legacy42.run_with_subject(subject_params)

    manifest_payload = result if isinstance(result, dict) else {}
    manifest_payload.setdefault("description", DESCRIPTION)
    manifest_payload.setdefault(
        "task_contract",
        {
            "initial_state": "humanoid holding an upper working rung of a real physical ladder",
            "pre_fall_action": "deterministic 2-step downward descent on physical ladder rungs",
            "trigger": "mid-ladder pose unlock/release after two downward steps",
            "fall_direction": "backward/supine, away from ladder, using Task40/Task38 posterior COM push",
            "support_policy": "ladder rungs and rails are compiled physical MuJoCo geoms; no visual-only prop during descent or release",
            "settle": "grounded supine post-fall rest with bounded protective arms/legs",
            "start_rung_index_zero_based": 5,
            "trigger_rung_index_zero_based": 3,
            "visible_descent_steps": 2,
        },
    )

    manifest = write_run_manifest(output_dir, SCENARIO_ID, subject_params, manifest_payload)
    print(f"\n  [Scenario 42] Run manifest -> {manifest}")
    print(f"  [Scenario 42] Output folder -> {output_dir}")

    payload = {"scenario_id": SCENARIO_ID, "output_dir": str(output_dir), "manifest": str(manifest)}
    if isinstance(result, dict):
        payload.update(result)
    return payload


if __name__ == "__main__":
    run()
