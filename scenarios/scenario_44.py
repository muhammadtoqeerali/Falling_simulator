# -*- coding: utf-8 -*-
"""
scenarios/scenario_44.py
Thin wrapper for Scenario 44: Vertical fall while climbing up the ladder.

Runtime contract:
  physical ladder with real MuJoCo rung/rail contact -> stable first-rung hold
  -> 4-5 visible upward climbing steps -> high-rung vertical/downward release
  -> protective floor-impact settle.

Place this file at `scenarios/scenario_44.py` and keep `scenario44_legacy.py`
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

SCENARIO_ID = 44
DESCRIPTION = "Vertical fall while climbing up the ladder"


def run(subject_params: dict | None = None) -> dict:
    """Run Scenario 44 with standardized output handling."""
    if subject_params is None:
        subject_params = {"age": 75, "height": 1.65, "sex": "male", "weight": None}

    subject_params = dict(subject_params)
    project_root = Path(__file__).resolve().parents[1]
    output_dir = make_output_dir(project_root, SCENARIO_ID, subject_params)

    print(
        "  [Scenario 44] Ladder-climb vertical-fall runtime: physical ladder "
        "-> first-rung hold -> 4-5 upward climbing steps -> high-rung "
        "vertical/downward release -> protective floor-impact settle."
    )
    print(f"  [Scenario 44] Output folder -> {output_dir}\n")

    with standardized_run_context(project_root, output_dir, subject_params):
        import scenario44_legacy as legacy44
        result = legacy44.run_with_subject(subject_params)

    manifest_payload = result if isinstance(result, dict) else {}
    manifest_payload.setdefault("description", DESCRIPTION)
    manifest_payload.setdefault(
        "task_contract",
        {
            "initial_state": "humanoid holding the first rung of a real physical ladder",
            "pre_fall_action": "deterministic 4-5 step upward climb on physical ladder rungs",
            "trigger": "high-rung pose unlock/release after climb phase",
            "fall_direction": "vertical/downward fall while climbing up the ladder",
            "support_policy": "ladder rungs and rails are compiled physical MuJoCo geoms; no visual-only prop during climb or release",
            "settle": "grounded post-fall protective rest with bounded arms/legs",
            "trigger_rung_height_m": 1.81,
            "visible_climb_steps": 5,
        },
    )

    manifest = write_run_manifest(output_dir, SCENARIO_ID, subject_params, manifest_payload)
    print(f"\n  [Scenario 44] Run manifest -> {manifest}")
    print(f"  [Scenario 44] Output folder -> {output_dir}")

    payload = {"scenario_id": SCENARIO_ID, "output_dir": str(output_dir), "manifest": str(manifest)}
    if isinstance(result, dict):
        payload.update(result)
    return payload


if __name__ == "__main__":
    run()
