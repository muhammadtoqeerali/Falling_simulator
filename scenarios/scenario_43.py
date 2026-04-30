# -*- coding: utf-8 -*-
"""
scenarios/scenario_43.py
Thin wrapper for Scenario 43: Forward fall while climbing up the ladder.

Runtime contract:
  physical ladder geoms -> stable first-rung hold
  -> REFERENCE/MOCAP-style top-rung climb with discrete plant/swing/contact/load-transfer
  -> hand/foot Cartesian rung targets solved by Jacobian IK
  -> physical top-rung forward release and protective prone settle.

Place this file at `scenarios/scenario_43.py` and keep `scenario43_legacy.py`
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

SCENARIO_ID = 43
DESCRIPTION = "Forward fall while climbing up the ladder"


def run(subject_params: dict | None = None) -> dict:
    """Run Scenario 43 with standardized output handling."""
    if subject_params is None:
        subject_params = {"age": 75, "height": 1.65, "sex": "male", "weight": None}

    subject_params = dict(subject_params)
    project_root = Path(__file__).resolve().parents[1]
    output_dir = make_output_dir(project_root, SCENARIO_ID, subject_params)

    print(
        "  [Scenario 43] Ladder-climb forward-fall runtime: physical ladder "
        "-> first-rung hold -> reference/mocap-style discrete ladder climb with hand/foot rung locks "
        "-> top-rung forward fall using forward/down COM pitch logic -> protective prone settle."
    )
    print(f"  [Scenario 43] Output folder -> {output_dir}\n")

    with standardized_run_context(project_root, output_dir, subject_params):
        import scenario43_legacy as legacy43
        result = legacy43.run_with_subject(subject_params)

    manifest_payload = result if isinstance(result, dict) else {}
    manifest_payload.setdefault("description", DESCRIPTION)
    manifest_payload.setdefault(
        "task_contract",
        {
            "initial_state": "humanoid holding the first rung of a real physical ladder",
            "pre_fall_action": "reference/mocap-style top-rung climb: plant -> swing -> rung contact -> body/load transfer",
            "trigger": "top-rung reference pose unlock/release after climb phase",
            "fall_direction": "forward/prone, along ladder-facing climb direction, using forward/down COM pitch",
            "support_policy": "ladder rungs/rails stay physical; climb motion is reference-driven, not produced by vertical fake grip forces",
            "settle": "grounded protective prone post-fall rest with bounded protective arms/legs",
            "trigger_rung_index_zero_based": 6,
            "visible_climb_steps": 7,
        },
    )

    manifest = write_run_manifest(output_dir, SCENARIO_ID, subject_params, manifest_payload)
    print(f"\n  [Scenario 43] Run manifest -> {manifest}")
    print(f"  [Scenario 43] Output folder -> {output_dir}")

    payload = {"scenario_id": SCENARIO_ID, "output_dir": str(output_dir), "manifest": str(manifest)}
    if isinstance(result, dict):
        payload.update(result)
    return payload


if __name__ == "__main__":
    run()
