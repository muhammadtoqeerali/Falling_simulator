# -*- coding: utf-8 -*-
"""
scenarios/scenario_40.py
Thin wrapper for Scenario 40: Backward fall from height.

Runtime contract:
  quiet arms-down stand on a 1.80 m cube/platform -> one visible swing-leg
  backward step off the rear edge -> gravity/pitch-driven backward fall ->
  protective limb reaction -> grounded supine settle.

Place this file at `scenarios/scenario_40.py` and keep `scenario40_legacy.py`
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

SCENARIO_ID = 40
DESCRIPTION = "Backward fall from height"


def run(subject_params: dict | None = None) -> dict:
    """Run Scenario 40 with standardized output handling."""
    if subject_params is None:
        subject_params = {"age": 75, "height": 1.65, "sex": "male", "weight": None}

    subject_params = dict(subject_params)
    project_root = Path(__file__).resolve().parents[1]
    output_dir = make_output_dir(project_root, SCENARIO_ID, subject_params)

    print(
        "  [Scenario 40] Backward height-fall runtime: 1.80 m platform stand "
        "-> single swing-leg backward step -> backward fall clear of platform "
        "-> protective supine settle."
    )
    print(f"  [Scenario 40] Output folder -> {output_dir}\n")

    with standardized_run_context(project_root, output_dir, subject_params):
        import scenario40_legacy as legacy40
        result = legacy40.run_with_subject(subject_params)

    manifest_payload = result if isinstance(result, dict) else {}
    manifest_payload.setdefault("description", DESCRIPTION)
    manifest_payload.setdefault(
        "task_contract",
        {
            "initial_state": "humanoid standing upright at the rear edge of a 1.80 m cube/platform",
            "trigger": "one visible swing-leg backward step beyond support, then release of pose lock",
            "fall_direction": "backward/supine, with rearward clearance after the foot leaves support",
            "support_policy": "platform is physical during stand/step; no whole-body slide across the platform",
            "settle": "grounded supine post-fall rest with bounded protective arms/legs",
            "platform_height_m": 1.80,
        },
    )

    manifest = write_run_manifest(output_dir, SCENARIO_ID, subject_params, manifest_payload)
    print(f"\n  [Scenario 40] Run manifest -> {manifest}")
    print(f"  [Scenario 40] Output folder -> {output_dir}")

    payload = {"scenario_id": SCENARIO_ID, "output_dir": str(output_dir), "manifest": str(manifest)}
    if isinstance(result, dict):
        payload.update(result)
    return payload


if __name__ == "__main__":
    run()
