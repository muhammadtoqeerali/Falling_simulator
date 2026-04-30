# -*- coding: utf-8 -*-
"""
scenarios/scenario_39.py
Thin wrapper for Scenario 39: Forward fall from height.

Runtime contract:
  stand on a 1.80 m cube/platform -> platform support release / forward COM loss
  -> gravity-driven forward fall -> prone grounded settle.

Place this file at `scenarios/scenario_39.py` and keep `scenario39_legacy.py`
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

SCENARIO_ID = 39
DESCRIPTION = "Forward fall from height"


def run(subject_params: dict | None = None) -> dict:
    """Run Scenario 39 with standardized output handling.

    Defaults match the validated adult profile used in nearby fall tasks:
    32-year-old male, 1.75 m, 80 kg. The humanoid is initialized upright on a
    1.80 m platform/cube, the support is released, and MuJoCo gravity/contact
    physics drives the forward fall and grounded prone settle.
    """
    if subject_params is None:
        subject_params = {"age": 32, "height": 1.75, "sex": "male", "weight": 80.0}

    subject_params = dict(subject_params)
    project_root = Path(__file__).resolve().parents[1]
    output_dir = make_output_dir(project_root, SCENARIO_ID, subject_params)

    print(
        "  [Scenario 39] Forward height-fall runtime: 1.80 m cube/platform stand "
        "-> support release -> gravity-driven forward fall -> prone grounded settle."
    )
    print(f"  [Scenario 39] Output folder -> {output_dir}\n")

    with standardized_run_context(project_root, output_dir, subject_params):
        import scenario39_legacy as legacy39
        result = legacy39.run_with_subject(subject_params)

    manifest_payload = result if isinstance(result, dict) else {}
    manifest_payload.setdefault("description", DESCRIPTION)
    manifest_payload.setdefault(
        "task_contract",
        {
            "initial_state": "humanoid standing upright on a 1.80 m cube/platform",
            "trigger": "platform support release with small forward COM bias; no prescribed impact body",
            "fall_direction": "forward/prone, determined by gravity and contact physics",
            "support_policy": "platform support only before release; free ballistic/contact dynamics after release",
            "settle": "grounded prone post-fall rest, no sitting recovery",
            "platform_height_m": 1.80,
        },
    )

    manifest = write_run_manifest(output_dir, SCENARIO_ID, subject_params, manifest_payload)
    print(f"\n  [Scenario 39] Run manifest -> {manifest}")
    print(f"  [Scenario 39] Output folder -> {output_dir}")

    payload = {"scenario_id": SCENARIO_ID, "output_dir": str(output_dir), "manifest": str(manifest)}
    if isinstance(result, dict):
        payload.update(result)
    return payload


if __name__ == "__main__":
    run()
