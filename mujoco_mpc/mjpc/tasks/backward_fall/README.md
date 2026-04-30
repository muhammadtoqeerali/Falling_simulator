# BackwardFallHumanoid MJPC task scaffold

This folder contains a practical starting point for a custom MJPC task aimed at:
- backward-fall onset,
- controlled impact,
- stable settle / lying posture.

## What it is
A compile-adapt scaffold for the official MJPC `Task` / `BaseResidualFn` interface.
It follows the same overall pattern as upstream MJPC task files such as `cartpole.h`,
`cartpole.cc`, and task XML.

## What you still need to adapt locally
1. Register `BackwardFallHumanoid` in your MJPC build, following the task-registration
   pattern used in your local `mujoco_mpc` checkout.
2. Verify the body names `Pelvis` and `Head` in your humanoid MJCF.
3. Adjust residual weights and parameters in `task.xml` after the first rollouts.
4. Set `residual_PhaseGate` from Python or task logic when impact has occurred.

## Recommended Python env
After you build and register the task, use:

```bash
export MJPC_ENABLE=1
export MJPC_TASK_ID="BackwardFallHumanoid"
export MJPC_SEND_AS=xml
export MJPC_PLANNER_STEPS=8
export MJPC_USE_WALK_RECOVER=1
export MJPC_USE_PERTURB_RECOVER=1
export MJPC_USE_REACT_RECOVER=1
export MJPC_USE_SETTLE=1
```

## Notes
- This scaffold is intentionally minimal and honest. It is not guaranteed to compile
  without adapting it to your local MJPC repo layout and humanoid model names.
- The accompanying `contact_tuning_defaults.xml` gives conservative XML-level contact
  defaults for feet and impact-prone upper-body geoms.
