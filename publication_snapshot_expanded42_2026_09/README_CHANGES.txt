Refactor summary
================

What changed
------------
1. backward_fall_walking_best.py
   - Created as the shared validated backbone derived from the task-34 legacy source.
   - Made import-safe so fall_core can import its classes/functions without triggering interactive prompts.
   - Kept the legacy standalone execution path under __main__.

2. fall_core.py
   - Continues to own shared simulation/export/validation flow.
   - Added per-run output directory creation under outputs/.
   - Output folder name now includes scenario id, age, height, sex, weight, and timestamp.
   - Added run_manifest.json for each run.

3. scenarios/
   - Added thin wrapper modules for scenario_20.py through scenario_42.py for all registered scenarios.
   - Each wrapper exposes run(subject_params), matching fall_dispatcher expectations.

4. fall_dispatcher.py
   - Kept the dispatcher flow.
   - Updated the implemented marker to show 20 and 34 as implemented.

5. legacy_sources/
   - Preserved the uploaded standalone scenario_20.py and scenario_34.py for reference.

Important note
--------------
This package compiles cleanly in the container, but the container used for packaging does not have the user's MuJoCo/humenv runtime installed, so full end-to-end execution was not run here.
