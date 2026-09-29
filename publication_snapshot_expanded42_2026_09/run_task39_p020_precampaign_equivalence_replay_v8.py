#!/usr/bin/env python3
"""
TASK39 P020 PRE-CAMPAIGN SOURCE EQUIVALENCE REPLAY V8
======================================================

ONE controlled replay after V2-V6 diagnostic review.

Scientific purpose
------------------
Test the strongest genuine historical source candidate identified by V4:
  scenario39_legacy.py
  Git commit db9a7b67163edcf58d64d66afd4ed3271b34899b
  expected blob SHA256 ae5180d9afdc1a76b3b7bcc82e8d3b6f4fb65c005688734f96bdfd583d81f399

The current September-1 source that FAILED is NOT used.

This launcher:
  1) extracts the historical source directly from Git;
  2) verifies its SHA256;
  3) finds the already-proven V2 headless/instrumented replay driver;
  4) executes that driver with the historical scenario module injected in memory;
  5) does NOT overwrite scenario39_legacy.py on disk;
  6) preserves the previous failed replay output;
  7) independently recomputes the strict equivalence gate;
  8) marks figure pose/state ready ONLY if all original thresholds pass.

The canonical corrected396 CSV remains primary for paper signal panels.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
import re
import runpy
import shutil
import subprocess
import sys
import tempfile
import traceback
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

PROJECT = Path(os.environ.get(
    "MUJOCO_PROJECT_ROOT",
    "/mnt/hdd16T/ToqeerHomeBackup/mujoco_project"
)).resolve()

PYTHON = PROJECT / ".venv/bin/python"
COMMIT = "db9a7b67163edcf58d64d66afd4ed3271b34899b"
GIT_SOURCE = "scenario39_legacy.py"
EXPECTED_SOURCE_SHA256 = "ae5180d9afdc1a76b3b7bcc82e8d3b6f4fb65c005688734f96bdfd583d81f399"

OLD_OUT = PROJECT / "outputs/task39_p020_exact_pose_replay_v2_headless"
OLD_ZIP = PROJECT / "outputs/task39_p020_exact_pose_replay_v2_headless.zip"
OLD_SHA = PROJECT / "outputs/task39_p020_exact_pose_replay_v2_headless.zip.sha256"

NEW_OUT = PROJECT / "outputs/task39_p020_precampaign_equivalence_replay_v8"
NEW_ZIP = PROJECT / "outputs/task39_p020_precampaign_equivalence_replay_v8.zip"
NEW_SHA = PROJECT / "outputs/task39_p020_precampaign_equivalence_replay_v8.zip.sha256"

CANON = PROJECT / (
    "outputs/_highrate_overnight/campaign_highrate_truth_v3_rne_corrected396/"
    "runs/P020/task_39/scenario39_age49_h1p63_sex_female_w73p0_20260826_144048/"
    "fall_scenario39_age49_20260826_144048.csv"
)

CANON_PHASES = {
    "SETUP": 4.633333333333462,
    "FALL_ONSET": 5.633333333333462,
    "MAX_DESCENT": 6.0133,
    "FIRST_CONTACT": 6.1433,
    "PEAK_IMPACT": 6.166666666666878,
    "POST_IMPACT": 6.416666666666878,
    "REST": 7.4833,
}

THRESH = {
    "pelvis_height_rmse_m": 0.03,
    "sensor_position_3d_rmse_m": 0.04,
    "peak_impact_time_delta_s": 0.10,
    "total_duration_delta_s": 0.15,
}

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def write_json(p: Path, obj: Any):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")

def git_show(commit: str, rel: str) -> str:
    cp = subprocess.run(
        ["git", "-C", str(PROJECT), "show", f"{commit}:{rel}"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False
    )
    if cp.returncode != 0:
        raise RuntimeError(f"git show failed: {cp.stderr.strip()}")
    return cp.stdout

def find_replay_driver() -> tuple[Path, list[dict[str, Any]]]:
    """
    Resolve the ORIGINAL V2 headless exact-pose replay driver.

    V7 bug fixed here:
      V7's heuristic could select V7 itself. V8 hard-excludes this historical
      replay launcher family and strongly prefers the original
      exact_pose_replay_v2_headless driver.
    """
    self_path = Path(__file__).resolve()

    preferred_names = [
        "run_task39_p020_exact_pose_replay_v2_headless.py",
        "task39_p020_exact_pose_replay_v2_headless.py",
        "run_task39_p020_exact_pose_replay_bundle_v2_headless.py",
        "run_task39_p020_exact_pose_replay_v2.py",
    ]

    # 1) Exact preferred filenames first.
    for name in preferred_names:
        p = (PROJECT / name).resolve()
        if p.exists() and p.is_file() and p != self_path:
            txt = p.read_text(encoding="utf-8", errors="replace")
            required_any = [
                "MarkerKinematicsExporter",
                "full_state_frames",
                "phase_state_indices",
                "REPLAY_EQUIVALENCE",
                "headless_viewer_patch",
            ]
            if "scenario39_legacy" in txt and sum(m in txt for m in required_any) >= 2:
                return p, [{
                    "path": str(p),
                    "score": 1000,
                    "name": p.name,
                    "reason": "exact preferred V2 headless replay filename",
                }]

    # 2) Fallback discovery across project-root Python launchers.
    hard_exclude = [
        "precampaign_equivalence",
        "historical_source",
        "resolver_fix",
        "diagnos",
        "mismatch",
        "provenance",
        "adjudicat",
        "callchain",
        "residue",
        "collector",
        "extractor",
        "schema",
        "probe",
        "figure",
        "patch",
    ]
    markers = {
        "MarkerKinematicsExporter": 12,
        "full_state_frames": 12,
        "phase_state_indices": 10,
        "REPLAY_EQUIVALENCE": 12,
        "USE_FOR_PAPER": 8,
        "headless_viewer_patch": 10,
        "canonical_phase_timestamps": 8,
        "scenario39_legacy_SOURCE_USED": 8,
        "render_capture_log": 6,
        "phase_contacts": 6,
        "phase_markers_long": 6,
        "mujoco.Renderer": 4,
        "launch_passive": 3,
    }

    rows = []
    for p in PROJECT.glob("*.py"):
        try:
            rp = p.resolve()
        except Exception:
            continue
        if rp == self_path:
            continue

        lowname = p.name.lower()
        if any(x in lowname for x in hard_exclude):
            continue

        try:
            body = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        if "scenario39_legacy" not in body:
            continue

        score = sum(w for m, w in markers.items() if m in body)

        # Strong filename evidence for the original V2 launcher.
        if "exact_pose_replay" in lowname:
            score += 30
        if "v2" in lowname:
            score += 20
        if "headless" in lowname:
            score += 20
        if "p020" in lowname:
            score += 10
        if "task39" in lowname:
            score += 8

        # Require multiple instrumentation signatures so ordinary scenario
        # launchers cannot win.
        instrumentation_hits = sum(
            m in body for m in [
                "MarkerKinematicsExporter",
                "full_state_frames",
                "phase_state_indices",
                "REPLAY_EQUIVALENCE",
                "headless_viewer_patch",
                "render_capture_log",
            ]
        )
        if instrumentation_hits < 2:
            continue

        rows.append({
            "path": str(rp),
            "score": score,
            "name": p.name,
            "instrumentation_hits": instrumentation_hits,
        })

    rows.sort(key=lambda r: (-r["score"], r["name"]))

    if not rows:
        raise RuntimeError(
            "Could not locate the original V2 headless exact-pose replay Python "
            "driver after excluding V7/V8 and diagnostic/provenance scripts."
        )

    # Refuse ambiguous/weak selection rather than recursively executing V8.
    if rows[0]["score"] < 45:
        raise RuntimeError(f"V2 replay-driver confidence too low: {rows[:10]}")

    if "precampaign_equivalence" in Path(rows[0]["path"]).name.lower():
        raise RuntimeError("Safety stop: resolver selected a historical-equivalence launcher.")

    return Path(rows[0]["path"]), rows[:20]


def preserve_path(p: Path, tag: str) -> Path | None:
    if not p.exists():
        return None
    backup = p.with_name(p.name + f".preserved_{tag}")
    i = 1
    while backup.exists():
        backup = p.with_name(p.name + f".preserved_{tag}_{i}")
        i += 1
    p.rename(backup)
    return backup

def restore_path(original: Path, backup: Path | None):
    if backup is None or not backup.exists():
        return
    if original.exists():
        raise RuntimeError(f"Refusing restore because target already exists: {original}")
    backup.rename(original)

def read_csv_comment(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        rows = [line for line in f if not line.startswith("#") and line.strip()]
    reader = csv.DictReader(rows)
    cols: dict[str, list[float]] = {}
    for r in reader:
        for k, v in r.items():
            if k is None:
                continue
            try:
                fv = float(v)
            except Exception:
                continue
            cols.setdefault(k, []).append(fv)
    return {k: np.asarray(v, dtype=float) for k, v in cols.items()}

def csv_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            return next(csv.reader([line]))
    return []

def find_replay_main(root: Path) -> Path:
    raw = root / "01_REPLAY_RAW"
    search_root = raw if raw.exists() else root
    required = {"timestamp", "pelvis_height", "sensor_pos_x", "sensor_pos_y", "sensor_pos_z"}
    candidates = []
    for p in search_root.glob("*.csv"):
        low = p.name.lower()
        if any(x in low for x in [
            "highrate", "marker", "segment", "joint", "dynamic", "contact",
            "quality", "grf", "phase"
        ]):
            continue
        try:
            h = set(csv_header(p))
        except Exception:
            continue
        if required.issubset(h):
            candidates.append(p)
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        candidates.sort(key=lambda p: (len(p.name), p.name))
        return candidates[0]
    raise RuntimeError(f"No replay main CSV found under {search_root}")

def interp_at(t_src, y_src, t_dst):
    order = np.argsort(t_src)
    t_src = np.asarray(t_src)[order]
    y_src = np.asarray(y_src)[order]
    return np.interp(t_dst, t_src, y_src)

def rmse(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if not np.any(m):
        return float("nan")
    return float(np.sqrt(np.mean((a[m]-b[m])**2)))

def strict_gate(canon_path: Path, replay_path: Path) -> dict[str, Any]:
    c = read_csv_comment(canon_path)
    r = read_csv_comment(replay_path)
    req = [
        "timestamp", "pelvis_height", "sensor_pos_x", "sensor_pos_y", "sensor_pos_z",
        "impact_magnitude"
    ]
    missing_c = [x for x in req if x not in c]
    missing_r = [x for x in req if x not in r]
    if missing_c or missing_r:
        raise RuntimeError(f"Gate columns missing. canonical={missing_c}, replay={missing_r}")

    tc = c["timestamp"]
    tr = r["timestamp"]
    start = max(float(tc[0]), float(tr[0]))
    stop = min(float(tc[-1]), float(tr[-1]))
    mask = (tc >= start) & (tc <= stop)
    teval = tc[mask]

    cp = c["pelvis_height"][mask]
    rp = interp_at(tr, r["pelvis_height"], teval)
    pelvis_rmse = rmse(cp, rp)

    sq = np.zeros(len(teval), dtype=float)
    for ch in ["sensor_pos_x", "sensor_pos_y", "sensor_pos_z"]:
        cc = c[ch][mask]
        rr = interp_at(tr, r[ch], teval)
        sq += (cc-rr)**2
    sensor3d = float(np.sqrt(np.mean(sq)))

    canon_impact = float(tc[int(np.nanargmax(c["impact_magnitude"]))])
    replay_impact = float(tr[int(np.nanargmax(r["impact_magnitude"]))])
    impact_delta = abs(canon_impact - replay_impact)

    canon_duration = float(tc[-1] - tc[0])
    replay_duration = float(tr[-1] - tr[0])
    duration_delta = abs(canon_duration - replay_duration)

    metrics = {
        "pelvis_height_rmse_m": pelvis_rmse,
        "sensor_position_3d_rmse_m": sensor3d,
        "canonical_peak_impact_s": canon_impact,
        "replay_peak_impact_s": replay_impact,
        "peak_impact_time_delta_s": impact_delta,
        "canonical_duration_s": canon_duration,
        "replay_duration_s": replay_duration,
        "total_duration_delta_s": duration_delta,
    }
    checks = {
        k: metrics[k] <= THRESH[k]
        for k in THRESH
    }
    return {
        "thresholds": THRESH,
        "metrics": metrics,
        "checks": checks,
        "strict_equivalence_pass": bool(all(checks.values())),
    }

def patch_driver_source(txt: str, historical_path: Path) -> str:
    # Replace only exact string literals "scenario39_legacy.py"; do not alter
    # output audit names such as scenario39_legacy_SOURCE_USED.py.
    hp = str(historical_path).replace("\\", "\\\\")
    txt = re.sub(
        r'(["\'])scenario39_legacy\.py\1',
        lambda m: repr(hp),
        txt
    )
    return txt

def main():
    print("="*104)
    print("TASK39 P020 PRE-CAMPAIGN SOURCE EQUIVALENCE REPLAY V8 — ONE CONTROLLED REPLAY")
    print("="*104)
    print("Historical source commit:", COMMIT)
    print("Current September-1 failed source will NOT be used.")
    print("Canonical corrected396 signal remains primary.")

    if not CANON.exists():
        raise SystemExit(f"ERROR: canonical P020 CSV not found: {CANON}")

    tag = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    tmp_root = PROJECT / "outputs" / f"_task39_precampaign_runtime_tmp_{tag}"
    tmp_root.mkdir(parents=True, exist_ok=False)
    hist_source = tmp_root / "scenario39_legacy.py"

    source_txt = git_show(COMMIT, GIT_SOURCE)
    hist_source.write_text(source_txt, encoding="utf-8")
    got_hash = sha256_file(hist_source)
    print("Historical source SHA256:", got_hash)
    if got_hash != EXPECTED_SOURCE_SHA256:
        raise SystemExit(
            f"ERROR: historical source hash mismatch. expected={EXPECTED_SOURCE_SHA256} got={got_hash}"
        )
    print("HISTORICAL SOURCE HASH: PASS")

    driver, candidates = find_replay_driver()
    if driver.resolve() == Path(__file__).resolve():
        raise SystemExit("ERROR: V8 safety guard: replay driver resolved to V8 itself.")
    if "precampaign_equivalence" in driver.name.lower():
        raise SystemExit(
            f"ERROR: V8 safety guard: refusing historical-equivalence launcher as V2 driver: {driver}"
        )
    print("Resolved existing V2 replay driver:", driver)
    print("Driver discovery score:", candidates[0]["score"])
    write_json(tmp_root / "driver_discovery.json", {
        "selected": str(driver),
        "candidates": candidates,
    })

    # Preserve the old failed replay and any prior V7 result.
    if NEW_OUT.exists():
        preserve_path(NEW_OUT, "previous_v8")
    if NEW_ZIP.exists():
        preserve_path(NEW_ZIP, "previous_v8")
    if NEW_SHA.exists():
        preserve_path(NEW_SHA, "previous_v8")

    backups = {
        OLD_OUT: preserve_path(OLD_OUT, tag),
        OLD_ZIP: preserve_path(OLD_ZIP, tag),
        OLD_SHA: preserve_path(OLD_SHA, tag),
    }

    run_error = None
    new_generated_dir = False
    try:
        # Load historical source as the module name expected by the proven V2 driver.
        spec = importlib.util.spec_from_file_location("scenario39_legacy", hist_source)
        if spec is None or spec.loader is None:
            raise RuntimeError("Could not create import spec for historical scenario39 source")
        module = importlib.util.module_from_spec(spec)
        sys.modules["scenario39_legacy"] = module
        spec.loader.exec_module(module)

        # Give the V2 driver the same source through both module import and any exact
        # path literal it may use for source audit/loading, without modifying it on disk.
        driver_txt = driver.read_text(encoding="utf-8", errors="replace")
        patched_txt = patch_driver_source(driver_txt, hist_source)

        os.environ["TASK39_HISTORICAL_SOURCE_OVERRIDE"] = str(hist_source)
        os.environ["TASK39_HISTORICAL_SOURCE_COMMIT"] = COMMIT
        os.environ["TASK39_HISTORICAL_SOURCE_SHA256"] = got_hash
        os.environ.setdefault("MUJOCO_GL", "egl")

        glb = {
            "__name__": "__main__",
            "__file__": str(driver),
            "__package__": None,
            "__cached__": None,
        }
        print("-"*104)
        print("Executing the already-proven V2 headless/instrumented driver with historical source injected...")
        print("-"*104)
        try:
            exec(compile(patched_txt, str(driver), "exec"), glb, glb)
        except SystemExit as e:
            if e.code not in (None, 0):
                raise RuntimeError(f"Replay driver exited with status {e.code}") from e

        if not OLD_OUT.exists():
            raise RuntimeError(
                "Replay driver completed but expected output directory was not created: "
                f"{OLD_OUT}"
            )

        OLD_OUT.rename(NEW_OUT)
        new_generated_dir = True
        if OLD_ZIP.exists():
            OLD_ZIP.rename(NEW_ZIP)
        if OLD_SHA.exists():
            OLD_SHA.rename(NEW_SHA)

    except Exception as e:
        run_error = f"{type(e).__name__}: {e}"
        traceback.print_exc()
        # Preserve any partial newly generated replay output rather than deleting evidence.
        if OLD_OUT.exists() and not NEW_OUT.exists():
            OLD_OUT.rename(NEW_OUT)
            new_generated_dir = True
        if OLD_ZIP.exists() and not NEW_ZIP.exists():
            OLD_ZIP.rename(NEW_ZIP)
        if OLD_SHA.exists() and not NEW_SHA.exists():
            OLD_SHA.rename(NEW_SHA)
    finally:
        # Restore the prior failed V2 replay exactly where it was.
        for original, backup in backups.items():
            if backup is not None and backup.exists():
                if original.exists():
                    # Anything remaining here belongs to this run; preserve it separately.
                    stray = original.with_name(original.name + f".stray_{tag}")
                    original.rename(stray)
                backup.rename(original)

    decision_dir = NEW_OUT / "05_HISTORICAL_EQUIVALENCE_DECISION"
    decision_dir.mkdir(parents=True, exist_ok=True)

    replay_main = None
    gate = None
    if run_error is None and NEW_OUT.exists():
        try:
            replay_main = find_replay_main(NEW_OUT)
            gate = strict_gate(CANON, replay_main)
        except Exception as e:
            run_error = f"Independent gate error: {type(e).__name__}: {e}"
            traceback.print_exc()

    strict_pass = bool(gate and gate.get("strict_equivalence_pass"))
    paper_pose_ready = bool(strict_pass)

    decision = {
        "historical_source_commit": COMMIT,
        "historical_source_path_in_git": GIT_SOURCE,
        "historical_source_sha256": got_hash,
        "expected_historical_source_sha256": EXPECTED_SOURCE_SHA256,
        "existing_v2_driver_used": str(driver),
        "current_failed_source_was_used": False,
        "current_source_file_modified_on_disk": False,
        "previous_failed_replay_preserved_and_restored": True,
        "canonical_csv": str(CANON),
        "replay_main_csv": str(replay_main) if replay_main else None,
        "driver_or_gate_error": run_error,
        "strict_gate": gate,
        "paper_gate": "PASS_FOR_POSE_STATE_EVIDENCE" if paper_pose_ready else "BLOCK_REMAINS",
        "use_replay_pose_state_for_task39_figure": paper_pose_ready,
        "canonical_corrected396_signal_remains_primary": True,
        "canonical_phase_targets_s": CANON_PHASES,
        "scientific_scope_if_pass": (
            "Replay pose/qpos/body/marker/contact state may be used for the Task-39 figure "
            "only at the canonical phase targets. Canonical corrected396 CSV remains the "
            "primary source for signal/impact/IMU panels."
        ),
    }
    write_json(decision_dir / "HISTORICAL_SOURCE_REPLAY_DECISION.json", decision)

    # Figure-ready manifest, created only on strict PASS.
    if paper_pose_ready:
        files = []
        for p in NEW_OUT.rglob("*"):
            if not p.is_file():
                continue
            low = p.name.lower()
            if any(k in low for k in [
                "phase", "render", "marker", "joint", "contact", "dynamic",
                "full_state", "inventory", "validation", "replay_equivalence"
            ]):
                files.append(str(p.relative_to(NEW_OUT)))
        write_json(decision_dir / "FIGURE_EVIDENCE_READY.json", {
            "status": "READY",
            "canonical_phase_targets_s": CANON_PHASES,
            "validated_replay_files": sorted(files),
            "canonical_signal_source": str(CANON),
        })

    # Rebuild V7 ZIP so it contains our independent decision files.
    if NEW_ZIP.exists():
        NEW_ZIP.unlink()
    with zipfile.ZipFile(NEW_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(NEW_OUT.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(NEW_OUT.parent))
    digest = sha256_file(NEW_ZIP)
    NEW_SHA.write_text(f"{digest}  {NEW_ZIP.name}\n", encoding="utf-8")

    # Temporary historical source is already recorded by hash/commit. Remove temp
    # runtime directory only after output decision is safely written.
    try:
        shutil.rmtree(tmp_root)
    except Exception:
        pass

    print("="*104)
    print("V8 FINAL DECISION")
    print("="*104)
    print("Historical source hash: PASS")
    print("Replay driver error:", run_error or "NONE")
    if gate:
        m = gate["metrics"]
        print("Canonical duration [s]:", m["canonical_duration_s"])
        print("Replay duration [s]:", m["replay_duration_s"])
        print("Duration delta [s]:", m["total_duration_delta_s"])
        print("Pelvis-height RMSE [m]:", m["pelvis_height_rmse_m"])
        print("Sensor-position 3D RMSE [m]:", m["sensor_position_3d_rmse_m"])
        print("Peak-impact time delta [s]:", m["peak_impact_time_delta_s"])
        print("STRICT EQUIVALENCE:", "PASS" if strict_pass else "FAIL")
    else:
        print("STRICT EQUIVALENCE: NOT COMPUTED")

    print("Paper pose/state gate:", decision["paper_gate"])
    print("Canonical corrected396 signal remains primary: YES")
    print("ZIP:", NEW_ZIP)
    print("ZIP SHA256:", digest)
    print("="*104)

    if run_error:
        sys.exit(5)
    if not strict_pass:
        # Nonzero status signals that figure pose/state is still blocked, but the
        # diagnostic ZIP has been preserved for review.
        sys.exit(10)

if __name__ == "__main__":
    main()
