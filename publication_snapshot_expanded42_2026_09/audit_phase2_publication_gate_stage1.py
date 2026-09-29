from pathlib import Path
import pandas as pd
import numpy as np
import re
import json
import csv
import os
import sys
import traceback
from datetime import datetime

ROOT = Path("/mnt/hdd16T/ToqeerHomeBackup/mujoco_project")
REAL_ROOT = Path("/mnt/hdd16T/ToqeerHomeBackup/toqeer/uniVr-dataset")
UNIVR = REAL_ROOT / "UniVrFall_Dataset"
KFALL = REAL_ROOT / "KFall_oriented"
EXTENDED35 = (
    ROOT
    / "outputs"
    / "_highrate_overnight"
    / "campaign_highrate_truth_v2_extended35"
)
RUNNER = ROOT / "run_phase2_complete_final_campaign.py"

OUT = ROOT / "outputs" / "phase2_publication_gate_audit_20260826"
STAGE = OUT / "stage1_raw_data_audit"

STAGE.mkdir(parents=True, exist_ok=True)

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
TABULAR_EXT = {".csv", ".txt", ".tsv", ".xlsx", ".xls", ".json"}

def now():
    return datetime.now().isoformat(timespec="seconds")

def safe_read_csv(path, nrows=None):
    attempts = [
        {},
        {"comment": "#"},
        {"sep": None, "engine": "python"},
        {"comment": "#", "sep": None, "engine": "python"},
    ]
    last = None
    for kw in attempts:
        try:
            return pd.read_csv(path, nrows=nrows, **kw)
        except Exception as e:
            last = e
    raise last

def norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())

def find_sensor_columns(cols):
    result = {
        "ax": None, "ay": None, "az": None,
        "gx": None, "gy": None, "gz": None,
    }

    normalized = {norm(c): c for c in cols}

    aliases = {
        "ax": [
            "accx", "accelx", "accelerometerx", "accelerationx",
            "ax", "accxg", "accxmps2", "accxms2",
        ],
        "ay": [
            "accy", "accely", "accelerometery", "accelerationy",
            "ay", "accyg", "accymps2", "accyms2",
        ],
        "az": [
            "accz", "accelz", "accelerometerz", "accelerationz",
            "az", "acczg", "acczmps2", "acczms2",
        ],
        "gx": [
            "gyrox", "gyrx", "angularvelocityx", "wx",
            "gx", "gyroxrad", "gyroxdegs",
        ],
        "gy": [
            "gyroy", "gyry", "angularvelocityy", "wy",
            "gy", "gyroyrad", "gyroydegs",
        ],
        "gz": [
            "gyroz", "gyrz", "angularvelocityz", "wz",
            "gz", "gyrozrad", "gyrozdegs",
        ],
    }

    for key, names in aliases.items():
        for name in names:
            if name in normalized:
                result[key] = normalized[name]
                break

    if not all(result[k] for k in ["ax", "ay", "az"]):
        for c in cols:
            n = norm(c)
            if result["ax"] is None and ("acc" in n or "accel" in n) and n.endswith("x"):
                result["ax"] = c
            if result["ay"] is None and ("acc" in n or "accel" in n) and n.endswith("y"):
                result["ay"] = c
            if result["az"] is None and ("acc" in n or "accel" in n) and n.endswith("z"):
                result["az"] = c

    if not all(result[k] for k in ["gx", "gy", "gz"]):
        for c in cols:
            n = norm(c)
            if result["gx"] is None and ("gyro" in n or "gyr" in n) and n.endswith("x"):
                result["gx"] = c
            if result["gy"] is None and ("gyro" in n or "gyr" in n) and n.endswith("y"):
                result["gy"] = c
            if result["gz"] is None and ("gyro" in n or "gyr" in n) and n.endswith("z"):
                result["gz"] = c

    return result

def guess_subject_from_path(path):
    candidates = []
    for part in path.parts:
        low = part.lower()

        patterns = [
            r"^(s\d{1,4})$",
            r"^(p\d{1,4})$",
            r"^(subj(?:ect)?[_-]?\d+)$",
            r"^(participant[_-]?\d+)$",
        ]

        for pat in patterns:
            m = re.match(pat, low)
            if m:
                candidates.append(part)

    return "|".join(candidates)

def path_hint(path):
    s = str(path).lower()

    fall_words = [
        "fall", "faint", "slip", "trip", "collapse",
        "forward", "backward", "lateral",
    ]

    activity_words = [
        "adl", "activity", "walk", "walking", "sit",
        "stand", "stairs", "jog", "run", "lying",
        "pickup", "bend",
    ]

    fall = any(w in s for w in fall_words)
    activity = any(w in s for w in activity_words)

    if fall and not activity:
        return "fall_path_hint"
    if activity and not fall:
        return "activity_path_hint"
    if fall and activity:
        return "mixed_path_hint"
    return "unknown"

def count_rows(path):
    try:
        df = safe_read_csv(path)
        return len(df), list(df.columns)
    except Exception:
        return None, []

def inventory_dataset(root, out_csv, dataset_name):
    rows = []

    files = sorted(root.rglob("*.csv")) if root.exists() else []

    print(f"{dataset_name}: CSV files found = {len(files)}")

    for i, f in enumerate(files, 1):
        nrows = None
        cols = []

        try:
            df_head = safe_read_csv(f, nrows=5)
            cols = list(df_head.columns)
        except Exception:
            pass

        parent = f.parent.name
        grandparent = f.parent.parent.name if len(f.parents) > 1 else ""
        ggparent = f.parent.parent.parent.name if len(f.parents) > 2 else ""

        try:
            rel = f.relative_to(root)
        except Exception:
            rel = f

        rows.append({
            "dataset": dataset_name,
            "full_path": str(f),
            "relative_path": str(rel),
            "filename": f.name,
            "parent_parts_minus2": parent,
            "parts_minus3": grandparent,
            "parts_minus4": ggparent,
            "subject_regex_candidates": guess_subject_from_path(f),
            "path_semantic_hint_only": path_hint(f),
            "size_bytes": f.stat().st_size,
            "columns": "|".join(map(str, cols)),
        })

        if i % 500 == 0:
            print(f"  inventoried {i}/{len(files)}")

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return files, pd.DataFrame(rows)

def metadata_candidates(root, out_csv, dataset):
    rows = []

    if not root.exists():
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        return

    keywords = [
        "label", "annot", "task", "event", "meta",
        "info", "activity", "fall", "subject",
        "participant", "description", "protocol",
    ]

    for f in sorted(root.rglob("*")):
        if not f.is_file():
            continue
        if f.suffix.lower() not in TABULAR_EXT:
            continue

        low = f.name.lower()

        if any(k in low for k in keywords):
            rows.append({
                "dataset": dataset,
                "path": str(f),
                "filename": f.name,
                "suffix": f.suffix.lower(),
                "size_bytes": f.stat().st_size,
            })

    pd.DataFrame(rows).to_csv(out_csv, index=False)

def representative_files(files, n=12):
    if not files:
        return []

    if len(files) <= n:
        return files

    idx = np.linspace(0, len(files)-1, n, dtype=int)
    return [files[i] for i in idx]

def numeric_summary_for_file(path, dataset):
    row = {
        "dataset": dataset,
        "path": str(path),
        "status": "",
    }

    try:
        df = safe_read_csv(path)
        cols = find_sensor_columns(df.columns)

        row["columns"] = "|".join(map(str, df.columns))
        row["detected_ax"] = cols["ax"]
        row["detected_ay"] = cols["ay"]
        row["detected_az"] = cols["az"]
        row["detected_gx"] = cols["gx"]
        row["detected_gy"] = cols["gy"]
        row["detected_gz"] = cols["gz"]
        row["rows"] = len(df)

        for key in ["ax", "ay", "az", "gx", "gy", "gz"]:
            c = cols[key]
            if c is None:
                continue

            s = pd.to_numeric(df[c], errors="coerce").dropna()

            if len(s) == 0:
                continue

            row[f"{key}_min"] = float(s.min())
            row[f"{key}_p01"] = float(s.quantile(0.01))
            row[f"{key}_median"] = float(s.median())
            row[f"{key}_p99"] = float(s.quantile(0.99))
            row[f"{key}_max"] = float(s.max())

        if all(cols[k] for k in ["ax", "ay", "az"]):
            arr = df[[cols["ax"], cols["ay"], cols["az"]]].apply(
                pd.to_numeric, errors="coerce"
            ).dropna().to_numpy(float)

            if len(arr):
                mag = np.linalg.norm(arr, axis=1)
                row["acc_mag_median_stored_units"] = float(np.median(mag))
                row["acc_mag_p95_stored_units"] = float(np.quantile(mag, 0.95))
                row["acc_mag_max_stored_units"] = float(np.max(mag))

        if all(cols[k] for k in ["gx", "gy", "gz"]):
            arr = df[[cols["gx"], cols["gy"], cols["gz"]]].apply(
                pd.to_numeric, errors="coerce"
            ).dropna().to_numpy(float)

            if len(arr):
                mag = np.linalg.norm(arr, axis=1)
                row["gyro_mag_median_stored_units"] = float(np.median(mag))
                row["gyro_mag_p95_stored_units"] = float(np.quantile(mag, 0.95))
                row["gyro_mag_max_stored_units"] = float(np.max(mag))

        row["status"] = "OK"

    except Exception as e:
        row["status"] = f"ERROR: {e}"

    return row

def unit_diagnostics(univr_files, kfall_files):
    rows = []

    for f in representative_files(univr_files, 15):
        rows.append(numeric_summary_for_file(f, "UniVRFall"))

    for f in representative_files(kfall_files, 15):
        rows.append(numeric_summary_for_file(f, "KFall"))

    pd.DataFrame(rows).to_csv(
        STAGE / "physical_unit_range_samples.csv",
        index=False,
    )

def group_path_audit(inv_df, out_csv):
    rows = []

    if inv_df.empty:
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        return

    for _, r in inv_df.iterrows():
        rows.append({
            "dataset": r["dataset"],
            "full_path": r["full_path"],
            "parts_minus2_used_by_old_runner": r["parent_parts_minus2"],
            "parts_minus3": r["parts_minus3"],
            "parts_minus4": r["parts_minus4"],
            "subject_regex_candidates": r["subject_regex_candidates"],
        })

    pd.DataFrame(rows).to_csv(out_csv, index=False)

def extract_task_id(path):
    s = str(path)

    patterns = [
        r"task[_-]?(\d+)",
        r"scenario[_-]?(\d+)",
        r"scenario(\d+)",
    ]

    for pat in patterns:
        m = re.search(pat, s, flags=re.I)
        if m:
            return int(m.group(1))

    return None

def extract_profile_hint(path):
    s = str(path)

    patterns = [
        r"(P\d{2,3})",
        r"age[_-]?(\d+)",
        r"age(\d+)",
    ]

    vals = []

    for pat in patterns:
        for m in re.finditer(pat, s, flags=re.I):
            vals.append(m.group(0))

    return "|".join(dict.fromkeys(vals))

def current_phase2_sim_label(task):
    positive = set(list(range(20,35)) + list(range(37,43)))

    if task is None:
        return None

    return int(task in positive)

def extended35_inventory():
    files = []

    if EXTENDED35.exists():
        files = sorted(EXTENDED35.rglob("*highrate_truth*.csv"))

        if not files:
            files = sorted(EXTENDED35.rglob("*.csv"))

    rows = []

    print(f"Extended35 CSV candidates found = {len(files)}")

    for i, f in enumerate(files, 1):
        task = extract_task_id(f)

        nrows = None
        cols = []

        try:
            df = safe_read_csv(f)
            nrows = len(df)
            cols = list(df.columns)
        except Exception:
            pass

        rows.append({
            "full_path": str(f),
            "task_id": task,
            "profile_hint": extract_profile_hint(f),
            "rows": nrows,
            "columns": "|".join(map(str, cols)),
            "old_phase2_binary_label": current_phase2_sim_label(task),
        })

        if i % 100 == 0:
            print(f"  inventoried {i}/{len(files)}")

    df = pd.DataFrame(rows)

    df.to_csv(
        STAGE / "extended35_source_trial_inventory.csv",
        index=False,
    )

    if not df.empty:
        summary = (
            df.groupby(
                ["task_id", "old_phase2_binary_label"],
                dropna=False
            )
            .size()
            .reset_index(name="source_file_count")
        )

        summary.to_csv(
            STAGE / "extended35_task_label_summary.csv",
            index=False,
        )

def cache_label_audit():
    cache_root = ROOT / "outputs" / "phase2_complete_final_campaign" / "dataset_cache"

    rows = []

    if cache_root.exists():
        for p in sorted(cache_root.rglob("y.npy")):
            try:
                y = np.load(p, mmap_mode="r")
                u, c = np.unique(y, return_counts=True)

                rows.append({
                    "path": str(p),
                    "shape": str(y.shape),
                    "class_counts": json.dumps(
                        {str(k): int(v) for k, v in zip(u.tolist(), c.tolist())}
                    ),
                })
            except Exception as e:
                rows.append({
                    "path": str(p),
                    "shape": "",
                    "class_counts": f"ERROR: {e}",
                })

    pd.DataFrame(rows).to_csv(
        STAGE / "cached_label_counts.csv",
        index=False,
    )

def runner_evidence():
    out = STAGE / "current_phase2_runner_evidence.txt"

    if not RUNNER.exists():
        out.write_text(f"Runner missing: {RUNNER}\n")
        return

    lines = RUNNER.read_text(errors="ignore").splitlines()

    patterns = [
        "def create_windows",
        "label = 1",
        "label=1",
        "0.00981",
        "0.07",
        "fall_tasks",
        "fall_task",
        "EXT_GROUP",
        "GroupKFold",
        "cut=int",
        "cut = int",
        "np.random.permutation",
        "best_model.pth",
        "load_state_dict",
        "CNN_config",
        "sim_fraction",
    ]

    selected = []

    for idx, line in enumerate(lines, 1):
        if any(p in line for p in patterns):
            start = max(1, idx - 4)
            end = min(len(lines), idx + 8)

            selected.append(
                f"\n{'='*78}\nLINES {start}-{end}\n{'='*78}\n"
            )

            for j in range(start, end + 1):
                selected.append(
                    f"{j:05d}: {lines[j-1]}\n"
                )

    out.write_text("".join(selected))

def provenance_search():
    roots = [
        ROOT,
    ]

    name_patterns = [
        "*high*rate*imu*.py",
        "*sidecar*.py",
        "*sensor*.py",
        "*imu*.py",
    ]

    candidates = set()

    for root in roots:
        for pat in name_patterns:
            for f in root.rglob(pat):
                if not f.is_file():
                    continue

                s = str(f)

                if "/.venv/" in s or "/__pycache__/" in s:
                    continue

                candidates.add(f)

    rows = []
    snippets = []

    search_terms = [
        "MJOBJ_BODY",
        "MJOBJ_XBODY",
        "mjOBJ_BODY",
        "mjOBJ_XBODY",
        "ximat",
        "xmat",
        "xipos",
        "mj_objectVelocity",
        "mj_objectAcceleration",
        "sensor",
        "accel",
        "gyro",
    ]

    for f in sorted(candidates):
        try:
            text = f.read_text(errors="ignore")
        except Exception:
            continue

        hits = [term for term in search_terms if term in text]

        if not hits:
            continue

        rows.append({
            "path": str(f),
            "mtime": datetime.fromtimestamp(
                f.stat().st_mtime
            ).isoformat(timespec="seconds"),
            "matched_terms": "|".join(hits),
            "size_bytes": f.stat().st_size,
        })

        lines = text.splitlines()

        for i, line in enumerate(lines, 1):
            if any(term in line for term in search_terms[:8]):
                start = max(1, i - 3)
                end = min(len(lines), i + 5)

                snippets.append(
                    f"\n{'='*78}\n{f}\nLINES {start}-{end}\n{'='*78}\n"
                )

                for j in range(start, end + 1):
                    snippets.append(
                        f"{j:05d}: {lines[j-1]}\n"
                    )

    pd.DataFrame(rows).to_csv(
        STAGE / "extended35_sensor_code_candidates.csv",
        index=False,
    )

    (STAGE / "extended35_sensor_code_snippets.txt").write_text(
        "".join(snippets)
    )

def directory_summary():
    lines = []

    for name, p in [
        ("UNIVR", UNIVR),
        ("KFALL", KFALL),
        ("EXTENDED35", EXTENDED35),
    ]:
        lines.append(f"{name}: {p}\n")
        lines.append(f"exists: {p.exists()}\n")

        if p.exists():
            try:
                children = sorted(p.iterdir())
                lines.append("top-level entries:\n")

                for c in children[:100]:
                    lines.append(
                        f"  {'DIR ' if c.is_dir() else 'FILE'} {c.name}\n"
                    )

            except Exception as e:
                lines.append(f"listing error: {e}\n")

        lines.append("\n")

    (STAGE / "dataset_directory_summary.txt").write_text(
        "".join(lines)
    )

def main():
    print("=" * 80)
    print("PHASE-2 PUBLICATION GATE — STAGE 1 RAW-DATA AUDIT")
    print("=" * 80)
    print("Started:", now())
    print("Output :", STAGE)
    print()

    directory_summary()

    univr_files, univr_inv = inventory_dataset(
        UNIVR,
        STAGE / "univr_file_inventory.csv",
        "UniVRFall",
    )

    kfall_files, kfall_inv = inventory_dataset(
        KFALL,
        STAGE / "kfall_file_inventory.csv",
        "KFall",
    )

    metadata_candidates(
        UNIVR,
        STAGE / "univr_metadata_candidates.csv",
        "UniVRFall",
    )

    metadata_candidates(
        KFALL,
        STAGE / "kfall_metadata_candidates.csv",
        "KFall",
    )

    unit_diagnostics(
        univr_files,
        kfall_files,
    )

    group_path_audit(
        univr_inv,
        STAGE / "univr_group_path_audit.csv",
    )

    group_path_audit(
        kfall_inv,
        STAGE / "kfall_group_path_audit.csv",
    )

    extended35_inventory()
    cache_label_audit()
    runner_evidence()
    provenance_search()

    summary = []

    summary.append("# Phase-2 Publication Gate — Stage 1\n\n")
    summary.append(f"Generated: {now()}\n\n")
    summary.append("## Purpose\n\n")
    summary.append(
        "Read-only audit of physical-data file structure, candidate metadata, "
        "stored sensor ranges, grouping paths, Extended35 source trials, "
        "cached labels, Phase-2 runner evidence, and sensor-code provenance.\n\n"
    )

    summary.append("## Raw counts\n\n")
    summary.append(f"- UniVRFall CSV files: {len(univr_files)}\n")
    summary.append(f"- KFall CSV files: {len(kfall_files)}\n")

    ext_inv = STAGE / "extended35_source_trial_inventory.csv"

    if ext_inv.exists():
        try:
            ext_df = pd.read_csv(ext_inv)
            summary.append(
                f"- Extended35 candidate source CSV files: {len(ext_df)}\n"
            )
        except Exception:
            pass

    summary.append("\n## Important\n\n")
    summary.append(
        "This Stage-1 script does not modify data, train a CNN, or rerun MuJoCo.\n"
    )
    summary.append(
        "The path-based fall/activity field is only a search hint and must not "
        "be used as authoritative ground truth.\n"
    )
    summary.append(
        "Final UniVRFall/KFall labels must be resolved from authoritative "
        "dataset metadata/annotations.\n"
    )
    summary.append(
        "Sensor-unit conclusions must be based on actual local-file provenance "
        "plus numeric ranges, not on data-sheet sensitivities alone.\n"
    )

    (STAGE / "STAGE1_README.md").write_text(
        "".join(summary)
    )

    print()
    print("=" * 80)
    print("STAGE 1 AUDIT COMPLETE")
    print("=" * 80)

    print("Output:")
    print(STAGE)

    print()
    print("Key files:")
    for name in [
        "dataset_directory_summary.txt",
        "univr_file_inventory.csv",
        "univr_metadata_candidates.csv",
        "univr_group_path_audit.csv",
        "kfall_file_inventory.csv",
        "kfall_metadata_candidates.csv",
        "kfall_group_path_audit.csv",
        "physical_unit_range_samples.csv",
        "extended35_source_trial_inventory.csv",
        "extended35_task_label_summary.csv",
        "cached_label_counts.csv",
        "current_phase2_runner_evidence.txt",
        "extended35_sensor_code_candidates.csv",
        "extended35_sensor_code_snippets.txt",
        "STAGE1_README.md",
    ]:
        p = STAGE / name
        print(f"  {name}: {'OK' if p.exists() else 'MISSING'}")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(2)
