#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil
import zipfile

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT = Path("/mnt/hdd16T/ToqeerHomeBackup/mujoco_project")
PROTECHTO = Path("/mnt/hdd16T/ToqeerHomeBackup/toqeer/Protechto-master")
PHYSICAL_ROOT = Path("/mnt/hdd16T/ToqeerHomeBackup/toqeer/uniVr-dataset")

DEFAULT_STAGE = PROJECT / "outputs/paper_evidence_archive_v1"
DEFAULT_ZIP = PROJECT / "outputs/paper_evidence_archive_v1_FULL_V3.zip"

CATALOG_REL = Path(
    "09_SIMULATOR_VALIDATION_GALLERY/"
    "VALIDATION_FIGURE_AND_DATA_CATALOG.csv"
)

CANONICAL_PATH_TOKENS = (
    "phase2_publication_gate_audit_20260826",
    "campaign_highrate_truth_v3_rne_corrected396",
)

LEGACY_OR_SUPERSEDED_TOKENS = (
    "phase2_lightweight_paper_final_campaign_v1",
    "phase2_lightweight_paper_aligned_inputs_v1",
    "phase2_corrected_final_campaign_v1",
    "phase2_protechto_final_campaign_v1",
    "phase2_full_test_report",
    "invalid_do_not_use",
)

FIG_EXT = {".png", ".pdf", ".svg", ".jpg", ".jpeg"}

AGE_RE = re.compile(r"(?:^|[_/])age(?P<age>\d{1,3})(?:[_/]|$)", re.I)
HEIGHT_RE = re.compile(r"(?:^|[_/])h(?P<a>\d+)p(?P<b>\d+)(?:[_/]|$)", re.I)
WEIGHT_RE = re.compile(r"(?:^|[_/])w(?P<a>\d+)p(?P<b>\d+)(?:[_/]|$)", re.I)
SEX_RE = re.compile(r"(?:^|[_/])sex_(?P<sex>male|female)(?:[_/]|$)", re.I)
SCENARIO_RE = re.compile(r"(?:^|[/_])scenario(?P<n>\d+)(?:[_/]|$)", re.I)
TASK_RE = re.compile(r"(?:^|[/_])task_(?P<n>\d+)(?:[_/]|$)", re.I)
PROFILE_RE = re.compile(r"(?:^|[/_])P(?P<n>\d{3})(?:[_/]|$)")
FALL_RE = re.compile(
    r"(fall_(?:forward|backward|lateral|side|vertical|height)"
    r"(?:_[a-z0-9]+)*?)_age\d+",
    re.I,
)


def sha256_file(path: Path, chunk=4 * 1024 * 1024):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def copy_file(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def decimal_from_match(m):
    if not m:
        return np.nan
    a = m.group("a")
    b = m.group("b")
    return float(f"{int(a)}.{b}")


def parse_profile_from_path(path_text: str):
    age_m = AGE_RE.search(path_text)
    if not age_m:
        return None

    age = int(age_m.group("age"))
    if age < 1 or age > 120:
        return None

    hm = HEIGHT_RE.search(path_text)
    wm = WEIGHT_RE.search(path_text)
    sm = SEX_RE.search(path_text)
    scen = SCENARIO_RE.search(path_text)
    task = TASK_RE.search(path_text)
    prof = PROFILE_RE.search(path_text)
    fall = FALL_RE.search(path_text)

    return {
        "age_years": age,
        "height_m": decimal_from_match(hm),
        "weight_kg": decimal_from_match(wm),
        "sex": sm.group("sex").lower() if sm else "",
        "scenario_id": int(scen.group("n")) if scen else np.nan,
        "task_id": int(task.group("n")) if task else np.nan,
        "profile_id": f"P{prof.group('n')}" if prof else "",
        "fall_descriptor": fall.group(1).lower() if fall else "",
    }


def age_band(age):
    age = float(age)
    if age < 30:
        return "<30"
    if age < 50:
        return "30-49"
    if age < 65:
        return "50-64"
    return "65+"


def source_status(path_text: str):
    s = path_text.lower()

    if any(tok in s for tok in LEGACY_OR_SUPERSEDED_TOKENS):
        return "LEGACY_OR_SUPERSEDED"

    if any(tok in s for tok in CANONICAL_PATH_TOKENS):
        return "CANONICAL_QUANTITATIVE"

    # Scenario snapshots and individual profile visualizations are useful
    # illustrations but are not automatically quantitative validation evidence.
    if "skeleton_3d_snapshots" in s or re.search(r"scenario\d+_age\d+", s):
        return "ILLUSTRATIVE_PROFILE"

    if "publication_results_v1" in s:
        return "PAPER_CANDIDATE_NEEDS_PROVENANCE_CHECK"

    if "testing/outcomes_real_vs_sim" in s:
        return "PAPER_CANDIDATE_NEEDS_PROVENANCE_CHECK"

    return "NONCANONICAL_UNVERIFIED"


def build_profile_catalog(catalog: pd.DataFrame):
    rows = []

    for _, r in catalog.iterrows():
        src = str(r["source_path"])
        parsed = parse_profile_from_path(src)
        if parsed is None:
            continue

        status = source_status(src)

        row = {
            "source_path": src,
            "archive_path": r.get("archive_path", ""),
            "category": r.get("category", ""),
            "extension": r.get("extension", ""),
            "size_bytes": r.get("size_bytes", np.nan),
            "sha256": r.get("sha256", ""),
            "evidence_status": status,
            **parsed,
        }
        row["age_band"] = age_band(parsed["age_years"])
        rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    # Multiple files can belong to the same simulated profile/scenario.
    # Retain each file in the master catalog but build a unique-profile key.
    df["profile_key"] = (
        df["age_years"].astype(str)
        + "|"
        + df["sex"].astype(str)
        + "|"
        + df["height_m"].astype(str)
        + "|"
        + df["weight_kg"].astype(str)
        + "|"
        + df["scenario_id"].astype(str)
        + "|"
        + df["task_id"].astype(str)
        + "|"
        + df["profile_id"].astype(str)
    )

    return df.sort_values(
        ["age_years", "sex", "scenario_id", "source_path"],
        na_position="last",
    )


def representative_profiles(profile_files: pd.DataFrame):
    if profile_files.empty:
        return pd.DataFrame()

    # One row per simulated profile/scenario, preferring skeleton snapshot PNGs.
    unique_rows = []

    for key, g in profile_files.groupby("profile_key", sort=False):
        g = g.copy()
        g["_snapshot"] = g["source_path"].astype(str).str.contains(
            "skeleton_3d_snapshots", case=False, na=False
        ).astype(int)
        g["_png"] = (g["extension"].astype(str).str.lower() == ".png").astype(int)

        preferred = g.sort_values(
            ["_snapshot", "_png", "size_bytes"],
            ascending=[False, False, False],
        ).iloc[0]

        unique_rows.append(preferred.drop(labels=["_snapshot", "_png"]))

    profiles = pd.DataFrame(unique_rows)

    # Select up to 8 candidates per age band with diversity in sex and scenario/task.
    selected = []

    for band, g in profiles.groupby("age_band", sort=False):
        g = g.sort_values(
            ["age_years", "sex", "scenario_id", "task_id"],
            na_position="last",
        ).copy()

        # Candidate pool includes illustrative profile figures and canonical profiles.
        g = g[
            g["evidence_status"].isin(
                [
                    "ILLUSTRATIVE_PROFILE",
                    "CANONICAL_QUANTITATIVE",
                    "PAPER_CANDIDATE_NEEDS_PROVENANCE_CHECK",
                ]
            )
        ]

        if g.empty:
            continue

        # Greedy diversity: unique sex/scenario/fall combinations first.
        seen = set()
        keep_idx = []

        for idx, r in g.iterrows():
            key = (
                r.get("sex", ""),
                r.get("scenario_id", np.nan),
                r.get("fall_descriptor", ""),
            )
            if key in seen:
                continue
            seen.add(key)
            keep_idx.append(idx)
            if len(keep_idx) >= 8:
                break

        # Fill if needed.
        if len(keep_idx) < min(8, len(g)):
            for idx in g.index:
                if idx not in keep_idx:
                    keep_idx.append(idx)
                if len(keep_idx) >= min(8, len(g)):
                    break

        selected.append(g.loc[keep_idx])

    if not selected:
        return pd.DataFrame()

    out = pd.concat(selected, ignore_index=True)
    return out.sort_values(
        ["age_band", "age_years", "sex", "scenario_id"],
        na_position="last",
    )


def scan_physical_demographics():
    # Scan flexible text-like metadata formats. Do not infer ages from IDs.
    roots = [
        PHYSICAL_ROOT,
        PROTECHTO / "data",
    ]

    rows = []
    inspected = []

    subject_keys = (
        "subject", "subject_id", "subjectid", "participant",
        "participant_id", "participantid", "person", "person_id",
    )

    def find_col(cols, predicate):
        for c in cols:
            n = re.sub(r"[^a-z0-9]+", "_", str(c).lower()).strip("_")
            if predicate(n):
                return c
        return None

    for root in roots:
        if not root.exists():
            continue

        # CSV
        for p in root.rglob("*.csv"):
            try:
                if p.stat().st_size > 100 * 1024 * 1024:
                    continue
                df = pd.read_csv(p, nrows=100000, low_memory=False)
            except Exception:
                continue

            ac = find_col(
                df.columns,
                lambda n: n == "age" or n in {"age_years", "age_year", "years_old"} or n.startswith("age_"),
            )
            sc = find_col(
                df.columns,
                lambda n: n in subject_keys,
            )

            if ac is None or sc is None:
                continue

            ages = pd.to_numeric(df[ac], errors="coerce")
            valid = ages.between(1, 120)

            for idx in df.index[valid]:
                rows.append({
                    "source_file": str(p),
                    "subject_id": str(df.at[idx, sc]),
                    "age_years": float(ages.loc[idx]),
                    "age_band": age_band(float(ages.loc[idx])),
                    "format": "csv",
                })

            inspected.append({
                "source_file": str(p),
                "format": "csv",
                "subject_column": sc,
                "age_column": ac,
                "valid_rows": int(valid.sum()),
            })

        # JSON
        for p in root.rglob("*.json"):
            try:
                if p.stat().st_size > 20 * 1024 * 1024:
                    continue
                data = json.loads(p.read_text(encoding="utf-8", errors="replace"))
            except Exception:
                continue

            objects = []
            if isinstance(data, dict):
                objects = [data]
            elif isinstance(data, list):
                objects = [x for x in data[:100000] if isinstance(x, dict)]

            for obj in objects:
                norm = {
                    re.sub(r"[^a-z0-9]+", "_", str(k).lower()).strip("_"): (k, v)
                    for k, v in obj.items()
                }
                age_item = None
                sub_item = None
                for k, kv in norm.items():
                    if k == "age" or k in {"age_years", "age_year", "years_old"} or k.startswith("age_"):
                        age_item = kv
                    if k in subject_keys:
                        sub_item = kv
                if age_item and sub_item:
                    try:
                        a = float(age_item[1])
                    except Exception:
                        continue
                    if 1 <= a <= 120:
                        rows.append({
                            "source_file": str(p),
                            "subject_id": str(sub_item[1]),
                            "age_years": a,
                            "age_band": age_band(a),
                            "format": "json",
                        })

    raw = pd.DataFrame(rows)
    inspected_df = pd.DataFrame(inspected)

    if not raw.empty:
        raw = raw.drop_duplicates(
            subset=["source_file", "subject_id", "age_years"]
        ).sort_values(["age_years", "subject_id"])

    return raw, inspected_df


def create_age_figures(profile_catalog: pd.DataFrame, outdir: Path):
    if profile_catalog.empty:
        return

    profiles = profile_catalog.drop_duplicates("profile_key").copy()

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.hist(profiles["age_years"].astype(float), bins=12)
    ax.set_xlabel("Simulated profile age (years)")
    ax.set_ylabel("Number of unique simulated profiles/scenarios")
    ax.set_title("Age coverage of discovered simulated profiles")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    for ext in ("png", "pdf", "svg"):
        fig.savefig(
            (outdir / "fig_simulated_profile_age_distribution").with_suffix("." + ext),
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)

    coverage = (
        profiles.groupby(["age_band", "sex"], dropna=False)
        .size()
        .reset_index(name="unique_profiles")
    )
    coverage.to_csv(
        outdir / "plot_data_simulated_profile_age_sex_coverage.csv",
        index=False,
    )

    pivot = coverage.pivot(
        index="age_band",
        columns="sex",
        values="unique_profiles",
    ).fillna(0)

    if not pivot.empty:
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        pivot.plot(kind="bar", ax=ax)
        ax.set_xlabel("Age band")
        ax.set_ylabel("Unique simulated profiles/scenarios")
        ax.set_title("Simulated demographic-profile coverage")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        for ext in ("png", "pdf", "svg"):
            fig.savefig(
                (outdir / "fig_simulated_profile_age_sex_coverage").with_suffix("." + ext),
                dpi=300,
                bbox_inches="tight",
            )
        plt.close(fig)


def canonical_figure_shortlist(catalog: pd.DataFrame, profile_selected: pd.DataFrame):
    rows = []

    if not catalog.empty:
        for _, r in catalog.iterrows():
            src = str(r["source_path"])
            status = source_status(src)
            if r.get("extension", "") not in FIG_EXT:
                continue

            # Strict quantitative shortlist:
            # only canonical corrected/publication-gate artifacts.
            if status == "CANONICAL_QUANTITATIVE":
                rows.append({
                    "usage_tier": "CANONICAL_QUANTITATIVE",
                    "source_path": src,
                    "archive_path": r.get("archive_path", ""),
                    "category": r.get("category", ""),
                    "extension": r.get("extension", ""),
                    "age_years": np.nan,
                    "sex": "",
                    "scenario_id": np.nan,
                    "task_id": np.nan,
                    "note": "Suitable for quantitative validation claims after normal figure-quality review.",
                })

    if not profile_selected.empty:
        for _, r in profile_selected.iterrows():
            rows.append({
                "usage_tier": "ILLUSTRATIVE_SIMULATED_PROFILE",
                "source_path": r["source_path"],
                "archive_path": r["archive_path"],
                "category": r["category"],
                "extension": r["extension"],
                "age_years": r["age_years"],
                "sex": r["sex"],
                "scenario_id": r["scenario_id"],
                "task_id": r["task_id"],
                "note": (
                    "Use only as an illustrative simulator-profile example unless the same "
                    "case is separately supported by canonical quantitative validation."
                ),
            })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.drop_duplicates(
            subset=["usage_tier", "source_path"]
        ).sort_values(
            ["usage_tier", "category", "age_years", "source_path"],
            na_position="last",
        )
    return out


def copy_selected_profile_figures(selected: pd.DataFrame, stage: Path, outdir: Path):
    if selected.empty:
        return 0

    copied = 0
    for _, r in selected.iterrows():
        src = Path(str(r["source_path"]))
        if not src.exists() or src.suffix.lower() not in FIG_EXT:
            continue

        band = str(r["age_band"]).replace("+", "plus").replace("<", "under")
        age = int(r["age_years"])
        sex = r["sex"] if r["sex"] else "unknownsex"
        scenario = (
            f"scenario{int(r['scenario_id'])}"
            if pd.notna(r["scenario_id"])
            else "scenarioNA"
        )

        dst = (
            outdir
            / band
            / f"age{age}_{sex}_{scenario}_{src.name}"
        )
        copy_file(src, dst)
        copied += 1

    return copied


def update_manifest(stage: Path):
    d00 = stage / "00_README_AND_MANIFESTS"
    rows = []

    for p in sorted(stage.rglob("*")):
        if not p.is_file():
            continue
        if p.name == "FILE_MANIFEST_SHA256.csv":
            continue

        rows.append({
            "path": str(p.relative_to(stage)),
            "size_bytes": p.stat().st_size,
            "sha256": sha256_file(p),
        })

    pd.DataFrame(rows).to_csv(
        d00 / "FILE_MANIFEST_SHA256.csv",
        index=False,
    )


def rezip(stage: Path, zip_path: Path):
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(
        zip_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True,
    ) as zf:
        for p in sorted(stage.rglob("*")):
            if p.is_file():
                arc = Path(stage.name) / p.relative_to(stage)
                zf.write(p, arcname=str(arc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default=str(DEFAULT_STAGE))
    ap.add_argument("--zip", dest="zip_path", default=str(DEFAULT_ZIP))
    args = ap.parse_args()

    stage = Path(args.stage)
    zip_path = Path(args.zip_path)

    catalog_path = stage / CATALOG_REL

    if not stage.exists():
        raise SystemExit(f"Evidence stage missing: {stage}")

    if not catalog_path.exists():
        raise SystemExit(f"Validation catalog missing: {catalog_path}")

    print("=" * 122)
    print("PAPER EVIDENCE V3 — CANONICALIZATION + SIMULATED AGE/PROFILE DIVERSITY")
    print("=" * 122)

    catalog = pd.read_csv(catalog_path)

    v3 = stage / "13_CANONICAL_PAPER_EVIDENCE_V3"
    if v3.exists():
        shutil.rmtree(v3)
    v3.mkdir(parents=True)

    # ------------------------------------------------------------------
    # 1. Explicit status for every validation item
    # ------------------------------------------------------------------
    status_df = catalog.copy()
    status_df["evidence_status"] = status_df["source_path"].astype(str).map(
        source_status
    )
    status_df.to_csv(
        v3 / "ALL_VALIDATION_ITEMS_WITH_STATUS.csv",
        index=False,
    )

    status_counts = (
        status_df.groupby(
            ["evidence_status", "category", "extension"],
            dropna=False,
        )
        .size()
        .reset_index(name="count")
        .sort_values(
            ["evidence_status", "category", "extension"]
        )
    )
    status_counts.to_csv(
        v3 / "VALIDATION_STATUS_COUNTS.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # 2. Parse simulated demographics directly from profile paths
    # ------------------------------------------------------------------
    profile_files = build_profile_catalog(catalog)
    profile_files.to_csv(
        v3 / "SIMULATED_PROFILE_FILES_WITH_DEMOGRAPHICS.csv",
        index=False,
    )

    if profile_files.empty:
        profiles = pd.DataFrame()
    else:
        profiles = (
            profile_files
            .sort_values("source_path")
            .drop_duplicates("profile_key")
        )

    profiles.to_csv(
        v3 / "SIMULATED_UNIQUE_PROFILE_DEMOGRAPHICS.csv",
        index=False,
    )

    if not profiles.empty:
        diversity = (
            profiles.groupby(
                ["age_band", "sex"],
                dropna=False,
            )
            .agg(
                unique_profiles=("profile_key", "nunique"),
                min_age=("age_years", "min"),
                median_age=("age_years", "median"),
                max_age=("age_years", "max"),
                unique_scenarios=("scenario_id", "nunique"),
                unique_tasks=("task_id", "nunique"),
            )
            .reset_index()
        )
    else:
        diversity = pd.DataFrame()

    diversity.to_csv(
        v3 / "SIMULATED_PROFILE_DIVERSITY_SUMMARY.csv",
        index=False,
    )

    selected = representative_profiles(profile_files)
    selected.to_csv(
        v3 / "AGE_STRATIFIED_SIMULATED_PROFILE_CANDIDATES.csv",
        index=False,
    )

    copied = copy_selected_profile_figures(
        selected,
        stage,
        v3 / "selected_profile_figures",
    )

    # ------------------------------------------------------------------
    # 3. Physical age metadata scan, kept distinct from simulation
    # ------------------------------------------------------------------
    physical_demo, physical_sources = scan_physical_demographics()

    physical_demo.to_csv(
        v3 / "PHYSICAL_SUBJECT_AGE_METADATA_DISCOVERED.csv",
        index=False,
    )

    physical_sources.to_csv(
        v3 / "PHYSICAL_DEMOGRAPHIC_SOURCE_SCAN.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # 4. Strict canonical figure shortlist
    # ------------------------------------------------------------------
    strict = canonical_figure_shortlist(
        catalog,
        selected,
    )

    strict.to_csv(
        v3 / "PAPER_FIGURE_CANDIDATES_CANONICALIZED.csv",
        index=False,
    )

    # Legacy/superseded results get an explicit red-flag table.
    legacy = status_df[
        status_df["evidence_status"] == "LEGACY_OR_SUPERSEDED"
    ].copy()

    legacy.to_csv(
        v3 / "LEGACY_OR_SUPERSEDED_DO_NOT_USE_PRIMARY.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # 5. Age/diversity charts
    # ------------------------------------------------------------------
    create_age_figures(
        profile_files,
        v3,
    )

    # ------------------------------------------------------------------
    # 6. Guidance
    # ------------------------------------------------------------------
    guide = f"""# Canonical Paper Evidence V3

This directory resolves two important issues in the broad V2 evidence gallery.

## 1. Simulated demographic profiles

The previous demographic scanner found zero records because the simulator demographics
are encoded primarily in run/scenario paths such as:

`scenario42_age65_h1p75_sex_female_w78p0_...`

V3 parses those explicit path fields without inventing values.

Discovered simulated profile files: {len(profile_files)}
Unique simulated profile/scenario keys: {len(profiles)}
Age-stratified representative candidates: {len(selected)}
Representative figure files copied: {copied}

These are **simulated demographic profiles**, not claims about physical participants.

## 2. Physical participant ages

`PHYSICAL_SUBJECT_AGE_METADATA_DISCOVERED.csv` contains physical participant ages only
if an actual physical metadata CSV/JSON explicitly provided both a subject identifier
and an age. No age is inferred from physical subject IDs.

Physical subject-age records found automatically: {len(physical_demo)}

## 3. Quantitative validation vs illustrative visualization

Evidence status is explicit:

- `CANONICAL_QUANTITATIVE`
  Paths tied to the frozen publication-gate audit or corrected396 RNE-corrected campaign.
  These are the first-choice sources for quantitative validation claims.

- `ILLUSTRATIVE_PROFILE`
  Simulator profile/skeleton examples. These can illustrate parameterization, demographic
  variation, and fall mechanics, but are not automatically quantitative validation evidence.

- `PAPER_CANDIDATE_NEEDS_PROVENANCE_CHECK`
  Potentially useful historical paper/testing figures. Inspect provenance before use.

- `LEGACY_OR_SUPERSEDED`
  Older/superseded campaign material. Do not use in primary manuscript results.

- `NONCANONICAL_UNVERIFIED`
  Retained for completeness but not approved automatically for paper claims.

## 4. Why this matters

The broad V2 archive intentionally collected everything. V3 prevents paper writing from
accidentally selecting an attractive plot from an obsolete campaign.

Use:
`PAPER_FIGURE_CANDIDATES_CANONICALIZED.csv`

as the main figure-selection index.
"""

    (v3 / "README.md").write_text(
        guide,
        encoding="utf-8",
    )

    # Simple manifest specific to V3.
    rows = []
    for p in sorted(v3.rglob("*")):
        if p.is_file():
            rows.append({
                "path": str(p.relative_to(stage)),
                "size_bytes": p.stat().st_size,
                "sha256": sha256_file(p),
            })

    pd.DataFrame(rows).to_csv(
        v3 / "V3_FILE_MANIFEST_SHA256.csv",
        index=False,
    )

    # Overall archive manifest + ZIP
    update_manifest(stage)

    completion = {
        "validation_catalog_rows": int(len(catalog)),
        "simulated_profile_file_rows": int(len(profile_files)),
        "unique_simulated_profiles": int(len(profiles)),
        "age_stratified_simulated_profile_candidates": int(len(selected)),
        "physical_subject_age_records": int(len(physical_demo)),
        "selected_profile_figures_copied": int(copied),
        "legacy_or_superseded_items_flagged": int(len(legacy)),
    }

    (v3 / "V3_COMPLETION.json").write_text(
        json.dumps(completion, indent=2),
        encoding="utf-8",
    )

    update_manifest(stage)
    rezip(stage, zip_path)

    print()
    print("=" * 122)
    print("PAPER EVIDENCE V3 GATE: PASS")
    print("=" * 122)
    print("Validation catalog rows               :", len(catalog))
    print("Simulated profile files parsed        :", len(profile_files))
    print("Unique simulated profiles/scenarios   :", len(profiles))
    print("Age-stratified simulated candidates   :", len(selected))
    print("Selected profile figures copied       :", copied)
    print("Physical subject-age records found    :", len(physical_demo))
    print("Legacy/superseded items flagged       :", len(legacy))
    print("Final ZIP                             :", zip_path)
    print("Final ZIP bytes                       :", zip_path.stat().st_size)
    print("Final ZIP SHA256                      :", sha256_file(zip_path))
    print("NO TRAINING OR INFERENCE WAS RUN.")


if __name__ == "__main__":
    main()
