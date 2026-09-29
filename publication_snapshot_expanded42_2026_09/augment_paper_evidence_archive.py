#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil
import zipfile

import pandas as pd


PROJECT = Path("/mnt/hdd16T/ToqeerHomeBackup/mujoco_project")
PROTECHTO = Path("/mnt/hdd16T/ToqeerHomeBackup/toqeer/Protechto-master")
PHYSICAL_ROOT = Path("/mnt/hdd16T/ToqeerHomeBackup/toqeer/uniVr-dataset")

DEFAULT_STAGE = PROJECT / "outputs/paper_evidence_archive_v1"
DEFAULT_ZIP = PROJECT / "outputs/paper_evidence_archive_v1_FULL.zip"

FIG_EXT = {".png", ".pdf", ".svg", ".jpg", ".jpeg"}
TAB_EXT = {".csv", ".json", ".txt", ".md", ".tex", ".yaml", ".yml"}
MAX_COPY_BYTES = 100 * 1024 * 1024

VALIDATION_WORDS = re.compile(
    r"(valid|audit|figure|fig_|plot|chart|compare|comparison|metric|summary|report|"
    r"impact|event|imu|accel|acc_|gyro|gyr_|orientation|euler|profile|scenario|"
    r"distribution|hist|corr|error|agreement|waveform|timeseries|truth|rne|"
    r"confusion|classification|precision|recall|f1|paper|publication)",
    re.I,
)

SUBJECT_COLS = {
    "subject", "subject_id", "subjectid", "participant", "participant_id",
    "participantid", "person", "person_id", "id", "user", "user_id",
}
SEX_COLS = {"sex", "gender"}
HEIGHT_COLS = {"height", "height_cm", "height_m", "stature"}
WEIGHT_COLS = {"weight", "weight_kg", "mass", "mass_kg"}


def sha256_file(path: Path, chunk=4 * 1024 * 1024):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def safe_rel(path: Path, root: Path):
    try:
        return path.relative_to(root)
    except Exception:
        return Path(path.name)


def copy_file(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def classify_validation(path: Path):
    s = str(path).lower()
    if "confusion" in s:
        return "confusion_matrix"
    if "classification" in s or "precision" in s or "recall" in s or "f1" in s:
        return "classification_metrics"
    if "gyro" in s or "gyr" in s:
        return "gyroscope_validation"
    if "accel" in s or "/acc" in s or "_acc" in s:
        return "acceleration_validation"
    if "orientation" in s or "euler" in s:
        return "orientation_validation"
    if "impact" in s or "event" in s:
        return "event_impact_validation"
    if "profile" in s or "scenario" in s:
        return "profile_scenario_validation"
    if "hist" in s or "distribution" in s:
        return "distribution"
    if "corr" in s or "agreement" in s or "compare" in s:
        return "agreement_comparison"
    if path.suffix.lower() in FIG_EXT:
        return "figure_other"
    return "validation_table_other"


def discover_validation(stage: Path):
    gallery = stage / "09_SIMULATOR_VALIDATION_GALLERY"
    gallery.mkdir(parents=True, exist_ok=True)

    roots = [
        ("publication_gate", PROJECT / "outputs/phase2_publication_gate_audit_20260826"),
        ("corrected396", PROJECT / "outputs/_highrate_overnight/campaign_highrate_truth_v3_rne_corrected396"),
        ("project_outputs", PROJECT / "outputs"),
    ]

    rows = []
    seen_hash = set()

    for group, root in roots:
        if not root.exists():
            continue

        for p in root.rglob("*"):
            if not p.is_file():
                continue
            if stage in p.parents:
                continue
            if p.name.startswith("paper_evidence_archive_v1"):
                continue
            if p.name.startswith("full_campaign_") and p.suffix == ".log":
                continue
            if p.stat().st_size > MAX_COPY_BYTES:
                continue

            ext = p.suffix.lower()
            is_fig = ext in FIG_EXT
            is_table = ext in TAB_EXT and bool(VALIDATION_WORDS.search(p.name))
            if not (is_fig or is_table):
                continue

            try:
                h = sha256_file(p)
            except Exception:
                continue

            key = (h, p.stat().st_size)
            if key in seen_hash:
                continue
            seen_hash.add(key)

            rel = safe_rel(p, root)
            dest = gallery / "files" / group / rel
            copy_file(p, dest)

            rows.append({
                "source_group": group,
                "source_path": str(p),
                "archive_path": str(dest.relative_to(stage)),
                "category": classify_validation(p),
                "extension": ext,
                "size_bytes": p.stat().st_size,
                "sha256": h,
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["category", "source_group", "source_path"])
    df.to_csv(gallery / "VALIDATION_FIGURE_AND_DATA_CATALOG.csv", index=False)

    if not df.empty:
        counts = (
            df.groupby(["category", "extension"])
            .size()
            .reset_index(name="count")
            .sort_values(["category", "extension"])
        )
    else:
        counts = pd.DataFrame(columns=["category", "extension", "count"])
    counts.to_csv(gallery / "VALIDATION_CATALOG_COUNTS.csv", index=False)

    readme = f"""# Simulator Validation Gallery

Copied validation/figure/table files: {len(df)}

Sources searched:
- phase2 publication-gate audit
- corrected396 high-rate corrected campaign
- project outputs generally

Use `VALIDATION_FIGURE_AND_DATA_CATALOG.csv` to filter candidates by validation category.
The original source path and SHA256 are retained for every copied file.

This discovery process does not rank files by favorable outcomes.
"""
    (gallery / "README.md").write_text(readme, encoding="utf-8")
    return df


def normalize_name(c):
    return re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_")


def find_first_col(cols, candidates):
    norm = {normalize_name(c): c for c in cols}
    for cand in candidates:
        if cand in norm:
            return norm[cand]
    return None


def age_col(cols):
    for c in cols:
        n = normalize_name(c)
        if n == "age" or n in {"age_year", "age_years", "years_old"} or n.startswith("age_"):
            return c
    return None


def scan_demographics(stage: Path):
    ddir = stage / "10_SUBJECT_AND_PROFILE_DIVERSITY"
    ddir.mkdir(parents=True, exist_ok=True)

    roots = [
        ("physical_univr_kfall", PHYSICAL_ROOT),
        ("protechto_data", PROTECHTO / "data"),
        ("synthetic_dataset", PROJECT / "outputs/phase2_corrected_event_dataset_v1"),
        ("synthetic_cache", PROJECT / "outputs/protechto_exact_synthetic_cache_v1"),
    ]

    rows = []
    inspected = []

    for group, root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*.csv"):
            try:
                if p.stat().st_size > 50 * 1024 * 1024:
                    continue
                df = pd.read_csv(p, nrows=20000, low_memory=False)
            except Exception:
                continue

            ac = age_col(df.columns)
            if ac is None:
                continue

            sc = find_first_col(df.columns, SUBJECT_COLS)
            if sc is None:
                sc = find_first_col(
                    df.columns,
                    {"profile_id", "profile", "subject_profile", "participant_profile"},
                )
            if sc is None:
                continue

            sexc = find_first_col(df.columns, SEX_COLS)
            hc = find_first_col(df.columns, HEIGHT_COLS)
            wc = find_first_col(df.columns, WEIGHT_COLS)

            ages = pd.to_numeric(df[ac], errors="coerce")
            valid = ages.between(1, 120)

            for idx in df.index[valid]:
                rows.append({
                    "source_group": group,
                    "source_file": str(p),
                    "subject_or_profile_id": str(df.at[idx, sc]),
                    "age_years": float(ages.loc[idx]),
                    "sex_gender": str(df.at[idx, sexc]) if sexc else "",
                    "height": str(df.at[idx, hc]) if hc else "",
                    "weight": str(df.at[idx, wc]) if wc else "",
                })

            inspected.append({
                "source_group": group,
                "source_file": str(p),
                "subject_column": sc,
                "age_column": ac,
                "sex_column": sexc or "",
                "height_column": hc or "",
                "weight_column": wc or "",
                "valid_age_rows": int(valid.sum()),
            })

    pd.DataFrame(inspected).to_csv(ddir / "DEMOGRAPHIC_SOURCE_CATALOG.csv", index=False)

    raw = pd.DataFrame(rows)
    if raw.empty:
        raw.to_csv(ddir / "SUBJECT_DEMOGRAPHICS_DISCOVERED.csv", index=False)
        pd.DataFrame().to_csv(ddir / "AGE_STRATIFIED_SUBJECT_PROFILE_CANDIDATES.csv", index=False)
        (ddir / "README.md").write_text(
            "No CSV containing both an identifiable subject/profile field and an explicit numeric age field was discovered automatically.\n",
            encoding="utf-8",
        )
        return raw, pd.DataFrame()

    raw = raw.drop_duplicates(
        subset=["source_group", "source_file", "subject_or_profile_id", "age_years"]
    ).sort_values(["source_group", "age_years", "subject_or_profile_id"])
    raw.to_csv(ddir / "SUBJECT_DEMOGRAPHICS_DISCOVERED.csv", index=False)

    def band(a):
        if a < 30:
            return "<30"
        if a < 50:
            return "30-49"
        if a < 65:
            return "50-64"
        return "65+"

    raw["age_band"] = raw["age_years"].map(band)

    rep_parts = []
    for (_, _), g in raw.groupby(["source_group", "age_band"], sort=False):
        g = g.sort_values("age_years").drop_duplicates("subject_or_profile_id")
        if len(g) <= 3:
            chosen = g
        else:
            inds = sorted(set([0, len(g)//2, len(g)-1]))
            chosen = g.iloc[inds]
        rep_parts.append(chosen)

    reps = pd.concat(rep_parts, ignore_index=True) if rep_parts else pd.DataFrame()
    reps.to_csv(ddir / "AGE_STRATIFIED_SUBJECT_PROFILE_CANDIDATES.csv", index=False)

    summary = (
        raw.groupby(["source_group", "age_band"])
        .agg(
            unique_subjects_profiles=("subject_or_profile_id", "nunique"),
            min_age=("age_years", "min"),
            median_age=("age_years", "median"),
            max_age=("age_years", "max"),
        )
        .reset_index()
    )
    summary.to_csv(ddir / "AGE_DIVERSITY_SUMMARY.csv", index=False)

    gallery_cat = stage / "09_SIMULATOR_VALIDATION_GALLERY/VALIDATION_FIGURE_AND_DATA_CATALOG.csv"
    matches = []
    if gallery_cat.exists() and not reps.empty:
        cat = pd.read_csv(gallery_cat)
        for _, rr in reps.iterrows():
            sid = str(rr["subject_or_profile_id"])
            if len(sid) < 3:
                continue
            mask = cat["source_path"].astype(str).str.contains(
                re.escape(sid), regex=True, na=False
            )
            for _, cr in cat[mask].iterrows():
                matches.append({
                    "source_group": rr["source_group"],
                    "subject_or_profile_id": sid,
                    "age_years": rr["age_years"],
                    "age_band": rr["age_band"],
                    "validation_category": cr["category"],
                    "validation_archive_path": cr["archive_path"],
                    "validation_source_path": cr["source_path"],
                })

    pd.DataFrame(matches).to_csv(
        ddir / "REPRESENTATIVE_AGE_CASE_VALIDATION_MATCHES.csv", index=False
    )

    readme = """# Subject / Profile Diversity Evidence

These tables are discovery aids for choosing representative paper examples.

No age is inferred. A subject/profile is included only when a source CSV explicitly
contains both an age field and an identifiable subject/profile field.

Files:
- SUBJECT_DEMOGRAPHICS_DISCOVERED.csv
- AGE_DIVERSITY_SUMMARY.csv
- AGE_STRATIFIED_SUBJECT_PROFILE_CANDIDATES.csv
- REPRESENTATIVE_AGE_CASE_VALIDATION_MATCHES.csv
"""
    (ddir / "README.md").write_text(readme, encoding="utf-8")
    return raw, reps


def create_figure_shortlist(stage: Path, catalog: pd.DataFrame):
    d = stage / "11_PAPER_FIGURE_SHORTLIST"
    d.mkdir(parents=True, exist_ok=True)

    if catalog.empty:
        pd.DataFrame().to_csv(d / "FIGURE_SHORTLIST.csv", index=False)
        return

    fig = catalog[catalog["extension"].isin(FIG_EXT)].copy()
    if fig.empty:
        fig.to_csv(d / "FIGURE_SHORTLIST.csv", index=False)
        return

    preference = {".svg": 4, ".pdf": 3, ".png": 2, ".jpg": 1, ".jpeg": 1}
    category_score = {
        "acceleration_validation": 5,
        "gyroscope_validation": 5,
        "event_impact_validation": 5,
        "agreement_comparison": 5,
        "profile_scenario_validation": 4,
        "distribution": 3,
        "confusion_matrix": 3,
        "classification_metrics": 2,
        "figure_other": 1,
    }
    fig["rank_score"] = (
        fig["extension"].map(preference).fillna(0)
        + fig["category"].map(category_score).fillna(0)
    )

    parts = []
    for category, g in fig.groupby("category"):
        g = g.sort_values(["rank_score", "size_bytes"], ascending=[False, False])
        parts.append(g.head(12))

    short = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if not short.empty:
        short = short.sort_values(["rank_score", "category"], ascending=[False, True])
    short.to_csv(d / "FIGURE_SHORTLIST.csv", index=False)

    (d / "README.md").write_text(
        """# Paper Figure Shortlist

This shortlist is ranked only by validation category relevance and publication-friendly
file format. It is NOT ranked by favorable metric values.

The complete unfiltered candidate set remains in 09_SIMULATOR_VALIDATION_GALLERY.
""",
        encoding="utf-8",
    )


def create_evidence_map(stage: Path):
    d = stage / "12_CLAIM_TO_EVIDENCE_MAP"
    d.mkdir(parents=True, exist_ok=True)

    rows = [
        ["Exact lightweight Protechto pipeline",
         "01_METHOD_PROTECHTO/canonical_source",
         "07_ENVIRONMENT_AND_REPRODUCIBILITY/SOURCE_HASHES.csv"],
        ["Untouched physical pipeline / no manual 150-ms trim",
         "06_VALIDATION_AND_AUDITS/protechto_clean_lightweight_pipeline_audit.txt",
         "01_METHOD_PROTECHTO/canonical_source/preprocessing/windowing.py"],
        ["35-fold campaign completion",
         "06_VALIDATION_AND_AUDITS/completion_markers",
         "06_VALIDATION_AND_AUDITS/campaign_state/FINAL_CAMPAIGN_COMPLETE"],
        ["Segment-level classification results",
         "04_PAPER_TABLES/SEGMENT_LEVEL_PRIMARY_POOLED.csv",
         "03_CAMPAIGN_RESULTS/conditions"],
        ["Physical event-level results",
         "04_PAPER_TABLES/EVENT_LEVEL_PHYSICAL_PRIMARY_POOLED.csv",
         "04_PAPER_TABLES/EVENT_CLASSIFICATION_REPORT_PHYSICAL_POOLED.csv"],
        ["Synthetic-only event sensitivity",
         "04_PAPER_TABLES/EVENT_LEVEL_SIM_ONLY_DIAGNOSTIC.csv",
         "05_FIGURES_AND_PLOT_DATA/fig_sim_only_event_detection.pdf"],
        ["Corrected simulator validation and event policy",
         "02_METHOD_SIMULATOR/synthetic_event_policy",
         "09_SIMULATOR_VALIDATION_GALLERY"],
        ["Age/subject diversity candidate selection",
         "10_SUBJECT_AND_PROFILE_DIVERSITY/AGE_STRATIFIED_SUBJECT_PROFILE_CANDIDATES.csv",
         "10_SUBJECT_AND_PROFILE_DIVERSITY/REPRESENTATIVE_AGE_CASE_VALIDATION_MATCHES.csv"],
    ]
    pd.DataFrame(
        rows,
        columns=["paper_claim_or_section", "primary_evidence", "supporting_evidence"],
    ).to_csv(d / "CLAIM_TO_EVIDENCE_MAP.csv", index=False)


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
    pd.DataFrame(rows).to_csv(d00 / "FILE_MANIFEST_SHA256.csv", index=False)


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

    if not stage.exists():
        raise SystemExit(f"Base evidence staging folder does not exist: {stage}")

    print("=" * 118)
    print("AUGMENT PAPER EVIDENCE WITH SIMULATOR VALIDATION + SUBJECT/AGE DIVERSITY")
    print("=" * 118)

    catalog = discover_validation(stage)
    print("Validation files catalogued:", len(catalog))

    demog, reps = scan_demographics(stage)
    print("Demographic records discovered:", len(demog))
    print("Age-stratified candidates:", len(reps))

    create_figure_shortlist(stage, catalog)
    create_evidence_map(stage)

    raw_note = """# Raw corrected simulator data policy

The paper archive emphasizes validation outputs, figures, manifests, summary tables, exact
source/configuration, and reproducibility evidence. It does not blindly duplicate every
high-rate corrected raw trial because that can be extremely large.

Canonical corrected simulator root:
`/mnt/hdd16T/ToqeerHomeBackup/mujoco_project/outputs/_highrate_overnight/campaign_highrate_truth_v3_rne_corrected396`

The event manifest, validation outputs, figures, and source provenance are included.
"""
    (stage / "09_SIMULATOR_VALIDATION_GALLERY/RAW_DATA_POLICY.md").write_text(
        raw_note, encoding="utf-8"
    )

    complete = {
        "stage": str(stage),
        "zip": str(zip_path),
        "validation_catalog_items": int(len(catalog)),
        "demographic_records": int(len(demog)),
        "age_stratified_candidates": int(len(reps)),
    }
    (stage / "FULL_ARCHIVE_AUGMENTATION_COMPLETE.json").write_text(
        json.dumps(complete, indent=2), encoding="utf-8"
    )

    update_manifest(stage)
    rezip(stage, zip_path)
    h = sha256_file(zip_path)

    print()
    print("=" * 118)
    print("FULL PAPER EVIDENCE ARCHIVE AUGMENTATION GATE: PASS")
    print("=" * 118)
    print("ZIP:", zip_path)
    print("ZIP bytes:", zip_path.stat().st_size)
    print("ZIP SHA256:", h)
    print("Validation items:", len(catalog))
    print("Demographic records:", len(demog))
    print("Age-stratified candidates:", len(reps))
    print("NO TRAINING OR INFERENCE WAS RUN.")


if __name__ == "__main__":
    main()
