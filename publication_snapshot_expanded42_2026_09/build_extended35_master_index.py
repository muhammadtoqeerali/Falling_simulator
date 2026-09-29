import pandas as pd
from pathlib import Path


base = Path(
    "outputs/_highrate_overnight/"
    "campaign_highrate_truth_v2_extended35"
)


# ----------------------------
# Load source manifests
# ----------------------------

run = pd.read_csv(
    base / "campaign_run_manifest.csv"
)

qc = pd.read_csv(
    base / "highrate_truth_qc_manifest.csv"
)

profiles = pd.read_csv(
    base / "combined_35_profile_design.csv"
)

events = pd.read_csv(
    base / "campaign_event_manifest.csv"
)


print("Loaded:")
print("run:", run.shape)
print("qc:", qc.shape)
print("profiles:", profiles.shape)
print("events:", events.shape)


# ----------------------------
# Clean duplicate QC fields
# ----------------------------

remove_cols = [
    "truth_rows",
    "truth_rate_hz",
    "truth_qc_ok",
    "highrate_truth_csv"
]

run_clean = run.drop(
    columns=[
        c for c in remove_cols
        if c in run.columns
    ]
)


# ----------------------------
# Merge QC information
# ----------------------------

df = run_clean.merge(
    qc[
        [
            "profile_id",
            "scenario_id",
            "truth_rows",
            "truth_rate_hz",
            "truth_qc_ok",
            "highrate_truth_csv"
        ]
    ],
    on=[
        "profile_id",
        "scenario_id"
    ],
    how="left"
)


# ----------------------------
# Add profile parameters
# ----------------------------

profile_cols = [
    "profile_id",
    "age",
    "height",
    "sex",
    "weight",
    "bmi",
    "cohort_role",
    "age_band",
    "target_walk_speed_mps",
    "double_support_fraction",
    "muscle_strength_factor",
    "balance_impairment",
    "reaction_delay_s",
    "stand_stoop_deg",
    "walk_stoop_deg",
    "arm_gain",
    "proprioception_scale"
]

profile_cols = [
    c for c in profile_cols
    if c in profiles.columns
]


df = df.merge(
    profiles[profile_cols],
    on="profile_id",
    how="left"
)


# ----------------------------
# Add event information
# ----------------------------

event_cols = [
    "profile_id",
    "scenario_id",
    "event_available",
    "fall_duration_s",
    "lead_time_ms",
    "onset_time_s",
    "impact_time_s",
    "settle_time_s",
    "manifest_schema"
]

event_cols = [
    c for c in event_cols
    if c in events.columns
]


df = df.merge(
    events[event_cols],
    on=[
        "profile_id",
        "scenario_id"
    ],
    how="left"
)


# ----------------------------
# Final audit
# ----------------------------

print("\n==============================")
print("FINAL MASTER INDEX AUDIT")
print("==============================")

print("Shape:")
print(df.shape)

print("\nProfiles:")
print(df["profile_id"].nunique())

print("\nScenarios:")
print(df["scenario_id"].nunique())

print("\nQC:")
print(
    df["truth_qc_ok"]
    .value_counts(dropna=False)
)


print("\nMissing critical fields:")

critical = [
    "profile_id",
    "scenario_id",
    "highrate_truth_csv",
    "truth_rows",
    "truth_rate_hz"
]

for c in critical:
    print(
        c,
        df[c].isna().sum()
    )


# ----------------------------
# Save
# ----------------------------

out = (
    base /
    "Extended35_Master_Dataset_Index.csv"
)

df.to_csv(
    out,
    index=False
)


print("\nSaved:")
print(out)

