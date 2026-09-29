import pandas as pd
from pathlib import Path


base = Path(
    "outputs/_highrate_overnight/"
    "campaign_highrate_truth_v2_extended35"
)


src = (
    base /
    "Extended35_Master_Dataset_Index.csv"
)


df = pd.read_csv(src)


print("Input:")
print(df.shape)


# ----------------------------
# Resolve duplicate columns
# ----------------------------

def choose(primary, secondary):

    if primary in df.columns:
        return df[primary]

    if secondary in df.columns:
        return df[secondary]

    return pd.Series(
        [None] * len(df)
    )


clean = pd.DataFrame()


# Identity

clean["run_id"] = (
    df["profile_id"].astype(str)
    +
    "_scenario_"
    +
    df["scenario_id"].astype(str)
)


clean["profile_id"] = df["profile_id"]

clean["scenario_id"] = df["scenario_id"]


# Demographics

clean["age"] = choose(
    "age_x",
    "age_y"
)

clean["height_m"] = choose(
    "height_x",
    "height_y"
)

clean["sex"] = choose(
    "sex_x",
    "sex_y"
)

clean["weight_kg"] = choose(
    "weight_x",
    "weight_y"
)


# Scenario metadata

for c in [
    "description",
    "classification",
    "overall_score",
    "authenticity",
    "sisfall_compliant",
    "kfall_compliant"
]:

    if c in df.columns:
        clean[c] = df[c]


# QC fields

for c in [
    "truth_rows",
    "truth_rate_hz",
    "truth_qc_ok",
    "highrate_truth_csv"
]:

    if c in df.columns:
        clean[c] = df[c]


# Event fields

for c in [
    "event_available",
    "fall_duration_s",
    "lead_time_ms",
    "onset_time_s",
    "impact_time_s",
    "settle_time_s",
    "manifest_schema"
]:

    if c in df.columns:
        clean[c] = df[c]


# Preserve output location

if "output_dir" in df.columns:
    clean["output_dir"] = df["output_dir"]


# ----------------------------
# Audit
# ----------------------------

print("\nClean index:")
print(clean.shape)


print("\nProfiles:")
print(clean.profile_id.nunique())


print("\nScenarios:")
print(clean.scenario_id.nunique())


print("\nDuplicate run IDs:")
print(
    clean.run_id.duplicated().sum()
)


print("\nQC:")
print(
    clean.truth_qc_ok.value_counts(
        dropna=False
    )
)


print("\nMissing:")
for c in [
    "profile_id",
    "scenario_id",
    "highrate_truth_csv",
    "truth_rate_hz"
]:
    print(
        c,
        clean[c].isna().sum()
    )


# Save

out = (
    base /
    "Extended35_Master_Dataset_Index_CLEAN.csv"
)


clean.to_csv(
    out,
    index=False
)


print("\nSaved:")
print(out)

