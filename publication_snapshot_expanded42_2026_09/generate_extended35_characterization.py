import pandas as pd
from pathlib import Path


base = Path(
"outputs/_highrate_overnight/"
"campaign_highrate_truth_v2_extended35"
)


master = pd.read_csv(
    base /
    "Extended35_Master_Dataset_Index_CLEAN.csv"
)


physics = pd.read_csv(
    base /
    "Extended35_IMU_Physics_Features.csv"
)


print("Master:")
print(master.shape)

print("\nPhysics:")
print(physics.shape)


# ---------------------------
# Dataset overview
# ---------------------------

overview = pd.DataFrame({

    "total_runs":
        [len(master)],

    "profiles":
        [master.profile_id.nunique()],

    "scenarios":
        [master.scenario_id.nunique()],

    "age_min":
        [master.age.min()],

    "age_max":
        [master.age.max()],

    "height_min_m":
        [master.height_m.min()],

    "height_max_m":
        [master.height_m.max()],

    "weight_min_kg":
        [master.weight_kg.min()],

    "weight_max_kg":
        [master.weight_kg.max()],


})


overview.to_csv(
    base /
    "Extended35_Dataset_Overview.csv",
    index=False
)


# ---------------------------
# Global physics summary
# ---------------------------

summary = physics.describe().T

summary.to_csv(
    base /
    "Extended35_Global_Physics_Summary.csv"
)


# ---------------------------
# Scenario analysis
# ---------------------------

scenario_summary = (
    physics
    .groupby("scenario_id")
    .agg(
        runs=("scenario_id","count"),

        duration_mean_s=
        ("duration_s","mean"),

        acc_peak_mean=
        ("acc_peak","mean"),

        acc_peak_max=
        ("acc_peak","max"),

        gyro_peak_mean=
        ("gyro_peak","mean"),

        gyro_peak_max=
        ("gyro_peak","max"),

        speed_max_mean=
        ("max_sensor_speed","mean")
    )
    .reset_index()
)


scenario_summary.to_csv(
    base /
    "Extended35_Scenario_Physics_Summary.csv",
    index=False
)


# ---------------------------
# Profile analysis
# ---------------------------

profile_summary = (
    physics
    .groupby("profile_id")
    .agg(
        runs=("profile_id","count"),

        duration_mean_s=
        ("duration_s","mean"),

        acc_peak_mean=
        ("acc_peak","mean"),

        gyro_peak_mean=
        ("gyro_peak","mean")
    )
    .reset_index()
)


profile_summary.to_csv(
    base /
    "Extended35_Profile_Physics_Summary.csv",
    index=False
)


print("\nCOMPLETE")

print(
"Created:"
)

for f in [
"Extended35_Dataset_Overview.csv",
"Extended35_Global_Physics_Summary.csv",
"Extended35_Scenario_Physics_Summary.csv",
"Extended35_Profile_Physics_Summary.csv"
]:

    print("-",f)

