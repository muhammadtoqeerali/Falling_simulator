
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


profile = pd.read_csv(
    base /
    "Extended35_Profile_Diversity_Analysis.csv"
)


scenario = pd.read_csv(
    base /
    "Extended35_Scenario_Biomechanical_Comparison.csv"
)



rows = []


def add(section, metric, value):

    rows.append(
        {
            "section": section,
            "metric": metric,
            "value": value
        }
    )



# -------------------------
# Dataset
# -------------------------

add(
    "Dataset",
    "Total trials",
    len(master)
)

add(
    "Dataset",
    "Profiles",
    master.profile_id.nunique()
)

add(
    "Dataset",
    "Scenarios",
    master.scenario_id.nunique()
)

add(
    "Dataset",
    "Sampling frequency Hz",
    physics.sampling_rate_hz.mean()
)

add(
    "Dataset",
    "Total generated samples",
    int(physics.samples.sum())
)

add(
    "Dataset",
    "Total recording duration hours",
    round(
        physics.duration_s.sum()/3600,
        3
    )
)



# -------------------------
# Demographics
# -------------------------

add(
    "Participants",
    "Age range",
    f"{master.age.min()}-{master.age.max()} years"
)

add(
    "Participants",
    "Height range",
    f"{master.height_m.min():.2f}-{master.height_m.max():.2f} m"
)

add(
    "Participants",
    "Weight range",
    f"{master.weight_kg.min():.1f}-{master.weight_kg.max():.1f} kg"
)


# -------------------------
# Sensor
# -------------------------

add(
    "Sensor",
    "Sensor type",
    "Virtual IMU"
)

add(
    "Sensor",
    "Sensor location",
    "Pelvis segment (lower trunk approximation)"
)

add(
    "Sensor",
    "Signals",
    "3-axis acceleration + 3-axis angular velocity"
)


# -------------------------
# Physics
# -------------------------

add(
    "Dynamics",
    "Acceleration peak range",
    f"{physics.acc_peak.min():.2f}-{physics.acc_peak.max():.2f}"
)

add(
    "Dynamics",
    "Gyroscope peak range",
    f"{physics.gyro_peak.min():.2f}-{physics.gyro_peak.max():.2f}"
)

add(
    "Dynamics",
    "Maximum sensor speed",
    f"{physics.max_sensor_speed.max():.2f}"
)

add(
    "Dynamics",
    "Maximum displacement",
    f"{physics.sensor_dx.max():.2f}"
)



# -------------------------
# Scenarios
# -------------------------

add(
    "Fall scenarios",
    "Fall classes",
    scenario.scenario_id.nunique()
)

add(
    "Fall scenarios",
    "Trials per scenario",
    int(
        scenario.trials.mean()
    )
)


# Save

out = (
    base /
    "Extended35_Dataset_Methods_Table.csv"
)


pd.DataFrame(rows).to_csv(
    out,
    index=False
)


print("======================")
print("COMPLETE")
print("======================")

print(
pd.DataFrame(rows)
)


print("\nSaved:")
print(out)

