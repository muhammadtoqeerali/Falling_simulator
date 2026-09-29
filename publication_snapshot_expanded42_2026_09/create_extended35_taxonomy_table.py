
import pandas as pd
from pathlib import Path


base = Path(
"outputs/_highrate_overnight/"
"campaign_highrate_truth_v2_extended35"
)


scenario_file = (
    base /
    "Extended35_Scenario_Biomechanical_Comparison.csv"
)


df = pd.read_csv(
    scenario_file
)


rows=[]


for _, r in df.iterrows():

    desc = str(
        r["description"]
    ).lower()


    if "forward" in desc:
        direction = "Forward"

    elif "backward" in desc:
        direction = "Backward"

    elif "lateral" in desc:
        direction = "Lateral"

    else:
        direction = "Mixed"



    if "trip" in desc:
        mechanism = "Trip"

    elif "slip" in desc:
        mechanism = "Slip"

    elif "sit" in desc:
        mechanism = "Sitting transition"

    elif "height" in desc or "platform" in desc:
        mechanism = "Height fall"

    elif "ladder" in desc:
        mechanism = "Ladder fall"

    elif "faint" in desc:
        mechanism = "Fainting"

    elif "moving" in desc:
        mechanism = "Backward movement"

    else:
        mechanism = "Other"



    rows.append({

        "scenario_id":
            r["scenario_id"],

        "description":
            r["description"],

        "fall_direction":
            direction,

        "fall_mechanism":
            mechanism,

        "trials":
            r["trials"],

        "duration_mean_s":
            r["duration_mean_s"],

        "acc_peak_mean":
            r["acc_peak_mean"],

        "acc_peak_max":
            r["acc_peak_max"],

        "gyro_peak_mean":
            r["gyro_peak_mean"],

        "gyro_peak_max":
            r["gyro_peak_max"],

        "speed_mean":
            r["speed_mean"],

        "displacement_mean":
            r["displacement_x_mean"]

    })



out_df = pd.DataFrame(rows)


out = (
    base /
    "Extended35_Fall_Taxonomy_Table.csv"
)


out_df.to_csv(
    out,
    index=False
)


print("======================")
print("COMPLETE")
print("======================")

print(
out_df
)


print("\nSaved:")
print(out)

