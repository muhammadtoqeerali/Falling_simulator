import pandas as pd
import numpy as np
from pathlib import Path


base = Path(
    "outputs/_highrate_overnight/"
    "campaign_highrate_truth_v2_extended35"
)


index_file = (
    base /
    "Extended35_Master_Dataset_Index.csv"
)


index = pd.read_csv(index_file)


print("Loaded master index:")
print(index.shape)

print("\nAvailable demographic columns:")
print(
    [
        c for c in index.columns
        if c in [
            "age",
            "age_x",
            "age_y",
            "height",
            "height_x",
            "height_y",
            "sex",
            "sex_x",
            "sex_y",
            "weight",
            "weight_x",
            "weight_y"
        ]
    ]
)


# ----------------------------
# Resolve demographic columns
# ----------------------------

def pick_column(options):

    for c in options:
        if c in index.columns:
            return c

    return None


age_col = pick_column(
    ["age", "age_x", "age_y"]
)

height_col = pick_column(
    ["height", "height_x", "height_y"]
)

sex_col = pick_column(
    ["sex", "sex_x", "sex_y"]
)

weight_col = pick_column(
    ["weight", "weight_x", "weight_y"]
)


print("\nSelected:")
print("age:", age_col)
print("height:", height_col)
print("sex:", sex_col)
print("weight:", weight_col)


features = []


for i, r in index.iterrows():

    try:

        truth = pd.read_csv(
            r["highrate_truth_csv"],
            comment="#"
        )


        t = truth["timestamp"].values


        duration = (
            t[-1]-t[0]
            if len(t)>1
            else np.nan
        )


        fs = (
            1/np.mean(np.diff(t))
            if len(t)>1
            else np.nan
        )


        acc = truth["accel_true_mag"]


        gyro = truth["gyro_true_mag"]


        speed = np.sqrt(
            truth.sensor_vel_x**2 +
            truth.sensor_vel_y**2 +
            truth.sensor_vel_z**2
        )


        features.append({

            "profile_id":
                r["profile_id"],

            "scenario_id":
                r["scenario_id"],


            "age":
                r[age_col] if age_col else np.nan,

            "height":
                r[height_col] if height_col else np.nan,

            "sex":
                r[sex_col] if sex_col else np.nan,

            "weight":
                r[weight_col] if weight_col else np.nan,


            "samples":
                len(truth),

            "duration_s":
                duration,

            "sampling_rate_hz":
                fs,


            "acc_mean":
                acc.mean(),

            "acc_std":
                acc.std(),

            "acc_rms":
                np.sqrt(
                    np.mean(acc**2)
                ),

            "acc_peak":
                acc.max(),


            "gyro_mean":
                gyro.mean(),

            "gyro_std":
                gyro.std(),

            "gyro_rms":
                np.sqrt(
                    np.mean(gyro**2)
                ),

            "gyro_peak":
                gyro.max(),


            "acc_x_rms":
                np.sqrt(
                    np.mean(
                        truth.accel_true_x**2
                    )
                ),

            "acc_y_rms":
                np.sqrt(
                    np.mean(
                        truth.accel_true_y**2
                    )
                ),

            "acc_z_rms":
                np.sqrt(
                    np.mean(
                        truth.accel_true_z**2
                    )
                ),


            "gyro_x_rms":
                np.sqrt(
                    np.mean(
                        truth.gyro_true_x**2
                    )
                ),

            "gyro_y_rms":
                np.sqrt(
                    np.mean(
                        truth.gyro_true_y**2
                    )
                ),

            "gyro_z_rms":
                np.sqrt(
                    np.mean(
                        truth.gyro_true_z**2
                    )
                ),


            "max_sensor_speed":
                speed.max(),


            "sensor_dx":
                truth.sensor_pos_x.max()
                -
                truth.sensor_pos_x.min(),

            "sensor_dy":
                truth.sensor_pos_y.max()
                -
                truth.sensor_pos_y.min(),

            "sensor_dz":
                truth.sensor_pos_z.max()
                -
                truth.sensor_pos_z.min(),


            "pelvis_height_min":
                truth.pelvis_height.min(),

            "pelvis_height_max":
                truth.pelvis_height.max(),

        })


    except Exception as e:

        print(
            "FAILED:",
            r["profile_id"],
            r["scenario_id"],
            str(e)
        )


    if (i+1)%50==0:
        print(
            "Processed:",
            i+1,
            "/",
            len(index)
        )



out = (
    base /
    "Extended35_IMU_Physics_Features.csv"
)


result = pd.DataFrame(features)


print("\n====================")
print("RESULT")
print("====================")

print(
    "Rows:",
    len(result)
)

print(
    "Columns:",
    len(result.columns)
)


if len(result)>0:

    print("\nSampling rate:")
    print(
        result["sampling_rate_hz"]
        .describe()
    )


result.to_csv(
    out,
    index=False
)


print("\nSaved:")
print(out)

