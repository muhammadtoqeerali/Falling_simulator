
from pathlib import Path
import pandas as pd
import numpy as np


base = Path(
"outputs/_highrate_overnight/"
"campaign_highrate_truth_v2_extended35"
)


ACC_SCALE = 9.81/1024.0

# public dataset gyro raw conversion
GYRO_SCALE = 1/1000.0


def summarize(
    name,
    acc,
    gyro,
    duration,
    hz,
    samples
):

    return {

    "Dataset":name,

    "Trials":len(duration),

    "Total_samples":samples,

    "Sampling_Hz_mean":
        np.mean(hz),

    "Duration_mean_s":
        np.mean(duration),


    "Acceleration_mean_mps2":
        np.mean(acc),

    "Acceleration_RMS_mps2":
        np.sqrt(np.mean(acc**2)),

    "Acceleration_std_mps2":
        np.std(acc),

    "Acceleration_P95_mps2":
        np.percentile(acc,95),

    "Acceleration_range":
        f"{acc.min():.2f}-{acc.max():.2f}",


    "Gyroscope_mean_dps":
        np.mean(gyro),

    "Gyroscope_RMS_dps":
        np.sqrt(np.mean(gyro**2)),

    "Gyroscope_std_dps":
        np.std(gyro),

    "Gyroscope_P95_dps":
        np.percentile(gyro,95),

    "Gyroscope_range":
        f"{gyro.min():.2f}-{gyro.max():.2f}"

    }



results=[]


# ==========================
# Extended35 TRUE waveform
# ==========================

acc=[]
gyro=[]
duration=[]
hz=[]
samples=0


files=list(
(base/"runs").rglob("*_highrate_truth.csv")
)


print("Extended35 files:",len(files))


for f in files:

    df=pd.read_csv(
        f,
        comment="#"
    )


    a=np.sqrt(
        df.accel_true_x**2+
        df.accel_true_y**2+
        df.accel_true_z**2
    )


    g=np.sqrt(
        df.gyro_true_x**2+
        df.gyro_true_y**2+
        df.gyro_true_z**2
    )


    acc.extend(a.values)
    gyro.extend(g.values*57.2957795)


    t=df.timestamp.values

    duration.append(
        t[-1]-t[0]
    )


    hz.append(
        1/np.mean(np.diff(t))
    )


    samples+=len(df)



results.append(
summarize(
"Extended35",
np.array(acc),
np.array(gyro),
duration,
hz,
samples
)
)



out=pd.DataFrame(results)


print("==============================")
print("TRUE EXTENDED35 WAVEFORM")
print("==============================")

print(out.to_string(index=False))


save=base/"Extended35_TRUE_Waveform_Statistics.csv"

out.to_csv(
save,
index=False
)


print("\nSaved:")
print(save)

