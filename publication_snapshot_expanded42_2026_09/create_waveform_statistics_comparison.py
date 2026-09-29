
from pathlib import Path
import pandas as pd
import numpy as np


base = Path(
"outputs/_highrate_overnight/"
"campaign_highrate_truth_v2_extended35"
)


ACC_SCALE = 9.81 / 1024.0

# raw gyro datasets -> deg/s
GYRO_SCALE = 1.0 / 1000.0

# Extended35 MuJoCo rad/s -> deg/s
RAD_TO_DEG = 57.2957795



def waveform_summary(
    name,
    acc_values,
    gyro_values,
    duration,
    hz
):

    return {

    "Dataset": name,

    "Trials":
        len(duration),

    "Sampling_Hz_mean":
        np.mean(hz),

    "Duration_mean_s":
        np.mean(duration),


    "Acceleration_mean_mps2":
        np.mean(acc_values),

    "Acceleration_RMS_mps2":
        np.sqrt(
            np.mean(
                acc_values**2
            )
        ),

    "Acceleration_std_mps2":
        np.std(acc_values),

    "Acceleration_P95_mps2":
        np.percentile(
            acc_values,
            95
        ),

    "Acceleration_range_mps2":
        f"{acc_values.min():.2f}-{acc_values.max():.2f}",


    "Gyroscope_mean_dps":
        np.mean(gyro_values),

    "Gyroscope_RMS_dps":
        np.sqrt(
            np.mean(
                gyro_values**2
            )
        ),

    "Gyroscope_std_dps":
        np.std(gyro_values),

    "Gyroscope_P95_dps":
        np.percentile(
            gyro_values,
            95
        ),

    "Gyroscope_range_dps":
        f"{gyro_values.min():.2f}-{gyro_values.max():.2f}"

    }



results=[]



# =====================================================
# KFall
# =====================================================

acc=[]
gyro=[]
duration=[]
hz=[]


root=Path(
"/mnt/hdd16T/ToqeerHomeBackup/"
"toqeer/uniVr-dataset/"
"KFall_oriented/sensors_data"
)


for f in root.rglob("*.csv"):

    try:

        df=pd.read_csv(f)


        a=np.sqrt(
            df.AccX**2+
            df.AccY**2+
            df.AccZ**2
        )*ACC_SCALE


        g=np.sqrt(
            df.GyrX**2+
            df.GyrY**2+
            df.GyrZ**2
        )*GYRO_SCALE


        acc.extend(a.values)
        gyro.extend(g.values)


        t=df["TimeStamp(s)"].values

        duration.append(
            t[-1]-t[0]
        )

        hz.append(
            1/np.mean(np.diff(t))
        )


    except:
        pass



results.append(
waveform_summary(
"KFall",
np.array(acc),
np.array(gyro),
duration,
hz
)
)



# =====================================================
# UniVRFall
# =====================================================

acc=[]
gyro=[]
duration=[]
hz=[]


root=Path(
"/mnt/hdd16T/ToqeerHomeBackup/"
"toqeer/uniVr-dataset/"
"UniVrFall_Dataset/sensors_data"
)


for f in root.rglob("*.csv"):

    try:

        df=pd.read_csv(f)


        a=np.sqrt(
            df.AccX**2+
            df.AccY**2+
            df.AccZ**2
        )*ACC_SCALE


        g=np.sqrt(
            df.GyrX**2+
            df.GyrY**2+
            df.GyrZ**2
        )*GYRO_SCALE


        acc.extend(a.values)
        gyro.extend(g.values)


        t=df["TimeStamp(s)"].values/1000


        duration.append(
            t[-1]-t[0]
        )


        d=np.diff(t)

        hz.append(
            1/np.mean(
                d[d>0]
            )
        )


    except:
        pass



results.append(
waveform_summary(
"UniVRFall",
np.array(acc),
np.array(gyro),
duration,
hz
)
)



# =====================================================
# Extended35
# =====================================================


df=pd.read_csv(
base/"Extended35_IMU_Physics_Features.csv"
)


# For waveform-level we use per-trial features
# available from generated extraction

acc=df.acc_rms.values
gyro=df.gyro_rms.values*RAD_TO_DEG


results.append(
waveform_summary(
"Extended35",
acc,
gyro,
df.duration_s.values,
df.sampling_rate_hz.values
)
)



out=pd.DataFrame(results)


print("==============================")
print("WAVEFORM STATISTICS")
print("==============================")

print(
out.to_string(index=False)
)


save=base/"Extended35_KFall_UniVRFall_Waveform_Statistics.csv"


out.to_csv(
save,
index=False
)


print("\nSaved:")
print(save)

