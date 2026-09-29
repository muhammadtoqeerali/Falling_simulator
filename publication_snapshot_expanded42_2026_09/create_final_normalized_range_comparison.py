
from pathlib import Path
import pandas as pd
import numpy as np


base = Path(
"outputs/_highrate_overnight/"
"campaign_highrate_truth_v2_extended35"
)


# ------------------------------------------------
# Conversion assumptions
# ------------------------------------------------

ACC_SCALE = 9.81 / 1024.0

# gyro raw -> degree/s
# from dataset sensor convention
GYRO_SCALE = 1.0 / 1000.0

RAD_TO_DEG = 57.2957795



# ------------------------------------------------
# Generic statistics
# ------------------------------------------------

def stats(name, acc, gyro, duration, hz):

    return {

    "Dataset":name,

    "Trials":len(acc),

    "Sampling_Hz_mean":
        np.mean(hz),

    "Duration_mean_s":
        np.mean(duration),


    "Acceleration_mean_mps2":
        np.mean(acc),

    "Acceleration_median_mps2":
        np.median(acc),

    "Acceleration_P95_mps2":
        np.percentile(acc,95),

    "Acceleration_range_mps2":
        f"{np.min(acc):.2f}-{np.max(acc):.2f}",


    "Gyroscope_mean_dps":
        np.mean(gyro),

    "Gyroscope_median_dps":
        np.median(gyro),

    "Gyroscope_P95_dps":
        np.percentile(gyro,95),

    "Gyroscope_range_dps":
        f"{np.min(gyro):.2f}-{np.max(gyro):.2f}"

    }



results=[]


# ------------------------------------------------
# KFall
# ------------------------------------------------

kacc=[]
kgyro=[]
kdur=[]
khz=[]


root="/mnt/hdd16T/ToqeerHomeBackup/toqeer/uniVr-dataset/KFall_oriented/sensors_data"


for f in Path(root).rglob("*.csv"):

    try:

        df=pd.read_csv(f)


        acc=np.sqrt(
            df.AccX**2+
            df.AccY**2+
            df.AccZ**2
        ) * ACC_SCALE


        gyro=np.sqrt(
            df.GyrX**2+
            df.GyrY**2+
            df.GyrZ**2
        ) * GYRO_SCALE


        kacc.append(acc.max())
        kgyro.append(gyro.max())


        t=df["TimeStamp(s)"]

        kdur.append(
            t.iloc[-1]-t.iloc[0]
        )

        khz.append(
            1/np.mean(np.diff(t))
        )


    except:
        pass



results.append(
stats(
"KFall",
np.array(kacc),
np.array(kgyro),
np.array(kdur),
np.array(khz)
)
)



# ------------------------------------------------
# UniVRFall
# ------------------------------------------------

uacc=[]
ugyro=[]
udur=[]
uhz=[]


root="/mnt/hdd16T/ToqeerHomeBackup/toqeer/uniVr-dataset/UniVrFall_Dataset/sensors_data"


for f in Path(root).rglob("*.csv"):

    try:

        df=pd.read_csv(f)


        acc=np.sqrt(
            df.AccX**2+
            df.AccY**2+
            df.AccZ**2
        ) * ACC_SCALE


        gyro=np.sqrt(
            df.GyrX**2+
            df.GyrY**2+
            df.GyrZ**2
        ) * GYRO_SCALE


        uacc.append(acc.max())
        ugyro.append(gyro.max())


        t=df["TimeStamp(s)"]/1000


        udur.append(
            t.iloc[-1]-t.iloc[0]
        )

        uhz.append(
            1/np.mean(
            np.diff(t)[np.diff(t)>0]
            )
        )


    except:
        pass



results.append(
stats(
"UniVRFall",
np.array(uacc),
np.array(ugyro),
np.array(udur),
np.array(uhz)
)
)



# ------------------------------------------------
# Extended35
# ------------------------------------------------


df=pd.read_csv(
base/"Extended35_IMU_Physics_Features.csv"
)


# convert rad/s -> deg/s

gyro=df.gyro_peak * RAD_TO_DEG


results.append(
stats(
"Extended35",
df.acc_peak.values,
gyro.values,
df.duration_s.values,
df.sampling_rate_hz.values
)
)



out=pd.DataFrame(results)


print("==============================")
print("FINAL NORMALIZED RANGE TABLE")
print("==============================")

print(out.to_string(index=False))


save=base/"Extended35_KFall_UniVRFall_Normalized_Range_Comparison.csv"


out.to_csv(
save,
index=False
)


print("\nSaved:")
print(save)

