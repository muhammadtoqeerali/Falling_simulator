
from pathlib import Path
import pandas as pd
import numpy as np


BASE = Path(
"outputs/_highrate_overnight/"
"campaign_highrate_truth_v2_extended35"
)


def compute_stats(
    name,
    acc_values,
    gyro_values,
    durations,
    hz_values,
    samples
):

    return {

        "Dataset": name,
        "Trials": len(durations),
        "Total_samples": samples,

        "Sampling_Hz_mean": np.mean(hz_values),
        "Duration_mean_s": np.mean(durations),

        "Acceleration_mean_mps2":
            np.mean(acc_values),

        "Acceleration_RMS_mps2":
            np.sqrt(np.mean(acc_values**2)),

        "Acceleration_std_mps2":
            np.std(acc_values),

        "Acceleration_P95_mps2":
            np.percentile(acc_values,95),

        "Acceleration_P99_mps2":
            np.percentile(acc_values,99),

        "Acceleration_peak_mps2":
            np.max(acc_values),


        "Gyroscope_mean_dps":
            np.mean(gyro_values),

        "Gyroscope_RMS_dps":
            np.sqrt(np.mean(gyro_values**2)),

        "Gyroscope_std_dps":
            np.std(gyro_values),

        "Gyroscope_P95_dps":
            np.percentile(gyro_values,95),

        "Gyroscope_P99_dps":
            np.percentile(gyro_values,99),

        "Gyroscope_peak_dps":
            np.max(gyro_values)

    }




def dynamic_gyro(x,y,z):

    x=x-x.mean()
    y=y-y.mean()
    z=z-z.mean()

    return np.sqrt(
        x*x+
        y*y+
        z*z
    )


results=[]


# =====================================================
# KFall
# =====================================================

print("Processing KFall")


kfall_root = Path(
"/mnt/hdd16T/ToqeerHomeBackup/toqeer/"
"uniVr-dataset/KFall_oriented/sensors_data"
)


acc=[]
gyro=[]
duration=[]
hz=[]
samples=0


for f in kfall_root.rglob("*.csv"):

    df=pd.read_csv(f)


    # KFall raw accelerometer:
    # convert to m/s2
    a=np.sqrt(
        df.AccX**2+
        df.AccY**2+
        df.AccZ**2
    ) * 9.81/1024


    # raw gyro -> deg/s
    g=dynamic_gyro(
        df.GyrX/16.4,
        df.GyrY/16.4,
        df.GyrZ/16.4
    )


    acc.extend(a.values)
    gyro.extend(g.values)


    if len(df)>1:
        duration.append(
            df["TimeStamp(s)"].iloc[-1] -
            df["TimeStamp(s)"].iloc[0]
        )

        hz.append(
            1/
            np.mean(
                np.diff(df["TimeStamp(s)"])
            )
        )

    samples += len(df)


results.append(
    compute_stats(
        "KFall",
        np.array(acc),
        np.array(gyro),
        duration,
        hz,
        samples
    )
)


# =====================================================
# UniVRFall
# =====================================================

print("Processing UniVRFall")


univr_root = Path(
"/mnt/hdd16T/ToqeerHomeBackup/toqeer/"
"uniVr-dataset/UniVrFall_Dataset/sensors_data"
)


acc=[]
gyro=[]
duration=[]
hz=[]
samples=0


for f in univr_root.rglob("*.csv"):

    df=pd.read_csv(f)


    a=np.sqrt(
        df.AccX**2+
        df.AccY**2+
        df.AccZ**2
    ) * 9.81/1024


    g=dynamic_gyro(
        df.GyrX/16.4,
        df.GyrY/16.4,
        df.GyrZ/16.4
    )


    acc.extend(a.values)
    gyro.extend(g.values)


    # UniVRFall timestamp column compatibility
    time_col = None

    for c in [
        "TimeStamp(s)",
        "Timestamp(s)",
        "Timestamp",
        "timestamp",
        "TimeStamp"
    ]:
        if c in df.columns:
            time_col=c
            break


    if time_col is not None:

        t=df[time_col]


        # UniVR timestamps are usually milliseconds
        dt=np.mean(np.diff(t))/1000


        if dt>0:

            hz.append(1/dt)

            duration.append(
                (t.iloc[-1]-t.iloc[0])/1000
            )

    else:

        # fallback
        hz.append(np.nan)
        duration.append(np.nan)


    samples+=len(df)


results.append(
    compute_stats(
        "UniVRFall",
        np.array(acc),
        np.array(gyro),
        duration,
        hz,
        samples
    )
)



# =====================================================
# Extended35 TRUE waveform
# =====================================================

print("Processing Extended35")


ext_root=BASE/"runs"


acc=[]
gyro=[]
duration=[]
hz=[]
samples=0


for f in ext_root.rglob("*_highrate_truth.csv"):

    df=pd.read_csv(
        f,
        comment="#"
    )


    a=np.sqrt(
        df.accel_true_x**2+
        df.accel_true_y**2+
        df.accel_true_z**2
    )


    g=dynamic_gyro(
        df.gyro_true_x*57.2957795,
        df.gyro_true_y*57.2957795,
        df.gyro_true_z*57.2957795
    )


    acc.extend(a.values)
    gyro.extend(g.values)


    duration.append(
        df.timestamp.iloc[-1] -
        df.timestamp.iloc[0]
    )


    hz.append(
        1/
        np.mean(
            np.diff(df.timestamp)
        )
    )


    samples += len(df)


results.append(
    compute_stats(
        "Extended35",
        np.array(acc),
        np.array(gyro),
        duration,
        hz,
        samples
    )
)



out=pd.DataFrame(results)


print("\n==============================")
print("FINAL TRUE WAVEFORM COMPARISON")
print("==============================")

print(out.to_string(index=False))


save=BASE/"Extended35_KFall_UniVRFall_TRUE_Waveform_Final_Comparison.csv"

out.to_csv(
    save,
    index=False
)


print("\nSaved:")
print(save)

