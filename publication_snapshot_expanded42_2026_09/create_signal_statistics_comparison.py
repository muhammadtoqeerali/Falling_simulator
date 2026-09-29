
from pathlib import Path
import pandas as pd
import numpy as np


# ==================================================
# KFall
# ==================================================

kfall_root = Path(
"/mnt/hdd16T/ToqeerHomeBackup/toqeer/uniVr-dataset/KFall_oriented/sensors_data"
)


kfall_results=[]


for f in kfall_root.rglob("*.csv"):

    try:

        df=pd.read_csv(f)


        t=df["TimeStamp(s)"].values


        duration=t[-1]-t[0]

        dt=np.diff(t)

        hz=1/np.mean(dt)


        acc=np.sqrt(
            df.AccX**2+
            df.AccY**2+
            df.AccZ**2
        )


        gyro=np.sqrt(
            df.GyrX**2+
            df.GyrY**2+
            df.GyrZ**2
        )


        kfall_results.append(
            {
            "duration_s":duration,
            "sampling_hz":hz,
            "acc_rms":acc.std(),
            "acc_peak":acc.max(),
            "gyro_rms":gyro.std(),
            "gyro_peak":gyro.max()
            }
        )


    except Exception as e:
        pass



kfall=pd.DataFrame(kfall_results)



# ==================================================
# UniVRFall
# ==================================================

univr_root=Path(
"/mnt/hdd16T/ToqeerHomeBackup/toqeer/uniVr-dataset/UniVrFall_Dataset/sensors_data"
)


univr_results=[]


for f in univr_root.rglob("*.csv"):

    try:

        df=pd.read_csv(f)


        t=df["TimeStamp(s)"].values


        duration=t[-1]-t[0]

        hz=1/np.mean(np.diff(t))


        acc=np.sqrt(
            df.AccX**2+
            df.AccY**2+
            df.AccZ**2
        )


        gyro=np.sqrt(
            df.GyrX**2+
            df.GyrY**2+
            df.GyrZ**2
        )


        univr_results.append(
            {
            "duration_s":duration,
            "sampling_hz":hz,
            "acc_rms":acc.std(),
            "acc_peak":acc.max(),
            "gyro_rms":gyro.std(),
            "gyro_peak":gyro.max()
            }
        )


    except:
        pass



univr=pd.DataFrame(univr_results)



# ==================================================
# Extended35
# ==================================================

ext=pd.read_csv(
"outputs/_highrate_overnight/"
"campaign_highrate_truth_v2_extended35/"
"Extended35_IMU_Physics_Features.csv"
)



extended35=pd.DataFrame(
{
"duration_s":ext.duration_s,
"sampling_hz":ext.sampling_rate_hz,
"acc_rms":ext.acc_rms,
"acc_peak":ext.acc_peak,
"gyro_rms":ext.gyro_rms,
"gyro_peak":ext.gyro_peak
}
)



# ==================================================
# Summary
# ==================================================

summary=[]


for name,data in [

("KFall",kfall),
("UniVRFall",univr),
("Extended35",extended35)

]:

    summary.append(
        {
        "Dataset":name,
        "Trials":len(data),
        "Sampling_Hz_mean":data.sampling_hz.mean(),
        "Duration_mean_s":data.duration_s.mean(),
        "Acc_RMS_mean":data.acc_rms.mean(),
        "Acc_Peak_mean":data.acc_peak.mean(),
        "Acc_Peak_max":data.acc_peak.max(),
        "Gyro_RMS_mean":data.gyro_rms.mean(),
        "Gyro_Peak_mean":data.gyro_peak.mean(),
        "Gyro_Peak_max":data.gyro_peak.max()
        }
    )



out=pd.DataFrame(summary)


print("==============================")
print("SIGNAL COMPARISON")
print("==============================")

print(out)



save=Path(
"outputs/_highrate_overnight/"
"campaign_highrate_truth_v2_extended35/"
"Extended35_KFall_UniVRFall_Signal_Statistics.csv"
)


out.to_csv(
save,
index=False
)


print("\nSaved:")
print(save)

