
from pathlib import Path
import pandas as pd
import numpy as np


ACC_SCALE = 9.81/1024.0
GYRO_SCALE = 2000.0/32768.0


def extract_dataset(root, name):

    results=[]

    files=list(Path(root).rglob("*.csv"))

    print(name, "files:", len(files))


    for f in files:

        try:

            df=pd.read_csv(f)


            t=df["TimeStamp(s)"].values


            # timestamp correction
            if name=="UniVRFall":

                t=t/1000.0


            duration=t[-1]-t[0]


            hz=1/np.mean(
                np.diff(t)[np.diff(t)>0]
            )


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


            # calibration
            acc=acc*ACC_SCALE

            gyro=gyro*GYRO_SCALE


            results.append(
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


    return pd.DataFrame(results)



# -------------------------
# KFall
# -------------------------

kfall=extract_dataset(
"/mnt/hdd16T/ToqeerHomeBackup/toqeer/uniVr-dataset/KFall_oriented/sensors_data",
"KFall"
)



# -------------------------
# UniVRFall
# -------------------------

univr=extract_dataset(
"/mnt/hdd16T/ToqeerHomeBackup/toqeer/uniVr-dataset/UniVrFall_Dataset/sensors_data",
"UniVRFall"
)



# -------------------------
# Extended35
# -------------------------

ext=pd.read_csv(
"outputs/_highrate_overnight/"
"campaign_highrate_truth_v2_extended35/"
"Extended35_IMU_Physics_Features.csv"
)


extended=pd.DataFrame(
{
"duration_s":ext.duration_s,
"sampling_hz":ext.sampling_rate_hz,
"acc_rms":ext.acc_rms,
"acc_peak":ext.acc_peak,
"gyro_rms":ext.gyro_rms,
"gyro_peak":ext.gyro_peak
}
)



summary=[]


for name,data in [

("KFall",kfall),
("UniVRFall",univr),
("Extended35",extended)

]:


    summary.append(
    {
    "Dataset":name,
    "Trials":len(data),
    "Sampling_Hz_mean":data.sampling_hz.mean(),
    "Duration_mean_s":data.duration_s.mean(),

    "Acc_RMS_mean_mps2":
        data.acc_rms.mean(),

    "Acc_Peak_mean_mps2":
        data.acc_peak.mean(),

    "Acc_Peak_max_mps2":
        data.acc_peak.max(),

    "Gyro_RMS_mean_dps":
        data.gyro_rms.mean(),

    "Gyro_Peak_mean_dps":
        data.gyro_peak.mean(),

    "Gyro_Peak_max_dps":
        data.gyro_peak.max()
    }
    )



out=pd.DataFrame(summary)


print("\n==============================")
print("NORMALIZED SIGNAL COMPARISON")
print("==============================")

print(out)


save=Path(
"outputs/_highrate_overnight/"
"campaign_highrate_truth_v2_extended35/"
"Extended35_KFall_UniVRFall_Signal_Statistics_NORMALIZED.csv"
)


out.to_csv(
save,
index=False
)


print("\nSaved:")
print(save)

