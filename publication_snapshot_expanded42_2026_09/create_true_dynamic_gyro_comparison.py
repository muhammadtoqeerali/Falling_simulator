from pathlib import Path
import pandas as pd
import numpy as np


OUT = Path(
    "outputs/_highrate_overnight/"
    "campaign_highrate_truth_v2_extended35"
)


def dynamic_gyro(x, y, z):

    x = x - np.mean(x)
    y = y - np.mean(y)
    z = z - np.mean(z)

    return np.sqrt(
        x*x +
        y*y +
        z*z
    )


def summarize(values):

    return {
        "Gyro_mean_dps": np.mean(values),
        "Gyro_RMS_dps": np.sqrt(np.mean(values**2)),
        "Gyro_std_dps": np.std(values),
        "Gyro_P95_dps": np.percentile(values,95),
        "Gyro_P99_dps": np.percentile(values,99),
        "Gyro_peak_dps": np.max(values)
    }


results=[]


# ============================
# KFall
# ============================

print("Processing KFall")

root = Path(
"/mnt/hdd16T/ToqeerHomeBackup/"
"toqeer/uniVr-dataset/KFall_oriented/"
"sensors_data"
)

files=list(root.rglob("*.csv"))

trial_stats=[]

for f in files:

    df=pd.read_csv(f)

    gx=df["GyrX"]/16.4
    gy=df["GyrY"]/16.4
    gz=df["GyrZ"]/16.4

    g=dynamic_gyro(
        gx,
        gy,
        gz
    )

    
trial_stats.append(
    summarize(g)
)



row={
"Dataset":"KFall",
"Trials":len(files)
}

row.update(
    pd.DataFrame(trial_stats).mean().to_dict()
)

results.append(row)



# ============================
# UniVRFall
# ============================

print("Processing UniVRFall")


root = Path(
"/mnt/hdd16T/ToqeerHomeBackup/"
"toqeer/uniVr-dataset/"
"UniVrFall_Dataset/sensors_data"
)


files=list(root.rglob("*.csv"))

trial_stats=[]


for f in files:

    df=pd.read_csv(f)


    gx=df["GyrX"]/16.4
    gy=df["GyrY"]/16.4
    gz=df["GyrZ"]/16.4


    g=dynamic_gyro(
        gx,
        gy,
        gz
    )


    
trial_stats.append(
    summarize(g)
)




row={
"Dataset":"UniVRFall",
"Trials":len(files)
}

row.update(
    pd.DataFrame(trial_stats).mean().to_dict()
)


results.append(row)



# ============================
# Extended35
# ============================

print("Processing Extended35")


root=OUT/"runs"

files=list(
    root.rglob("*_highrate_truth.csv")
)


trial_stats=[]


for f in files:

    try:
        df=pd.read_csv(f)

    except pd.errors.ParserError:
        df=pd.read_csv(
            f,
            comment="#"
        )


    if "gyro_true_x" not in df.columns:
        continue


    gx=df["gyro_true_x"]*57.2957795
    gy=df["gyro_true_y"]*57.2957795
    gz=df["gyro_true_z"]*57.2957795


    g=dynamic_gyro(
        gx,
        gy,
        gz
    )


    
trial_stats.append(
    summarize(g)
)




row={
"Dataset":"Extended35",
"Trials":len(files)
}


row.update(
    summarize(
        pd.DataFrame(trial_stats)
    )
)


results.append(row)



# ============================
# Save
# ============================


out=pd.DataFrame(results)


print()
print("==============================")
print("TRUE DYNAMIC GYRO COMPARISON")
print("==============================")

print(out.to_string(index=False))


save=OUT/"Extended35_KFall_UniVRFall_TRUE_DYNAMIC_GYRO_Comparison.csv"

out.to_csv(
    save,
    index=False
)


print()
print("Saved:")
print(save)



txt=OUT/"Extended35_KFall_UniVRFall_TRUE_DYNAMIC_GYRO_Methods.txt"


txt.write_text(
"""Gyroscope comparison methodology

Gyroscope signals were converted from dataset-specific raw units into degrees per second.

To remove orientation-dependent sensor offsets, the mean angular velocity of each recording was subtracted independently for every axis.

Dynamic angular velocity magnitude was then calculated as:

sqrt(wx^2 + wy^2 + wz^2)

Statistics reported:
- Mean angular velocity
- RMS angular velocity
- Standard deviation
- 95th percentile
- 99th percentile
- Maximum dynamic angular velocity

This procedure enables fair comparison between real-world datasets and Extended35 simulated IMU signals.
"""
)


print(txt)
