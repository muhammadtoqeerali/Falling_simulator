from pathlib import Path
import pandas as pd
import numpy as np


OUT=Path(
"outputs/_highrate_overnight/"
"campaign_highrate_truth_v2_extended35"
)


def accel_mag(x,y,z):

    return np.sqrt(
        np.asarray(x)**2+
        np.asarray(y)**2+
        np.asarray(z)**2
    )


def trial_metrics(a):

    return {
        "acc_mean":
            np.mean(a),

        "acc_rms":
            np.sqrt(
                np.mean(a*a)
            ),

        "acc_std":
            np.std(a),

        "acc_p95":
            np.percentile(a,95),

        "acc_p99":
            np.percentile(a,99),

        "acc_peak":
            np.max(a)
    }



def summary(name,trials):

    df=pd.DataFrame(trials)

    out={
        "Dataset":name,
        "Trials":len(df)
    }

    for c in df.columns:

        out[c+"_median"]=df[c].median()
        out[c+"_Q25"]=df[c].quantile(.25)
        out[c+"_Q75"]=df[c].quantile(.75)
        out[c+"_P95"]=df[c].quantile(.95)

    return out



results=[]


# =====================
# KFall
# =====================

print("Processing KFall")

trials=[]

root=Path(
"/mnt/hdd16T/ToqeerHomeBackup/"
"toqeer/uniVr-dataset/KFall_oriented/"
"sensors_data"
)


for f in root.rglob("*.csv"):

    df=pd.read_csv(f)

    a=accel_mag(
        df.AccX,
        df.AccY,
        df.AccZ
    )

    # raw KFall acceleration conversion
    # counts -> m/s2 approximately
    a=a*9.81/1024

    trials.append(
        trial_metrics(a)
    )


results.append(
summary(
"KFall",
trials
)
)



# =====================
# UniVRFall
# =====================

print("Processing UniVRFall")


trials=[]

root=Path(
"/mnt/hdd16T/ToqeerHomeBackup/"
"toqeer/uniVr-dataset/"
"UniVrFall_Dataset/sensors_data"
)


for f in root.rglob("*.csv"):

    df=pd.read_csv(f)

    a=accel_mag(
        df.AccX,
        df.AccY,
        df.AccZ
    )

    a=a*9.81/1024


    trials.append(
        trial_metrics(a)
    )


results.append(
summary(
"UniVRFall",
trials
)
)



# =====================
# Extended35
# =====================

print("Processing Extended35")


trials=[]

root=OUT/"runs"


for f in root.rglob("*_highrate_truth.csv"):

    try:
        df=pd.read_csv(f)

    except:
        df=pd.read_csv(
            f,
            comment="#"
        )


    if "accel_true_x" not in df:
        continue


    a=accel_mag(
        df.accel_true_x,
        df.accel_true_y,
        df.accel_true_z
    )


    trials.append(
        trial_metrics(a)
    )


results.append(
summary(
"Extended35",
trials
)
)



out=pd.DataFrame(results)


print()
print("==============================")
print("ROBUST ACCELERATION COMPARISON")
print("==============================")

print(
out.to_string(index=False)
)


save=OUT/"Extended35_KFall_UniVRFall_ROBUST_ACCELERATION_Comparison.csv"


out.to_csv(
save,
index=False
)


print()
print("Saved:")
print(save)
