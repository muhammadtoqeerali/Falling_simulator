from pathlib import Path
import pandas as pd
import numpy as np


OUT = Path(
    "outputs/_highrate_overnight/"
    "campaign_highrate_truth_v2_extended35"
)


def dynamic_gyro(x,y,z):

    x=np.asarray(x)-np.mean(x)
    y=np.asarray(y)-np.mean(y)
    z=np.asarray(z)-np.mean(z)

    return np.sqrt(
        x*x+y*y+z*z
    )


def trial_metrics(g):

    return {
        "gyro_rms":
            np.sqrt(np.mean(g*g)),

        "gyro_p95":
            np.percentile(g,95),

        "gyro_p99":
            np.percentile(g,99),

        "gyro_peak":
            np.max(g)
    }



def dataset_summary(name, trials):

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


# ======================
# KFall
# ======================

print("Processing KFall")


trials=[]

root=Path(
"/mnt/hdd16T/ToqeerHomeBackup/"
"toqeer/uniVr-dataset/KFall_oriented/"
"sensors_data"
)


for f in root.rglob("*.csv"):

    df=pd.read_csv(f)

    g=dynamic_gyro(
        df.GyrX/16.4,
        df.GyrY/16.4,
        df.GyrZ/16.4
    )

    trials.append(
        trial_metrics(g)
    )


results.append(
    dataset_summary(
        "KFall",
        trials
    )
)



# ======================
# UniVRFall
# ======================

print("Processing UniVRFall")


trials=[]

root=Path(
"/mnt/hdd16T/ToqeerHomeBackup/"
"toqeer/uniVr-dataset/"
"UniVrFall_Dataset/sensors_data"
)


for f in root.rglob("*.csv"):

    df=pd.read_csv(f)

    g=dynamic_gyro(
        df.GyrX/16.4,
        df.GyrY/16.4,
        df.GyrZ/16.4
    )

    trials.append(
        trial_metrics(g)
    )


results.append(
    dataset_summary(
        "UniVRFall",
        trials
    )
)



# ======================
# Extended35
# ======================

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

    if "gyro_true_x" not in df:
        continue


    g=dynamic_gyro(
        df.gyro_true_x*57.2957795,
        df.gyro_true_y*57.2957795,
        df.gyro_true_z*57.2957795
    )


    trials.append(
        trial_metrics(g)
    )


results.append(
    dataset_summary(
        "Extended35",
        trials
    )
)



out=pd.DataFrame(results)


print()
print("==============================")
print("ROBUST DYNAMIC GYRO COMPARISON")
print("==============================")

print(
    out.to_string(index=False)
)


save=OUT/"Extended35_KFall_UniVRFall_ROBUST_DYNAMIC_GYRO_Comparison.csv"

out.to_csv(
    save,
    index=False
)


print()
print("Saved:")
print(save)


