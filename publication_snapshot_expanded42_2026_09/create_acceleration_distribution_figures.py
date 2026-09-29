from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


root=Path(
"outputs/_highrate_overnight/"
"campaign_highrate_truth_v2_extended35"
)


df=pd.read_csv(
root/
"Extended35_KFall_UniVRFall_ROBUST_ACCELERATION_Comparison.csv"
)


datasets=df["Dataset"]


# =========================
# RMS acceleration
# =========================

metrics=[
"acc_rms_Q25",
"acc_rms_median",
"acc_rms_Q75",
"acc_rms_P95"
]


plt.figure(figsize=(8,5))


for i,row in df.iterrows():

    plt.plot(
        [i]*4,
        row[metrics].values,
        marker="o"
    )


plt.xticks(
range(len(datasets)),
datasets
)

plt.ylabel(
"Acceleration RMS (m/s²)"
)

plt.title(
"Acceleration RMS Distribution Comparison"
)

plt.grid(True)

plt.tight_layout()


plt.savefig(
root/
"Fig_Acceleration_RMS_Distribution.png",
dpi=300
)


plt.close()



# =========================
# Peak acceleration
# =========================


metrics=[
"acc_peak_Q25",
"acc_peak_median",
"acc_peak_Q75",
"acc_peak_P95"
]


plt.figure(figsize=(8,5))


for i,row in df.iterrows():

    plt.plot(
        [i]*4,
        row[metrics].values,
        marker="o"
    )


plt.xticks(
range(len(datasets)),
datasets
)


plt.ylabel(
"Peak acceleration (m/s²)"
)


plt.title(
"Peak Acceleration Distribution Comparison"
)


plt.grid(True)

plt.tight_layout()


plt.savefig(
root/
"Fig_Peak_Acceleration_Distribution.png",
dpi=300
)


plt.close()



print("Figures saved")


