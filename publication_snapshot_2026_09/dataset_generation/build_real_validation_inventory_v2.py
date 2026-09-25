from pathlib import Path
import pandas as pd
import re

CANONICAL = {
     1:20,  2:21,  3:22,  4:23,  5:24,
     6:25,  7:26,  8:27,  9:28, 10:29,
    11:30, 12:31, 13:32, 14:33, 15:34,
    16:37, 17:38, 18:39, 19:40, 20:41,
    21:42,
}

DATASETS = {
    "UniVrFall": Path(
        "/mnt/hdd16T/ToqeerHomeBackup/toqeer/"
        "uniVr-dataset/UniVrFall_Dataset"
    ),
    "KFall": Path(
        "/mnt/hdd16T/ToqeerHomeBackup/toqeer/"
        "uniVr-dataset/KFall_oriented"
    ),
}

OUT = Path("outputs/validation_v2")
OUT.mkdir(parents=True, exist_ok=True)

rows = []

for dataset, root in DATASETS.items():

    for lab in sorted((root / "labels_data").glob("*.xlsx")):

        sm = re.search(r"SA(\d+)", lab.name, re.I)
        if not sm:
            continue

        subject = int(sm.group(1))

        labels = pd.read_excel(lab)

        labels["Task Code (Task ID)"] = (
            labels["Task Code (Task ID)"].ffill()
        )

        labels["Description"] = (
            labels["Description"].ffill()
        )

        for _, r in labels.iterrows():

            code = str(r["Task Code (Task ID)"])

            mf = re.search(r"F\s*0*(\d+)", code, re.I)
            mt = re.search(r"\((\d+)\)", code)

            if not mf or not mt:
                continue

            fnum = int(mf.group(1))
            raw_task_id = int(mt.group(1))

            canonical_task = CANONICAL.get(fnum)

            # We only want fall tasks represented by the canonical map.
            if canonical_task is None:
                continue

            try:
                trial = int(r["Trial ID"])
            except Exception:
                continue

            # IMPORTANT:
            # physical filename T number = raw ID in parentheses,
            # NOT the semantic F number.
            sensor = (
                root /
                "sensors_data" /
                f"SA{subject:02d}" /
                f"S{subject:02d}T{raw_task_id:02d}R{trial:02d}.csv"
            )

            onset = pd.to_numeric(
                pd.Series([r["Fall_onset_frame"]]),
                errors="coerce"
            ).iloc[0]

            impact = pd.to_numeric(
                pd.Series([r["Fall_impact_frame"]]),
                errors="coerce"
            ).iloc[0]

            exists = sensor.exists()

            valid_event = (
                pd.notna(onset)
                and pd.notna(impact)
                and impact > onset
            )

            rows.append({
                "dataset": dataset,
                "subject_id": subject,

                "semantic_fnum": fnum,

                # ID used by the original physical dataset/file.
                "raw_task_id": raw_task_id,

                # ID used for comparison with simulator.
                "canonical_task_id": canonical_task,

                "trial_id": trial,
                "description": str(r["Description"]),

                "fall_onset_frame": onset,
                "fall_impact_frame": impact,

                "sensor_exists": exists,
                "valid_event_label": valid_event,

                "sensor_csv": str(sensor),
                "label_xlsx": str(lab),
            })


df = pd.DataFrame(rows)

out_csv = OUT / "real_inventory_canonical.csv"
df.to_csv(out_csv, index=False)

print("=" * 100)
print("CANONICAL REAL VALIDATION INVENTORY")
print("=" * 100)

print("\nTotal label records:", len(df))

for dataset in DATASETS:

    x = df[df.dataset == dataset]

    print("\n" + "=" * 80)
    print(dataset)
    print("=" * 80)

    print("Records              :", len(x))
    print("Subjects             :", x.subject_id.nunique())
    print("Sensor files found   :", int(x.sensor_exists.sum()))
    print("Sensor files missing :", int((~x.sensor_exists).sum()))

    print(
        "Valid onset/impact   :",
        int((x.sensor_exists & x.valid_event_label).sum())
    )

    print(
        "Invalid event labels :",
        int((x.sensor_exists & ~x.valid_event_label).sum())
    )

    print("\nRecords per canonical task:")
    print(
        x.groupby("canonical_task_id")
         .size()
         .to_string()
    )

    print("\nFOUND sensor records per canonical task:")
    print(
        x[x.sensor_exists]
         .groupby("canonical_task_id")
         .size()
         .to_string()
    )

    bad = x[~x.sensor_exists]

    if len(bad):
        print("\nMISSING SENSOR FILES:")
        print(
            bad[[
                "subject_id",
                "semantic_fnum",
                "raw_task_id",
                "canonical_task_id",
                "trial_id",
                "sensor_csv",
            ]]
            .to_string(index=False)
        )


print("\n" + "=" * 100)
print("TASK COVERAGE FOR VALIDATION")
print("=" * 100)

u = set(
    df[
        (df.dataset == "UniVrFall") &
        df.sensor_exists
    ].canonical_task_id
)

k = set(
    df[
        (df.dataset == "KFall") &
        df.sensor_exists
    ].canonical_task_id
)

print("UniVrFall tasks :", sorted(u))
print("KFall tasks     :", sorted(k))
print("Both datasets   :", sorted(u & k))
print("UniVrFall only  :", sorted(u - k))

print("\nSaved:")
print(out_csv)
