from pathlib import Path
import pandas as pd

ROOT = Path("/mnt/hdd16T/ToqeerHomeBackup/mujoco_project")

M = (
    ROOT
    / "outputs/_highrate_overnight/"
      "campaign_highrate_truth_v1/"
      "highrate_event_manifest_PUBLICATION_QC.csv"
)

OUT = (
    ROOT
    / "outputs/validation_v2/"
      "legacy_channel_semantics_audit.csv"
)


def norm(x):
    p = Path(str(x))
    return p if p.is_absolute() else ROOT / p


def has_triplet(cols, names):
    c = {str(x).strip().lower() for x in cols}
    return all(x.lower() in c for x in names)


m = pd.read_csv(M)

rows = []

for _, r in m.iterrows():

    task = int(r["scenario_id"])
    age = int(r["sim_age_profile"])
    path = norm(r["legacy_csv"])

    df = pd.read_csv(
        path,
        comment="#",
        nrows=3,
    )

    cols = list(df.columns)

    rows.append({
        "scenario_id": task,
        "sim_age_profile": age,
        "n_columns": len(cols),

        "has_accel_true_xyz":
            has_triplet(
                cols,
                [
                    "accel_true_x",
                    "accel_true_y",
                    "accel_true_z",
                ],
            ),

        "has_accel_raw_xyz":
            has_triplet(
                cols,
                [
                    "accel_raw_x",
                    "accel_raw_y",
                    "accel_raw_z",
                ],
            ),

        "has_accel_xyz":
            has_triplet(
                cols,
                [
                    "accel_x",
                    "accel_y",
                    "accel_z",
                ],
            ),

        "has_ax_ay_az":
            has_triplet(
                cols,
                ["ax","ay","az"],
            ),

        "has_gyro_true_xyz":
            has_triplet(
                cols,
                [
                    "gyro_true_x",
                    "gyro_true_y",
                    "gyro_true_z",
                ],
            ),

        "has_gyro_raw_xyz":
            has_triplet(
                cols,
                [
                    "gyro_raw_x",
                    "gyro_raw_y",
                    "gyro_raw_z",
                ],
            ),

        "has_gyro_xyz":
            has_triplet(
                cols,
                [
                    "gyro_x",
                    "gyro_y",
                    "gyro_z",
                ],
            ),

        "has_gx_gy_gz":
            has_triplet(
                cols,
                ["gx","gy","gz"],
            ),

        "columns":
            "|".join(cols),
    })


out = pd.DataFrame(rows)

out.to_csv(
    OUT,
    index=False,
)


flags = [
    "has_accel_true_xyz",
    "has_accel_raw_xyz",
    "has_accel_xyz",
    "has_ax_ay_az",
    "has_gyro_true_xyz",
    "has_gyro_raw_xyz",
    "has_gyro_xyz",
    "has_gx_gy_gz",
]


print("=" * 110)
print("LEGACY SIGNAL-SEMANTICS AUDIT")
print("=" * 110)

print("records:", len(out))
print()

print("Overall availability:")
for c in flags:
    print(
        f"{c:25s}",
        int(out[c].sum()),
        "/",
        len(out),
    )


print()
print("Availability by semantic task:")

summary = (
    out.groupby("scenario_id")[flags]
    .sum()
    .astype(int)
)

print(
    summary.to_string()
)


print()
print("Representative relevant columns by task:")

for task in sorted(
    out.scenario_id.unique()
):

    row = out[
        out.scenario_id == task
    ].iloc[0]

    cols = row["columns"].split("|")

    interesting = [
        c for c in cols
        if any(
            k in c.lower()
            for k in [
                "accel",
                "gyro",
                "gyr",
                "impact",
                "ax",
                "ay",
                "az",
                "gx",
                "gy",
                "gz",
            ]
        )
    ]

    print()
    print(
        f"TASK {task}: "
        + ", ".join(interesting)
    )


print()
print("Audit CSV:", OUT)
print("LEGACY_CHANNEL_SEMANTICS_AUDIT_OK")
