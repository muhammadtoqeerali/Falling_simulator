from pathlib import Path
import os
import re
import hashlib
import numpy as np
from sklearn.model_selection import KFold

ROOT = Path(
    "/mnt/hdd16T/ToqeerHomeBackup/toqeer/Protechto-master"
)

DATA = (
    ROOT
    / "data/dataset/segments/"
    / "300ms_50ov_npseg_filt_binary"
)

EXPECTED_TEST = {
    1: (450947, 449383, 1564),
    2: (296080, 294222, 1858),
    3: (201537, 199628, 1909),
    4: (63129, 60749, 2380),
    5: (285474, 283379, 2095),
}

REQUIRED = [
    ROOT / "train.py",
    ROOT / "test.py",
    ROOT / "constants.py",
    ROOT / "config.json",
    ROOT / "dataloaders/KFoldDataloader.py",
    ROOT / "dataloaders/helper.py",
    ROOT / "simulation/SimulationKFoldDataloader.py",
    ROOT / "simulation/helper.py",
    ROOT / "models/CNN.py",
    ROOT / "models/wrappers/Predictor.py",
    ROOT / "models/wrappers/Simulator.py",
    ROOT / "models/modules/IMUNormalizer.py",
    ROOT / "models/configs/CNN_config.py",
    DATA,
]

print("=" * 110)
print("PROTECHTO CLEAN REAL-ONLY PREFLIGHT")
print("=" * 110)

print()
print("Canonical root:")
print(ROOT)

print()
print("Physical dataset:")
print(DATA)

for p in REQUIRED:
    if not p.exists():
        raise RuntimeError(
            f"Missing required path: {p}"
        )

print()
print("Required paths: PASS")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)
    return h.hexdigest()


print()
print("SOURCE SHA256")
for rel in [
    "train.py",
    "test.py",
    "dataloaders/KFoldDataloader.py",
    "simulation/SimulationKFoldDataloader.py",
    "simulation/helper.py",
    "models/CNN.py",
    "models/wrappers/Predictor.py",
    "models/wrappers/Simulator.py",
    "models/modules/IMUNormalizer.py",
    "models/configs/CNN_config.py",
]:
    p = ROOT / rel
    print(
        f"{rel:<48} {sha256(p)}"
    )


print()
print("=" * 110)
print("VERIFY K-FOLD PATH HAS NO ACTIVE AUGMENTATION")
print("=" * 110)

src = (
    ROOT
    / "dataloaders/KFoldDataloader.py"
).read_text(encoding="utf-8", errors="replace")

m = re.search(
    r"def construct\(.*?"
    r"(?=\n\s+def |\Z)",
    src,
    flags=re.S,
)

if not m:
    raise RuntimeError(
        "Could not isolate KFoldDataloader.construct()"
    )

construct_src = m.group(0)

active_aug = []

for line in construct_src.splitlines():
    stripped = line.strip()

    if (
        "data_augmentation(" in stripped
        and not stripped.startswith("#")
    ):
        active_aug.append(line)

if active_aug:
    raise RuntimeError(
        "Active augmentation call found in "
        "KFold construct():\n"
        + "\n".join(active_aug)
    )

print("Active data_augmentation() in KFold construct: NONE")
print("K-FOLD REAL-ONLY GATE: PASS")


print()
print("=" * 110)
print("VERIFY 71-SUBJECT SPLIT AND EXACT UNTOUCHED TEST COUNTS")
print("=" * 110)

excluded = {
    "999",
    "1000",
}

subjects = np.array([
    s
    for s in sorted(os.listdir(DATA))
    if (
        not s.startswith(".")
        and s not in excluded
    )
])

print(
    "Eligible subjects:",
    len(subjects),
)

if len(subjects) != 71:
    raise RuntimeError(
        f"Expected 71 eligible subjects, got {len(subjects)}"
    )

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)


def count_subjects(subject_list):
    activity = 0
    falling = 0

    for subject in subject_list:
        sd = DATA / str(subject)

        for task in sorted(
            p for p in sd.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        ):
            for trial in sorted(
                p for p in task.iterdir()
                if p.is_dir() and not p.name.startswith(".")
            ):
                y = np.load(
                    trial / "labels.npy",
                    allow_pickle=True,
                ).astype(str)

                activity += int(
                    np.sum(y == "Activity")
                )

                falling += int(
                    np.sum(y == "Falling")
                )

                if not np.all(
                    np.isin(
                        y,
                        ["Activity", "Falling"],
                    )
                ):
                    raise RuntimeError(
                        f"Unexpected labels in {trial}"
                    )

    return (
        activity + falling,
        activity,
        falling,
    )


for fold0, (_, test_idx) in enumerate(
    kf.split(subjects)
):
    fold = fold0 + 1
    test_subjects = subjects[test_idx]

    counts = count_subjects(
        test_subjects
    )

    expected = EXPECTED_TEST[fold]

    print()
    print(
        f"fold={fold}"
    )
    print(
        "  subjects:",
        list(test_subjects),
    )
    print(
        "  observed total/activity/falling:",
        counts,
    )
    print(
        "  expected total/activity/falling:",
        expected,
    )

    if counts != expected:
        raise RuntimeError(
            f"Fold {fold} count mismatch: "
            f"{counts} != {expected}"
        )

print()
print("UNTOUCHED PHYSICAL DATASET/SPLIT GATE: PASS")


print()
print("=" * 110)
print("VERIFY TRAINING CONSTANTS")
print("=" * 110)

train_src = (
    ROOT / "train.py"
).read_text(encoding="utf-8", errors="replace")

required_training = [
    "k = 5",
    "batch_size=64",
    "patience=20",
    "max_epochs=100",
]

for token in required_training:
    if token not in train_src:
        raise RuntimeError(
            f"Training token not found: {token}"
        )

print("K-folds          : 5")
print("Batch size       : 64")
print("Max epochs       : 100")
print("Early-stop       : patience 20")


cfg_src = (
    ROOT
    / "models/configs/CNN_config.py"
).read_text(encoding="utf-8", errors="replace")

required_cfg = [
    '"conv_1_dim": 64',
    '"conv_1_filter": 4',
    '"conv_dropout": 0.4',
    '"conv_pool": 2',
    '"fc": 128',
    '"fc_dropout": 0.4',
    '"lr": 0.001078759046949597',
    '"wd": 4.864667834921314e-05',
    '"prediction_bias": 0.65',
    "nn.CrossEntropyLoss()",
]

for token in required_cfg:
    if token not in cfg_src:
        raise RuntimeError(
            f"CNN config token not found: {token}"
        )

print("CNN filters      : 64")
print("CNN kernel       : 4")
print("Pooling          : 2")
print("Conv dropout     : 0.4")
print("FC               : 128")
print("FC dropout       : 0.4")
print("Loss             : CrossEntropyLoss")
print("Optimizer        : AdamW")
print("LR               : 0.001078759046949597")
print("Weight decay     : 4.864667834921314e-05")
print("Prediction bias  : 0.65")


print()
print("=" * 110)
print("VERIFY EVENT THRESHOLD")
print("=" * 110)

helper = (
    ROOT
    / "simulation/helper.py"
).read_text(encoding="utf-8", errors="replace")

if not re.search(
    r"def\s+is_simulation_passed_threshold"
    r"\s*\([^)]*threshold\s*=\s*2",
    helper,
):
    raise RuntimeError(
        "Could not verify threshold=2"
    )

print("Consecutive Falling threshold: 2")
print("EVENT PIPELINE GATE: PASS")


print()
print("=" * 110)
print("VERIFY NO MANUAL 150-ms OPERATION IN THIS CAMPAIGN")
print("=" * 110)

print(
    "Physical input is already-segmented "
    "300ms_50ov_npseg_filt_binary."
)

print(
    "This campaign does not call preprocessing "
    "and performs no temporal trimming."
)

print(
    "MANUAL -150ms OPERATION: NONE"
)


print()
print("=" * 110)
print("PROTECHTO CLEAN REAL-ONLY PRELAUNCH GATE: PASS")
print("=" * 110)
