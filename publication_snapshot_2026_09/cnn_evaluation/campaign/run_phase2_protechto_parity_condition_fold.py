from pathlib import Path
import argparse
import inspect
import json
import random
import sys
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT = Path(
    "/mnt/hdd16T/ToqeerHomeBackup/mujoco_project"
)

PROTECHTO = Path(
    "/mnt/hdd16T/ToqeerHomeBackup/toqeer/uniVr-dataset/UniVR_Protechto_dataset/Protechto_Code"
)

DATA = (
    PROJECT /
    "outputs/phase2_corrected_event_dataset_v1"
)

SPLITS = (
    PROJECT /
    "outputs/phase2_protechto_original_kfold_manifests_v1"
)

AUG = (
    PROJECT /
    "outputs/phase2_corrected_augmentation_manifests_v1"
)

CNN_CONFIG = {
    "conv_1_dim": 64,
    "conv_1_filter": 4,
    "conv_pool": 2,
    "conv_dropout": 0.4,
    "fc": 128,
    "fc_dropout": 0.4,
}

BATCH_SIZE = 64
FULL_MAX_EPOCHS = 100
FULL_PATIENCE = 20
LR = 0.001078759046949597
WD = 4.864667834921314e-05
FOLD = int(
    __import__("os").environ["PHASE2_FOLD"]
)

ALL_EXPERIMENTS = [
    "EXP01_PHYSICAL_ONLY",
    "EXP02_SIM_ONLY_CLOSED_DOMAIN",
    "EXP03_MIX20",
    "EXP04_MIX50",
    "EXP05_MIX70",
    "EXP06_MIX100",
]

EXPERIMENT_NAME = (
    __import__("os")
    .environ["PHASE2_EXPERIMENT"]
)

if EXPERIMENT_NAME not in ALL_EXPERIMENTS:
    raise RuntimeError(
        f"Unknown experiment: {EXPERIMENT_NAME}"
    )

EXPERIMENTS = [
    EXPERIMENT_NAME
]


sys.path.insert(
    0,
    str(PROTECHTO),
)

import importlib.util as _importlib_util
import types as _types
import sys as _sys
from pathlib import Path as _Path

_ORIGINAL_PROTECHTO = _Path(
    "/mnt/hdd16T/ToqeerHomeBackup/toqeer/"
    "uniVr-dataset/UniVR_Protechto_dataset/"
    "Protechto_Code"
)

_ORIGINAL_MODELS = (
    _ORIGINAL_PROTECHTO / "models"
)

for _name in [
    "models.CNN",
    "models.helper",
    "models.modules.IMUNormalizer",
    "models.modules",
    "models",
]:
    _sys.modules.pop(_name, None)

_models_pkg = _types.ModuleType("models")
_models_pkg.__path__ = [
    str(_ORIGINAL_MODELS)
]
_sys.modules["models"] = _models_pkg

_modules_pkg = _types.ModuleType(
    "models.modules"
)
_modules_pkg.__path__ = [
    str(_ORIGINAL_MODELS / "modules")
]
_sys.modules[
    "models.modules"
] = _modules_pkg


def _load_original_module(
    module_name,
    file_path,
):
    spec = (
        _importlib_util
        .spec_from_file_location(
            module_name,
            file_path,
        )
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise RuntimeError(
            f"Cannot load {module_name} "
            f"from {file_path}"
        )

    module = (
        _importlib_util
        .module_from_spec(spec)
    )

    _sys.modules[
        module_name
    ] = module

    spec.loader.exec_module(
        module
    )

    return module


_original_helper = _load_original_module(
    "models.helper",
    _ORIGINAL_MODELS / "helper.py",
)

_original_normalizer = _load_original_module(
    "models.modules.IMUNormalizer",
    _ORIGINAL_MODELS
    / "modules"
    / "IMUNormalizer.py",
)

_original_cnn = _load_original_module(
    "models.CNN",
    _ORIGINAL_MODELS / "CNN.py",
)

CNN = _original_cnn.CNN



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(
        torch.backends,
        "cudnn",
    ):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def smoke_subset(
    X,
    y,
    meta,
    limit,
    seed,
):
    if len(y) <= limit:
        return X, y, meta.reset_index(drop=True)

    rng = np.random.default_rng(seed)

    y = np.asarray(y)

    pos = np.flatnonzero(
        y == 1
    )

    neg = np.flatnonzero(
        y == 0
    )

    frac_pos = (
        len(pos) /
        max(1, len(y))
    )

    n_pos = int(
        round(
            limit * frac_pos
        )
    )

    n_pos = max(
        32,
        n_pos,
    )

    n_pos = min(
        len(pos),
        n_pos,
        limit - 1,
    )

    n_neg = min(
        len(neg),
        limit - n_pos,
    )

    if n_pos + n_neg < limit:

        extra_pos = min(
            len(pos) - n_pos,
            limit - n_pos - n_neg,
        )

        n_pos += max(
            0,
            extra_pos,
        )

    pos_sel = rng.choice(
        pos,
        size=n_pos,
        replace=False,
    )

    neg_sel = rng.choice(
        neg,
        size=n_neg,
        replace=False,
    )

    idx = np.concatenate(
        [
            pos_sel,
            neg_sel,
        ]
    )

    rng.shuffle(idx)

    return (
        np.asarray(
            X[idx],
            dtype=np.float32,
        ),
        np.asarray(
            y[idx],
            dtype=np.int64,
        ),
        meta.iloc[
            idx
        ].reset_index(
            drop=True
        ),
    )


def make_loader(
    X,
    y,
    *,
    shuffle,
    seed,
):
    ds = TensorDataset(
        torch.from_numpy(
            np.asarray(
                X,
                dtype=np.float32,
            )
        ),
        torch.from_numpy(
            np.asarray(
                y,
                dtype=np.int64,
            )
        ),
    )

    g = torch.Generator()
    g.manual_seed(seed)

    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=(
            g if shuffle
            else None
        ),
    )


def evaluate(
    model,
    X,
    y,
    device,
):
    loader = make_loader(
        X,
        y,
        shuffle=False,
        seed=0,
    )

    preds = []
    probs = []

    model.eval()

    with torch.no_grad():

        for xb, _ in loader:

            xb = xb.to(
                device,
                non_blocking=True,
            )

            logits = model(xb)

            p = torch.softmax(
                logits,
                dim=1,
            )

            probs.append(
                p.cpu().numpy()
            )

            preds.append(
                torch.argmax(
                    p,
                    dim=1,
                )
                .cpu()
                .numpy()
            )

    pred = np.concatenate(
        preds
    )

    prob = np.concatenate(
        probs
    )

    y = np.asarray(
        y,
        dtype=int,
    )

    cm = confusion_matrix(
        y,
        pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = (
        cm.ravel()
    )

    metrics = {
        "accuracy":
            float(
                accuracy_score(
                    y,
                    pred,
                )
            ),

        "balanced_accuracy":
            float(
                balanced_accuracy_score(
                    y,
                    pred,
                )
            ),

        "precision_fall":
            float(
                precision_score(
                    y,
                    pred,
                    zero_division=0,
                )
            ),

        "recall_fall":
            float(
                recall_score(
                    y,
                    pred,
                    zero_division=0,
                )
            ),

        "f1_fall":
            float(
                f1_score(
                    y,
                    pred,
                    zero_division=0,
                )
            ),

        "specificity":
            float(
                tn /
                max(
                    1,
                    tn + fp,
                )
            ),

        "tn":
            int(tn),

        "fp":
            int(fp),

        "fn":
            int(fn),

        "tp":
            int(tp),
    }

    if len(
        np.unique(y)
    ) == 2:

        metrics[
            "roc_auc"
        ] = float(
            roc_auc_score(
                y,
                prob[:, 1],
            )
        )

        metrics[
            "average_precision"
        ] = float(
            average_precision_score(
                y,
                prob[:, 1],
            )
        )

    return (
        metrics,
        cm,
        pred,
        prob,
    )


def load_sources():

    physical_X = np.load(
        DATA / "physical_X.npy",
        mmap_mode="r",
    )

    physical_y = np.load(
        DATA / "physical_y.npy",
        mmap_mode="r",
    )

    physical_meta = pd.read_csv(
        DATA / "physical_metadata.csv"
    )

    synthetic_X = np.load(
        DATA / "synthetic_X.npy",
        mmap_mode="r",
    )

    synthetic_y = np.load(
        DATA / "synthetic_y.npy",
        mmap_mode="r",
    )

    synthetic_meta = pd.read_csv(
        DATA / "synthetic_metadata.csv"
    )


    # Original Protechto raw-unit model boundary.
    # Stored dataset: Acc=m/s^2, Gyro=deg/s.
    # Original model input: Acc=mg, Gyro=mdps.
    physical_X = np.array(
        physical_X,
        dtype=np.float32,
        copy=True,
    )
    synthetic_X = np.array(
        synthetic_X,
        dtype=np.float32,
        copy=True,
    )

    physical_X[:, :, 0:3] /= np.float32(0.00980665)
    physical_X[:, :, 3:6] *= np.float32(1000.0)

    synthetic_X[:, :, 0:3] /= np.float32(0.00980665)
    synthetic_X[:, :, 3:6] *= np.float32(1000.0)

    return (
        physical_X,
        physical_y,
        physical_meta,
        synthetic_X,
        synthetic_y,
        synthetic_meta,
    )


def role_indices(
    meta,
    assignment_file,
    id_col,
    fold,
):
    split = pd.read_csv(
        assignment_file
    )

    split = split[
        split["outer_fold"]
        == fold
    ].copy()

    role_map = dict(
        zip(
            split[
                id_col
            ].astype(str),
            split[
                "role"
            ].astype(str),
        )
    )

    role = (
        meta[id_col]
        .astype(str)
        .map(role_map)
    )

    if role.isna().any():
        raise RuntimeError(
            f"Missing split assignment for {id_col}"
        )

    return {
        r:
            np.flatnonzero(
                role.to_numpy()
                == r
            )
        for r in [
            "train",
            "val",
            "test",
        ]
    }


def build_experiment(
    experiment,
    sources,
):
    (
        physical_X,
        physical_y,
        physical_meta,
        synthetic_X,
        synthetic_y,
        synthetic_meta,
    ) = sources

    if experiment in [
        "EXP01_PHYSICAL_ONLY",
        "EXP03_MIX20",
        "EXP04_MIX50",
        "EXP05_MIX70",
        "EXP06_MIX100",
    ]:

        idx = role_indices(
            physical_meta,
            SPLITS /
            "physical_nested_split_assignments.csv",
            "group_id",
            FOLD,
        )

        train_X = np.asarray(
            physical_X[
                idx["train"]
            ],
            dtype=np.float32,
        )

        train_y = np.asarray(
            physical_y[
                idx["train"]
            ],
            dtype=np.int64,
        )

        train_meta = (
            physical_meta.iloc[
                idx["train"]
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        train_meta[
            "training_source"
        ] = "PHYSICAL"

        val_X = np.asarray(
            physical_X[
                idx["val"]
            ],
            dtype=np.float32,
        )

        val_y = np.asarray(
            physical_y[
                idx["val"]
            ],
            dtype=np.int64,
        )

        val_meta = (
            physical_meta.iloc[
                idx["val"]
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        test_X = np.asarray(
            physical_X[
                idx["test"]
            ],
            dtype=np.float32,
        )

        test_y = np.asarray(
            physical_y[
                idx["test"]
            ],
            dtype=np.int64,
        )

        test_meta = (
            physical_meta.iloc[
                idx["test"]
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        if experiment != "EXP01_PHYSICAL_ONLY":

            mix_name = {
                "EXP03_MIX20": "MIX20",
                "EXP04_MIX50": "MIX50",
                "EXP05_MIX70": "MIX70",
                "EXP06_MIX100": "MIX100",
            }[experiment]

            draws = pd.read_csv(
                AUG /
                f"fold_{FOLD}_{mix_name}_synthetic_draws.csv"
            )

            syn_idx = pd.to_numeric(
                draws[
                    "synthetic_window_index"
                ],
                errors="raise",
            ).astype(int).to_numpy()

            if not (
                synthetic_y[
                    syn_idx
                ] == 1
            ).all():
                raise RuntimeError(
                    f"{mix_name} contains non-Falling synthetic draws."
                )

            syn_X = np.asarray(
                synthetic_X[
                    syn_idx
                ],
                dtype=np.float32,
            )

            syn_y = np.asarray(
                synthetic_y[
                    syn_idx
                ],
                dtype=np.int64,
            )

            syn_meta = (
                synthetic_meta.iloc[
                    syn_idx
                ]
                .copy()
                .reset_index(
                    drop=True
                )
            )

            syn_meta[
                "training_source"
            ] = "SYNTHETIC"

            syn_meta[
                "synthetic_draw_position"
            ] = draws[
                "draw_position"
            ].to_numpy()

            syn_meta[
                "synthetic_window_index"
            ] = syn_idx

            syn_meta[
                "synthetic_window_use_number"
            ] = draws[
                "window_use_number"
            ].to_numpy()

            train_X = np.concatenate(
                [
                    train_X,
                    syn_X,
                ],
                axis=0,
            )

            train_y = np.concatenate(
                [
                    train_y,
                    syn_y,
                ],
                axis=0,
            )

            train_meta = pd.concat(
                [
                    train_meta,
                    syn_meta,
                ],
                ignore_index=True,
                sort=False,
            )

    elif experiment == "EXP02_SIM_ONLY_CLOSED_DOMAIN":

        idx = role_indices(
            synthetic_meta,
            SPLITS /
            "synthetic_nested_split_assignments.csv",
            "profile_id",
            FOLD,
        )

        train_X = np.asarray(
            synthetic_X[
                idx["train"]
            ],
            dtype=np.float32,
        )

        train_y = np.asarray(
            synthetic_y[
                idx["train"]
            ],
            dtype=np.int64,
        )

        train_meta = (
            synthetic_meta.iloc[
                idx["train"]
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        train_meta[
            "training_source"
        ] = "SYNTHETIC"

        val_X = np.asarray(
            synthetic_X[
                idx["val"]
            ],
            dtype=np.float32,
        )

        val_y = np.asarray(
            synthetic_y[
                idx["val"]
            ],
            dtype=np.int64,
        )

        val_meta = (
            synthetic_meta.iloc[
                idx["val"]
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        test_X = np.asarray(
            synthetic_X[
                idx["test"]
            ],
            dtype=np.float32,
        )

        test_y = np.asarray(
            synthetic_y[
                idx["test"]
            ],
            dtype=np.int64,
        )

        test_meta = (
            synthetic_meta.iloc[
                idx["test"]
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

    else:
        raise ValueError(
            experiment
        )

    return {
        "train_X":
            train_X,

        "train_y":
            train_y,

        "train_meta":
            train_meta,

        "val_X":
            val_X,

        "val_y":
            val_y,

        "val_meta":
            val_meta,

        "test_X":
            test_X,

        "test_y":
            test_y,

        "test_meta":
            test_meta,
    }


def train_one(
    experiment,
    data,
    output_dir,
    *,
    smoke,
):
    seed = (
        20260827
        + 100 * FOLD
        + EXPERIMENTS.index(
            experiment
        )
    )

    set_seed(seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    max_epochs = (
        2
        if smoke
        else FULL_MAX_EPOCHS
    )

    patience = (
        2
        if smoke
        else FULL_PATIENCE
    )

    train_X = data[
        "train_X"
    ]

    train_y = data[
        "train_y"
    ]

    train_meta = data[
        "train_meta"
    ]

    val_X = data[
        "val_X"
    ]

    val_y = data[
        "val_y"
    ]

    val_meta = data[
        "val_meta"
    ]

    test_X = data[
        "test_X"
    ]

    test_y = data[
        "test_y"
    ]

    test_meta = data[
        "test_meta"
    ]

    if smoke:

        train_X, train_y, train_meta = (
            smoke_subset(
                train_X,
                train_y,
                train_meta,
                4096,
                seed + 1,
            )
        )

        val_X, val_y, val_meta = (
            smoke_subset(
                val_X,
                val_y,
                val_meta,
                2048,
                seed + 2,
            )
        )

        test_X, test_y, test_meta = (
            smoke_subset(
                test_X,
                test_y,
                test_meta,
                2048,
                seed + 3,
            )
        )

    for name, y in [
        ("train", train_y),
        ("val", val_y),
        ("test", test_y),
    ]:
        classes = set(
            np.unique(
                y
            ).tolist()
        )

        if classes != {0, 1}:
            raise RuntimeError(
                f"{experiment} {name} missing a class: "
                f"{classes}"
            )

    model = CNN(
        n_features=9,
        n_classes=2,
        config=CNN_CONFIG,
    ).to(device)

    probe = torch.zeros(
        (
            8,
            30,
            9,
        ),
        dtype=torch.float32,
        device=device,
    )

    with torch.no_grad():
        probe_out = model(
            probe
        )

    if tuple(
        probe_out.shape
    ) != (8, 2):
        raise RuntimeError(
            f"Bad model output shape: "
            f"{tuple(probe_out.shape)}"
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WD,
    )

    criterion = nn.CrossEntropyLoss()

    train_loader = make_loader(
        train_X,
        train_y,
        shuffle=True,
        seed=seed,
    )

    val_loader = make_loader(
        val_X,
        val_y,
        shuffle=False,
        seed=seed,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_path = (
        output_dir /
        "best_model.pth"
    )

    history = []

    best_loss = np.inf
    best_epoch = -1
    patience_count = 0

    start_time = time.time()

    for epoch in range(
        max_epochs
    ):

        model.train()

        train_loss_sum = 0.0
        train_n = 0

        for xb, yb in train_loader:

            xb = xb.to(
                device,
                non_blocking=True,
            )

            yb = yb.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                xb
            )

            loss = criterion(
                logits,
                yb,
            )

            loss.backward()

            optimizer.step()

            n = int(
                yb.shape[0]
            )

            train_loss_sum += (
                float(
                    loss.item()
                )
                * n
            )

            train_n += n

        model.eval()

        val_loss_sum = 0.0
        val_n = 0

        with torch.no_grad():

            for xb, yb in val_loader:

                xb = xb.to(
                    device,
                    non_blocking=True,
                )

                yb = yb.to(
                    device,
                    non_blocking=True,
                )

                logits = model(
                    xb
                )

                loss = criterion(
                    logits,
                    yb,
                )

                n = int(
                    yb.shape[0]
                )

                val_loss_sum += (
                    float(
                        loss.item()
                    )
                    * n
                )

                val_n += n

        train_loss = (
            train_loss_sum /
            max(
                1,
                train_n,
            )
        )

        val_loss = (
            val_loss_sum /
            max(
                1,
                val_n,
            )
        )

        history.append({
            "epoch":
                epoch + 1,

            "train_loss":
                train_loss,

            "val_loss":
                val_loss,
        })

        print(
            f"{experiment} "
            f"epoch={epoch + 1:03d} "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f}"
        )

        if val_loss < best_loss:

            best_loss = val_loss
            best_epoch = epoch + 1
            patience_count = 0

            torch.save(
                model.state_dict(),
                best_path,
            )

        else:

            patience_count += 1

        if (
            patience_count
            >= patience
        ):
            break

    if not best_path.exists():
        raise RuntimeError(
            "Best checkpoint was not created."
        )

    state = torch.load(
        best_path,
        map_location=device,
    )

    model.load_state_dict(
        state
    )

    test_metrics, cm, pred, prob = (
        evaluate(
            model,
            test_X,
            test_y,
            device,
        )
    )

    elapsed = (
        time.time()
        - start_time
    )

    history_df = pd.DataFrame(
        history
    )

    history_df.to_csv(
        output_dir /
        "training_history.csv",
        index=False,
    )

    pred_df = (
        test_meta.copy()
        .reset_index(
            drop=True
        )
    )

    pred_df[
        "true_label"
    ] = test_y

    pred_df[
        "predicted_label"
    ] = pred

    pred_df[
        "prob_activity"
    ] = prob[:, 0]

    pred_df[
        "prob_fall"
    ] = prob[:, 1]

    pred_df[
        "fold"
    ] = FOLD

    pred_df[
        "experiment"
    ] = experiment

    pred_df.to_csv(
        output_dir /
        "predictions.csv",
        index=False,
    )

    pd.DataFrame(
        cm,
        index=[
            "true_activity",
            "true_fall",
        ],
        columns=[
            "pred_activity",
            "pred_fall",
        ],
    ).to_csv(
        output_dir /
        "confusion_matrix.csv"
    )

    train_meta.to_csv(
        output_dir /
        "training_window_manifest.csv",
        index=False,
    )

    composition = {
        "train_windows":
            int(len(train_y)),

        "train_activity":
            int(
                (
                    train_y == 0
                ).sum()
            ),

        "train_falling":
            int(
                (
                    train_y == 1
                ).sum()
            ),

        "val_windows":
            int(len(val_y)),

        "val_activity":
            int(
                (
                    val_y == 0
                ).sum()
            ),

        "val_falling":
            int(
                (
                    val_y == 1
                ).sum()
            ),

        "test_windows":
            int(len(test_y)),

        "test_activity":
            int(
                (
                    test_y == 0
                ).sum()
            ),

        "test_falling":
            int(
                (
                    test_y == 1
                ).sum()
            ),
    }

    if (
        "training_source"
        in train_meta.columns
    ):
        composition[
            "training_source_counts"
        ] = (
            train_meta[
                "training_source"
            ]
            .value_counts()
            .to_dict()
        )

    run_config = {
        "experiment":
            experiment,

        "fold":
            FOLD,

        "smoke":
            bool(smoke),

        "seed":
            seed,

        "device":
            str(device),

        "cnn_signature":
            str(
                inspect.signature(
                    CNN
                )
            ),

        "cnn_config":
            CNN_CONFIG,

        "window_samples":
            30,

        "features":
            9,

        "classes":
            2,

        "batch_size":
            BATCH_SIZE,

        "max_epochs":
            max_epochs,

        "patience":
            patience,

        "optimizer":
            "Adam",

        "learning_rate":
            LR,

        "loss":
            "unweighted CrossEntropyLoss",

        "checkpoint_selection":
            "minimum validation loss",

        "checkpoint_reloaded_before_test":
            True,

        "elapsed_seconds":
            elapsed,

        "best_epoch":
            int(best_epoch),

        "best_val_loss":
            float(best_loss),

        "composition":
            composition,
    }

    with open(
        output_dir /
        "run_config.json",
        "w",
    ) as f:
        json.dump(
            run_config,
            f,
            indent=2,
        )

    with open(
        output_dir /
        "metrics.json",
        "w",
    ) as f:
        json.dump(
            test_metrics,
            f,
            indent=2,
        )

    pd.DataFrame(
        [
            {
                "experiment":
                    experiment,
                "fold":
                    FOLD,
                **test_metrics,
            }
        ]
    ).to_csv(
        output_dir /
        "metrics.csv",
        index=False,
    )

    fig = plt.figure(
        figsize=(6, 4)
    )

    plt.plot(
        history_df[
            "epoch"
        ],
        history_df[
            "train_loss"
        ],
        label="train",
    )

    plt.plot(
        history_df[
            "epoch"
        ],
        history_df[
            "val_loss"
        ],
        label="validation",
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Cross-entropy loss"
    )

    plt.legend()

    plt.tight_layout()

    fig.savefig(
        output_dir /
        "training_history.png",
        dpi=160,
    )

    plt.close(
        fig
    )

    fig = plt.figure(
        figsize=(4.5, 4)
    )

    plt.imshow(
        cm
    )

    plt.xticks(
        [0, 1],
        [
            "Activity",
            "Fall",
        ],
    )

    plt.yticks(
        [0, 1],
        [
            "Activity",
            "Fall",
        ],
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "True"
    )

    for i in range(2):
        for j in range(2):
            plt.text(
                j,
                i,
                str(
                    cm[i, j]
                ),
                ha="center",
                va="center",
            )

    plt.tight_layout()

    fig.savefig(
        output_dir /
        "confusion_matrix.png",
        dpi=160,
    )

    plt.close(
        fig
    )

    return {
        "experiment":
            experiment,

        "best_epoch":
            int(best_epoch),

        "best_val_loss":
            float(best_loss),

        "elapsed_seconds":
            float(elapsed),

        **test_metrics,
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--smoke",
        action="store_true",
    )

    args = parser.parse_args()

    mode = (
        "smoke"
        if args.smoke
        else "full_pilot"
    )

    output_root = (
        PROJECT /
        "outputs" /
        (
            "phase2_protechto_parity_fold0_smoke_v1"
            if args.smoke
            else
            "phase2_protechto_parity_fold0_v1"
        )
    )

    print("=" * 88)
    print(
        f"CORRECTED PHASE-2 FINAL CAMPAIGN | "
        f"{EXPERIMENT_NAME} | FOLD {FOLD}"
    )
    print("=" * 88)

    print(
        "Mode:",
        mode
    )

    print(
        "Torch:",
        torch.__version__
    )

    print(
        "CUDA available:",
        torch.cuda.is_available()
    )

    print(
        "Device count:",
        torch.cuda.device_count()
    )

    print(
        "CNN:",
        CNN
    )

    print(
        "CNN signature:",
        inspect.signature(
            CNN
        )
    )

    sources = load_sources()

    results = []

    for experiment in EXPERIMENTS:

        print()
        print("=" * 88)
        print(experiment)
        print("=" * 88)

        data = build_experiment(
            experiment,
            sources,
        )

        result = train_one(
            experiment,
            data,
            output_root /
            experiment /
            f"fold_{FOLD}",
            smoke=args.smoke,
        )

        results.append(
            result
        )

        print()
        print(
            "RESULT:",
            json.dumps(
                result,
                indent=2,
            )
        )

    result_df = pd.DataFrame(
        results
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df.to_csv(
        output_root /
        f"{EXPERIMENT_NAME}_fold_{FOLD}_summary.csv",
        index=False,
    )

    print()
    print("=" * 88)
    print("PILOT SUMMARY")
    print("=" * 88)

    print(
        result_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    required = []

    for experiment in EXPERIMENTS:

        d = (
            output_root /
            experiment /
            f"fold_{FOLD}"
        )

        required.extend(
            [
                d /
                "best_model.pth",

                d /
                "metrics.json",

                d /
                "confusion_matrix.csv",

                d /
                "predictions.csv",

                d /
                "training_history.csv",

                d /
                "run_config.json",
            ]
        )

    missing = [
        str(x)
        for x in required
        if not x.exists()
    ]

    print()
    print("=" * 88)

    if missing:

        print(
            "PILOT ARTIFACT GATE: FAIL"
        )

        for x in missing:
            print(
                "MISSING:",
                x
            )

    else:

        print(
            "PILOT ARTIFACT GATE: PASS"
        )

    print("=" * 88)

    print()
    print(
        "Output:",
        output_root
    )


if __name__ == "__main__":
    main()
