#!/usr/bin/env python3
"""
Phase-2 CNN Simulation Validation Campaign

Purpose:
Evaluate whether Extended35 simulated fall data improves
CNN fall detection performance when added to real KFall +
UniVRFall datasets.

Experiments:

EXP01:
    Real only

EXP02:
    Simulation only

EXP03:
    Real + 20% simulation

EXP04:
    Real + 50% simulation

EXP05:
    Real + 70% simulation

EXP06:
    Real + 100% simulation


Protocol:
- Protechto CNN architecture
- 9 input channels
- 30 sample windows
- subject-independent 5-fold evaluation
- no VAE/GAN augmentation
- real-only test sets

"""

import os
import json
import time
import random
import shutil
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


# ==========================================================
# PATH CONFIGURATION
# ==========================================================


MUJOCO_ROOT = Path(
    "/mnt/hdd16T/ToqeerHomeBackup/mujoco_project"
)


PROTECHTO_ROOT = Path(
    "/mnt/hdd16T/ToqeerHomeBackup/toqeer/Protechto_master"
)


OUTPUT_ROOT = (
    MUJOCO_ROOT
    /
    "outputs"
    /
    "phase2_pilot_EXP02_EXP03"
)


OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


EXTENDED35_ROOT = (
    MUJOCO_ROOT
    /
    "outputs"
    /
    "_highrate_overnight"
    /
    "campaign_highrate_truth_v2_extended35"
)


REAL_DATA_ROOT = Path(
    "/mnt/hdd16T/ToqeerHomeBackup/toqeer/uniVr-dataset"
)


KFALL_ROOT = (
    REAL_DATA_ROOT
    /
    "KFall_oriented"
)


UNIVR_ROOT = (
    REAL_DATA_ROOT
    /
    "UniVrFall_Dataset"
)


# ==========================================================
# CNN CONFIGURATION
# ==========================================================


WINDOW_SIZE = 30

BATCH_SIZE = 64

MAX_EPOCHS = 10

PATIENCE = 5


N_FEATURES = 9

N_CLASSES = 2


RANDOM_SEED = 42


EXPERIMENTS = {

    "EXP01_REAL_ONLY":
    {
        "real_fraction":1.0,
        "sim_fraction":0.0
    },

    "EXP02_SIM_ONLY":
    {
        "real_fraction":0.0,
        "sim_fraction":1.0
    },

    "EXP03_MIX20":
    {
        "real_fraction":1.0,
        "sim_fraction":0.2
    }

}


# ==========================================================
# RANDOM CONTROL
# ==========================================================


def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)

    except Exception:
        pass



set_seed(RANDOM_SEED)



# ==========================================================
# LOGGING
# ==========================================================


LOG_FILE = (
    OUTPUT_ROOT
    /
    "phase2_campaign.log"
)


def log(msg):

    text = (
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
        f"{msg}"
    )

    print(text)

    with open(
        LOG_FILE,
        "a"
    ) as f:

        f.write(text+"\n")



# ==========================================================
# SAVE CONFIG
# ==========================================================


CONFIG_SAVE = (
    OUTPUT_ROOT
    /
    "experiment_config.json"
)


with open(CONFIG_SAVE,"w") as f:

    json.dump(
        {
            "window_size":WINDOW_SIZE,
            "batch_size":BATCH_SIZE,
            "epochs":MAX_EPOCHS,
            "patience":PATIENCE,
            "experiments":EXPERIMENTS
        },
        f,
        indent=4
    )


log("Phase-2 configuration initialized")
# ==========================================================
# DATASET PREPARATION
# ==========================================================

import torch


def convert_extended35_file(csv_file):
    """
    Convert Extended35 high-rate truth file
    into Protechto CNN 9-channel format.

    Output:
        [AccX,AccY,AccZ,
         GyrX,GyrY,GyrZ,
         EulerX,EulerY,EulerZ]
    """

    df = pd.read_csv(
        csv_file,
        comment="#"
    )


    x = np.column_stack(
        [
            df["accel_true_x"].values,
            df["accel_true_y"].values,
            df["accel_true_z"].values,

            df["gyro_true_x"].values * 57.295779513,
            df["gyro_true_y"].values * 57.295779513,
            df["gyro_true_z"].values * 57.295779513,

            np.zeros(len(df)),
            np.zeros(len(df)),
            np.zeros(len(df)),

        ]
    )


    return x



def create_windows(
    signal,
    label,
    window=30
):
    """
    Same window length as CNN.
    """

    X=[]
    y=[]


    for i in range(
        0,
        len(signal)-window+1
    ):

        segment = signal[
            i:i+window
        ]

        X.append(segment)
        y.append(label)


    return (
        np.array(X),
        np.array(y)
    )



def prepare_extended35_dataset():

    log(
        "Preparing Extended35 dataset"
    )


    files=list(
        (
            EXTENDED35_ROOT
            /
            "runs"
        )
        .rglob(
            "*_highrate_truth.csv"
        )
    )


    all_x=[]
    all_y=[]
    subjects=[]


    log(
        f"Extended35 files found: {len(files)}"
    )


    for f in files:


        try:

            x = convert_extended35_file(
                f
            )


            # scenario id from path
            path=str(f)


            if "task_" in path:

                task=int(
                    path.split("task_")[1]
                    .split("/")[0]
                )

            else:

                task=-1


            label = (
                1
                if task in [
                    20,21,22,23,24,
                    25,26,27,28,29,
                    30,31,32,33,34,
                    37,38,39,40,41,42
                ]
                else 0
            )


            Xw,Yw=create_windows(
                x,
                label,
                WINDOW_SIZE
            )


            all_x.append(
                Xw
            )

            all_y.append(
                Yw
            )


            subjects.append(
                f.parts[-4]
            )


        except Exception as e:

            log(
                f"Extended35 failed {f}: {e}"
            )


    X=np.concatenate(
        all_x,
        axis=0
    )

    y=np.concatenate(
        all_y,
        axis=0
    )


    out=(
        OUTPUT_ROOT
        /
        "dataset_cache"
        /
        "Extended35"
    )

    out.mkdir(
        parents=True,
        exist_ok=True
    )


    np.save(
        out/"X.npy",
        X
    )

    np.save(
        out/"y.npy",
        y
    )


    pd.DataFrame(
        {
            "samples":[len(X)],
            "falls":[int(y.sum())]
        }
    ).to_csv(
        out/"dataset_summary.csv",
        index=False
    )


    log(
        f"Extended35 windows: {X.shape}"
    )


    return X,y



# ==========================================================
# CHECK CACHE OR BUILD
# ==========================================================


EXT_CACHE = (
    OUTPUT_ROOT
    /
    "dataset_cache"
    /
    "Extended35"
    /
    "X.npy"
)


if EXT_CACHE.exists():

    log(
        "Loading Extended35 cache"
    )

    EXT_X=np.load(
        EXT_CACHE
    )

    EXT_Y=np.load(
        EXT_CACHE.parent/"y.npy"
    )


else:

    EXT_X,EXT_Y = (
        prepare_extended35_dataset()
    )


# Simulation subject groups for GroupKFold
EXT_GROUP = np.array(
    [
        f"SIM_{i}"
        for i in range(len(EXT_Y))
    ]
)

# ==========================================================
# CNN TRAINING ENGINE
# ==========================================================


import torch.nn as nn
from torch.utils.data import (
    TensorDataset,
    DataLoader
)

from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# Add Protechto path

import sys

if str(PROTECHTO_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROTECHTO_ROOT)
    )


import sys

sys.path.insert(
    0,
    "/mnt/hdd16T/ToqeerHomeBackup/toqeer/Protechto_master"
)

from models.CNN import CNN



try:

    from models.configs.CNN_config import config as CNN_CONFIG

except Exception:

    CNN_CONFIG = {

        "conv_1_dim":32,
        "conv_1_filter":3,
        "conv_pool":2,
        "conv_dropout":0.2,
        "fc":128,
        "fc_dropout":0.2

    }



# ==========================================================
# MODEL CREATION
# ==========================================================


def create_model():

    model=CNN(
        n_features=N_FEATURES,
        n_classes=N_CLASSES,
        config=CNN_CONFIG
    )

    return model



# ==========================================================
# DATASET MIXING
# ==========================================================


def create_training_dataset(
        real_X,
        real_y,
        sim_fraction
):

    """
    Add Extended35 samples.

    sim_fraction:

        0.0
        0.2
        0.5
        0.7
        1.0

    """

    if sim_fraction == 0:

        return (
            real_X,
            real_y
        )


    n_sim=int(
        len(real_X)
        *
        sim_fraction
    )


    if n_sim > len(EXT_X):

        idx=np.random.choice(
            len(EXT_X),
            n_sim,
            replace=True
        )

    else:

        idx=np.random.choice(
            len(EXT_X),
            n_sim,
            replace=False
        )


    sim_x=EXT_X[idx]

    sim_y=EXT_Y[idx]


    X=np.concatenate(
        [
            real_X,
            sim_x
        ],
        axis=0
    )


    y=np.concatenate(
        [
            real_y,
            sim_y
        ],
        axis=0
    )


    order=np.random.permutation(
        len(y)
    )


    return (
        X[order],
        y[order]
    )



# ==========================================================
# TRAINING FUNCTION
# ==========================================================


def train_one_fold(
        train_X,
        train_y,
        val_X,
        val_y,
        save_dir
):


    save_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    model=create_model()


    device=(
        "cuda"
        if torch.cuda.is_available()
        else
        "cpu"
    )


    model.to(device)


    train_loader=DataLoader(

        TensorDataset(

            torch.tensor(
                train_X,
                dtype=torch.float32
            ),

            torch.tensor(
                train_y,
                dtype=torch.long
            )

        ),

        batch_size=BATCH_SIZE,
        shuffle=True

    )


    val_loader=DataLoader(

        TensorDataset(

            torch.tensor(
                val_X,
                dtype=torch.float32
            ),

            torch.tensor(
                val_y,
                dtype=torch.long
            )

        ),

        batch_size=BATCH_SIZE,
        shuffle=False

    )


    optimizer=torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )


    criterion=nn.CrossEntropyLoss()


    best_loss=np.inf

    patience_count=0


    history=[]


    for epoch in range(MAX_EPOCHS):


        model.train()

        train_loss=0


        for xb,yb in train_loader:

            xb=xb.to(device)

            yb=yb.to(device)


            optimizer.zero_grad()


            out=model(xb)


            loss=criterion(
                out,
                yb
            )


            loss.backward()

            optimizer.step()


            train_loss += (
                loss.item()
            )


        model.eval()


        val_loss=0


        with torch.no_grad():

            for xb,yb in val_loader:

                xb=xb.to(device)

                yb=yb.to(device)


                out=model(xb)


                loss=criterion(
                    out,
                    yb
                )

                val_loss += loss.item()



        val_loss /= max(
            len(val_loader),
            1
        )


        history.append(
            {
                "epoch":epoch,
                "train_loss":train_loss,
                "val_loss":val_loss
            }
        )


        status_dir = save_dir.parent / "status"
        status_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        (status_dir / "current_epoch.txt").write_text(
            str(epoch+1)
        )

        (status_dir / "status.txt").write_text(
            f"epoch={epoch+1}/{MAX_EPOCHS}\nval_loss={val_loss}"
        )


        if val_loss < best_loss:


            best_loss=val_loss

            patience_count=0


            torch.save(

                model.state_dict(),

                save_dir
                /
                "best_model.pth"

            )


        else:

            patience_count +=1


        if patience_count >= PATIENCE:

            break



    pd.DataFrame(
        history
    ).to_csv(

        save_dir
        /
        "training_history.csv",

        index=False

    )


    return model



# ==========================================================
# EVALUATION
# ==========================================================


def evaluate_model(
        model,
        X,
        y,
        save_dir
):


    device=next(
        model.parameters()
    ).device


    model.eval()


    loader=DataLoader(

        TensorDataset(

            torch.tensor(
                X,
                dtype=torch.float32
            ),

            torch.tensor(
                y,
                dtype=torch.long
            )

        ),

        batch_size=BATCH_SIZE

    )


    preds=[]


    with torch.no_grad():

        for xb,yb in loader:

            xb=xb.to(device)

            out=model(xb)

            p=torch.argmax(
                out,
                dim=1
            )


            preds.extend(
                p.cpu().numpy()
            )


    result={

        "accuracy":
        accuracy_score(
            y,
            preds
        ),

        "precision":
        precision_score(
            y,
            preds,
            zero_division=0
        ),

        "recall":
        recall_score(
            y,
            preds,
            zero_division=0
        ),

        "f1":
        f1_score(
            y,
            preds,
            zero_division=0
        )

    }


    pd.DataFrame(
        [result]
    ).to_csv(

        save_dir
        /
        "metrics.csv",

        index=False

    )


    np.savetxt(

        save_dir
        /
        "confusion_matrix.csv",

        confusion_matrix(
            y,
            preds
        ),

        fmt="%d"

    )


    return result
# ==========================================================
# REAL DATASET PREPARATION
# ==========================================================


def load_real_dataset():

    """
    Load KFall + UniVRFall.

    Creates unified CNN format:

    [AccX AccY AccZ
     GyrX GyrY GyrZ
     EulerX EulerY EulerZ]

    """

    log(
        "Preparing real datasets"
    )


    X_all=[]
    y_all=[]
    group_all=[]


    # ------------------------------
    # KFall
    # ------------------------------

    kfiles=list(
        (
            KFALL_ROOT
            /
            "sensors_data"
        )
        .rglob(
            "*.csv"
        )
    )


    log(
        f"KFall files: {len(kfiles)}"
    )


    for f in kfiles:


        try:

            df=pd.read_csv(
                f
            )


            if "Fall Type" not in df.columns:
                continue


            signal=np.column_stack(
                [

                df["AccX"].values * 0.00981,
                df["AccY"].values * 0.00981,
                df["AccZ"].values * 0.00981,

                df["GyrX"].values * 0.07,
                df["GyrY"].values * 0.07,
                df["GyrZ"].values * 0.07,

                np.zeros(len(df)),
                np.zeros(len(df)),
                np.zeros(len(df)),

                ]
            )


            label=(
                1
                if str(
                    df["Fall Type"].iloc[0]
                ).startswith("F")
                else 0
            )


            Xw,Yw=create_windows(
                signal,
                label,
                WINDOW_SIZE
            )


            X_all.append(Xw)
            y_all.append(Yw)
            group_all.extend([str(f.parts[-2])] * len(Yw))



        except Exception:

            continue



    # ------------------------------
    # UniVRFall
    # ------------------------------

    ufiles=list(
        (
            UNIVR_ROOT
            /
            "sensors_data"
        )
        .rglob(
            "*.csv"
        )
    )


    log(
        f"UniVRFall files: {len(ufiles)}"
    )


    for f in ufiles:


        try:

            df=pd.read_csv(
                f
            )


            signal=np.column_stack(
                [

                df["AccX"].values * 0.00981,
                df["AccY"].values * 0.00981,
                df["AccZ"].values * 0.00981,

                df["GyrX"].values * 0.07,
                df["GyrY"].values * 0.07,
                df["GyrZ"].values * 0.07,

                np.zeros(len(df)),
                np.zeros(len(df)),
                np.zeros(len(df)),

                ]
            )


            # UniVR fall files are all falls
            label=1


            Xw,Yw=create_windows(
                signal,
                label,
                WINDOW_SIZE
            )


            X_all.append(Xw)
            y_all.append(Yw)
            group_all.extend([str(f.parts[-2])] * len(Yw))



        except Exception:

            continue



    X=np.concatenate(
        X_all,
        axis=0
    )

    y=np.concatenate(
        y_all,
        axis=0
    )


    log(
        f"Real dataset windows {X.shape}"
    )


    return X,y,np.array(group_all)



# ==========================================================
# EXPERIMENT EXECUTION
# ==========================================================


def run_experiment(
        name,
        sim_fraction,
        base_X,
        base_y,
        base_group
):


    log(
        f"START {name}"
    )


    exp_dir=(
        OUTPUT_ROOT
        /
        "experiments"
        /
        name
    )


    exp_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    results=[]


    kfold=GroupKFold(
        n_splits=5
    )


    indices=np.arange(
        len(base_y)
    )


    for fold,(train_idx,test_idx) in enumerate(
        kfold.split(
            indices,
            groups=base_group
        )
    ):


        log(
            f"{name} fold {fold}"
        )


        fold_dir=(
            exp_dir
            /
            f"fold_{fold}"
        )


        fold_dir.mkdir(
            exist_ok=True
        )


        train_X=real_X[
            train_idx
        ]

        train_y=real_y[
            train_idx
        ]


        test_X=real_X[
            test_idx
        ]

        test_y=real_y[
            test_idx
        ]



        train_X,train_y=create_training_dataset(
            train_X,
            train_y,
            sim_fraction
        )



        # validation split

        cut=int(
            len(train_y)*0.8
        )


        val_X=train_X[cut:]

        val_y=train_y[cut:]


        train_X=train_X[:cut]

        train_y=train_y[:cut]



        model=train_one_fold(
            train_X,
            train_y,
            val_X,
            val_y,
            fold_dir
        )


        metric=evaluate_model(
            model,
            test_X,
            test_y,
            fold_dir
        )


        metric["experiment"]=name

        metric["fold"]=fold


        results.append(
            metric
        )



    return pd.DataFrame(
        results
    )



# ==========================================================
# MAIN PIPELINE
# ==========================================================


if __name__=="__main__":


    log(
        "PHASE 2 CNN CAMPAIGN START"
    )


    REAL_X,REAL_Y,REAL_GROUP = load_real_dataset()



    all_results=[]


    for exp,cfg in EXPERIMENTS.items():


        if exp=="EXP02_SIM_ONLY":


            # simulation only

            rx=np.zeros_like(
                EXT_X
            )

            ry=np.zeros_like(
                EXT_Y
            )


            df=run_experiment(
                exp,
                1.0,
                EXT_X,
                EXT_Y,
                EXT_GROUP
            )


        else:


            df=run_experiment(
                exp,
                cfg["sim_fraction"],
                REAL_X,
                REAL_Y,
                REAL_GROUP
            )


        all_results.append(
            df
        )



    final=pd.concat(
        all_results,
        ignore_index=True
    )


    final_dir=(
        OUTPUT_ROOT
        /
        "FINAL_RESULTS"
    )

    final_dir.mkdir(
        exist_ok=True
    )


    final.to_csv(
        final_dir
        /
        "CNN_segment_results.csv",
        index=False
    )


    log(
        "PHASE 2 COMPLETE"
    )
