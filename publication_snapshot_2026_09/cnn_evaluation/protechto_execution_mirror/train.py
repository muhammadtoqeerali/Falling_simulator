#!/usr/bin/env python3
import datetime
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from dataloaders.KFoldDataloader import KFoldDataloader
from dataloaders.TrainTestDataloader import TrainTestDataloader
from models import *
from models.wrappers import Predictor
import argparse
import json

with open("config.json", "r") as config_file:
    settings = json.load(config_file)

torch.set_float32_matmul_precision('high')


def train(model_module, dataset_path, gpu, config):
    now = str(datetime.datetime.now()).replace(" ", "_").replace(":", "_").split(".")[0]

    # 1) prepare the train/test split
    dataloader = TrainTestDataloader(
        root_directory=f"{settings['data']}/{dataset_path}",
        test_size=0.2,
        batch_size=64
    )
    dataloader.setup(verbose=True)

    # ---------------------------- NEW ----------------------------
    # grab the per-class weights from the dataloader and stick them into cfg
    #config["class_weights"] = dataloader.class_weights.tolist()
    # --------------------------------------------------------------

    model = Predictor(
        model_module,
        n_features=dataloader.input_shape[1],
        n_classes=dataloader.label_encoder.classes_.shape[0],
        classes=dataloader.label_encoder.classes_,
        experiment=f"{dataloader.input_shape[0] * 10}ms/{now}",
        fold="train",
        cfg=config,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{settings['root']}/checkpoints/{model.model_name}/{dataloader.input_shape[0] * 10}ms/{now}",
        filename="best-checkpoint",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=20
    )
    logger = TensorBoardLogger("lightning_logs", name="UniVrFall")

    trainer = Trainer(
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        max_epochs=100,
        accelerator="gpu",
        devices=[gpu],
        enable_progress_bar=True
    )

    trainer.fit(model, dataloader)
    trainer.test(model, dataloaders=dataloader.test_dl)


def k_fold_train(model_module, dataset_path, gpu, config):
    now = str(datetime.datetime.now()).replace(" ", "_").replace(":", "_").split(".")[0]
    k = 5

    dataloader = KFoldDataloader(
        root_directory=f"{settings['data']}/{dataset_path}",
        k=k,
        batch_size=64
    )

    for fold in range(k):
        print(f"Fold: {fold + 1}")

        # 1) build the fold's train/val/test
        dataloader.construct(fold_index=fold, verbose=True)

        # ---------------------------- NEW ----------------------------
        # grab the per-class weights from the dataloader and stick them into cfg
        #config["class_weights"] = dataloader.class_weights.tolist()
        # --------------------------------------------------------------

        model = Predictor(
            model_module,
            n_features=dataloader.input_shape[1],
            n_classes=dataloader.label_encoder.classes_.shape[0],
            classes=dataloader.label_encoder.classes_,
            experiment=f"{dataloader.input_shape[0] * 10}ms/{now}",
            fold=fold + 1,
            cfg=config,
        )

        checkpoint_callback = ModelCheckpoint(
            dirpath=f"{settings['root']}/checkpoints/{model.model_name}/{dataloader.input_shape[0] * 10}ms/{now}",
            filename=f"best-checkpoint-fold_{fold + 1}",
            save_top_k=1,
            verbose=True,
            monitor="val_loss",
            mode="min",
        )
        early_stopping_callback = EarlyStopping(
            monitor="val_loss",
            patience=20
        )
        logger = TensorBoardLogger("lightning_logs", name="UniVrFall")

        trainer = Trainer(
            logger=logger,
            callbacks=[checkpoint_callback, early_stopping_callback],
            max_epochs=100,
            accelerator="gpu",
            devices=[gpu],
            enable_progress_bar=True
        )

        trainer.fit(model, dataloader)
        trainer.test(model, dataloaders=dataloader.test_dl)


def main():
    train_techniques = {
        "k-fold": k_fold_train,
        "train": train
    }
    parser = argparse.ArgumentParser(
        description="python train.py -m CNN -t k-fold -d /path/to/dataset -g 0"
    )
    parser.add_argument('-m', '--model', type=str, required=True)
    parser.add_argument('-t', '--technique',
                        choices=["k-fold", "train"], required=True)
    parser.add_argument('-d', '--dataset', type=str, required=True)
    parser.add_argument('-g', '--gpu', type=int, required=True)

    args = parser.parse_args()
    print(f"Training of {args.model} using {args.technique} on {args.dataset}")
    print(f"Using configuration:\n{globals()[f'{args.model}_cfg']}")

    train_techniques[args.technique](
        model_module=globals()[args.model],
        dataset_path=args.dataset,
        gpu=args.gpu,
        config=globals()[f"{args.model}_cfg"]
    )


if __name__ == '__main__':
    main()
