import torch
import torch.nn as nn 
import os
import lightning as pl
from torchmetrics.functional import accuracy, precision, recall, f1_score
import numpy as np 
from models.helper import generate_confusion_matrix
import json
with open("config.json", "r") as config_file:
    settings = json.load(config_file)


class Predictor(pl.LightningModule):

    def __init__(self, model, n_features: int, n_classes: int, classes, experiment, cfg: dict, fold=None):
        super().__init__()
        self.save_hyperparameters()
        self.n_features = n_features
        self.n_classes = n_classes
        self.classes = classes
        self.fold = fold
        self.cfg = cfg

        # Model
        self.model = model(n_features=n_features, n_classes=n_classes, config=self.cfg)
        self.model_name = self.model.get_name()

        # ------------------------------------------------------------------
        if "class_weights" in self.cfg:               # list or np array
            w = torch.tensor(self.cfg["class_weights"], dtype=torch.float32)
            # register as buffer ? automatically moved to GPU with the module
            self.register_buffer("ce_weights", w)
            self.criterion = nn.CrossEntropyLoss(weight=self.ce_weights)
            print(f"[Predictor] using CrossEntropyLoss with weights = {self.ce_weights.tolist()}")
        else:
            # fall back to whatever was already in the tuning config
            self.criterion = self.cfg["criterion"]
        # ------------------------------------------------------------------

        # Loss function
        #self.criterion = self.cfg["criterion"] # FocalLoss(gamma=2, alpha=[0.75, 0.25]) # nn.CrossEntropyLoss()

        # History validation
        self.val_y_pred = []
        self.val_y = []

        # History test
        self.test_y_pred = []
        self.test_y = []

        # Save results to
        self.output_dir = f"{settings['results']}/{self.model_name}/{experiment}"

    def forward(self, x, y=None):
        y_hat = self.model(x)
        loss = 0
        if y is not None:
            loss = self.criterion(y_hat, y)

        return loss, y_hat

    def training_step(self, batch, batch_idx):
        # Data and labels form the batch
        x, y = batch

        loss, y_hat = self(x, y)
        y_pred = torch.argmax(y_hat, dim=1)

        # Evaluation metrics
        step_accuracy = accuracy(
            y_pred, y, num_classes=self.n_classes, task='multiclass', average='macro', threshold=0.5
        )

        # Logging
        self.log('train_loss', loss, prog_bar=True, logger=True)
        self.log('train_accuracy', step_accuracy, prog_bar=True, logger=True)

        return {'loss': loss, 'accuracy': step_accuracy}

    def validation_step(self, batch, batch_idx):
        # Data and labels form the batch
        x, y = batch

        loss, y_hat = self(x, y)
        y_pred = torch.argmax(y_hat, dim=1)

        # Evaluation metrics
        step_accuracy = accuracy(
            y_pred, y, num_classes=self.n_classes , task='multiclass', average='macro', threshold=0.5
        )

        # Logging
        self.log('val_loss', loss, prog_bar=True, logger=True)
        self.log('val_accuracy', step_accuracy, prog_bar=True, logger=True)

        # Save
        self.val_y_pred.append(y_pred)
        self.val_y.append(y)

        return {'loss': loss, 'accuracy': step_accuracy}

    def on_validation_epoch_end(self):
        val_y_pred = torch.concat(self.val_y_pred)
        val_y = torch.concat(self.val_y)

        # Metrics
        val_accuracy = accuracy(
            val_y_pred, val_y, task="multiclass", num_classes=self.n_classes, average='macro', threshold=0.5
        )
        val_precision = precision(
            val_y_pred, val_y, task="multiclass", num_classes=self.n_classes, average='macro', threshold=0.5
        )
        val_recall = recall(
            val_y_pred, val_y, task="multiclass", num_classes=self.n_classes, average='macro', threshold=0.5
        )
        val_f1 = f1_score(
            val_y_pred, val_y, task="multiclass", num_classes=self.n_classes, average='macro', threshold=0.5
        )

        # Log
        self.log('val_accuracy', val_accuracy, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_precision', val_precision, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_recall', val_recall, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_f1', val_f1, prog_bar=True, logger=True, sync_dist=True)

        # Clear stored values for the next epoch
        self.val_y_pred.clear()
        self.val_y.clear()

    def test_step(self, batch, batch_idx):
        # Data and labels form the batch
        x, y = batch

        loss, y_hat = self(x, y)
        y_pred = torch.argmax(y_hat, dim=1)

        # Evaluation metrics
        step_accuracy = accuracy(
            y_pred, y, num_classes=self.n_classes, task='multiclass', average='macro', threshold=0.5
        )

        # Logging
        self.log('test_loss', loss, prog_bar=True, logger=True)
        self.log('test_accuracy', step_accuracy, prog_bar=True, logger=True)

        # Save
        self.test_y_pred.append(y_pred)
        self.test_y.append(y)

        return {'loss': loss, 'accuracy': step_accuracy}

    def on_test_epoch_end(self):
        test_y_pred = torch.concat(self.test_y_pred)
        test_y = torch.concat(self.test_y)

        # Metrics
        test_accuracy = accuracy(
            test_y_pred, test_y, task="multiclass", num_classes=self.n_classes, average='macro', threshold=0.5
        )
        test_precision = precision(
            test_y_pred, test_y, task="multiclass", num_classes=self.n_classes, average='macro', threshold=0.5
        )
        test_recall = recall(
            test_y_pred, test_y, task="multiclass", num_classes=self.n_classes, average='macro', threshold=0.5
        )
        test_f1 = f1_score(
            test_y_pred, test_y, task="multiclass", num_classes=self.n_classes, average='macro', threshold=0.5
        )

        # Log
        self.log('test_accuracy', test_accuracy, prog_bar=True, logger=True, sync_dist=True)
        self.log('test_precision', test_precision, prog_bar=True, logger=True, sync_dist=True)
        self.log('test_recall', test_recall, prog_bar=True, logger=True, sync_dist=True)
        self.log('test_f1', test_f1, prog_bar=True, logger=True, sync_dist=True)

        generate_confusion_matrix(
            test_y_pred.detach().cpu().numpy(),
            test_y.detach().cpu().numpy(),
            f"{self.output_dir}{f'/{self.fold}' if self.fold else ''}/{self.model_name}_confusion_matrix",
            self.classes,
            specific_title=f"{self.model_name} {f'fold {self.fold}' if self.fold else ''}",
        )
        
        fold_dir = f"{self.output_dir}/{self.fold if self.fold is not None else 'single'}"
        os.makedirs(fold_dir, exist_ok=True)
        np.save(os.path.join(fold_dir, "y_pred.npy"), test_y_pred.cpu().numpy())
        np.save(os.path.join(fold_dir, "y_true.npy"), test_y.cpu().numpy()) 
     
        # Clear stored values for the next epoch
        self.test_y_pred.clear()
        self.test_y.clear()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.cfg["lr"], weight_decay=self.cfg["wd"])
