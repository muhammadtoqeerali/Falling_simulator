# -*- coding: utf-8 -*-
"""models/wrappers/Simulator.py - emits per-fold predictions so that test.py
can build a global confusion matrix. ASCII-only file to avoid encoding issues.
"""

import os
import time
import pdfkit
import json
import numpy as np
import pandas as pd
import torch
import lightning as pl
import constants as const

from simulation.helper import (
    extract_from_batch,
    is_simulation_passed_threshold,
    process_results,
)
from models.helper import generate_confusion_matrix

with open("config.json", "r") as cf:
    settings = json.load(cf)


class Simulator(pl.LightningModule):

    def __init__(
        self,
        model,
        n_features: int,
        n_classes: int,
        classes,
        best_checkpoint_path: str,
        cfg: dict,
        fold: int | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.n_features = n_features
        self.n_classes = n_classes
        self.classes = classes
        self.cfg = cfg
        self.fold = fold

        # ---------------- model -----------------
        self.model = model(n_features=n_features, n_classes=n_classes, config=cfg)
        ckpt = torch.load(best_checkpoint_path, map_location="cpu", weights_only=False)
        self.load_state_dict(ckpt["state_dict"], strict=False)
        self.model.eval()
        self.model_name = self.model.get_name()

        self.criterion = cfg.get("criterion")         

        # ---------------- logging ---------------
        self.results: list[dict] = []
        self.test_y_pred: list[torch.Tensor] = []
        self.test_y: list[torch.Tensor] = []
        self.test_tasks: list[torch.Tensor] = []   # NEW for segments-analyses

        rel = "/".join(best_checkpoint_path.split(os.sep)[len(settings["root"].split(os.sep)) + 1 : -1])
        self.output_dir = os.path.join(settings["results"], rel)

    # -------------------------------------------------------
    def forward(self, x, y=None):
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y) if y is not None else 0
        return loss, y_hat

    # -------------------------------------------------------
    def test_step(self, batch, batch_idx):
        subject, task, trial, x, y = extract_from_batch(batch)
        if x.shape[0] == 0:
            return {"loss": 0}

        loss, y_hat = self(x, y)
        y_pred = self.get_output(
            torch.nn.functional.softmax(y_hat, dim=1).detach().cpu().numpy(),
            prediction_bias=self.cfg["prediction_bias"]
        )
        y_pred_filtered = y_pred
        passed = is_simulation_passed_threshold(y, y_pred)

        # ------------------------------------------------------------------
        # event-level prediction derived from the SAME rule you already
        # evaluate with `passed` (so it matches event_stats__bias_0.6.*).
        true_event = 1 if task in const.FALL_TASKS else 0
        predicted_event = true_event if passed else 1 - true_event

        self.results.append(
            {
                "subject": subject,
                "task": task,
                "trial": trial + 1,
                "passed": passed,
                "pred_event": predicted_event,   # NEW ? for confusion-matrix

            }
        )

        self.test_y_pred.append(torch.tensor(y_pred, dtype=torch.long))
        self.test_y.append(y)
        self.test_tasks.append(torch.full_like(y, fill_value=task))
        return {"loss": loss}

    @staticmethod
    def get_output(probabilities, prediction_bias=0.6):
        output = np.zeros(probabilities.shape[0])
        for i in range(len(probabilities)):
            if np.max(probabilities[i]) > prediction_bias:
                output[i] = np.argmax(probabilities[i])

        return output.astype(np.int64)

    def simulate_one_session(self, x, y_pred):
        i = 0
        y_pred = y_pred.detach().cpu().numpy()
        for x_i, y_i in zip(x.detach().cpu().numpy(), y_pred):
            if self.threshold_filter(x_i, y_i) == 0:
                y_pred[i] = 0
            i += 1

        return  y_pred

    @staticmethod
    def _smooth(preds: np.ndarray, window_size: int) -> np.ndarray:
        if window_size % 2 == 0:
            raise ValueError("window_size must be odd for majority voting")
        half = window_size // 2
        padded = np.pad(preds, pad_width=half, mode="edge")
        smoothed = []
        for i in range(len(preds)):
            block = padded[i : i + window_size]
            # count ones
            ones = block.sum()
            smoothed.append(1 if ones > half else 0)
        return np.array(smoothed, dtype=np.int64)


    @staticmethod
    def threshold_filter(data, pred_label):
        acc_data, gyro_data, angle_data = np.split(data, 3, axis=1)

        #Accelerometer magnitude
        acc_mag = np.sqrt(np.sum(acc_data ** 2, axis=1))

        # Gyroscope magnitude
        gyro_mag = np.sqrt(np.sum(gyro_data ** 2, axis=1))

        # Theta angle
        theta = np.arctan(
            np.sqrt(gyro_data[:, 1] ** 2 + gyro_data[:, 2] ** 2) / gyro_data[:, 0]
        ) * 180 / 3.14159

        d_theta = np.gradient(theta)
        mean_theta_grad = np.mean(d_theta)

        if np.any(acc_mag > 900) and np.abs(mean_theta_grad) <= 1.5:
            return 0
        else:
            return -1

    # -------------------------------------------------------
    def on_test_epoch_end(self):
        # save standard simulation stats
        import constants as const 
        import pandas as pd          # ? add
        import numpy  as np 

        fold_suffix = f"/{self.fold}" if self.fold is not None else ""
        dest_dir = os.path.join(self.output_dir, fold_suffix.lstrip("/"))
        process_results(pd.DataFrame(self.results), dest_dir, cfg=self.cfg)

        # ------------------- EVENT-LEVEL CONFUSION-MATRIX -------------------
        import sklearn.metrics as skm

        df_ev = pd.DataFrame(self.results)
        # ground-truth: was this simulation a FALL task (1) or an ACTIVITY (0)?
        df_ev["true_event"] = df_ev["task"].apply(
            lambda t: 1 if t in const.FALL_TASKS else 0
        )

        cm_ev = skm.confusion_matrix(df_ev["true_event"], df_ev["pred_event"])
        report_ev = skm.classification_report(
            df_ev["true_event"],
            df_ev["pred_event"],
            target_names=["No Fall", "Fall"],
            output_dict=True,
        )

        # pretty PDF/CSV export (reuse helpers already in test.py)
        from models.helper import generate_confusion_matrix   # same helper you use
        from test import save_classification_report           # already defined

        generate_confusion_matrix(
            df_ev["pred_event"],
            df_ev["true_event"],
            os.path.join(dest_dir, f"{self.model_name}_EVENT_confusion_matrix"),
            self.classes,
            specific_title=(f"{self.model_name} EVENT fold {self.fold}"
                            if self.fold is not None else
                            f"{self.model_name} EVENT"),
        )

        save_classification_report(
            df_ev["true_event"],
            df_ev["pred_event"],
            labels=["Activity", "Falling"],
            save_path=os.path.join(dest_dir, f"{self.model_name}_EVENT_classification_report"),
            model_name=f"{self.model_name} EVENT",
        )        # ---------------------------------------------------------------------


        # save arrays + fold-level confusion matrix
        if self.test_y_pred:
            y_pred_all = torch.cat(self.test_y_pred).cpu().numpy()
            y_true_all = torch.cat(self.test_y).cpu().numpy()

            os.makedirs(dest_dir, exist_ok=True)
            np.save(os.path.join(dest_dir, "y_pred.npy"), y_pred_all)
            np.save(os.path.join(dest_dir, "y_true.npy"), y_true_all)
            # save matching task id array  <-- NEW
            np.save(os.path.join(dest_dir, "task_ids.npy"),
                    torch.cat(self.test_tasks).cpu().numpy())        # <-- NEW
     
            generate_confusion_matrix(
                y_pred_all,
                y_true_all,
                os.path.join(dest_dir, f"{self.model_name}_confusion_matrix"),
                self.classes,
                specific_title=(f"{self.model_name} fold {self.fold}" if self.fold else self.model_name),
            )

        # clear for next fold
        self.test_y_pred.clear()
        self.test_y.clear()
        self.test_tasks.clear()                                       # <-- NEW

    # -------------------------------------------------------
    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.model.parameters(), lr=self.cfg.get("lr", 1e-3), weight_decay=self.cfg.get("wd", 0)
        )
