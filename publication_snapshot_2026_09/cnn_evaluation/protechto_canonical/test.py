# -*- coding: utf-8 -*-
"""
test.py - evaluate checkpoints and build a global confusion matrix & classification report
"""

import os, glob, json, warnings, argparse
import numpy as np
import pandas as pd
import torch
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import classification_report
from matplotlib.ticker import PercentFormatter
import seaborn as sns
from sklearn.metrics import confusion_matrix
from pathlib import Path
import constants as const 

from simulation.SimulationDataloader import SimulationDataloader
from simulation.SimulationKFoldDataloader import SimulationKFoldDataloader
from simulation.helper import process_results
from models import *
from models.wrappers import Simulator

with open("config.json", "r") as cf:
    settings = json.load(cf)

torch.set_float32_matmul_precision("high")
warnings.filterwarnings("ignore", category=FutureWarning, module=r".*")


def results_root(ckpt_root: str) -> str:
    parts = ckpt_root.strip(os.sep).split(os.sep)
    if "checkpoints" in parts:
        parts[parts.index("checkpoints")] = "results"
    else:
        parts.insert(0, "results")
    return os.sep.join(parts)


def generate_confusion_matrix(y_pred, y_true, filename, labels, specific_title=None, fig_size=(17, 14)):
    sns.set(font_scale=2)
    cm = confusion_matrix(y_true, y_pred)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                annot[i, j] = f"{p:.2f}%\n{c}/{int(cm_sum[i,0])}"
            else:
                annot[i, j] = f"{p:.2f}%\n{c}"
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm = cm * 100
    cm.index.name = 'True Label'
    cm.columns.name = 'Predicted Label'
    fig, ax = plt.subplots(figsize=fig_size)
    plt.yticks(va='center')

    sns.heatmap(cm, annot=annot, fmt='', ax=ax, xticklabels="auto", yticklabels="auto",
                cbar=True, cbar_kws={'format': PercentFormatter()}, cmap="Blues")

    if specific_title:
        plt.title(specific_title, fontsize=40, fontweight="bold")

    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{filename}.png", bbox_inches='tight', dpi=300)
    cm.to_csv(f"{filename}.csv")
    plt.close()

def save_classification_report(y_true, y_pred, labels, save_path, model_name="Model"):
    """
    Save classification report as a styled PDF table with bold headers and nice borders.
    """
    from matplotlib.backends.backend_pdf import PdfPages
    from sklearn.metrics import classification_report
    import matplotlib.pyplot as plt
    import pandas as pd

    report_dict = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose().round(4)

    fig, ax = plt.subplots(figsize=(10, 3 + 0.5 * len(labels)))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(
        cellText=report_df.values,
        colLabels=report_df.columns,
        rowLabels=report_df.index,
        cellLoc='center',
        loc='center'
    )

    # Styling
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)

    # Enhance headers
    for key, cell in table.get_celld().items():
        row, col = key
        cell.set_linewidth(1.5)
        if row == 0 or col == -1:  # Header row or first column
            cell.get_text().set_weight('bold')  # Set bold text
            cell.set_facecolor('#f0f0f0')       # Light gray background
        if row == 0:
            cell.set_linewidth(2.0)             # Thicker top row
        if col == 0:
            cell.set_linewidth(2.0)             # Thicker left column

    # Title
    plt.title(f"{model_name} - Classification Report (Global)", fontsize=16, fontweight='bold', pad=20)

    # Save to PDF
    with PdfPages(f"{save_path}.pdf") as pdf:
        pdf.savefig(fig, bbox_inches='tight')

    plt.close()
   # print(f"Classification report saved to {save_path}.pdf")


def test_k_fold(model_cls, ckpt_root, dataset, gpu, cfg):
    dl = SimulationKFoldDataloader(
        root_directory=f"{settings['data']}/{dataset}", k=5, batch_size=64
    )

    global_pred, global_true, rows = [], [], []

    for fold in range(dl.k):
        torch.cuda.empty_cache()
        print(f"Fold {fold + 1}")
        dl.construct(fold_index=fold)
        ckpt_path = os.path.join(
            settings["root"],
            ckpt_root,
            f"best-checkpoint-fold_{fold + 1}.ckpt",
        )

        model = Simulator(
            model_cls,
            n_features=9,
            n_classes=dl.label_encoder.classes_.shape[0],
            classes=dl.label_encoder.classes_,
            best_checkpoint_path=ckpt_path,
            fold=fold + 1,
            cfg=cfg,
        )

        Trainer(
            logger=TensorBoardLogger(f"{settings['root']}/lightning_logs", name="KFall"),
            accelerator="gpu",
            devices=[gpu],
            enable_progress_bar=True,
            max_epochs=1,
        ).test(model, dataloaders=dl.test_dl)

        fold_dir = os.path.join(results_root(ckpt_root), str(fold + 1))
        p, t = os.path.join(fold_dir, "y_pred.npy"), os.path.join(fold_dir, "y_true.npy")
        if os.path.exists(p) and os.path.exists(t):
            global_pred.append(np.load(p))
            global_true.append(np.load(t))

        rows.extend(model.results)

    out_root = results_root(ckpt_root)
    process_results(pd.DataFrame(rows), out_root, cfg=cfg)
    # ---------- GLOBAL  EVENT-LEVEL CONFUSION-MATRIX -------------
    df_rows = pd.DataFrame(rows)
    df_rows["true_event"] = df_rows["task"].apply(
        lambda t: 1 if t in const.FALL_TASKS else 0
    )
    cm_ev_glob = confusion_matrix(df_rows["true_event"], df_rows["pred_event"])
    generate_confusion_matrix(
        df_rows["pred_event"],
        df_rows["true_event"],
        os.path.join(out_root, f"{model_cls.__name__}_EVENT_confusion_matrix_GLOBAL"),
        dl.label_encoder.classes_,
        specific_title=f"{model_cls.__name__} EVENT GLOBAL (all folds)",
    )

    save_classification_report(
        df_rows["true_event"],
        df_rows["pred_event"],
        labels=["Activity", "Falling"],
        save_path=os.path.join(out_root, f"{model_cls.__name__}_EVENT_classification_report_GLOBAL"),
        model_name=f"{model_cls.__name__} EVENT",
    )

    if global_pred:
        y_pred_all = np.concatenate(global_pred)
        y_true_all = np.concatenate(global_true)
    # ---------- NEW: per-task segment mis-classification stats ----------
    task_files = sorted(glob.glob(os.path.join(out_root, "*", "task_ids.npy")))
    if task_files:                                                     # safety-check
        task_all = np.concatenate([np.load(f) for f in task_files])

        # build a row for every single segment
        df_seg = pd.DataFrame(
            dict(task=task_all, true=y_true_all, pred=y_pred_all)
        ).assign(
            TP=lambda d: (d.true == 1) & (d.pred == 1),
            TN=lambda d: (d.true == 0) & (d.pred == 0),
            FP=lambda d: (d.true == 0) & (d.pred == 1),
            FN=lambda d: (d.true == 1) & (d.pred == 0),
        )

        # aggregate per task
        seg_summary = (
            df_seg.groupby("task")[["TP", "TN", "FP", "FN"]]
                  .sum()
                  .reset_index()
        )

        # derive totals & percentages
        seg_summary["total_samples"] = seg_summary[["TP", "TN", "FP", "FN"]].sum(axis=1)
        seg_summary["misclassified"] = seg_summary["FP"] + seg_summary["FN"]
        seg_summary["perc_mis"] = (
            seg_summary["misclassified"] / seg_summary["total_samples"] * 100
        ).round(2)

        # add human-readable task names
        seg_summary["task_description"] = seg_summary["task"].map(const.TASKS_DESCRIPTIONS)

        # reorder / select displayed columns
        cols = [
            "task", "TP", "TN", "FP", "FN",
            "total_samples", "misclassified", "perc_mis",
            "task_description",
        ]
        seg_summary = seg_summary[cols]

        totals = seg_summary[["TP", "TN", "FP", "FN", "total_samples", "misclassified"]].sum()

        overall_row = {
            "task":            "Overall",         
            "TP":              totals["TP"],
            "TN":              totals["TN"],
            "FP":              totals["FP"],
            "FN":              totals["FN"],
            "total_samples":   totals["total_samples"],
            "misclassified":   totals["misclassified"],
            "perc_mis":        round(
                                   totals["misclassified"] / totals["total_samples"] * 100, 2
                               ),
            "task_description": "Overall"
        }

        seg_summary = pd.concat(
            [seg_summary, pd.DataFrame([overall_row])],
            ignore_index=True         
        )


        # ------------------------------------------------ PDF + CSV export
        seg_summary.to_csv(
            os.path.join(out_root, "segment_misclassified_by_task.csv"),
            index=False,
        )

        from simulation.helper import highlight_rows          # reuse colouring rule
        import pdfkit

        html_content = (
            seg_summary.style
            .apply(highlight_rows, axis=1)
            .format({"perc_mis": "{:.2f}%"})
            .to_html(index=False)
        )
        pdf_path = os.path.join(out_root, "segment_misclassified_by_task.pdf")
        pdfkit.from_string(html_content, pdf_path)

        print(f" Segment mis-classification report written to: {pdf_path}")
    # ------------------------------------------------------------------

        generate_confusion_matrix(
            y_pred_all,
            y_true_all,
            os.path.join(out_root, f"{model_cls.__name__}_confusion_matrix_GLOBAL"),
            dl.label_encoder.classes_,
            specific_title=f"{model_cls.__name__} GLOBAL (all folds)",
        )

        save_classification_report(
            y_true_all,
            y_pred_all,
            labels=dl.label_encoder.classes_,
            save_path=os.path.join(out_root, f"{model_cls.__name__}_classification_report_GLOBAL"),
            model_name=model_cls.__name__
        )

        print(f"Global confusion matrix and classification report saved to: {out_root}")
    else:
        print("Warning: no per-fold prediction arrays found.")

def test_single(model_cls, ckpt_root, dataset, gpu, cfg):
    sdl = SimulationDataloader(
        root_directory=f"{settings['data']}/{dataset}", test_size=0.2, batch_size=64
    )
    sdl.setup()

    ckpt_path = os.path.join(settings["root"], ckpt_root, "best-checkpoint.ckpt")
    model = Simulator(
        model_cls,
        n_features=9,
        n_classes=sdl.label_encoder.classes_.shape[0],
        classes=sdl.label_encoder.classes_,
        best_checkpoint_path=ckpt_path,
        cfg=cfg,
    )

    Trainer(
        logger=TensorBoardLogger(f"{settings['root']}/lightning_logs", name="KFall"),
        accelerator="gpu",
        devices=[gpu],
        enable_progress_bar=True,
        max_epochs=1,
    ).test(model, dataloaders=sdl.test_dl)


def main():
    parser = argparse.ArgumentParser("Evaluate checkpoints")
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("-c", "--checkpoint", required=True)
    parser.add_argument("-t", "--technique", choices=["k-fold", "train"], required=True)
    parser.add_argument("-d", "--dataset", required=True)
    parser.add_argument("-g", "--gpu", type=int, required=True)
    args = parser.parse_args()

    cfg = globals()[f"{args.model}_cfg"]
    model_cls = globals()[args.model]

    if args.technique == "k-fold":
        test_k_fold(model_cls, args.checkpoint, args.dataset, args.gpu, cfg)
    else:
        test_single(model_cls, args.checkpoint, args.dataset, args.gpu, cfg)


if __name__ == "__main__":
    main()
