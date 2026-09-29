#!/usr/bin/env python3

from pathlib import Path
import csv
import json
import shutil
import time

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(
    "/mnt/hdd16T/ToqeerHomeBackup/mujoco_project"
)

CAMPAIGN = (
    ROOT
    / "outputs"
    / "phase2_complete_final_campaign"
)

FINAL = (
    CAMPAIGN
    / "FINAL_RESULTS"
)

OUT = (
    ROOT
    / "outputs"
    / "PHASE2_FULL_TEST_REPORT"
)

ARCHIVES = (
    ROOT
    / "archives"
)

EXPERIMENTS = [
    "EXP01_REAL_ONLY",
    "EXP02_SIM_ONLY",
    "EXP03_MIX20",
    "EXP04_MIX50",
    "EXP05_MIX70",
    "EXP06_MIX100",
]

DISPLAY_NAMES = {
    "EXP01_REAL_ONLY": "Real only",
    "EXP02_SIM_ONLY": "Simulation only",
    "EXP03_MIX20": "Real + 20% Sim",
    "EXP04_MIX50": "Real + 50% Sim",
    "EXP05_MIX70": "Real + 70% Sim",
    "EXP06_MIX100": "Real + 100% Sim",
}

LABELS = [
    "Activity",
    "Fall",
]


def log(x):
    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {x}",
        flush=True
    )


# ============================================================
# ROBUST CONFUSION MATRIX LOADER
# ============================================================

def load_confusion_matrix(path):
    """
    Load binary confusion matrix in any format used by this campaign.

    Expected output convention:

        [[TN, FP],
         [FN, TP]]
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    def check(arr):
        arr = np.asarray(arr, dtype=float)

        if arr.shape != (2, 2):
            return None

        if not np.isfinite(arr).all():
            return None

        if (arr < 0).any():
            return None

        if not np.allclose(arr, np.rint(arr)):
            return None

        return np.rint(arr).astype(np.int64)

    # Campaign format:
    #
    # 342527 51027
    # 138588 416414
    #
    # np.loadtxt handles arbitrary whitespace automatically.
    try:
        arr = np.loadtxt(path)

        good = check(arr)

        if good is not None:
            return good

        if np.asarray(arr).size == 4:
            good = check(
                np.asarray(arr).reshape(2, 2)
            )

            if good is not None:
                return good

    except Exception:
        pass

    # Comma-separated fallback.
    try:
        arr = np.loadtxt(
            path,
            delimiter=","
        )

        good = check(arr)

        if good is not None:
            return good

        if np.asarray(arr).size == 4:
            good = check(
                np.asarray(arr).reshape(2, 2)
            )

            if good is not None:
                return good

    except Exception:
        pass

    # Pandas fallback for labelled/header formats.
    try:
        df = pd.read_csv(path)

        cols = {
            str(c).strip().lower(): c
            for c in df.columns
        }

        if all(
            x in cols
            for x in ["tn", "fp", "fn", "tp"]
        ) and len(df):

            r = df.iloc[0]

            good = check([
                [r[cols["tn"]], r[cols["fp"]]],
                [r[cols["fn"]], r[cols["tp"]]],
            ])

            if good is not None:
                return good

        numeric = (
            df.apply(pd.to_numeric, errors="coerce")
            .dropna(axis=0, how="all")
            .dropna(axis=1, how="all")
        )

        arr = numeric.to_numpy()

        good = check(arr)

        if good is not None:
            return good

        vals = arr[np.isfinite(arr)]

        if vals.size == 4:
            good = check(vals.reshape(2, 2))

            if good is not None:
                return good

    except Exception:
        pass

    raw = path.read_text(errors="replace")

    raise RuntimeError(
        f"\nCould not parse confusion matrix:\n{path}\n\n"
        f"Raw contents:\n{raw}"
    )


# ============================================================
# CLASSIFICATION REPORT FROM CONFUSION MATRIX
# ============================================================

def classification_from_cm(cm):

    rows = []

    total = cm.sum()

    for i, label in enumerate(LABELS):

        tp = int(cm[i, i])

        fn = int(
            cm[i, :].sum()
            - tp
        )

        fp = int(
            cm[:, i].sum()
            - tp
        )

        support = int(
            cm[i, :].sum()
        )

        precision = (
            tp / (tp + fp)
            if tp + fp
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if tp + fn
            else 0.0
        )

        f1 = (
            2 * precision * recall
            / (precision + recall)
            if precision + recall
            else 0.0
        )

        rows.append({
            "class": label,
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
            "support": support,
        })

    accuracy = (
        np.trace(cm) / total
        if total
        else 0.0
    )

    macro_precision = np.mean(
        [x["precision"] for x in rows]
    )

    macro_recall = np.mean(
        [x["recall"] for x in rows]
    )

    macro_f1 = np.mean(
        [x["f1-score"] for x in rows]
    )

    supports = np.array(
        [x["support"] for x in rows],
        dtype=float
    )

    weights = (
        supports / supports.sum()
        if supports.sum()
        else np.ones(len(rows)) / len(rows)
    )

    weighted_precision = np.sum(
        weights
        * np.array([
            x["precision"]
            for x in rows
        ])
    )

    weighted_recall = np.sum(
        weights
        * np.array([
            x["recall"]
            for x in rows
        ])
    )

    weighted_f1 = np.sum(
        weights
        * np.array([
            x["f1-score"]
            for x in rows
        ])
    )

    rows.extend([
        {
            "class": "accuracy",
            "precision": np.nan,
            "recall": np.nan,
            "f1-score": accuracy,
            "support": int(total),
        },
        {
            "class": "macro avg",
            "precision": macro_precision,
            "recall": macro_recall,
            "f1-score": macro_f1,
            "support": int(total),
        },
        {
            "class": "weighted avg",
            "precision": weighted_precision,
            "recall": weighted_recall,
            "f1-score": weighted_f1,
            "support": int(total),
        },
    ])

    return pd.DataFrame(rows)


# ============================================================
# CONFUSION MATRIX FIGURE
# ============================================================

def save_confusion_matrix(cm, outfile, title):

    row_sum = cm.sum(
        axis=1,
        keepdims=True
    )

    norm = np.divide(
        cm,
        row_sum,
        out=np.zeros_like(
            cm,
            dtype=float
        ),
        where=row_sum != 0
    )

    fig, ax = plt.subplots(
        figsize=(7.0, 6.0)
    )

    im = ax.imshow(norm)

    ax.set_xticks(
        range(2),
        labels=LABELS
    )

    ax.set_yticks(
        range(2),
        labels=LABELS
    )

    ax.set_xlabel(
        "Predicted Label"
    )

    ax.set_ylabel(
        "True Label"
    )

    ax.set_title(
        title
    )

    for i in range(2):
        for j in range(2):

            ax.text(
                j,
                i,
                (
                    f"{int(cm[i,j]):,}\n"
                    f"({norm[i,j]*100:.2f}%)"
                ),
                ha="center",
                va="center",
            )

    fig.colorbar(
        im,
        ax=ax,
        fraction=0.046,
        pad=0.04,
        label="Row-normalized proportion"
    )

    fig.tight_layout()

    fig.savefig(
        outfile,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# CLASSIFICATION REPORT PDF
# ============================================================

def save_report_pdf(report, outfile, title):

    with PdfPages(outfile) as pdf:

        fig, ax = plt.subplots(
            figsize=(9.0, 5.8)
        )

        ax.axis("off")

        ax.set_title(
            title,
            fontsize=14,
            pad=20
        )

        table_data = []

        for _, r in report.iterrows():

            table_data.append([
                r["class"],
                (
                    ""
                    if pd.isna(r["precision"])
                    else f"{r['precision']:.4f}"
                ),
                (
                    ""
                    if pd.isna(r["recall"])
                    else f"{r['recall']:.4f}"
                ),
                f"{r['f1-score']:.4f}",
                f"{int(r['support']):,}",
            ])

        table = ax.table(
            cellText=table_data,
            colLabels=[
                "Class",
                "Precision",
                "Recall",
                "F1-score",
                "Support",
            ],
            loc="center",
            cellLoc="center",
        )

        table.auto_set_font_size(
            False
        )

        table.set_fontsize(
            10
        )

        table.scale(
            1.0,
            1.6
        )

        pdf.savefig(
            fig,
            bbox_inches="tight"
        )

        plt.close(fig)


# ============================================================
# TRAINING HISTORY
# ============================================================

def save_training_history(history_file, outfile, title):

    df = pd.read_csv(
        history_file
    )

    if len(df) == 0:
        return False

    numeric = []

    for c in df.columns:

        series = pd.to_numeric(
            df[c],
            errors="coerce"
        )

        if series.notna().sum() > 0:
            numeric.append(c)

    epoch_candidates = [
        c for c in numeric
        if "epoch" in c.lower()
    ]

    if epoch_candidates:
        xcol = epoch_candidates[0]
        x = pd.to_numeric(
            df[xcol],
            errors="coerce"
        )
    else:
        xcol = "epoch"
        x = np.arange(
            1,
            len(df) + 1
        )

    plot_cols = [
        c for c in numeric
        if c != xcol
        and (
            "loss" in c.lower()
            or "acc" in c.lower()
            or "f1" in c.lower()
        )
    ]

    # If the script uses unusual names,
    # plot remaining numeric training statistics.
    if not plot_cols:

        plot_cols = [
            c for c in numeric
            if c != xcol
        ][:6]

    if not plot_cols:
        return False

    fig, ax = plt.subplots(
        figsize=(8.0, 5.2)
    )

    for c in plot_cols:

        y = pd.to_numeric(
            df[c],
            errors="coerce"
        )

        ax.plot(
            x,
            y,
            label=c
        )

    ax.set_xlabel(
        "Epoch"
    )

    ax.set_ylabel(
        "Metric value"
    )

    ax.set_title(
        title
    )

    ax.grid(
        True,
        alpha=0.25
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        outfile,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    return True


# ============================================================
# FINAL CONDITION COMPARISON
# ============================================================

def save_summary_figures():

    summary_file = (
        FINAL
        / "experiment_summary_mean_std.csv"
    )

    summary = pd.read_csv(
        summary_file
    )

    summary = summary.set_index(
        "experiment"
    ).loc[EXPERIMENTS].reset_index()

    labels = [
        DISPLAY_NAMES[x]
        for x in summary["experiment"]
    ]

    metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1",
    ]

    # --------------------------------------------------------
    # One figure per metric
    # --------------------------------------------------------

    for metric in metrics:

        mean_col = (
            f"{metric}_mean"
        )

        std_col = (
            f"{metric}_std"
        )

        if mean_col not in summary.columns:
            continue

        fig, ax = plt.subplots(
            figsize=(9.5, 5.7)
        )

        x = np.arange(
            len(labels)
        )

        values = (
            summary[mean_col]
            .to_numpy()
            * 100
        )

        errors = (
            summary[std_col]
            .to_numpy()
            * 100
            if std_col in summary.columns
            else None
        )

        bars = ax.bar(
            x,
            values,
            yerr=errors,
            capsize=4
        )

        ax.set_xticks(
            x,
            labels=labels,
            rotation=25,
            ha="right"
        )

        ax.set_ylabel(
            f"{metric.capitalize()} (%)"
        )

        ax.set_title(
            (
                f"CNN {metric.capitalize()} "
                "Across Training Conditions"
            )
        )

        ax.grid(
            axis="y",
            alpha=0.25
        )

        for bar, value in zip(
            bars,
            values
        ):

            ax.text(
                bar.get_x()
                + bar.get_width()/2,
                bar.get_height()
                + 0.5,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        fig.tight_layout()

        fig.savefig(
            OUT
            / "summary"
            / f"CNN_{metric}_by_training_condition.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(fig)

    # --------------------------------------------------------
    # Mixed-domain dose response only
    # --------------------------------------------------------

    mixed = summary[
        summary["experiment"].isin([
            "EXP01_REAL_ONLY",
            "EXP03_MIX20",
            "EXP04_MIX50",
            "EXP05_MIX70",
            "EXP06_MIX100",
        ])
    ].copy()

    fractions = [
        0,
        20,
        50,
        70,
        100,
    ]

    fig, ax = plt.subplots(
        figsize=(8.0, 5.4)
    )

    for metric in metrics:

        col = (
            f"{metric}_mean"
        )

        if col in mixed.columns:

            ax.plot(
                fractions,
                mixed[col].to_numpy() * 100,
                marker="o",
                label=metric.capitalize(),
            )

    ax.set_xlabel(
        "Simulated augmentation (%)"
    )

    ax.set_ylabel(
        "Performance (%)"
    )

    ax.set_title(
        "Synthetic-data Dose–Response on Held-out Real Data"
    )

    ax.grid(
        True,
        alpha=0.25
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUT
        / "summary"
        / "CNN_simulation_augmentation_dose_response.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():

    if not CAMPAIGN.exists():
        raise SystemExit(
            f"Campaign not found: {CAMPAIGN}"
        )

    if OUT.exists():
        shutil.rmtree(
            OUT
        )

    (
        OUT
        / "summary"
    ).mkdir(
        parents=True,
        exist_ok=True
    )

    (
        OUT
        / "tables"
    ).mkdir(
        parents=True,
        exist_ok=True
    )

    audit_rows = []

    global_cm_rows = []

    log(
        "Generating Phase-2 CNN visual/test report..."
    )

    for exp in EXPERIMENTS:

        log(
            f"Processing {exp}"
        )

        exp_out = (
            OUT
            / exp
        )

        exp_out.mkdir(
            parents=True,
            exist_ok=True
        )

        aggregate_cm = np.zeros(
            (2, 2),
            dtype=np.int64
        )

        for fold in range(5):

            src = (
                CAMPAIGN
                / "experiments"
                / exp
                / f"fold_{fold}"
            )

            dst = (
                exp_out
                / f"fold_{fold}"
            )

            dst.mkdir(
                parents=True,
                exist_ok=True
            )

            cm_file = (
                src
                / "confusion_matrix.csv"
            )

            metrics_file = (
                src
                / "metrics.csv"
            )

            hist_file = (
                src
                / "training_history.csv"
            )

            if not cm_file.exists():
                raise RuntimeError(
                    f"Missing {cm_file}"
                )

            cm = load_confusion_matrix(
                cm_file
            )

            aggregate_cm += cm

            # Save raw confusion-matrix copy
            shutil.copy2(
                cm_file,
                dst
                / "confusion_matrix.csv"
            )

            if metrics_file.exists():

                shutil.copy2(
                    metrics_file,
                    dst
                    / "metrics.csv"
                )

            if hist_file.exists():

                shutil.copy2(
                    hist_file,
                    dst
                    / "training_history.csv"
                )

            # Original-test style confusion matrix
            cm_png = (
                dst
                / "CNN_confusion_matrix.png"
            )

            save_confusion_matrix(
                cm,
                cm_png,
                (
                    f"{DISPLAY_NAMES[exp]} "
                    f"— Fold {fold + 1}"
                )
            )

            report = (
                classification_from_cm(
                    cm
                )
            )

            report.to_csv(
                dst
                / "CNN_classification_report.csv",
                index=False
            )

            save_report_pdf(
                report,
                dst
                / "CNN_classification_report.pdf",
                (
                    "CNN Classification Report\n"
                    f"{DISPLAY_NAMES[exp]} "
                    f"— Fold {fold + 1}"
                )
            )

            if hist_file.exists():

                save_training_history(
                    hist_file,
                    dst
                    / "CNN_training_history.png",
                    (
                        f"CNN Training History — "
                        f"{DISPLAY_NAMES[exp]}, "
                        f"Fold {fold + 1}"
                    )
                )

            audit_rows.append({
                "experiment":
                    exp,
                "fold":
                    fold,
                "tn":
                    int(cm[0,0]),
                "fp":
                    int(cm[0,1]),
                "fn":
                    int(cm[1,0]),
                "tp":
                    int(cm[1,1]),
                "n":
                    int(cm.sum()),
            })

        # ====================================================
        # GLOBAL / 5-FOLD AGGREGATE
        # ====================================================

        save_confusion_matrix(
            aggregate_cm,
            exp_out
            / "CNN_confusion_matrix_GLOBAL.png",
            (
                f"{DISPLAY_NAMES[exp]}\n"
                "Aggregated 5-Fold Confusion Matrix"
            )
        )

        report = (
            classification_from_cm(
                aggregate_cm
            )
        )

        report.to_csv(
            exp_out
            / "CNN_classification_report_GLOBAL.csv",
            index=False
        )

        save_report_pdf(
            report,
            exp_out
            / "CNN_classification_report_GLOBAL.pdf",
            (
                "CNN GLOBAL Classification Report\n"
                f"{DISPLAY_NAMES[exp]}"
            )
        )

        np.savetxt(
            exp_out
            / "CNN_confusion_matrix_GLOBAL.csv",
            aggregate_cm,
            fmt="%d",
            delimiter=","
        )

        global_cm_rows.append({
            "experiment":
                exp,
            "tn":
                int(aggregate_cm[0,0]),
            "fp":
                int(aggregate_cm[0,1]),
            "fn":
                int(aggregate_cm[1,0]),
            "tp":
                int(aggregate_cm[1,1]),
            "n":
                int(aggregate_cm.sum()),
        })

    # ========================================================
    # SUMMARY FIGURES
    # ========================================================

    save_summary_figures()

    pd.DataFrame(
        audit_rows
    ).to_csv(
        OUT
        / "tables"
        / "all_fold_confusion_counts.csv",
        index=False
    )

    pd.DataFrame(
        global_cm_rows
    ).to_csv(
        OUT
        / "tables"
        / "global_confusion_counts.csv",
        index=False
    )

    for filename in [
        "all_30_fold_metrics.csv",
        "experiment_summary_mean_std.csv",
        "augmentation_gain_vs_real_only.csv",
        "CNN_segment_results.csv",
    ]:

        src = FINAL / filename

        if src.exists():

            shutil.copy2(
                src,
                OUT
                / "tables"
                / filename
            )

    readme = """
PHASE-2 FULL CNN TEST REPORT

This directory is a post-hoc rendering of the completed Phase-2 CNN
campaign. No models were retrained.

Source campaign:
outputs/phase2_complete_final_campaign

Created from saved:
- confusion_matrix.csv
- metrics.csv
- training_history.csv

For each of 6 experiments x 5 folds it provides:
- CNN_confusion_matrix.png
- CNN_classification_report.csv
- CNN_classification_report.pdf
- CNN_training_history.png
- original numeric CSVs

For each experiment it also provides:
- CNN_confusion_matrix_GLOBAL.png
- CNN_confusion_matrix_GLOBAL.csv
- CNN_classification_report_GLOBAL.csv
- CNN_classification_report_GLOBAL.pdf

Summary includes:
- Accuracy comparison
- Precision comparison
- Recall comparison
- F1 comparison
- Simulation-augmentation dose-response

IMPORTANT:
These are SEGMENT-LEVEL CNN results.

Event-level evaluation is NOT reconstructed here because event-level
predictions require trial/event structure and must be evaluated separately.

No training was performed by this reporting script.
""".strip()

    (
        OUT
        / "README_REPORT.txt"
    ).write_text(
        readme + "\n"
    )

    metadata = {
        "created":
            time.strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
        "source_campaign":
            str(CAMPAIGN),
        "experiments":
            EXPERIMENTS,
        "folds_per_experiment":
            5,
        "total_folds":
            30,
        "report_level":
            "CNN segment level",
        "retraining":
            False,
    }

    (
        OUT
        / "report_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2
        )
    )

    # ========================================================
    # ZIP
    # ========================================================

    timestamp = time.strftime(
        "%Y%m%d_%H%M%S"
    )

    zip_base = (
        ARCHIVES
        / (
            "PHASE2_FULL_CNN_TEST_REPORT_"
            + timestamp
        )
    )

    zip_file = shutil.make_archive(
        str(zip_base),
        "zip",
        root_dir=OUT
    )

    log("=" * 72)
    log("REPORT GENERATION COMPLETE")
    log(f"Output: {OUT}")
    log(f"ZIP   : {zip_file}")
    log("=" * 72)

    print()
    print("Generated PNG files:")
    print(
        len(
            list(
                OUT.rglob("*.png")
            )
        )
    )

    print(
        "Generated PDF reports:",
        len(
            list(
                OUT.rglob("*.pdf")
            )
        )
    )

    print(
        "Generated CSV tables:",
        len(
            list(
                OUT.rglob("*.csv")
            )
        )
    )


if __name__ == "__main__":
    main()

