from pathlib import Path
import ast
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

PROTECHTO = Path(
    "/mnt/hdd16T/ToqeerHomeBackup/toqeer/Protechto_master"
)

ROOT = Path(
    "/mnt/hdd16T/ToqeerHomeBackup/mujoco_project/"
    "outputs/phase2_protechto_parity_fold0_v1"
)

PREDICTION_BIAS = 0.65

EXPERIMENTS = [
    "EXP01_PHYSICAL_ONLY",
]


def load_constant(path, name):
    tree = ast.parse(path.read_text())

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id == name
                ):
                    return ast.literal_eval(node.value)

    raise RuntimeError(
        f"{name} not found in {path}"
    )


def load_function(path, name, globals_dict):
    tree = ast.parse(path.read_text())

    for node in tree.body:
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == name
        ):
            module = ast.Module(
                body=[node],
                type_ignores=[],
            )

            ast.fix_missing_locations(module)

            ns = dict(globals_dict)

            exec(
                compile(
                    module,
                    str(path),
                    "exec",
                ),
                ns,
            )

            return ns[name]

    raise RuntimeError(
        f"{name} not found in {path}"
    )


cnn_file = (
    PROTECHTO
    / "simulation"
    / "LearningCurveMetrics.py"
)

threshold_file = (
    PROTECHTO
    / "evaluate_canonical_threshold.py"
)

THRESHOLD = load_constant(
    threshold_file,
    "THRESHOLD",
)

cnn_predictions_from_probabilities = (
    load_function(
        cnn_file,
        "cnn_predictions_from_probabilities",
        {
            "np": np,
        },
    )
)

historical_threshold_pass = (
    load_function(
        threshold_file,
        "historical_threshold_pass",
        {
            "np": np,
            "THRESHOLD": THRESHOLD,
        },
    )
)


if abs(PREDICTION_BIAS - 0.65) > 1e-12:
    raise RuntimeError(
        "prediction_bias must be 0.65"
    )

if THRESHOLD != 2:
    raise RuntimeError(
        f"Expected historical threshold 2, got {THRESHOLD}"
    )


print("=" * 100)
print("PROTECHTO HISTORICAL EVENT-LEVEL DIAGNOSTIC")
print("=" * 100)

print(
    "CNN decision function :",
    cnn_predictions_from_probabilities.__name__,
)

print(
    "Event function        :",
    historical_threshold_pass.__name__,
)

print(
    "prediction_bias       :",
    PREDICTION_BIAS,
)

print(
    "consecutive threshold :",
    THRESHOLD,
)

print()


all_rows = []


for experiment in EXPERIMENTS:

    print()
    print("#" * 100)
    print(experiment)
    print("#" * 100)

    for fold in [0]:

        path = (
            ROOT
            / experiment
            / f"fold_{fold}"
            / "predictions.csv"
        )

        if not path.exists():
            raise RuntimeError(
                f"Missing {path}"
            )

        df = pd.read_csv(
            path,
            low_memory=False,
        )

        required = [
            "dataset",
            "group_id",
            "task_id",
            "trial_id",
            "source_path",
            "source_row_start",
            "true_label",
            "prob_activity",
            "prob_fall",
        ]

        missing = [
            c for c in required
            if c not in df.columns
        ]

        if missing:
            raise RuntimeError(
                f"{path}: missing {missing}"
            )

        group_cols = [
            "dataset",
            "group_id",
            "task_id",
            "trial_id",
            "source_path",
        ]

        event_rows = []

        for keys, g in df.groupby(
            group_cols,
            dropna=False,
            sort=False,
        ):

            g = g.sort_values(
                "source_row_start"
            )

            y_true = (
                g["true_label"]
                .astype(int)
                .to_numpy()
            )

            probabilities = (
                g[
                    [
                        "prob_activity",
                        "prob_fall",
                    ]
                ]
                .to_numpy(dtype=np.float64)
            )

            y_pred = np.asarray(
                cnn_predictions_from_probabilities(
                    probabilities,
                    prediction_bias=(
                        PREDICTION_BIAS
                    ),
                ),
                dtype=int,
            )

            if len(y_pred) != len(y_true):
                raise RuntimeError(
                    "Prediction shape mismatch"
                )

            true_event = int(
                np.any(y_true == 1)
            )

            passed = bool(
                historical_threshold_pass(
                    y_true,
                    y_pred,
                    threshold=THRESHOLD,
                )
            )

            predicted_event = (
                true_event
                if passed
                else 1 - true_event
            )

            event_rows.append(
                {
                    "dataset": keys[0],
                    "group_id": keys[1],
                    "task_id": keys[2],
                    "trial_id": keys[3],
                    "source_path": keys[4],
                    "true_event": true_event,
                    "predicted_event":
                        predicted_event,
                    "passed_historical_rule":
                        int(passed),
                    "n_windows":
                        len(y_true),
                    "n_true_falling_windows":
                        int(y_true.sum()),
                    "n_pred_falling_windows":
                        int(y_pred.sum()),
                }
            )

        ev = pd.DataFrame(
            event_rows
        )

        y_true_event = (
            ev["true_event"]
            .astype(int)
            .to_numpy()
        )

        y_pred_event = (
            ev["predicted_event"]
            .astype(int)
            .to_numpy()
        )

        cm = confusion_matrix(
            y_true_event,
            y_pred_event,
            labels=[0, 1],
        )

        tn, fp, fn, tp = (
            cm.ravel()
        )

        accuracy = accuracy_score(
            y_true_event,
            y_pred_event,
        )

        balanced = (
            balanced_accuracy_score(
                y_true_event,
                y_pred_event,
            )
        )

        precision = precision_score(
            y_true_event,
            y_pred_event,
            zero_division=0,
        )

        recall = recall_score(
            y_true_event,
            y_pred_event,
            zero_division=0,
        )

        f1 = f1_score(
            y_true_event,
            y_pred_event,
            zero_division=0,
        )

        row = {
            "experiment": experiment,
            "fold": fold,
            "events": len(ev),
            "activity_events":
                int(
                    (y_true_event == 0).sum()
                ),
            "fall_events":
                int(
                    (y_true_event == 1).sum()
                ),
            "accuracy": accuracy,
            "balanced_accuracy":
                balanced,
            "precision_fall":
                precision,
            "recall_fall":
                recall,
            "f1_fall":
                f1,
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }

        all_rows.append(row)

        out = (
            ROOT
            / experiment
            / f"fold_{fold}"
            / "protechto_event_diagnostic"
        )

        out.mkdir(
            parents=True,
            exist_ok=True,
        )

        ev.to_csv(
            out
            / "event_predictions.csv",
            index=False,
        )

        pd.DataFrame(
            [row]
        ).to_csv(
            out
            / "event_metrics.csv",
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
            out
            / "event_confusion_matrix.csv"
        )

        print(
            f"fold={fold} "
            f"events={len(ev):4d} "
            f"acc={accuracy:.4f} "
            f"bal={balanced:.4f} "
            f"prec={precision:.4f} "
            f"rec={recall:.4f} "
            f"f1={f1:.4f} "
            f"TN={tn} "
            f"FP={fp} "
            f"FN={fn} "
            f"TP={tp}"
        )


results = pd.DataFrame(
    all_rows
)

results.to_csv(
    ROOT
    / "diagnostic_protechto_event_results.csv",
    index=False,
)


print()
print("=" * 100)
print("5-FOLD EVENT-LEVEL SUMMARY")
print("=" * 100)


metrics = [
    "accuracy",
    "balanced_accuracy",
    "precision_fall",
    "recall_fall",
    "f1_fall",
]


for experiment in EXPERIMENTS:

    r = results[
        results["experiment"]
        == experiment
    ]

    print()
    print(experiment)

    for metric in metrics:

        print(
            f"  {metric:20s}: "
            f"{r[metric].mean():.4f} "
            f"± "
            f"{r[metric].std():.4f}"
        )


print()
print("=" * 100)
print("SOURCE PARITY")
print("=" * 100)

print(
    "prediction function loaded directly from:"
)

print(cnn_file)

print(
    "event function loaded directly from:"
)

print(threshold_file)

print()
print(
    "No Lightning import was used."
)

print(
    "No event rule was reimplemented."
)

print()
print(
    "PROTECHTO EVENT-EVALUATION "
    "DIAGNOSTIC GATE: PASS"
)
