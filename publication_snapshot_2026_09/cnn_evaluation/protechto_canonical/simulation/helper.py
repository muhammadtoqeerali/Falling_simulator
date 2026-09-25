import os
import pdfkit
import constants as const
import pandas as pd
import torch
import numpy as np
from pathlib import Path


def load_simulation(subjects, encoder, target_directory):
    sessions = []
    for subject in subjects:
        for task in sorted(
                [t for t in os.listdir(os.path.join(target_directory, str(subject))) if not t.startswith(".")]
        ):
            for trial in sorted(
                    [t for t in os.listdir(os.path.join(target_directory, str(subject), str(task))) if
                     not t.startswith(".")]
            ):
                x_i = np.load(os.path.join(target_directory, subject, task, trial, "segments.npy"))
                y_i = np.load(os.path.join(target_directory, subject, task, trial, "labels.npy"))
                session = {
                    "subject": subject,
                    "task": task,
                    "trial": trial,
                    "x": torch.tensor(x_i, dtype=torch.float32),
                    "y": torch.tensor(encoder.transform(y_i)).long()
                }
                sessions.append(session)
    return sessions

def extract_from_batch(batch):
    subject, task, trial, x, y = batch.values()
    subject = int(subject[0])
    task = int(task[0])
    trial = int(trial[0])
    x = x[0]
    y = y[0]

    return subject, task, trial, x, y

def add_percentage(value):
    if value is not None:
        return f"{round(value, 2)}%"
    else:
        return "-"


def is_simulation_passed(y_true, y_pred):
    for i in range(len(y_true)):
        if y_pred[i] == y_true[i] and y_pred[i] == 1:
            return True

        if y_pred[i] == 1 and y_true[i] == 0:
            if 1 in y_true[i: i+3]:
                return True
            else:
                return False

        if y_pred[i] == 0 and y_true[i] == 1:
            if 1 in y_pred[i: ]:
                return True
            else:
                return False

    return True


def is_simulation_passed_threshold(y_true, y_pred, threshold=2):

    # Case A: no real fall & no pred ? pass
    if 1 not in y_true and 1 not in y_pred:
        return True

    # Case B: real fall but no pred ? fail
    if 1 in y_true and 1 not in y_pred:
        return False

    # Case C: no real fall but we predicted some fall ? fail if any run = threshold
    if 1 not in y_true and 1 in y_pred:
        for i in range(len(y_pred) - threshold + 1):
            if y_pred[i : i + threshold].sum() >= threshold:
                return False
        return True

    # Case D: real fall and we also predicted falls ? pass if any run = threshold
    if 1 in y_true and 1 in y_pred:
        for i in range(len(y_pred) - threshold + 1):
            if y_pred[i : i + threshold].sum() >= threshold:
                # require that run overlaps the true-fall region
                if 1 in y_true[i : i + threshold]:
                    return True
        return False


def highlight_rows(row):
    if row["task"] in const.FALL_TASKS:
        return ['background-color: rgba(255, 0, 0, 0.5)'] * len(row)
    else:
        return ['background-color: rgba(0, 255, 0, 0.3)'] * len(row)

def process_results(df, output_path, cfg):
    Path(output_path).mkdir(parents=True, exist_ok=True)
    stats = []
    df.sort_values(by=['subject', 'task'], ascending=[True, True], inplace=True)
    for task in sorted(df["task"].unique()):
        total = df[df["task"] == task].shape[0]
        passed = df[df["task"] == task]["passed"].sum()
        perc_missed = f"{round((total - passed) / total * 100, 2): 3.2f}%"
        task_description = const.TASKS_DESCRIPTIONS[task]
        stats.append(
            {
                "task": task,
                "passed_simulations": passed,
                "missed_simulations": total - passed,
                "perc_missed": perc_missed,
                "task_description": task_description
            }
        )
    stats_df = pd.DataFrame(stats)
    stats_df.sort_values(by=['task'], ascending=[True], inplace=True)
    stats_df.reset_index(drop=True, inplace=True)

    passed_simulations = stats_df['passed_simulations'].sum()
    missed_simulations = stats_df['missed_simulations'].sum()
    total_simulations = stats_df['passed_simulations'].sum() + stats_df['missed_simulations'].sum()
    perc_simulations_passed = (passed_simulations/ total_simulations) * 100
    perc_simulations_missed = (missed_simulations/ total_simulations) * 100

    stats_df.loc[len(stats_df)] = ["Overall",f"{passed_simulations}/{total_simulations} ({add_percentage(perc_simulations_passed)})", f"{missed_simulations}/{total_simulations} ({add_percentage(perc_simulations_missed)})", "--", "--"]

    styled_df = stats_df.style.apply(highlight_rows, axis=1)
    html_content = styled_df.to_html(index=False)
    pdfkit.from_string(html_content, f"{output_path}/event_stats__bias_{cfg['prediction_bias']}.pdf")
    stats_df.to_csv(f"{output_path}/event_stats__bias_{cfg['prediction_bias']}.csv", index=False)

    tex = stats_df.to_latex(index=False, escape=True, caption="Event Based Stats", label="tab:ov_perf")
    tex = tex.replace(
        "\\label{tab:ov_perf}", "\\begin{adjustbox}{max width=\\textwidth}"
    ).replace(
        "\\end{tabular}","\\end{tabular}\n\\end{adjustbox}\n"
    )
    with open(f"{output_path}/event_stats__bias_{cfg['prediction_bias']}.tex", "w") as f_tex:
        f_tex.write(tex)

    activities_df = df[df['task'].isin(const.ACTIVITY_TASKS)]
    falling_df = df[df['task'].isin(const.FALL_TASKS)]
    backward_df = df[df['task'].isin(const.BACKWARD_FALLS)]
    forward_df = df[df['task'].isin(const.FORWARD_FALLS)]
    lateral_df = df[df['task'].isin(const.LATERAL_FALLS)]
    height_df = df[df['task'].isin(const.FALLS_FROM_HEIGHT)]

    activities_passed_sim = activities_df["passed"].sum() * 100 / activities_df["passed"].shape[0]
    falling_passed_sim = falling_df["passed"].sum() * 100 / falling_df["passed"].shape[0]
    backward_passed_sim = backward_df["passed"].sum() * 100 / backward_df["passed"].shape[0]
    forward_passed_sim = forward_df["passed"].sum() * 100 / forward_df["passed"].shape[0]
    lateral_passed_sim = lateral_df["passed"].sum() * 100 / lateral_df["passed"].shape[0]
    height_passed_sim = height_df["passed"].sum() * 100 / height_df["passed"].shape[0]
    overall_passed_sim = df["passed"].sum() * 100 / df["passed"].shape[0]

    # Save a latex table:
    overall_stats = {
        "Overall passed simulations": add_percentage(overall_passed_sim),
        "Activities passed simulations": add_percentage(activities_passed_sim),
        "Falling passed simulations": add_percentage(falling_passed_sim),
        "Backward passed simulations": add_percentage(backward_passed_sim),
        "Height passed simulations": add_percentage(height_passed_sim),
        "Forward passed simulations": add_percentage(forward_passed_sim),
        "Lateral passed simulations": add_percentage(lateral_passed_sim)
    }
    tex_code = pd.DataFrame.from_dict(overall_stats.items()).to_latex(header=["Statistic", "Percentage"], index=False, escape=True, caption="Event Based Overall Performance", label="tab:ov_perf")
    with open(f"{output_path}/report_sim_stats__bias_{cfg['prediction_bias']}.tex", "w") as f_tex:
        f_tex.write(tex_code)

    with open(f"{output_path}/overall_sim_stats__bias_{cfg['prediction_bias']}.csv", "a+") as f:
        f.write(f"Overall passed simulations,{add_percentage(overall_passed_sim)}\n")
        f.write(f"Activities passed simulations,{add_percentage(activities_passed_sim)}\n")
        f.write(f"Falling passed simulations,{add_percentage(falling_passed_sim)}\n")
        f.write(f"Backward passed simulations,{add_percentage(backward_passed_sim)}\n")
        f.write(f"Height passed simulations,{add_percentage(height_passed_sim)}\n")
        f.write(f"Forward passed simulations,{add_percentage(forward_passed_sim)}\n")
        f.write(f"Lateral passed simulations,{add_percentage(lateral_passed_sim)}\n")
        f.write("\n\n")
        f.write("subject,passed_activities,passed_falling,forward,lateral,backward,from_height\n")


    overall_sbj_stats = []
    for subject in df["subject"].unique():
        sbj_passed_falling = None
        sbj_passed_falling_forward = None
        sbj_passed_falling_lateral = None

        sbj_passed_activities = (activities_df[activities_df["subject"] == subject]["passed"].sum() * 100 /
                                 activities_df[activities_df["subject"] == subject].shape[0])

        if falling_df[falling_df["subject"] == subject].shape[0] > 0:
            sbj_passed_falling = (falling_df[falling_df["subject"] == subject]["passed"].sum() * 100 / falling_df[falling_df["subject"] == subject].shape[0])

        if sbj_passed_falling:
            backward_res = falling_df[falling_df["subject"] == subject][falling_df[falling_df["subject"] == subject]["task"].isin(const.BACKWARD_FALLS)]
            height_res = falling_df[falling_df["subject"] == subject][falling_df[falling_df["subject"] == subject]["task"].isin(const.FALLS_FROM_HEIGHT)]
            lateral_res = falling_df[falling_df["subject"] == subject][falling_df[falling_df["subject"] == subject]["task"].isin(const.LATERAL_FALLS)]
            forward_res = falling_df[falling_df["subject"] == subject][falling_df[falling_df["subject"] == subject]["task"].isin(const.FORWARD_FALLS)]

            sbj_passed_falling_backward = backward_res["passed"].sum() * 100 / backward_res.shape[0]
            sbj_passed_falling_lateral = lateral_res["passed"].sum() * 100 / lateral_res.shape[0]
            sbj_passed_falling_forward = forward_res["passed"].sum() * 100 / forward_res.shape[0]
            if height_res.shape[0] > 0:
                sbj_passed_height = height_res["passed"].sum() * 100 / height_res.shape[0]
            else:
                sbj_passed_height = None

        else:
            sbj_passed_falling = None
            sbj_passed_falling_backward = None
            sbj_passed_height = None

        overall_sbj_stats.append(
            [
                subject,
                add_percentage(sbj_passed_activities),
                add_percentage(sbj_passed_falling),
                add_percentage(sbj_passed_falling_forward),
                add_percentage(sbj_passed_falling_lateral),
                add_percentage(sbj_passed_falling_backward),
                add_percentage(sbj_passed_height)
            ]
        )
        with open(f"{output_path}/overall_sim_stats__bias_{cfg['prediction_bias']}.csv", "a+") as f:
            f.write(f"{subject},{add_percentage(sbj_passed_activities)},{add_percentage(sbj_passed_falling)},{add_percentage(sbj_passed_falling_forward)},{add_percentage(sbj_passed_falling_lateral)},{add_percentage(sbj_passed_falling_backward)},{add_percentage(sbj_passed_height)}\n")

    tex_code = pd.DataFrame(overall_sbj_stats, columns=["subject","passed_activities","passed_falling","forward","lateral","backward","from_height"]).to_latex(index=False, escape=True, caption="Subject Based Simulations Results", label="tab:ov_perf")
    tex_code =  tex_code.replace(
        "\\label{tab:ov_perf}", "\\begin{adjustbox}{max width=\\textwidth}"
    ).replace(
        "\\end{tabular}","\\end{tabular}\n\\end{adjustbox}\n"
    )
    with open(f"{output_path}/overall_sim_stats__bias_{cfg['prediction_bias']}.tex", "w") as f_tex:
        f_tex.write(tex_code)