import argparse
import os
import pandas as pd
import numpy as np
from preprocessing.helper import low_pass_filter, metadata_from_filename, metadata_from_raw, read_convert_data
from pathlib import Path
import json
with open("config.json", "r") as config_file:
    settings = json.load(config_file)


def windowing(x, y, window_size, sampling_rate, overlap_percentage, apply_filter):
    """
    Segments the data and labels into windows with the option to specify overlap.

    Parameters:
    - x: Input data, expected to be in the shape of (n_samples, n_features).
    - y: Labels for the data, expected to be in the shape of (n_samples,).
    - window_size: Size of each window in milliseconds (default is 50ms).
    - sampling_rate: Sampling frequency of the data in Hz (default is 100Hz).
    - overlap_percentage: Desired percentage of overlap between windows (0-100).

    Returns:
    - segmented_x: List of segmented windows from the data.
    - segmented_y: List of labels for each window, based on the label of the first sample in the window.
    """
    # Calculate the number of samples per window
    samples_per_window = int((window_size / 1000) * sampling_rate)

    # Calculate step size based on overlap
    step_size = int(samples_per_window * (1 - (overlap_percentage / 100)))

    # Initialize the lists to store segmented data and labels
    segmented_x = []
    segmented_y = []

    # Segment the data based on calculated step size and window size
    for start in range(0, len(x) - samples_per_window + 1, step_size):
        end = start + samples_per_window
        if end > len(x):  # Ensure the last window doesn't exceed data length
            break
        window = x[start:end]

        if apply_filter:
            window = low_pass_filter(window, 5, sampling_rate)

        label = y[start]


        segmented_x.append(window)
        segmented_y.append(label)

    return np.array(segmented_x), np.array(segmented_y)


def create_dataset(path, window_size, overlap, binary_class=False, subject_offset=0):
    path = Path(path)
    skip = 0
    data_path = path / "sensors_data"
    labels_path = path / "labels_data"

    subjects = sorted(data_path.glob("*"))
    labels = sorted(labels_path.glob("*"))

    backward_falls = [21, 27, 34, 37, 38, 40, 41, 42]

    for subject_path, label_path in zip(subjects, labels):
        print(f"\rProcessing {subject_path.name}", end="", flush=True)
        label_annotation_df = pd.read_excel(
            label_path,
            sheet_name="Sheet1"
        ).ffill()

        processed_files = []

        # Files with falls
        for row in label_annotation_df.itertuples():
            task_id, start_fall_frame, end_fall_frame, trial = metadata_from_raw(row)
            file_name = f"S{subject_path.name[2:]}T{task_id}R0{trial}.csv"
            file_path = subject_path / file_name
            df = pd.read_csv(file_path, index_col=1) # or 2

            # Add labels
            if  binary_class:
                df.loc[: start_fall_frame, "label"] = "Activity"
                df.loc[start_fall_frame:end_fall_frame, "label"] = "Falling"
                # df.loc[end_fall_frame - 15:end_fall_frame, "label"] = "Activity"
                df.loc[end_fall_frame:, "label"] = "Activity"
            else:
                df.loc[: start_fall_frame, "label"] = "Activity"
                if int(task_id) in backward_falls:
                    df.loc[start_fall_frame:end_fall_frame, "label"] = "BackwardFalling"
                else:
                    df.loc[start_fall_frame:end_fall_frame, "label"] = "OtherFalling"
                # else:
                #     df.loc[start_fall_frame:end_fall_frame, "label"] = "Falling"
                df.loc[end_fall_frame - 15:end_fall_frame, "label"] = "PreImpact"
                df.loc[end_fall_frame: , "label"] = "Activity"

            # Exclude 150ms before impact and Post-Fall
            # df = df.loc[:end_fall_frame - 16]

            # Extract data and labels
            x = df[["AccX", "AccY", "AccZ", "GyrX", "GyrY", "GyrZ", "EulerX", "EulerY", "EulerZ"]].values
            y = df["label"].values

            # Perform segmentation
            x_activity, y_activity = windowing(x[:start_fall_frame], y[:start_fall_frame], window_size, 100, overlap, True)
            x_falling, y_falling = windowing(x[start_fall_frame:end_fall_frame], y[start_fall_frame:end_fall_frame], window_size, 100, overlap, True)
            # x_pre_impact, y_pre_impact = windowing(x[end_fall_frame - 15:end_fall_frame], y[end_fall_frame - 15:end_fall_frame], window_size, 100, overlap, True)
            # x_post_fall, y_post_fall = windowing(x[end_fall_frame:], y[end_fall_frame:], window_size, 100, overlap, True)

            try:
                x = np.concatenate([x_activity, x_falling], axis=0)
                y = np.concatenate([y_activity, y_falling], axis=0)
            except Exception as e:
                print(e)
                skip += 1
                continue

            output_dir_path = \
                f"{settings['data']}/{path.name}/segments/{window_size}ms_{overlap}ov_npseg_filt{'_binary' if binary_class else ''}/{int(subject_path.name[2:]) + subject_offset}/{task_id}/{trial}"
            Path(output_dir_path).mkdir(parents=True, exist_ok=True)
            np.save(f"{output_dir_path}/segments.npy", x)
            np.save(f"{output_dir_path}/labels.npy", y)

            processed_files.append(file_name)

        for file_path in sorted(subject_path.glob("*")):
            if file_path.name.endswith(".csv") and file_path.name not in processed_files:
                subject_id, task_id, trial = metadata_from_filename(file_path.name)
                df = pd.read_csv(file_path, index_col=1)
                df["label"] = "Activity"

                # Extract data and labels
                x = df[["AccX", "AccY", "AccZ", "GyrX", "GyrY", "GyrZ", "EulerX", "EulerY", "EulerZ"]].values
                y = df["label"].values

                # Perform segmentation
                x, y = windowing(x, y, window_size, 100, overlap, True)

                output_dir_path = \
                    f"{settings['data']}/{path.name}/segments/{window_size}ms_{overlap}ov_npseg_filt{'_binary' if binary_class else ''}/{subject_id + subject_offset}/{task_id}/{trial}"
                Path(output_dir_path).mkdir(parents=True, exist_ok=True)
                np.save(f"{output_dir_path}/segments.npy", x)
                np.save(f"{output_dir_path}/labels.npy", y)

                processed_files.append(file_path.name)
    print(f"\nSkipped: {skip}")


def create_on_field(path, window_size, overlap, binary_class=False, subject_offset=1000):
    dates = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    for date in dates:
        files =  [f for f in os.listdir(os.path.join(path, date)) if f.endswith(".csv")]
        for file in files:
            print(f"\rProcessing {file}", end="", flush=True)
            df = read_convert_data(os.path.join(path, date, file))

            df["label"] = "Activity"

            # Extract data and labels
            x = df[["accX", "accY", "accZ", "gyrX", "gyrY", "gyrZ", "roll", "pitch", "yaw"]].values
            y = df["label"].values

            # Perform segmentation
            x, y = windowing(x, y, window_size, 100, overlap, True)

            print(f" --> {x.shape[0]} windows")

            subject_id = int(file.split("_")[1].split(".")[0])
            task_id = 88
            task_dir = f"{settings['data']}/OnField/segments/{window_size}ms_{overlap}ov_npseg_filt_binary/{subject_id}/{task_id}"
            trial = len(os.listdir(task_dir)) + 1 if os.path.isdir(task_dir) else 1

            output_dir_path = f"{task_dir}/{trial}"
            Path(output_dir_path).mkdir(parents=True, exist_ok=True)
            np.save(f"{output_dir_path}/segments.npy", x)
            np.save(f"{output_dir_path}/labels.npy", y)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="This script segments the dataset given as input into windows. Usage example: python windowing.py -d /<PATH>/KFall -w 300 -o 50 -b 1 -s 100"
    )
    # Add arguments
    parser.add_argument('-d', '--dataset', type=str, help='Path to the dataset', required=True)
    parser.add_argument('-w', '--window_size', type=int, help='Size of the time window', required=True)
    parser.add_argument('-o', '--overlap', type=int, help='Percentage of window overlap', required=True)
    parser.add_argument('-b', '--binary_class', type=int, help='Binary or multiclass labels', required=True)
    parser.add_argument('-s', '--subject_offset', type=int, help='Offset for subject, to distinguish subject of other datasets', required=True)
    parser.add_argument('-f', '--on_field', type=int, help='Generate on field', required=False, default=0)

    # Parse the arguments
    args = parser.parse_args()

    print(f"Creating windows form {args.dataset}, window size: {args.window_size}ms, overlap: {args.overlap}%, using {'binary' if bool(args.binary_class) else 'multiclass'} labels, subject offset: {args.subject_offset}")

    create_dataset(
        path=args.dataset,
        window_size=args.window_size,
        overlap=args.overlap,
        binary_class=bool(args.binary_class),
        subject_offset=args.subject_offset
    )

    if args.on_field:
        create_on_field("/mnt/hdd16T/protechto/ThirdPartyDatasets/OnFieldRecordings", args.window_size, args.overlap, True, 0)
