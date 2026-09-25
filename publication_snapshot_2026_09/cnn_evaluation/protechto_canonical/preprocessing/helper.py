import argparse
import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path
from scipy.signal import butter, filtfilt
import constants as const


def read_convert_data(file):
    df = pd.read_csv(file, index_col=0)
    df = df[df["type"] == 1]
    df.drop(
        ["type", "reserved1", "reserved2", "reserved3", "vIn", "vBatt", "hall", "inputs", "outputs", "aiOut0", "aiOut1"],
        axis=1,
        inplace=True
    )
    return df

def metadata_from_filename(filename):
    # Subject_ID, Task_ID, Trial
    return int(filename[1:3]), int(filename[4:6]), int(filename[7:-4])

def metadata_from_raw(line):
    # Task_ID, Start fall frame, End fall frame, Trial
    return line[1].split("(")[1].split(")")[0], line[-2], line[-1], line[3]

def find_mode(char_array):
    # Use Counter to count the frequency of each character
    counter = Counter(char_array)

    # Find the character(s) with the highest frequency
    max_count = max(counter.values())

    # Find all characters that have the max frequency
    mode_chars = [char for char, count in counter.items() if count == max_count]

    return mode_chars

def prioritize_falling(y):
    return "Falling" if "Falling" in y else "Activity"

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

def low_pass_filter(data, cutoff_frequency, sampling_rate):
    """
    Apply a low-pass filter to the data.

    :param data: The input signal data as a numpy array.
    :param cutoff_frequency: The cutoff frequency for the low-pass filter.
    :param sampling_rate: The sampling rate of the data.
    :return: The filtered data as a numpy array.
    """
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_frequency / nyquist

    # Get the filter coefficients
    b, a = butter(1, normal_cutoff, btype='low', analog=False)

    # Apply the filter to the data
    filtered_data = np.zeros_like(data)
    for timeseries in range(data.shape[1]):
        filtered_data[:, timeseries] = filtfilt(b, a, data[:, timeseries])

    return filtered_data

def create_binary_dataset(path, window_size, overlap, subject_offset=0):
    path = Path(path)
    skip = 0
    data_path = path / "sensors_data"
    labels_path = path / "labels_data"

    subjects = sorted(data_path.glob("*"))
    labels = sorted(labels_path.glob("*"))

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
            df = pd.read_csv(file_path, index_col=1)

            # Add labels
            df.loc[: start_fall_frame, "label"] = "Activity"
            df.loc[start_fall_frame:end_fall_frame, "label"] = "Falling"
            df.loc[end_fall_frame - 15:end_fall_frame, "label"] = "Activity"
            df.loc[end_fall_frame:, "label"] = "Activity"

            # Extract data and labels
            x = df[["AccX", "AccY", "AccZ", "GyrX", "GyrY", "GyrZ", "EulerX", "EulerY", "EulerZ"]].values
            y = df["label"].values

            # Perform segmentation
            x_activity, y_activity = windowing(
                x[:start_fall_frame],
                y[:start_fall_frame],
                window_size,
                100,
                overlap,
                True
            )
            x_falling, y_falling = windowing(
                x[start_fall_frame:end_fall_frame - 15],
                y[start_fall_frame:end_fall_frame],
                window_size,
                100,
                overlap,
                True
            )

            try:
                x = np.concatenate([x_activity, x_falling], axis=0)
                y = np.concatenate([y_activity, y_falling], axis=0)
            except Exception as _:
                skip += 1
                continue

            output_dir_path = f"../data/{path.name}/segments/{window_size}ms_{overlap}ov_filt_binary/{int(subject_path.name[2:]) + subject_offset}/{task_id}/{trial}"
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
                    f"../data/{path.name}/segments/{window_size}ms_{overlap}ov_filt_binary/{subject_id + subject_offset}/{task_id}/{trial}"
                Path(output_dir_path).mkdir(parents=True, exist_ok=True)
                np.save(f"{output_dir_path}/segments.npy", x)
                np.save(f"{output_dir_path}/labels.npy", y)

                processed_files.append(file_path.name)
    print(f"\nSkipped: {skip}")

def create_full_multiclass_dataset(path, window_size, overlap, subject_offset=0):
    path = Path(path)
    skip = 0
    data_path = path / "sensors_data"
    labels_path = path / "labels_data"

    subjects = sorted(data_path.glob("*"))
    labels = sorted(labels_path.glob("*"))

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
            df = pd.read_csv(file_path, index_col=1)

            # Add labels
            df.loc[start_fall_frame:end_fall_frame, "label"] = row.Description

            # Extract data and labels
            x = df[["AccX", "AccY", "AccZ", "GyrX", "GyrY", "GyrZ", "EulerX", "EulerY", "EulerZ"]].values
            y = df["label"].values

            try:
                # Perform segmentation
                x_falling, y_falling = windowing(
                    x[start_fall_frame:end_fall_frame - 15],
                    y[start_fall_frame:end_fall_frame],
                    window_size,
                    100,
                    overlap,
                    True
                )

                x = x_falling
                y = y_falling
            except Exception as _:
                skip += 1
                continue

            output_dir_path = f"../data/{path.name}/segments/{window_size}ms_{overlap}ov_filt_fullmulti/{int(subject_path.name[2:]) + subject_offset}/{task_id}/{trial}"
            Path(output_dir_path).mkdir(parents=True, exist_ok=True)
            np.save(f"{output_dir_path}/segments.npy", x)
            np.save(f"{output_dir_path}/labels.npy", y)

            processed_files.append(file_name)

        for file_path in sorted(subject_path.glob("*")):
            if file_path.name.endswith(".csv") and file_path.name not in processed_files:
                subject_id, task_id, trial = metadata_from_filename(file_path.name)
                df = pd.read_csv(file_path, index_col=1)
                df["label"] = const.TASKS_DESCRIPTIONS[task_id]

                # Extract data and labels
                x = df[["AccX", "AccY", "AccZ", "GyrX", "GyrY", "GyrZ", "EulerX", "EulerY", "EulerZ"]].values
                y = df["label"].values

                # Perform segmentation
                x, y = windowing(x, y, window_size, 100, overlap, True)

                output_dir_path = \
                    f"../data/{path.name}/segments/{window_size}ms_{overlap}ov_filt_fullmulti/{subject_id + subject_offset}/{task_id}/{trial}"
                Path(output_dir_path).mkdir(parents=True, exist_ok=True)
                np.save(f"{output_dir_path}/segments.npy", x)
                np.save(f"{output_dir_path}/labels.npy", y)

                processed_files.append(file_path.name)
    print(f"\nSkipped: {skip}")

def create_cluster_multiclass_fall_dataset(path, window_size, overlap, subject_offset=0):
    path = Path(path)
    skip = 0
    data_path = path / "sensors_data"
    labels_path = path / "labels_data"

    subjects = sorted(data_path.glob("*"))
    labels = sorted(labels_path.glob("*"))

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
            df = pd.read_csv(file_path, index_col=1)

            label = "Falling"
            if int(task_id) in const.BACKWARD_FALLS:
                label = "Backward Fall"
            if int(task_id) in const.FORWARD_FALLS:
                label = "Forward Fall"
            if int(task_id) in const.LATERAL_FALLS:
                label = "Lateral Fall"

            # Add labels
            df.loc[start_fall_frame:end_fall_frame, "label"] = label

            # Extract data and labels
            x = df[["AccX", "AccY", "AccZ", "GyrX", "GyrY", "GyrZ", "EulerX", "EulerY", "EulerZ"]].values
            y = df["label"].values

            try:
                # Perform segmentation
                x_falling, y_falling = windowing(
                    x[start_fall_frame:end_fall_frame - 15],
                    y[start_fall_frame:end_fall_frame],
                    window_size,
                    100,
                    overlap,
                    True
                )

                x = x_falling
                y = y_falling
            except Exception as _:
                skip += 1
                continue

            output_dir_path = f"../data/{path.name}/segments/{window_size}ms_{overlap}ov_filt_clusterfall/{int(subject_path.name[2:]) + subject_offset}/{task_id}/{trial}"
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
                    f"../data/{path.name}/segments/{window_size}ms_{overlap}ov_filt_clusterfall/{subject_id + subject_offset}/{task_id}/{trial}"
                Path(output_dir_path).mkdir(parents=True, exist_ok=True)
                np.save(f"{output_dir_path}/segments.npy", x)
                np.save(f"{output_dir_path}/labels.npy", y)

                processed_files.append(file_path.name)
    print(f"\nSkipped: {skip}")


if __name__ == '__main__':
    create_binary_dataset("/home/shared/data/protechto/KFall/", 300, 50, 100)