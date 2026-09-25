import os
import torch
import constants as const
import numpy as np
from multiprocessing import cpu_count
from torch.utils.data import TensorDataset, DataLoader
import json
from preprocessing.time_warping import time_warping

with open("config.json", "r") as config_file:
    settings = json.load(config_file)


def load(subjects, target_directory):
    x, y = [], []
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

                x.extend(x_i)
                y.extend(y_i)

    x = np.array(x)
    y = np.array(y)

    return x, y

def load_on_field(target_directory, encoder):
    sessions = []

    for subject in sorted(os.listdir(target_directory)):
        for task in sorted(
                [t for t in os.listdir(os.path.join(target_directory, str(subject))) if not t.startswith(".")]
        ):
            for trial in sorted(
                    [t for t in os.listdir(os.path.join(target_directory, str(subject), str(task))) if
                     not t.startswith(".")]
            ):
                # timestamps = np.load(os.path.join(target_directory, subject, task, trial, "timestamps.npy"))
                x_i = np.load(os.path.join(target_directory, subject, task, trial, "segments.npy"))
                y_i = np.load(os.path.join(target_directory, subject, task, trial, "labels.npy"))
                session = {
                    "subject": subject,
                    "task": task,
                    "trial": trial,
                    # "timestamps": timestamps.astype(str),
                    "x": torch.tensor(x_i, dtype=torch.float32),
                    "y": torch.tensor(encoder.transform(y_i)).long()
                }
                sessions.append(session)
    return sessions


def data_augmentation(x, y, encoder):
    falling_idx = np.where(y > 0)[0]

    x_falling = x[falling_idx]
    y_falling = y[falling_idx]

    # Time warping with sigma 1 and 2
    # x_falling_warped_0 = np.array([time_warping(x_falling[i], sigma=0.5) for i in range(x_falling.shape[0])])
    x_falling_warped_1 = np.array([time_warping(x_falling[i], sigma=1) for i in range(x_falling.shape[0])])
    x_falling_warped_2 = np.array([time_warping(x_falling[i], sigma=2) for i in range(x_falling.shape[0])])
    # x_falling_warped_3 = np.array([time_warping(x_falling[i], sigma=1.5) for i in range(x_falling.shape[0])])

    # Augment with a session of on field data
    x_field_20241106, y_field_20241106 = load(const.DATA_AUGMENTATION_SUBJECTS, f"{settings['data']}/OnField/segments/300ms_50ov_npseg_filt_binary")
    y_field_20241106 = encoder.transform(y_field_20241106)

    # x_field_1223, y_field_1223 = load(["10027", "10083"],"data/OnFieldRecordings/20241223/segments/300ms_50ov_npseg_filt")
    # y_field_1223 = encoder.transform(y_field_1223)

    x_augmented = np.concat([x, x_falling_warped_1, x_falling_warped_2, x_field_20241106], axis=0) # x_falling_warped_0, x_falling_warped_3], axis=0)
    y_augmented = np.concat([y, y_falling, y_falling, y_field_20241106])
    return x_augmented, y_augmented


def dataloader(x: np.ndarray, y: np.ndarray, shuffle: bool, batch_size: int):
    return DataLoader(
        TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y).long()),
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=cpu_count()
    )

def print_subjects_distribution(train_subjects, validation_subjects, test_subjects):
    print("-----------------------------------------------------------------------------------------------------------")
    print("Subjects distribution:")
    print(f"    Train subjects: {sorted([int(s) for s in train_subjects])}")
    print(f"    Validation subjects: {sorted([int(s) for s in validation_subjects])}")
    print(f"    Test subjects: {sorted([int(s) for s in test_subjects])}")
    print("-----------------------------------------------------------------------------------------------------------")


def print_labels_distribution(y_train, y_val, y_test):
    print("-----------------------------------------------------------------------------------------------------------")
    print("Labels distribution:")
    for _set, name in zip([y_train, y_val, y_test], ["Train", "Validation", "Test"]):
        print(f"    {name}")
        labels, counts = np.unique(_set, return_counts=True)
        for label, count in zip(labels, counts):
            print(f"        └── {label}: {count}")
    print("-----------------------------------------------------------------------------------------------------------")