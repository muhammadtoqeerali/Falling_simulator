import os
import lightning as pl
import numpy as np
import constants as const
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from multiprocessing import cpu_count
from simulation.helper import load_simulation
from dataloaders.helper import load
from sklearn.model_selection import KFold


class SimulationKFoldDataloader(pl.LightningDataModule):
    def __init__(self, root_directory, k: int, batch_size: int):
        super().__init__()
        self.root_directory = root_directory
        self.subjects = np.array([s for s in sorted(os.listdir(root_directory)) if s not in const.DATA_AUGMENTATION_SUBJECTS])
        self.k = k
        self.k_fold = KFold(n_splits=self.k, shuffle=True, random_state=42)
        self.batch_size = batch_size

        self.label_encoder = LabelEncoder()
        self.test_dl = None
        self.input_shape = None

    def construct(self, fold_index: int):
        # Split the subjects in two sets, one for training and one for test
        _, test_subjects_index = list(self.k_fold.split(self.subjects))[fold_index]
        test_subjects = self.subjects[test_subjects_index]

        _, y_test = load(test_subjects, self.root_directory)
        self.label_encoder.fit_transform(y_test)
        del y_test

        # Load data for test
        sessions = load_simulation(test_subjects, self.label_encoder, self.root_directory)

        test = DataLoader(
            sessions,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            num_workers=cpu_count()
        )
        self.test_dl = test

    def test_dataloader(self):
        return self.test_dl
