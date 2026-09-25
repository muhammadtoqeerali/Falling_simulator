import os, json
import numpy as np
import torch
import constants as const 
import lightning as pl
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
from dataloaders.helper import (
    load,
    dataloader,
    print_labels_distribution,
    print_subjects_distribution,
    data_augmentation
)

class KFoldDataloader(pl.LightningDataModule):
    def __init__(self, root_directory, k: int, batch_size: int):
        super().__init__()
        self.root_directory = root_directory
        self.subjects = np.array([
            s for s in sorted(os.listdir(root_directory))
            if s not in const.DATA_AUGMENTATION_SUBJECTS
        ])
        self.k = k
        self.k_fold = KFold(n_splits=self.k, shuffle=True, random_state=42)
        self.batch_size = batch_size

        self.label_encoder = LabelEncoder()
        self.train_dl = None
        self.validation_dl = None
        self.test_dl = None
        self.input_shape = None
        self.class_weights = None

    def construct(self, fold_index: int, verbose=False):
        # 1) split subjects
        train_idx, test_idx = list(self.k_fold.split(self.subjects))[fold_index]
        train_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=42)
        train_subjects = self.subjects[train_idx]
        validation_subjects = self.subjects[val_idx]
        test_subjects = self.subjects[test_idx]

        # 2) load raw data
        x_train, y_train = load(train_subjects, self.root_directory)
        x_val,   y_val   = load(validation_subjects, self.root_directory)
        x_test,  y_test  = load(test_subjects, self.root_directory)

        # 3) record for model
        self.input_shape = x_train.shape[1:]

        # 5) optional diagnostics
        if verbose:
            print_subjects_distribution(train_subjects, validation_subjects, test_subjects)
            print_labels_distribution(y_train, y_val, y_test)

        # 6) encode labels
        y_train = self.label_encoder.fit_transform(y_train)
        y_val   = self.label_encoder.transform(y_val)
        y_test  = self.label_encoder.transform(y_test)

        print("Labels distribution AFTER VAE Augmentation:")
        print_labels_distribution(y_train, y_val, y_test)

        # 8) compute inverse-frequency *weights* for CrossEntropyLoss
        #counts  = np.bincount(y_train)                # e.g. [N_act , N_fall]

        #weights = torch.tensor([0.30, 0.70], dtype=torch.float32)
        #self.class_weights = weights.clone().detach()

        #weights = len(y_train) / (len(counts) * counts) # auto-compute per-class weights
        #self.class_weights = torch.tensor(weights, dtype=torch.float32)


        # 9) wrap in PyTorch DataLoaders
        self.train_dl      = dataloader(x_train, y_train, batch_size=self.batch_size, shuffle=True)
        self.validation_dl = dataloader(x_val,   y_val,   batch_size=self.batch_size, shuffle=False)
        self.test_dl       = dataloader(x_test,  y_test,  batch_size=self.batch_size, shuffle=False)

    def train_dataloader(self):
        return self.train_dl

    def val_dataloader(self):
        return self.validation_dl

    def test_dataloader(self):
        return self.test_dl
