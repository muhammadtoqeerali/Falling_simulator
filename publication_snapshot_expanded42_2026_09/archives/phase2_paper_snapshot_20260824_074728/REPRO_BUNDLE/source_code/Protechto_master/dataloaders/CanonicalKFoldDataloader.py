"""
Canonical five-fold dataloader for the reconstructed
IEEE Sensors experiments.

Historical subject splitting is preserved:

    subjects:
        sorted dataset directories
        excluding DATA_AUGMENTATION_SUBJECTS

    outer split:
        KFold(
            n_splits=5,
            shuffle=True,
            random_state=42
        )

    within each outer-training fold:
        train_test_split(
            ...,
            test_size=0.2,
            random_state=42
        )

Only the REAL training windows are passed to canonical
augmentation.

Validation and test data are always real and unchanged.
"""

import os

import numpy as np
import lightning as pl
import torch

from sklearn.model_selection import (
    KFold,
    train_test_split,
)
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import (
    DataLoader,
    TensorDataset,
)

import constants as const

from dataloaders.helper import load

from dataloaders.canonical_augmentation import (
    apply_canonical_augmentation,
)


VALID_MODES = {
    "original",
    "vae",
    "ctgan",
    "vaegan",
}


class CanonicalKFoldDataloader(
    pl.LightningDataModule
):
    """
    Full-real-data canonical five-fold loader.

    Parameters
    ----------
    root_directory:
        Segmented dataset directory.

    fold_index:
        Zero-based fold index: 0..4.

    mode:
        original / vae / ctgan / vaegan

    batch_size:
        Historical CNN training batch size. Canonical
        experiments use 64.

    augmentation_seed:
        Seed controlling augmentation-side randomness.
        CNN/model RNG must be reset separately by the
        training runner before model construction.
    """

    def __init__(
        self,
        root_directory,
        fold_index,
        mode,
        batch_size=64,
        augmentation_seed=42,
    ):

        super().__init__()

        self.root_directory = str(
            root_directory
        )

        self.fold_index = int(
            fold_index
        )

        self.mode = (
            str(mode)
            .strip()
            .lower()
            .replace("-", "")
            .replace("_", "")
        )

        aliases = {
            "original": "original",
            "orig": "original",
            "vae": "vae",
            "ctgan": "ctgan",
            "gan": "ctgan",
            "vaegan": "vaegan",
        }

        if self.mode not in aliases:

            raise ValueError(
                f"Unknown mode: {mode!r}"
            )

        self.mode = aliases[
            self.mode
        ]

        if self.mode not in VALID_MODES:

            raise ValueError(
                f"Unsupported canonical mode: "
                f"{self.mode}"
            )

        if self.fold_index not in range(5):

            raise ValueError(
                "fold_index must be 0..4"
            )

        self.batch_size = int(
            batch_size
        )

        self.augmentation_seed = int(
            augmentation_seed
        )

        # -------------------------------------------------
        # Historical subject universe.
        # -------------------------------------------------

        self.subjects = np.array(
            [
                subject
                for subject
                in sorted(
                    os.listdir(
                        self.root_directory
                    )
                )
                if (
                    subject
                    not in
                    const.DATA_AUGMENTATION_SUBJECTS
                    and os.path.isdir(
                        os.path.join(
                            self.root_directory,
                            subject,
                        )
                    )
                )
            ]
        )

        self.k_fold = KFold(
            n_splits=5,
            shuffle=True,
            random_state=42,
        )

        self.label_encoder = (
            LabelEncoder()
        )

        # -------------------------------------------------
        # Explicitly no class weighting.
        # -------------------------------------------------

        self.class_weights = None

        # -------------------------------------------------
        # Populated by construct().
        # -------------------------------------------------

        self.train_subjects = None
        self.validation_subjects = None
        self.test_subjects = None

        self.input_shape = None

        self.x_train = None
        self.y_train = None

        self.x_val = None
        self.y_val = None

        self.x_test = None
        self.y_test = None

        self.train_dl = None
        self.validation_dl = None
        self.test_dl = None

        self.real_train_windows = None
        self.real_train_falling = None

        self.final_train_windows = None
        self.final_train_falling = None

        self.validation_windows = None
        self.test_windows = None

        self.augmentation_metadata = None


    def _subject_split(
        self
    ):
        """
        Reproduce the historical subject split exactly.
        """

        folds = list(
            self.k_fold.split(
                self.subjects
            )
        )

        outer_train_idx, test_idx = (
            folds[
                self.fold_index
            ]
        )

        (
            train_idx,
            validation_idx,
        ) = train_test_split(
            outer_train_idx,
            test_size=0.2,
            random_state=42,
        )

        train_subjects = (
            self.subjects[
                train_idx
            ]
        )

        validation_subjects = (
            self.subjects[
                validation_idx
            ]
        )

        test_subjects = (
            self.subjects[
                test_idx
            ]
        )

        return (
            train_subjects,
            validation_subjects,
            test_subjects,
        )


    @staticmethod
    def _make_loader(
        x,
        y,
        *,
        batch_size,
        shuffle,
    ):

        dataset = TensorDataset(
            torch.tensor(
                x,
                dtype=torch.float32,
            ),
            torch.tensor(
                y,
                dtype=torch.long,
            ),
        )

        return DataLoader(
            dataset,
            batch_size=int(
                batch_size
            ),
            shuffle=bool(
                shuffle
            ),
            pin_memory=True,
        )


    def construct(
        self,
        verbose=True,
    ):
        """
        Build one canonical fold.

        Augmentation touches TRAINING DATA ONLY.
        """

        (
            self.train_subjects,
            self.validation_subjects,
            self.test_subjects,
        ) = self._subject_split()

        # -------------------------------------------------
        # Load full REAL datasets.
        # -------------------------------------------------

        x_train, y_train = load(
            self.train_subjects,
            self.root_directory,
        )

        x_val, y_val = load(
            self.validation_subjects,
            self.root_directory,
        )

        x_test, y_test = load(
            self.test_subjects,
            self.root_directory,
        )

        self.input_shape = (
            x_train.shape[1:]
        )

        # -------------------------------------------------
        # Historical label-encoder ordering.
        # Fit TRAIN only; transform validation/test.
        # -------------------------------------------------

        y_train = (
            self.label_encoder
            .fit_transform(
                y_train
            )
        )

        y_val = (
            self.label_encoder
            .transform(
                y_val
            )
        )

        y_test = (
            self.label_encoder
            .transform(
                y_test
            )
        )

        # -------------------------------------------------
        # Record REAL training state before augmentation.
        # -------------------------------------------------

        self.real_train_windows = int(
            len(x_train)
        )

        self.real_train_falling = int(
            np.sum(
                y_train > 0
            )
        )

        self.validation_windows = int(
            len(x_val)
        )

        self.test_windows = int(
            len(x_test)
        )

        # -------------------------------------------------
        # CANONICAL AUGMENTATION
        #
        # original:
        #     real only
        #
        # vae:
        #     real + two warps + 2x VAE
        #
        # ctgan:
        #     real + two warps + 1x GAN
        #
        # vaegan:
        #     real + two warps + 2x VAE + 1x GAN
        #
        # No OnField.
        # No class weights.
        # -------------------------------------------------

        (
            x_train,
            y_train,
            augmentation_metadata,
        ) = apply_canonical_augmentation(
            x_train,
            y_train,
            mode=self.mode,
            seed=self.augmentation_seed,
            verbose=verbose,
        )

        self.augmentation_metadata = (
            augmentation_metadata
        )

        self.final_train_windows = int(
            len(x_train)
        )

        self.final_train_falling = int(
            np.sum(
                y_train > 0
            )
        )

        # -------------------------------------------------
        # Preserve arrays for forensic validation.
        # -------------------------------------------------

        self.x_train = x_train
        self.y_train = y_train

        self.x_val = x_val
        self.y_val = y_val

        self.x_test = x_test
        self.y_test = y_test

        # -------------------------------------------------
        # DataLoaders.
        # -------------------------------------------------

        self.train_dl = (
            self._make_loader(
                x_train,
                y_train,
                batch_size=(
                    self.batch_size
                ),
                shuffle=True,
            )
        )

        self.validation_dl = (
            self._make_loader(
                x_val,
                y_val,
                batch_size=(
                    self.batch_size
                ),
                shuffle=False,
            )
        )

        self.test_dl = (
            self._make_loader(
                x_test,
                y_test,
                batch_size=(
                    self.batch_size
                ),
                shuffle=False,
            )
        )

        if verbose:

            print(
                "=" * 90
            )

            print(
                "CANONICAL FIVE-FOLD DATASET"
            )

            print(
                "=" * 90
            )

            print(
                f"Fold             : "
                f"{self.fold_index + 1}"
            )

            print(
                f"Mode             : "
                f"{self.mode}"
            )

            print(
                f"Augmentation seed: "
                f"{self.augmentation_seed}"
            )

            print(
                f"Batch size       : "
                f"{self.batch_size}"
            )

            print()

            print(
                "Train subjects:"
            )

            print(
                list(
                    self.train_subjects
                )
            )

            print()

            print(
                "Validation subjects:"
            )

            print(
                list(
                    self.validation_subjects
                )
            )

            print()

            print(
                "Test subjects:"
            )

            print(
                list(
                    self.test_subjects
                )
            )

            print()

            print(
                f"Real train windows : "
                f"{self.real_train_windows}"
            )

            print(
                f"Real train Falling : "
                f"{self.real_train_falling}"
            )

            print(
                f"Final train windows: "
                f"{self.final_train_windows}"
            )

            print(
                f"Final train Falling: "
                f"{self.final_train_falling}"
            )

            print(
                f"Validation windows : "
                f"{self.validation_windows}"
            )

            print(
                f"Test windows       : "
                f"{self.test_windows}"
            )

            print()

            print(
                "Encoded classes:",
                list(
                    self.label_encoder.classes_
                )
            )

            print(
                "Class weights:",
                self.class_weights
            )

            print(
                "=" * 90
            )

        return self


    def train_dataloader(
        self
    ):

        return self.train_dl


    def val_dataloader(
        self
    ):

        return self.validation_dl


    def test_dataloader(
        self
    ):

        return self.test_dl
