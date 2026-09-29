from ray import tune
from torch import nn
from losses.FocalLoss import FocalLoss
from losses.CostSensitiveLoss import CostSensitiveLoss
import numpy as np


def create_tuning_config():
    alpha = np.random.uniform(0.1, 0.9)
    tc = {
        "conv_1_dim": tune.choice([32, 64, 128]),
        "conv_1_filter": tune.choice([2, 3, 4]),
        "conv_pool": tune.choice([2, 3, 4]),
        "conv_dropout": tune.choice([0.1, 0.3, 0.5, 0.7]),
        "fc": tune.choice([32, 64, 128, 256]),
        "fc_dropout": tune.choice([0.1, 0.3, 0.5, 0.7]),
        "alpha_value": tune.uniform(0, 1),
        "criterion": tune.choice([
            FocalLoss(gamma=2, alpha=[alpha, 1.0 - alpha]),
            nn.CrossEntropyLoss()
        ]),

        "batch_size": tune.choice([32, 64, 128]),
        "lr": tune.loguniform(1e-5, 1e-1),
        "wd": tune.loguniform(1e-6, 1e-2),
    }
    return tc

tuning_config = create_tuning_config()

best_config = {
    "alpha_value": 0.01397150045307849,
    "batch_size": 128,
    "conv_1_dim": 64,
    "conv_1_filter": 4,
    "conv_dropout": 0.4,
    "conv_pool": 2,
    "criterion": nn.CrossEntropyLoss(),
    "fc": 128,
    "fc_dropout": 0.4,
    "lr": 0.001078759046949597,
    "wd": 4.864667834921314e-05,

    "prediction_bias": 0.65
}