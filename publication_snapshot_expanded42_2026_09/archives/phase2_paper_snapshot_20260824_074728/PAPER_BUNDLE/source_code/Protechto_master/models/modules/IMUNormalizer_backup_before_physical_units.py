import torch

class IMUNormalizer(torch.nn.Module):
    def __init__(self):
        super(IMUNormalizer, self).__init__()

    def forward(self, x):
        acc_data, gyro_data, angle_data = torch.split(x, 3, dim=2)

        # Normalization and convertion to dps
        gyro_data = gyro_data / torch.tensor(1000, dtype=torch.float32).to(x.device)

        # Normalization ± 4000 mg
        acc_data = acc_data / torch.tensor([4_000, 4_000, 4_000], dtype=torch.float32).to(x.device)

        # Normalization mdps milli-degrees per second, 1 mdps = 0.001 degrees per second (°/s). 2000 dps -> 2_000_000 mdps
        gyro_data = gyro_data / torch.tensor([1_800, 1_800, 1_800], dtype=torch.float32).to(x.device)

        # Normalization yaw = 0-360, pitch = +- 180, roll = +-90
        # angle_data = angle_data / torch.tensor([360, 180, 90], dtype=torch.float32).to(x.device)

        x = torch.concat([acc_data, gyro_data], dim=-1)
        return x
