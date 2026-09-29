import torch


class IMUNormalizer(torch.nn.Module):

    def __init__(self):
        super(IMUNormalizer, self).__init__()


    def forward(self, x):

        acc_data, gyro_data, angle_data = torch.split(
            x,
            3,
            dim=2
        )


        acc_data = acc_data / torch.tensor(
            [39.24,39.24,39.24],
            dtype=torch.float32,
            device=x.device
        )


        gyro_data = gyro_data / torch.tensor(
            [1800,1800,1800],
            dtype=torch.float32,
            device=x.device
        )


        x = torch.concat(
            [
                acc_data,
                gyro_data
            ],
            dim=-1
        )


        return x
