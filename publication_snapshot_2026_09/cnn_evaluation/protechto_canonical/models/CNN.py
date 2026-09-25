from torch import nn
from models.helper import calculate_maxpool1d_output_length, calculate_conv1d_output_length
from models.modules.IMUNormalizer import IMUNormalizer



class CNN(nn.Module):
    def __init__(self, n_features: int, n_classes: int, config):
        super(CNN, self).__init__()
        self.name = 'CNN'
        self.n_features = n_features - 3
        self.n_classes = n_classes

        # Model components
        conv_1_dim = config["conv_1_dim"]
        conv_1_filter = config["conv_1_filter"]

        # Normalization
        self.normalizer = IMUNormalizer()

        # Convolutions
        self.conv_1 =  nn.Sequential(
            nn.Conv1d(self.n_features, conv_1_dim, conv_1_filter),
            nn.BatchNorm1d(conv_1_dim),
            nn.PReLU(),
            nn.MaxPool1d(config["conv_pool"]),
            nn.Dropout(p=0.1)
        )

        self.conv_2 =  nn.Sequential(
            nn.Conv1d(conv_1_dim, conv_1_dim, conv_1_filter),
            nn.BatchNorm1d(conv_1_dim),
            nn.PReLU(),
            nn.MaxPool1d(config["conv_pool"]),
            nn.Dropout(p=config["conv_dropout"])
        )

        # Calculate the output dimension of the convolutions
        self.conv_1_out = calculate_maxpool1d_output_length(calculate_conv1d_output_length(30, conv_1_filter), config["conv_pool"])
        self.conv_2_out = calculate_maxpool1d_output_length(calculate_conv1d_output_length(self.conv_1_out, conv_1_filter), config["conv_pool"])

        # Multi-head Attention Layer
        # self.attention = nn.MultiheadAttention(embed_dim=self.conv_2_out, num_heads=1, batch_first=True)

        # Fully connected
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.conv_2_out * conv_1_dim, config["fc"]),  # self.dim * 6 , 704
            nn.PReLU(),
            nn.Dropout(p=config["fc_dropout"]),
            nn.Linear(config["fc"], self.n_classes)
        )

    def forward(self, x):
        x = self.normalizer(x)

        x = x.permute(0, 2, 1)
        x = self.conv_1(x)
        x = self.conv_2(x)
        # x = x.permute(0, 2, 1)
        # x, _ = self.attention(x, x, x)
        output = self.fc(x)

        return output

    def get_name(self):
        return self.name
