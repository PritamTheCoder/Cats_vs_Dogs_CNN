import torch
from torch.nn import Module, Conv2d, Linear, MaxPool2d, AdaptiveAvgPool1d
from torch.nn.functional import relu, dropout

class Network(Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv_1 = Conv2d(in_channels=3, out_channels=64, kernel_size=5)
        self.conv_2 = Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.conv_3 = Conv2d(in_channels=128, out_channels=256, kernel_size=5)

        self.maxPooling = MaxPool2d(kernel_size=4)
        self.adPooling = AdaptiveAvgPool1d(256)

        self.fc1 = Linear(in_features=256, out_features=128)
        self.fc2 = Linear(in_features=128, out_features=64)
        self.out = Linear(in_features=64, out_features=2)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.maxPooling(x)
        x = relu(x)

        x = self.conv_2(x)
        x = self.maxPooling(x)
        x = relu(x)

        x = self.conv_3(x)
        x = self.maxPooling(x)
        x = relu(x)

        x = dropout(x)

        # Flatten to (batch, features)
        x = x.view(x.size(0), -1)

        # Reshape to (batch, 1, features) for AdaptiveAvgPool1d
        x = x.unsqueeze(1)  # add channel dimension

        # Adaptive average pooling to output length=256
        x = self.adPooling(x).squeeze(1)  # remove channel dim

        x = self.fc1(x)
        x = relu(x)

        x = self.fc2(x)
        x = relu(x)

        return self.out(x)


# Note:
# This file is now architecture-only.
# It does NOT load any datasets or run training code.
# app.py will handle model loading and inference.
