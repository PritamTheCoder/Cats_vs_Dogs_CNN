from torch.nn import Module, Conv2d, Linear, MaxPool2d, AdaptiveAvgPool1d
from torch.nn.functional import relu, dropout
from dataset_loader import *


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
        x = x.view(x.size(0), -1)
        x = self.adPooling(x).squeeze()

        x = self.fc1(x)
        x = relu(x)

        x = self.fc2(x)
        x = relu(x)

        return self.out(x)


imageLoader = ImageLoader(trainData, transform)

dataLoader = DataLoader(imageLoader, batch_size=10, shuffle= True)

data = iter(dataLoader)

images = next(data)

network = Network()
out = network(images[0])
print(out)
