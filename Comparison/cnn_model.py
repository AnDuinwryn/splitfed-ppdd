import yaml
import os
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self, n_channels: int, n_frequencies: int, n_timeFrames: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=n_channels, out_channels=32, kernel_size=(3, 3), padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 3), padding=(0, 1))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 3))
        self.drop1 = nn.Dropout(.25)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 3), padding=(0, 1))
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 3), padding=(0, 1))
        self.flatten = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(n_frequencies * 16, 128)
        self.drop2 = nn.Dropout(.5)
        self.fc2 = nn.Linear(128, 2)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)

        x= self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.drop1(x)

        x = self.relu(self.conv3(x))
        x = self.pool3(x)

        x = self.relu(self.conv4(x))
        x = self.pool4(x)

        x = self.flatten(x)

        x = self.relu(self.fc1(x))
        x = self.drop2(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    pass
