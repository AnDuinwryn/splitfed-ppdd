import yaml
import os
import torch.nn as nn

class ClientCNNModel(nn.Module):
    def __init__(self, n_channels: int, n_frequencies: int, n_timeFrames: int):
        super().__init__()
        # Client model - Front part with conv1, conv2, conv3 and corresponding pooling layers
        self.conv1 = nn.Conv2d(in_channels=n_channels, out_channels=32, kernel_size=(3, 3), padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 3), padding=(0, 1))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 3))
        self.drop1 = nn.Dropout(.25)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 3), padding=(0, 1))
        
        # ReLU activations after each convolution layer
        self.relu = nn.ReLU()
        # Further appended...
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 3), padding=(0, 1))
        self.flatten = nn.Flatten(start_dim=1)
        
        # This will output the "smashed data" to be sent to the server
        

    def forward(self, x):
        x = self.relu(self.conv1(x))  # ReLU after conv1
        x = self.pool1(x)
        x = self.relu(self.conv2(x))  # ReLU after conv2
        x = self.pool2(x)
        x = self.drop1(x)
        x = self.relu(self.conv3(x))  # ReLU after conv3
        x = self.pool3(x)
        x = self.relu(self.conv4(x))
        x = self.pool4(x)
        x = self.flatten(x)
        return x

class ServerCNNModel(nn.Module):
    def __init__(self, n_frequencies: int):
        super().__init__()
        # Server model - Last two conv layers, fully connected layers, and dropout
        
        self.fc1 = nn.Linear(n_frequencies * 16, 128)
        self.drop2 = nn.Dropout(.5)
        self.fc2 = nn.Linear(128, 2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        

        x = self.relu(self.fc1(x))
        x = self.drop2(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    # Configuration loading
    with open(os.path.join(os.getcwd(), 'Config', 'backboneModel.yaml'), "r") as file:
        config = yaml.safe_load(file)

    input_shape = config['model']['input_size'][0]['mel_spectrogram']
    batch_size = config['training']['batch_size']
    model = ClientCNNModel(input_shape['n_channels'], input_shape['n_frequencies'], input_shape['n_timeFrames'])
    server_model = ServerCNNModel(input_shape['n_frequencies'])
