import os
import yaml
import pickle
import torch
from torch.utils.data import DataLoader, Subset
from torch import nn, optim
from Data.feature2tensor_wrapper import AudioDataset

class ClientCNNModel(nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=n_channels, out_channels=32, kernel_size=(3, 3), padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 3), padding=(0, 1))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 3))
        self.drop1 = nn.Dropout(.25)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 3), padding=(0, 1))
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.drop1(x)
        x = self.relu(self.conv3(x))
        x = self.pool3(x)
        return x

class ServerCNNModel(nn.Module):
    def __init__(self, n_frequencies: int):
        super().__init__()
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 3), padding=(0, 1))
        self.fc1 = nn.Linear(n_frequencies * 16, 128)
        self.drop2 = nn.Dropout(.5)
        self.fc2 = nn.Linear(128, 2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten(start_dim=1)

    def forward(self, x):
        x = self.relu(self.conv4(x))
        x = self.pool4(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.drop2(x)
        x = self.fc2(x)
        return x

def train_split_learning(client_models, server_model, train_loaders, val_loader, num_epochs, init_learning_rate, client_data_sizes, device):
    criterion = nn.CrossEntropyLoss()
    client_optimizers = [optim.Adam(model.parameters(), lr=init_learning_rate) for model in client_models]
    server_optimizer = optim.Adam(server_model.parameters(), lr=init_learning_rate)

    server_model.to(device)
    for model in client_models: model.to(device)

    total_data_size = sum(client_data_sizes)

    for epoch in range(num_epochs):
        server_model.train()
        for model in client_models: model.train()

        # 假设所有 train_loaders 的长度相同
        for batches in zip(*[iter(train_loader) for train_loader in train_loaders]):
            # 初始化服务器梯度
            server_optimizer.zero_grad()

            # 初始化客户端梯度
            for client_optimizer in client_optimizers:
                client_optimizer.zero_grad()

            # 遍历每个客户端
            for client_idx, (batch_data, batch_labels) in enumerate(batches):
                # 将数据移动到设备
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)

                # 客户端前向传播
                smashed_data = client_models[client_idx](batch_data)

                # 服务器前向传播
                outputs = server_model(smashed_data)
                loss = criterion(outputs, batch_labels)

                # 服务器反向传播
                loss.backward(retain_graph=True)

                # 计算 smashed_data 的梯度
                smashed_data_grad = torch.autograd.grad(loss, smashed_data, retain_graph=True)[0]

                # 客户端反向传播
                smashed_data.backward(smashed_data_grad)

            # 更新服务器参数
            server_optimizer.step()

            # 更新每个客户端参数
            for client_optimizer in client_optimizers:
                client_optimizer.step()

        # 验证阶段
        server_model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)
                smashed_data = client_models[0](batch_data)  # 使用任意客户端模型进行验证
                outputs = server_model(smashed_data)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += batch_labels.size(0)
                correct += predicted.eq(batch_labels).sum().item()

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {val_loss/len(val_loader)}, Accuracy: {100.*correct/total}%')

    return server_model










if __name__ == "__main__":
    # Example usage (you can adjust based on your needs):
    with open(os.path.join(os.getcwd(), 'Config', 'backboneModel.yaml'), "r") as file:
        config = yaml.safe_load(file)

    input_shape = config['model']['input_size'][0]['mel_spectrogram']
    batch_size = config['training']['batch_size']
    n_epochs = config['training']['n_epochs']
    init_learning_rate = config['training']['init_learning_rate']

    # Initialize the global model (server model)
    server_model = ServerCNNModel(128)

    # Assume we have a preprocessed dataset
    # Load data
    to_dir = 'Data/EENT/Processed'
    mel_path = os.path.join(to_dir, 'mel_data.pkl')
    
    with open(mel_path, 'rb') as f:
        loaded_data = pickle.load(f)

    # Initialize dataset and DataLoader
    tensor_data = AudioDataset(loaded_data)

    # Split data into training and validation sets
    train_size = int(0.8 * len(tensor_data))
    val_size = len(tensor_data) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(tensor_data, [train_size, val_size])

    # Split the training dataset into 3 parts for 3 clients
    num_clients = 3
    client_train_size = train_size // num_clients
    client_data_sizes = [client_train_size] * num_clients

    train_loaders = []
    for i in range(num_clients):
        client_train_dataset = Subset(train_dataset, range(i * client_train_size, (i + 1) * client_train_size))
        train_loader = DataLoader(client_train_dataset, batch_size=batch_size, shuffle=True)
        train_loaders.append(train_loader)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize client models (each client has its own model)
    client_models = [ClientCNNModel(input_shape["n_channels"]) for _ in range(num_clients)]

    # Train the split learning model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    server_model = train_split_learning(
        client_models=client_models,
        server_model=server_model,
        train_loaders=train_loaders,
        val_loader=val_loader,
        num_epochs=n_epochs,
        init_learning_rate=init_learning_rate,
        client_data_sizes=client_data_sizes,
        device=device
    )