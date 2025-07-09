import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from ast_model import AST_full  # 引入AST_full模型
from dataWrap import dat_wrap, fetch_mel_dat_ast  # 引入dataWrap和fetch_mel_dat_ast函数
import os
import pickle
import copy

# 配置参数
n_clients = 5  # 客户端数量
batch_size = 128
num_epochs = 50
lr_cli = 1e-4  # 客户端学习率
local_epochs = 3  # 每个客户端本地训练的epoch数量
store_pkl_dir = 'Results'  # 结果保存路径
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载数据
mel_dat = fetch_mel_dat_ast(n_cli=n_clients)  # 获取数据，返回的是客户端分割数据

# 使用 datawrap 处理数据
train_data = [dat_wrap(mel_dat[i]['Train']) for i in range(n_clients)]
valid_data = [dat_wrap(mel_dat[i]['Valid']) for i in range(n_clients)]

train_loaders = [DataLoader(train_data[i], batch_size=batch_size, shuffle=True) for i in range(n_clients)]
valid_loaders = [DataLoader(valid_data[i], batch_size=batch_size, shuffle=False) for i in range(n_clients)]

# 初始化全局模型
global_cli_net = AST_full(input_fdim=128, input_tdim=259, model_size='tiny224')
global_cli_net.to(device)

# 优化器
optimizer = optim.Adam(global_cli_net.parameters(), lr=lr_cli)
criterion = nn.CrossEntropyLoss()

# 存储测试精度
test_accuracies = []

# 测试函数
def test_accuracy(model, dataloaders):
    model.eval()
    correct_preds = 0
    total_preds = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for loader in dataloaders:
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)

                correct_preds += (predicted == labels).sum().item()
                total_preds += labels.size(0)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

    accuracy = correct_preds / total_preds
    return accuracy

# 联邦平均函数
def fed_avg(local_weights, n_clients):
    global_weights = copy.deepcopy(local_weights[0])
    for key in global_weights:
        global_weights[key] = sum([local_weights[i][key] for i in range(n_clients)]) / n_clients
    return global_weights

# 训练循环
for epoch in range(num_epochs):
    print(f"Global Epoch [{epoch+1}/{num_epochs}]")

    # 本地训练：每个客户端独立训练
    local_weights = []
    for i in range(n_clients):
        # 每个客户端基于当前全局模型进行训练
        model = AST_full(input_fdim=128, input_tdim=259, model_size='tiny224')
        model.to(device)
        model.load_state_dict(global_cli_net.state_dict())  # 使用当前全局模型的参数
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=lr_cli)

        for local_epoch in range(local_epochs):  # 每个客户端的本地训练epoch
            print(f"  Client {i+1}, Local Epoch {local_epoch+1}/{local_epochs}")
            for inputs, labels in train_loaders[i]:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # 保存每个客户端的训练结果
        local_weights.append(copy.deepcopy(model.state_dict()))

    # 聚合模型权重：联邦平均
    global_weights = fed_avg(local_weights, n_clients)
    global_cli_net.load_state_dict(global_weights)  # 更新全局模型

    # 评估：计算测试精度
    acc = test_accuracy(global_cli_net, valid_loaders)
    test_accuracies.append(acc)

    print(f"Global Test Accuracy: {acc:.4f}")

# 保存结果到pkl文件
os.makedirs(store_pkl_dir, exist_ok=True)
pkl_file = 'test_fed_5cli_ast_5.pkl'
pkl_path = os.path.join(store_pkl_dir, pkl_file)

with open(pkl_path, 'wb') as f:
    pickle.dump({'test_accuracies': test_accuracies}, f, protocol=pickle.HIGHEST_PROTOCOL)

print("Training finished and test accuracy saved!")
