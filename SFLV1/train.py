import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from ast_model import AST_full  # 引入你的模型
from dataWrap import dat_wrap, fetch_mel_dat_ast  # 引入dataWrap和fetch_mel_dat_ast函数
import os
import pickle

# 参数设置
n_clients = 1  # 单个客户端
batch_size = 128
num_epochs = 50

# 加载数据
dat_cli = fetch_mel_dat_ast(n_clients)  # 获取数据，返回的是客户端分割数据

# 使用 datawrap 处理数据
train_data = dat_cli[0]['Train']
valid_data = dat_cli[0]['Valid']

train_dataset = dat_wrap(train_data)
valid_dataset = dat_wrap(valid_data)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# 创建模型
model = AST_full()  # 这里你可以设置标签的维度
model = model.cuda()  # 如果使用GPU

# 优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# 存储测试准确度
test_accuracies = []

# 测试过程
def test_accuracy(model, dataloader):
    model.eval()
    correct_preds = 0
    total_preds = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.cuda(), labels.cuda()

            outputs = model(inputs)
            
            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)

    accuracy = correct_preds / total_preds
    return accuracy

# 训练循环
for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.cuda(), labels.cuda()

        optimizer.zero_grad()
        
        outputs = model(inputs)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # 测试阶段，获取 test accuracy
    test_acc = test_accuracy(model, valid_loader)  # 使用验证集作为测试集
    test_accuracies.append(test_acc)
    print(f"Epoch [{epoch+1}/{num_epochs}], Test Accuracy: {test_acc:.4f}")

# 将测试结果保存到pkl文件
store_pkl_dir = 'Results'  # 确保你的文件夹存在
# 定义pkl文件保存路径
pkl_file = 'test_cent_5cli_ast_4.pkl'
pkl_path = os.path.join(store_pkl_dir, pkl_file)

# 保存test accuracy到pkl文件
with open(pkl_path, 'wb') as f:
    pickle.dump({'test_accuracies': test_accuracies}, f, protocol=pickle.HIGHEST_PROTOCOL)

print("Training finished and test accuracy saved!")
