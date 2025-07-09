import torch
import os
from ast_split_model import input_tdim, input_fdim, ASTClient, ASTServer
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import dataWrap
import client

n_cli = 3
device_test = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 初始化模型结构
global_cli_net = ASTClient(input_fdim=input_fdim, input_tdim=input_tdim).to(device)
global_serv_net = ASTServer(input_fdim=input_fdim, input_tdim=input_tdim).to(device)

# 加载模型
model_dir = 'Results/models'
store_cli_finm = 'split_ast_5cli_clinet_997train.pth'
store_serv_finm = 'split_ast_5cli_servnet_997train.pth'
global_cli_net.load_state_dict(torch.load(os.path.join(model_dir, store_cli_finm)))
global_serv_net.load_state_dict(torch.load(os.path.join(model_dir, store_serv_finm)))

# 载入数据
mel_dat = dataWrap.fetch_mel_dat_ast(n_cli=n_cli)

# 初始化用于记录所有真实标签和预测标签
all_preds = []
all_labels = []

for cli_idx in range(n_cli):
    val_loader = DataLoader(dataWrap.dat_wrap(mel_dat[cli_idx]['Test']), batch_size=client.batch_size, shuffle=False)
    
    global_cli_net.eval()
    global_serv_net.eval()
    
    with torch.no_grad():
        for mel_test, labels in val_loader:
            mel_test, labels = mel_test.to(device_test), labels.to(device_test)
            
            hx = global_cli_net(mel_test)
            fx = global_serv_net(hx)
            probs = torch.softmax(fx, dim=1)
            all_preds.extend(torch.argmax(probs, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

# 输出混淆矩阵
print("混淆矩阵：")
print(confusion_matrix(all_labels, all_preds))
print("\n分类报告：")
print(classification_report(all_labels, all_preds, target_names=["Healthy", "Dysphonia"]))
