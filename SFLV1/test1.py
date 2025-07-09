import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pickle
import os
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, cohen_kappa_score, confusion_matrix

from ast_split_model import ASTFullModel  # 你需要把前面拼接好的类保存为 ast_model.py
import dataWrap  # 假设你已有 fetch_ast_mel_dat 和 dat_wrap
import timm
print(timm.list_models('vit*'))
#-----------------------------超参数设置-----------------------------#
batch_size = 128
n_epoch = 200
init_lr = 1e-4
store_pkl_dir = 'Results'
store_pkl_finm = 'fed_ast_res.pkl'

#----------------------------训练函数----------------------------#
def train_model(model, train_loader, val_loader, test_loader, num_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=init_lr)
    criterion = nn.CrossEntropyLoss()

    test_loss_ls = []
    test_acc_ls = []
    auroc_ls = []
    auprc_ls = []
    f1_ls = []
    kappa_ls = []
    confusion_matrices_data = {}

    for epoch in range(num_epochs):
        model.train()
        for batch_data, batch_labels in train_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            output = model(batch_data)
            loss = criterion(output, batch_labels)
            loss.backward()
            optimizer.step()

        # 验证
        model.eval()
        with torch.no_grad():
            total_val = 0
            correct_val = 0
            for batch_data, batch_labels in val_loader:
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
                output = model(batch_data)
                pred = torch.argmax(output, dim=1)
                correct_val += (pred == batch_labels).sum().item()
                total_val += batch_labels.size(0)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Acc: {correct_val / total_val:.4f}")

        # 测试
        total_correct = 0
        total_samples = 0
        all_labels = []
        all_preds = []
        all_probs = []
        sum_loss = 0

        with torch.no_grad():
            for batch_data, batch_labels in test_loader:
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
                output = model(batch_data)
                probs = torch.softmax(output, dim=1)

                sum_loss += criterion(output, batch_labels).item()
                total_correct += (torch.argmax(output, dim=1) == batch_labels).sum().item()
                total_samples += batch_labels.size(0)

                all_labels.extend(batch_labels.cpu().numpy())
                all_preds.extend(torch.argmax(probs, dim=1).cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())

        acc = total_correct / total_samples
        auroc = roc_auc_score(all_labels, all_probs)
        auprc = average_precision_score(all_labels, all_probs)
        f1 = f1_score(all_labels, all_preds)
        kappa = cohen_kappa_score(all_labels, all_preds)

        print(f"Test Accuracy: {acc:.4f} | AUROC: {auroc:.4f} | AUPRC: {auprc:.4f} | F1: {f1:.4f} | Kappa: {kappa:.4f}")

        test_loss_ls.append(sum_loss)
        test_acc_ls.append(acc)
        auroc_ls.append(auroc)
        auprc_ls.append(auprc)
        f1_ls.append(f1)
        kappa_ls.append(kappa)

        if epoch == n_epoch - 1:
            cm = confusion_matrix(all_labels, all_preds)
            confusion_matrices_data[0] = cm.tolist()

    results = {
        "test_loss": test_loss_ls,
        "test_accuracy": test_acc_ls,
        "auroc": auroc_ls,
        "auprc": auprc_ls,
        "f1": f1_ls,
        "kappa": kappa_ls,
        "confusion_matrices_data": confusion_matrices_data
    }

    os.makedirs(store_pkl_dir, exist_ok=True)
    with open(os.path.join(store_pkl_dir, store_pkl_finm), 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

#-----------------------------运行入口-----------------------------#
if __name__ == "__main__":
    model = ASTFullModel(input_fdim=128, input_tdim=259)
    mel_dat = dataWrap.fetch_mel_dat_ast(n_cli=1)
    train_loader = DataLoader(dataWrap.dat_wrap(mel_dat[0]['Train']), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataWrap.dat_wrap(mel_dat[0]['Valid']), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataWrap.dat_wrap(mel_dat[0]['Test']), batch_size=batch_size, shuffle=False)

    train_model(model, train_loader, val_loader, test_loader, n_epoch)
