import torch
import copy
import pickle
import os
import client

import fedAverage
import dataWrap
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, cohen_kappa_score
from sklearn.metrics import confusion_matrix
# -----------------------------------------------------------------------------------------------------------------------------#
n_globalEpoch = 50

device_cli = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_cli = client.server.n_cli
lr_cli = 1e-4
device_test = device_cli
criterion = nn.CrossEntropyLoss()
store_pkl_dir = 'Results'

# -----------------------------------------------------------------------------------------------------------------------------#
# Options:
from ast_split_model import input_tdim, input_fdim, ASTClient

# n_channel = 3
mel_dat = dataWrap.fetch_mel_dat_ast(n_cli=n_cli)
# mel_dat = dataWrap.fetch_mel_dat(n_cli=n_cli)
# from model import cli_cnn_net
# global_cli_net = cli_cnn_net(n_channel)  # Initialize global cli model for future distribution
global_cli_net = ASTClient(input_fdim=input_fdim, input_tdim=input_tdim)
store_pkl_finm = 'test_split_5cli_ast_5.pkl'
# store_cli_finm = 'split_cnn_5cli_clinet_997train.pth'
# store_serv_finm = 'split_cnn_5cli_servnet_997train.pth'
# -----------------------------------------------------------------------------------------------------------------------------#
def pipeline():
    
    
    # test_loss_ls = []
    test_acc_ls = []
    # auroc_ls = []
    # auprc_ls = []
    # f1_ls = []
    # kappa_ls = []
    confusion_matrices_data = {}

    for epoch in range(n_globalEpoch):
        print(f"Global epoch: {epoch}")
        local_cli_w_list = []

        for cli_idx in range(n_cli):
            local_cli = client.client(cli_idx, device_cli, mel_dat[cli_idx]['Train'], mel_dat[cli_idx]['Valid'], lr_cli)
            local_cli_w_list.append(copy.deepcopy(local_cli.train_cli(copy.deepcopy(global_cli_net).to(device_cli))))

        global_cli_net.load_state_dict(fedAverage.fed_avg(local_cli_w_list, n_cli))
        # Start evaluation
        global_cli_net_cpy = copy.deepcopy(global_cli_net).to(device_test)
        global_serv_net_cpy = copy.deepcopy(client.server.global_serv_net).to(device_test)

        sum_loss = 0
        total_correct = 0
        total_samples = 0
        all_labels = []
        all_preds = []
        all_probs = []
        for cli_idx in range(n_cli):
            val_loader = DataLoader(dataWrap.dat_wrap(mel_dat[cli_idx]['Test']), batch_size=client.batch_size, shuffle=True)
            global_cli_net_cpy.eval()
            global_serv_net_cpy.eval()
            with torch.no_grad():
                for mel_test, labels in val_loader:
                    mel_test, labels = mel_test.to(device_test), labels.to(device_test)

                    hx = global_cli_net_cpy(mel_test)
                    fx = global_serv_net_cpy(hx)
                    probs = torch.softmax(fx, dim=1)

                    sum_loss += criterion(fx, labels).item() / len(val_loader)

                    total_correct += (torch.argmax(fx, dim=1) == labels).sum().item()
                    total_samples += labels.size(0)

                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(torch.argmax(probs, dim=1).cpu().numpy())
                    all_probs.extend(probs[:, 1].cpu().numpy())

        sum_loss /= n_cli
        acc = total_correct / total_samples if total_samples > 0 else 0.0

        # auroc = roc_auc_score(all_labels, all_probs)
        # auprc = average_precision_score(all_labels, all_probs)
        # f1 = f1_score(all_labels, all_preds)
        # kappa = cohen_kappa_score(all_labels, all_preds)

        print(f"Test Accuracy: {acc:.4f}")

        # test_loss_ls.append(sum_loss)
        test_acc_ls.append(acc)
        # auroc_ls.append(auroc)
        # auprc_ls.append(auprc)
        # f1_ls.append(f1)
        # kappa_ls.append(kappa)

        if epoch == n_globalEpoch - 1:
            cm = confusion_matrix(all_labels, all_preds)
            confusion_matrices_data[cli_idx] = cm.tolist()
        
            model_dir = os.path.join(store_pkl_dir, 'models')
            os.makedirs(model_dir, exist_ok=True)
            
            # torch.save(global_cli_net.state_dict(), os.path.join(model_dir, store_cli_finm))
            # torch.save(client.server.global_serv_net.state_dict(), os.path.join(model_dir, store_serv_finm))

    results = {
        # "test_loss": test_loss_ls,
        "test_accuracy": test_acc_ls
        # "auroc": auroc_ls,
        # "auprc": auprc_ls,
        # "f1": f1_ls,
        # "kappa": kappa_ls,
        # "confusion_matrices_data": confusion_matrices_data
    }

    with open(os.path.join(store_pkl_dir, store_pkl_finm), 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

# -----------------------------------------------------------------------------------------------------------------------------#
if __name__ == "__main__":
    pipeline()
