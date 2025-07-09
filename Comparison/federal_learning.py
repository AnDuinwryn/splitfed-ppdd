import torch
import torch.nn as nn
import torch.optim as optim
import copy
import os
import pickle
#-----------------------------------------------------------------------------------------------------------------------------#
from torch.utils.data import DataLoader
from cnn_model import CNNModel
import dataWrap
#-----------------------------------------------------------------------------------------------------------------------------#
batch_size = 128
n_epoch = 200
init_lr = 1e-4
n_cli = 5
store_pkl_dir = 'Results'
# store_net_finm = 'fed_cnn_net.pth'
store_pkl_finm = 'fed_cnn_res_9.pkl'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, cohen_kappa_score
from sklearn.metrics import confusion_matrix
#-----------------------------------------------------------------------------------------------------------------------------#
def fed_avg(w, n_cli):                                                    # Federal average designed for equal client sample size
    assert len(w) == n_cli, "The length of weight list should equal the number of clients."
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for iter in range(1, n_cli):
            w_avg[k] += w[iter][k]
        w_avg[k] = torch.div(w_avg[k], n_cli)
    return w_avg
#-----------------------------------------------------------------------------------------------------------------------------#

def train_federated_model(global_model, train_loaders, val_loaders, test_loaders, num_epochs, init_learning_rate, device):
    criterion = nn.CrossEntropyLoss()
    global_model.to(device)

    test_loss_ls = []
    test_acc_ls = []
    auroc_ls = []
    auprc_ls = []
    f1_ls = []
    kappa_ls = []
    confusion_matrices_data = {}

    for epoch in range(num_epochs):
        # Train
        local_model_weights = []
        for i in range(n_cli):
            local_model = copy.deepcopy(global_model)
            optimizer = optim.Adam(local_model.parameters(), lr=init_learning_rate)

            local_model.to(device)
            local_model.train()
            for batch_data, batch_labels in train_loaders[i]:
                optimizer.zero_grad()

                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
                output = local_model(batch_data)
                loss = criterion(output, batch_labels)
                loss.backward()
                optimizer.step()

            local_model_weights.append(local_model.state_dict())
        global_model.load_state_dict(fed_avg(local_model_weights, n_cli))

        # Valid
        # global_model.eval()
        # val_loss = 0.0
        # val_accuracy = 0.0
        # total_correct = 0
        # total_samples = 0

        # with torch.no_grad():
        #     for val_loader in val_loaders:
        #         for batch_data, batch_labels in val_loader:
        #             batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
        #             output = global_model(batch_data)
        #             loss = criterion(output, batch_labels)
        #             val_loss += loss.item() / len(val_loader)
        #             predictions = torch.argmax(output, dim=1)
        #             total_correct += (predictions == batch_labels).sum().item()
        #             total_samples += batch_labels.size(0)

        # val_loss /= len(val_loaders)
        # val_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        # print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        # Test
        sum_loss = 0
        total_correct = 0
        total_samples = 0
        all_labels = []
        all_preds = []
        all_probs = []
        for cli_idx in range(n_cli):
            test_ldr = test_ldr_list[cli_idx]
            global_model.eval()
            with torch.no_grad():
                for mel_test, labels in test_ldr:
                    mel_test, labels = mel_test.to(device), labels.to(device)

                    fx = global_model(mel_test)
                    probs = torch.softmax(fx, dim=1)

                    sum_loss += criterion(fx, labels).item() / len(test_ldr)

                    total_correct += (torch.argmax(fx, dim=1) == labels).sum().item()
                    total_samples += labels.size(0)

                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(torch.argmax(probs, dim=1).cpu().numpy())
                    all_probs.extend(probs[:, 1].cpu().numpy())

        sum_loss /= n_cli
        acc = total_correct / total_samples if total_samples > 0 else 0.0

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
            confusion_matrices_data[cli_idx] = cm.tolist()
        
            model_dir = os.path.join(store_pkl_dir, 'models')
            os.makedirs(model_dir, exist_ok=True)

    results = {
        "test_loss": test_loss_ls,
        "test_accuracy": test_acc_ls,
        "auroc": auroc_ls,
        "auprc": auprc_ls,
        "f1": f1_ls,
        "kappa": kappa_ls,
        "confusion_matrices_data": confusion_matrices_data
    }
    with open(os.path.join(store_pkl_dir, store_pkl_finm), 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    return
#-----------------------------------------------------------------------------------------------------------------------------#
if __name__ == "__main__":
    global_model = CNNModel(3, 128, 259)
    train_ldr_list = []
    val_ldr_list = []
    test_ldr_list = []
    mel_dat = dataWrap.fetch_mel_dat(n_cli=n_cli)
    for client_data in mel_dat:
        train_ldr_list.append(DataLoader(dataWrap.dat_wrap(client_data['Train']), batch_size=batch_size, shuffle=True))
        val_ldr_list.append(DataLoader(dataWrap.dat_wrap(client_data['Valid']), batch_size=batch_size, shuffle=True))
        test_ldr_list.append(DataLoader(dataWrap.dat_wrap(client_data['Test']), batch_size=batch_size, shuffle=True))

    train_federated_model(
        global_model=global_model,
        train_loaders=train_ldr_list,
        val_loaders=val_ldr_list,
        test_loaders=test_ldr_list,
        num_epochs=n_epoch,
        init_learning_rate=init_lr,
        device=device
    )