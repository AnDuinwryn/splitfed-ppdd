import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pickle
import os
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, cohen_kappa_score
from sklearn.metrics import confusion_matrix
#-----------------------------------------------------------------------------------------------------------------------------#
from cnn_model import CNNModel
import dataWrap
#-----------------------------------------------------------------------------------------------------------------------------#
batch_size = 128
n_epoch = 200
init_lr = 1e-4
store_pkl_dir = 'Results'
store_net_finm = 'cent_cnn_net.pth'
store_pkl_finm = 'cent_cnn_res_9.pkl'
#-----------------------------------------------------------------------------------------------------------------------------#
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

    # Training
    for epoch in range(num_epochs):
        model.train()
        # total_train_loss = 0
        # correct_train = 0
        # total_train = 0

        for batch_data, batch_labels in train_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            output = model(batch_data)
            train_pred = torch.argmax(output, dim=1)
            loss = criterion(output, batch_labels)
            loss.backward()
            optimizer.step()
            
            # total_train_loss += loss.item()
            
            # correct_train += (predictions == batch_labels).sum().item()
            # total_train += batch_labels.size(0)

        # train_accuracy = correct_train / total_train
        # avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
                val_outputs = model(batch_data)
                loss = criterion(val_outputs, batch_labels)
                total_val_loss += loss.item()

                train_pred = torch.argmax(val_outputs, dim=1)
                correct_val += (train_pred == batch_labels).sum().item()
                total_val += batch_labels.size(0)

        val_accuracy = correct_val / total_val
        avg_val_loss = total_val_loss / len(val_loader)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        # Test
        sum_loss = 0
        total_correct = 0
        total_samples = 0
        all_labels = []
        all_preds = []
        all_probs = []

        model_cpy = copy.deepcopy(model).to(device)
        for cli_idx in range(1):
            model_cpy.eval()
            with torch.no_grad():
                for mel_test, labels in test_loader:
                    mel_test, labels = mel_test.to(device), labels.to(device)

                    fx = model(mel_test)
                    probs = torch.softmax(fx, dim=1)

                    sum_loss += criterion(fx, labels).item() / len(val_loader)

                    total_correct += (torch.argmax(fx, dim=1) == labels).sum().item()
                    total_samples += labels.size(0)

                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(torch.argmax(probs, dim=1).cpu().numpy())
                    all_probs.extend(probs[:, 1].cpu().numpy())

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


if __name__ == "__main__":
    model = CNNModel(n_channels=3, n_frequencies=128, n_timeFrames=259)
    mel_dat = dataWrap.fetch_mel_dat(n_cli=1)
    train_ldr = DataLoader(dataWrap.dat_wrap(mel_dat[0]['Train']), batch_size=batch_size, shuffle=True)
    val_ldr = DataLoader(dataWrap.dat_wrap(mel_dat[0]['Valid']), batch_size=batch_size, shuffle=True)
    test_ldr = DataLoader(dataWrap.dat_wrap(mel_dat[0]['Test']), batch_size=batch_size, shuffle=True)

    # Train the model
    train_model(
        model=model,
        train_loader=train_ldr,
        val_loader=val_ldr,
        test_loader=test_ldr,
        num_epochs=n_epoch
    )