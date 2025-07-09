import torch
from torch import nn
import copy
import numpy as np
from model import serv_cnn_net
import fedAverage

#-----------------------------------------------------------------------------------------------------------------------------#
device_serv = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_localEpoch = 5
n_cli = 5
lr_serv = 1e-04
criterion = nn.CrossEntropyLoss()
#-----------------------------------------------------------------------------------------------------------------------------#
# Selectable options
# n_freq = 128
# global_serv_net = serv_cnn_net(n_freq)                                              # Initialize global model
from ast_split_model import input_tdim, input_fdim, ASTServer
global_serv_net = ASTServer(input_fdim=input_fdim, input_tdim=input_tdim)
#-----------------------------------------------------------------------------------------------------------------------------#
global_serv_net.to(device_serv)
serv_w_list = []                                                                    # Store the state_dicts of all clients which finished their local epoches
cli_idx_list = []

batch_acc_train_ls = []
batch_loss_train_ls = []
#-----------------------------------------------------------------------------------------------------------------------------#
def train_serv(cli_idx, smashed_dat, labels, local_iter, is_last_batch):
    global global_serv_net
    global serv_w_list, cli_idx_list
    global batch_acc_train_ls, batch_loss_train_ls
    local_serv_net = copy.deepcopy(global_serv_net).to(device_serv)
    smashed_dat = smashed_dat.to(device_serv); labels = labels.to(device_serv)

    local_serv_net.train()
    servOptimizer = torch.optim.Adam(local_serv_net.parameters(), lr=lr_serv)
    servOptimizer.zero_grad()
    
    fx = local_serv_net(smashed_dat)
    loss = criterion(fx, labels)

    #TODO: Should calculate accuracy here
    preds = torch.argmax(fx, dim=1)
    acc_train = (preds == labels).float().mean().item()

    

    loss.backward()

    d_hx = smashed_dat.grad.clone().detach()
    batch_acc_train_ls.append(acc_train)
    batch_loss_train_ls.append(loss.detach().cpu())
    if is_last_batch:                                                               # All batches for one epoch finished
        avgacc_train = np.mean(batch_acc_train_ls)
        avgloss_train = np.mean(batch_loss_train_ls)
        batch_acc_train_ls = []; batch_loss_train_ls = []
        print("Client_{} train local epoch : {} \tAcc: {:.3f} \t Loss: {:.4f}".format(cli_idx, local_iter, avgacc_train, avgloss_train))
        if local_iter == n_localEpoch - 1:                                          # If all local epoches finished, need to pass the state_dicts to fed server.
            servOptimizer.step()                          
            serv_w_list.append(copy.deepcopy(local_serv_net.state_dict()))
            if cli_idx not in cli_idx_list: cli_idx_list.append(cli_idx)
        if len(cli_idx_list) == n_cli:                                              # Server update
            new_global_serv_w = fedAverage.fed_avg(serv_w_list, n_cli)
            global_serv_net.load_state_dict(new_global_serv_w)
            # Need to re-initialize
            serv_w_list = []; cli_idx_list = []
    return d_hx
#-----------------------------------------------------------------------------------------------------------------------------#
# def eval_serv(cli_idx, smashed_dat, labels, ): # server.eval_serv(self.idx, hx, labels, batch_iter == n_batch - 1)
    
#-----------------------------------------------------------------------------------------------------------------------------#
def fetch_global_serv_state_dict():
    return copy.deepcopy(global_serv_net.state_dict())
#-----------------------------------------------------------------------------------------------------------------------------#
def fetch_global_serv_net():
    return copy.deepcopy(global_serv_net)
#-----------------------------------------------------------------------------------------------------------------------------#
if __name__ == "__main__":
    pass
