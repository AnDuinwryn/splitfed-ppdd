import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import dataWrap
import server

# -----------------------------------------------------------------------------------------------------------------------------#
n_localEpoch = server.n_localEpoch  # Global variable, ought to remain fixed
batch_size = 128
# -----------------------------------------------------------------------------------------------------------------------------#

class client(object):
    def __init__(self, cli_idx: int, device, dat_train, dat_val, lr):
        self.idx = cli_idx  # From 0 to n_cli - 1
        self.device = device
        self.ldr_train = DataLoader(dataWrap.dat_wrap(dat_train), batch_size=batch_size, shuffle=True) 
        self.ldr_val = DataLoader(dataWrap.dat_wrap(dat_val), batch_size=batch_size, shuffle=True)
        self.lr = lr

    def train_cli(self, net):
        net.train()  # Train with specific model, net params given
        cliOptimizer = torch.optim.Adam(net.parameters(), lr=self.lr)

        for epoch_iter in range(n_localEpoch):
            n_batch = len(self.ldr_train)
            for batch_iter, (mel_train, labels) in enumerate(self.ldr_train):
                mel_train, labels = mel_train.to(self.device), labels.to(self.device)

                cliOptimizer.zero_grad()
                hx = net(mel_train)
                smashed_dat = hx.clone().detach().requires_grad_(True)

                dAk_t = server.train_serv(self.idx, smashed_dat, labels, epoch_iter, batch_iter == n_batch - 1)
                hx.backward(dAk_t)
                cliOptimizer.step()

        return net.state_dict()

# -----------------------------------------------------------------------------------------------------------------------------#
if __name__ == "__main__":
    pass
