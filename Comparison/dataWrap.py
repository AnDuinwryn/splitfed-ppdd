import torch
from torch.utils.data import Dataset
import os
import pickle
import numpy as np
#-----------------------------------------------------------------------------------------------------------------------------#
processed_base_dir = os.path.join('Data', 'EENT', 'Processed')
nm_pkl = "mel.pkl"
#-----------------------------------------------------------------------------------------------------------------------------#
class dat_wrap(Dataset):
    def __init__(self, dat):
        if dat is None: raise ValueError("dat_wrap received `None` as dat.")
        self.dat = dat
    
    def __len__(self):
        return len(self.dat)
    
    def __getitem__(self, idx):
        item = self.dat[idx]
        return torch.tensor(item['Feature']), torch.tensor(item['Label'], dtype=torch.long)
#-----------------------------------------------------------------------------------------------------------------------------#
def fetch_mel_dat(n_cli, verbose=False):
    with open(os.path.join(processed_base_dir, nm_pkl), 'rb') as f:
        mel_dat = pickle.load(f)
    if verbose: print("Type of mel_dat: {}".format(type(mel_dat)))
    np.random.shuffle(mel_dat)
    # np_mel_stack = np.stack([item["Feature"] for item in mel_dat])

    # mel_mean = np.mean(np_mel_stack, axis=(0, 2, 3), keepdims=True)
    # mel_std = np.std(np_mel_stack, axis=(0, 2, 3), keepdims=True)
    # np_mel_stack = (np_mel_stack - mel_mean) / (mel_std + 1e-6)
    
    if verbose: print("Len of mel_dat: {}".format(len(mel_dat)))
    #TODO: Evenly splitted into `n_cli` folds, and divide each fold into train set and validation set respectively
    fn_dat_clis = []
    n_mel_row = len(mel_dat)
    np.random.shuffle(mel_dat)
    mel_dat = mel_dat[:n_mel_row // n_cli * n_cli]
    mel_dat_clis = np.array_split(mel_dat, n_cli)
    for dat_cli in mel_dat_clis:
        n_cli_train_share = len(dat_cli)
        np.random.shuffle(dat_cli)
        fn_dat_clis.append({"Train": dat_cli[:int(.64 * n_cli_train_share)], 
                            "Valid": dat_cli[int(.64 * n_cli_train_share):int(.8 * n_cli_train_share)],
                            "Test": dat_cli[int(.8 * n_cli_train_share):]})
        
    if verbose:
        for i, dat_cli in enumerate(fn_dat_clis):
            train_labels = [item['Label'] for item in dat_cli['Train']]
            ls_unique, cnts = np.unique(train_labels, return_counts=True)
            print(f"Client_{i} label distribution: {dict(zip(ls_unique, cnts))}")
    return fn_dat_clis
#-----------------------------------------------------------------------------------------------------------------------------#
if __name__ == "__main__":
    # print(fetch_ast_mel_dat())
    pass