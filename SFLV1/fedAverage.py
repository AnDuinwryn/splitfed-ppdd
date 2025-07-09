import copy
import torch
#-----------------------------------------------------------------------------------------------------------------------------#
def fed_avg(w, n_cli):                                                    # Federal average designed for equal client sample size
    assert len(w) == n_cli, "The length of weight list should equal the number of clients."
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for iter in range(1, n_cli):
            w_avg[k] += w[iter][k]
        w_avg[k] = torch.div(w_avg[k], n_cli)
    return w_avg
    