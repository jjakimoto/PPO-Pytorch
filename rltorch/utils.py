import torch


def flatten_batch(x):
    n_batch = x.size(0)
    x = torch.cat([x_i[0] for x_i in torch.chunk(x, n_batch, 0)], 0)
    return x
