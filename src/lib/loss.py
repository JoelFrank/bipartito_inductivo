import torch
import torch.nn.functional as F

def cca_ssg_loss(z1, z2, lambd, num_nodes):
    # CCA-SSG loss function for self-supervised learning
    c = torch.mm(z1.T, z2)
    c = c / num_nodes
    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = off_diagonal(c).pow_(2).sum()
    loss = on_diag + lambd * off_diag
    return loss

def barlow_twins_loss(z1, z2):
    # Barlow Twins loss function
    batch_size = z1.size(0)
    z1_norm = (z1 - z1.mean(0)) / z1.std(0)
    z2_norm = (z2 - z2.mean(0)) / z2.std(0)
    
    c = torch.mm(z1_norm.T, z2_norm) / batch_size
    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = off_diagonal(c).pow_(2).sum()
    loss = on_diag + 0.005 * off_diag
    return loss

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
