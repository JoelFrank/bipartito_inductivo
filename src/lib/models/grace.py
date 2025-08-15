import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GraceEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, activation, base_model=GCNConv, k=2):
        super().__init__()
        self.base_model, self.k, self.activation = base_model, k, activation
        assert k >= 2
        self.conv = [base_model(in_channels, 2 * out_channels)]
        for _ in range(1, k - 1):
            self.conv.append(base_model(2 * out_channels, 2 * out_channels))
        self.conv.append(base_model(2 * out_channels, out_channels))
        self.conv = nn.ModuleList(self.conv)
    def forward(self, x, edge_index):
        for i in range(self.k): x = self.activation(self.conv[i](x, edge_index))
        return x

class GraceModel(torch.nn.Module):
    def __init__(self, encoder, num_hidden, num_proj_hidden, tau=0.5):
        super().__init__()
        self.encoder, self.tau = encoder, tau
        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)
    def forward(self, x, edge_index): return self.encoder(x, edge_index)
    def projection(self, z): return self.fc2(F.elu(self.fc1(z)))
    def sim(self, z1, z2): return torch.mm(F.normalize(z1), F.normalize(z2).t())
    def semi_loss(self, z1, z2):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim, between_sim = f(self.sim(z1, z1)), f(self.sim(z1, z2))
        return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
    def loss(self, z1, z2, mean=True):
        l1, l2 = self.semi_loss(self.projection(z1), self.projection(z2)), self.semi_loss(self.projection(z2), self.projection(z1))
        ret = (l1 + l2) * 0.5
        return ret.mean() if mean else ret.sum()
