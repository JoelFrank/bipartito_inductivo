import torch

class GraphBarlowTwins(torch.nn.Module):
    def __init__(self, encoder, has_features):
        super().__init__()
        self.encoder = encoder
        self.has_features = has_features

    def trainable_parameters(self):
        return list(self.encoder.parameters())

    def forward(self, online_x, target_x):
        rep_a = self.encoder(online_x)
        rep_b = self.encoder(target_x)
        return rep_a, rep_b
