import copy
import torch

class BGRL(torch.nn.Module):
    def __init__(self, encoder, predictor, has_features):
        super().__init__()
        self.online_encoder = encoder
        self.predictor = predictor
        self.has_features = has_features
        self.target_encoder = copy.deepcopy(encoder)
        self.target_encoder.reset_parameters()
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        if not self.has_features:
            self.target_encoder.node_feats = self.online_encoder.node_feats

    def trainable_parameters(self):
        return list(self.online_encoder.parameters()) + list(self.predictor.parameters())

    @torch.no_grad()
    def update_target_network(self, mm):
        assert 0.0 <= mm <= 1.0, "Momentum must be between 0.0 and 1.0"
        for param_q, param_k in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.mul_(mm).add_(param_q.data, alpha=1.0 - mm)

    def forward(self, online_x, target_x):
        online_y = self.online_encoder(online_x)
        online_q = self.predictor(online_y)
        with torch.no_grad():
            target_y = self.target_encoder(target_x).detach()
        return online_q, target_y
