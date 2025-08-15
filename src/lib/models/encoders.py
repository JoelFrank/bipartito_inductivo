import torch
import torch.nn as nn
from torch_geometric.nn import BatchNorm, GCNConv, LayerNorm, Sequential
from torch_geometric.data import Data
from enum import Enum

class EncoderModel(Enum):
    GCN = 'gcn'

class GCN(nn.Module):
    def __init__(self, layer_sizes, batchnorm=False, layernorm=False, use_feat=True, n_nodes=0):
        super().__init__()
        assert not (batchnorm and layernorm), "Cannot use both BatchNorm and LayerNorm"
        assert len(layer_sizes) >= 2
        self.n_layers = len(layer_sizes)
        self.use_feat = use_feat
        self.input_size, self.representation_size = layer_sizes[0], layer_sizes[-1]
        
        layers = []
        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append((GCNConv(in_dim, out_dim), 'x, edge_index -> x'))
            if batchnorm: layers.append(BatchNorm(out_dim))
            if layernorm: layers.append(LayerNorm(out_dim))
            layers.append(nn.PReLU())
        
        self.model = Sequential('x, edge_index', layers)
        
        if not self.use_feat:
            self.node_feats = nn.Embedding(n_nodes, layer_sizes[0])

    def forward(self, data):
        x = self.node_feats.weight if not self.use_feat else data.x
        return self.model(x, data.edge_index)

    def reset_parameters(self):
        self.model.reset_parameters()
        if not self.use_feat: self.node_feats.reset_parameters()

class EncoderZoo:
    models = {EncoderModel.GCN.value: GCN}
    def __init__(self, flags): self.flags = flags
    def _init_model(self, model_class, input_size, use_feat, n_nodes):
        flags = self.flags
        if model_class == GCN:
            # Si no hay features, el tamaño de entrada es la primera capa de GCN
            effective_input_size = input_size if use_feat else flags.graph_encoder_layer_dims[0]
            return GCN([effective_input_size] + flags.graph_encoder_layer_dims, batchnorm=True, use_feat=use_feat, n_nodes=n_nodes)
    @staticmethod
    def check_model(model_name: str):
        if model_name not in EncoderZoo.models: raise ValueError(f'Unknown encoder: "{model_name}"')
    def get_model(self, model_name, input_size, use_feat, n_nodes):
        self.check_model(model_name)
        return self._init_model(self.models[model_name], input_size, use_feat, n_nodes)
