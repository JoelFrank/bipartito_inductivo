from typing import List
import torch
from torch import nn
from enum import Enum

class DecoderModel(Enum):
    CONCAT_MLP = 'concat_mlp'
    PRODUCT_MLP = 'prod_mlp'

class MlpConcatDecoder(torch.nn.Module):
    def __init__(self, embedding_size, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
    def forward(self, x): return self.net(x)
    def predict(self, x): return torch.sigmoid(self.forward(x))

class MlpProdDecoder(torch.nn.Module):
    def __init__(self, embedding_size, hidden_size):
        super().__init__()
        self.embedding_size = embedding_size
        # La entrada es concatenación de embeddings (2 * embedding_size)
        # La salida del producto element-wise es embedding_size
        self.net = nn.Sequential(
            nn.Linear(embedding_size, hidden_size),  # Entrada: embedding_size (después del producto)
            nn.ReLU(), 
            nn.Dropout(0.5),  # AUMENTADO: Dropout más fuerte para evitar overfitting
            nn.Linear(hidden_size, 1)
        )
    def forward(self, x):
        # x es concatenación de embeddings [src_emb, dst_emb] con dimensión (batch, 2*embedding_size)
        # Separar y hacer producto element-wise
        left, right = x[:, : self.embedding_size], x[:, self.embedding_size :]
        product = left * right  # Resultado: (batch, embedding_size)
        return self.net(product)
    def predict(self, x): return torch.sigmoid(self.forward(x))

class DecoderZoo:
    models = {
        DecoderModel.CONCAT_MLP.value: MlpConcatDecoder,
        DecoderModel.PRODUCT_MLP.value: MlpProdDecoder,
    }
    def __init__(self, flags): self.flags = flags
    def init_model(self, model_class, embedding_size):
        flags = self.flags
        if model_class == MlpConcatDecoder:
            hidden_size = flags.link_mlp_hidden_size * 2 if flags.adjust_layer_sizes else flags.link_mlp_hidden_size
            return MlpConcatDecoder(embedding_size, hidden_size)
        elif model_class == MlpProdDecoder:
            return MlpProdDecoder(embedding_size, flags.link_mlp_hidden_size)
    @staticmethod
    def filter_models(models: List[str]):
        return [model for model in models if model in DecoderZoo.models]
    def check_model(self, model_name):
        if model_name not in self.models: raise ValueError(f'Unknown decoder: "{model_name}"')
        return True
    def get_model(self, model_name, embedding_size):
        self.check_model(model_name)
        return self.init_model(self.models[model_name], embedding_size)
