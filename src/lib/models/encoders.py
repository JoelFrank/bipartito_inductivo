import torch
import torch.nn as nn
# Importa SAGEConv y HeteroData
from torch_geometric.nn import BatchNorm, GCNConv, SAGEConv, LayerNorm, Sequential
from torch_geometric.data import Data, HeteroData
from enum import Enum

class EncoderModel(Enum):
    GCN = 'gcn'
    # NUEVO: añade el nuevo modelo
    BIPARTITE_SAGE = 'bipartite_sage'

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
        # Detectar si estamos usando embeddings (data.x contiene índices Long)
        if data.x.dtype == torch.long:
            # Convertir índices de embedding a embeddings reales usando embedding lookup
            # Para simplicidad, usar embedding inicial basado en índices
            if not hasattr(self, 'embedding_layer'):
                # Crear embedding layer dinámicamente si no existe
                max_idx = data.x.max().item()
                self.embedding_layer = nn.Embedding(max_idx + 1, self.input_size).to(data.x.device)
            x = self.embedding_layer(data.x.squeeze(-1))  # [num_nodes, embedding_dim]
        else:
            x = self.node_feats.weight if not self.use_feat else data.x.float()
        return self.model(x, data.edge_index)

    def reset_parameters(self):
        self.model.reset_parameters()
        if not self.use_feat: self.node_feats.reset_parameters()

# ==============================================================================
# NUEVO: Codificador Bipartito
# ==============================================================================
class BipartiteSAGE(nn.Module):
    def __init__(self, in_channels_src, in_channels_dst, hidden_channels, out_channels, num_layers=2, 
                 num_embeddings=None, embedding_dim=None):
        super().__init__()
        self.num_layers = num_layers
        self.in_channels_src = in_channels_src
        self.in_channels_dst = in_channels_dst
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        
        # Manejar embeddings si se especifica
        self.use_embeddings = num_embeddings is not None and embedding_dim is not None
        if self.use_embeddings:
            from ..utils import EmbeddingProcessor
            self.embedding_processor = EmbeddingProcessor(num_embeddings, embedding_dim)
            # Ajustar canales de entrada para usar embedding_dim
            actual_in_channels_src = embedding_dim
            actual_in_channels_dst = embedding_dim
        else:
            actual_in_channels_src = in_channels_src
            actual_in_channels_dst = in_channels_dst
        
        # Proyecciones iniciales para igualar dimensiones
        self.src_proj = nn.Linear(actual_in_channels_src, hidden_channels)
        self.dst_proj = nn.Linear(actual_in_channels_dst, hidden_channels)
        
        # Compatibilidad con T-BGRL: atributo node_feats dummy
        # En BipartiteSAGE usamos las proyecciones en lugar de node_feats
        self.node_feats = nn.Parameter(torch.empty(0))  # Placeholder para compatibilidad
        
        # Capas SAGE para el paso de mensajes bipartito (todas usan hidden_channels)
        self.convs = nn.ModuleList()
        
        for i in range(num_layers):
            if i == num_layers - 1:
                # Última capa: salida con out_channels
                self.convs.append(SAGEConv((hidden_channels, hidden_channels), out_channels))
            else:
                # Capas intermedias: hidden_channels
                self.convs.append(SAGEConv((hidden_channels, hidden_channels), hidden_channels))
        
        # Normalización (no en la última capa)
        self.norms = nn.ModuleList()
        for i in range(num_layers - 1):
            self.norms.append(BatchNorm(hidden_channels))
        
    def forward(self, data):
        # Manejar tanto HeteroData como Data regular
        if isinstance(data, HeteroData):
            # Modo HeteroData: usar estructura bipartita completa
            x_src = data['patrimonio'].x
            x_dst = data['localizacao'].x
            edge_index = data['patrimonio', 'located_at', 'localizacao'].edge_index
        else:
            # Modo compatibilidad: convertir Data regular a estructura bipartita
            device = data.edge_index.device  # Asegurar mismo device
            
            if hasattr(data, 'num_nodes_type_1') and hasattr(data, 'num_nodes_type_2'):
                # Datos bipartitos con metadatos
                num_nodes_type_1 = data.num_nodes_type_1
                if data.x is not None:
                    x_src = data.x[:num_nodes_type_1]
                    x_dst = data.x[num_nodes_type_1:]
                else:
                    # Crear features sintéticas en el mismo device
                    x_src = torch.randn(num_nodes_type_1, self.in_channels_src, device=device)
                    x_dst = torch.randn(data.num_nodes_type_2, self.in_channels_dst, device=device)
                edge_index = data.edge_index
            else:
                # Fallback: crear features sintéticas para pruebas
                total_nodes = data.num_nodes
                num_nodes_type_1 = total_nodes * 2 // 3  # Asumimos proporción 2:1
                num_nodes_type_2 = total_nodes - num_nodes_type_1
                
                x_src = torch.randn(num_nodes_type_1, self.in_channels_src, device=device)
                x_dst = torch.randn(num_nodes_type_2, self.in_channels_dst, device=device)
                edge_index = data.edge_index
        
        # Procesar embeddings si es necesario
        if self.use_embeddings:
            x_src = self.embedding_processor(x_src)
            x_dst = self.embedding_processor(x_dst)
        
        # Proyectar a dimensiones comunes
        x_src = self.src_proj(x_src)
        x_dst = self.dst_proj(x_dst)
        
        # Aplicar capas SAGE
        for i, conv in enumerate(self.convs):
            # Paso 1: patrimonio -> localizacao
            x_dst_new = conv((x_src, x_dst), edge_index)
            
            # Paso 2: localizacao -> patrimonio (usando edge_index invertido)
            edge_index_rev = edge_index[[1, 0]]
            x_src_new = conv((x_dst, x_src), edge_index_rev)
            
            # Aplicar normalización y activación (excepto en la última capa)
            if i < len(self.convs) - 1:
                x_src_new = torch.relu(self.norms[i](x_src_new))
                x_dst_new = torch.relu(self.norms[i](x_dst_new))
            
            x_src = x_src_new
            x_dst = x_dst_new
        
        # Retornar embeddings concatenados para compatibilidad
        return torch.cat([x_src, x_dst], dim=0)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms:
            norm.reset_parameters()
        if self.use_embeddings:
            self.embedding_processor.embedding.reset_parameters()

class EncoderZoo:
    models = {
        EncoderModel.GCN.value: GCN,
        EncoderModel.BIPARTITE_SAGE.value: BipartiteSAGE
    }
    def __init__(self, flags): self.flags = flags
    def _init_model(self, model_class, input_size, use_feat, n_nodes, data=None):
        flags = self.flags
        if model_class == GCN:
            # Si no hay features, el tamaño de entrada es la primera capa de GCN
            effective_input_size = input_size if use_feat else flags.graph_encoder_layer_dims[0]
            return GCN([effective_input_size] + flags.graph_encoder_layer_dims, batchnorm=True, use_feat=use_feat, n_nodes=n_nodes)
        elif model_class == BipartiteSAGE:
            # Para BipartiteSAGE necesitamos dimensiones específicas de entrada para cada tipo de nodo
            # Asumimos que input_size es una tupla (in_channels_src, in_channels_dst)
            if isinstance(input_size, tuple):
                in_channels_src, in_channels_dst = input_size
            else:
                # Fallback: usar la misma dimensión para ambos tipos de nodos
                in_channels_src = in_channels_dst = input_size
            
            hidden_channels = flags.graph_encoder_layer_dims[0] if flags.graph_encoder_layer_dims else 128
            out_channels = flags.graph_encoder_layer_dims[-1] if flags.graph_encoder_layer_dims else hidden_channels
            num_layers = len(flags.graph_encoder_layer_dims) if flags.graph_encoder_layer_dims else 2
            
            # Detectar si necesitamos embeddings
            num_embeddings = None
            embedding_dim = None
            if data is not None and hasattr(data, 'num_embedding_types'):
                num_embeddings = data.num_embedding_types
                embedding_dim = hidden_channels  # Usar hidden_channels como embedding_dim
                print(f"🔧 Configurando BipartiteSAGE con embeddings: {num_embeddings} tipos, dim={embedding_dim}")
            
            return BipartiteSAGE(
                in_channels_src=in_channels_src,
                in_channels_dst=in_channels_dst,
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                num_layers=num_layers,
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim
            )
    @staticmethod
    def check_model(model_name: str):
        if model_name not in EncoderZoo.models: raise ValueError(f'Unknown encoder: "{model_name}"')
    def get_model(self, model_name, input_size, use_feat, n_nodes, data=None):
        self.check_model(model_name)
        return self._init_model(self.models[model_name], input_size, use_feat, n_nodes, data)
