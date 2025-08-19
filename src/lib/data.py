# src/lib/data.py

from typing import Union, Optional
import torch
import torch.nn.functional as F
from torch_geometric.data import InMemoryDataset, HeteroData, Data
from torch_geometric.transforms import BaseTransform, Compose, NormalizeFeatures
import os

# ==============================================================================
# NUEVO: Transformaciones para normalización
# ==============================================================================
class ConvertToFloat(BaseTransform):
    def __call__(self, data):
        if hasattr(data, 'x') and data.x is not None:
            data.x = data.x.float()
        return data

# ==============================================================================
# NUEVO: Objeto de datos para el grafo bipartito
# ==============================================================================
class BipartiteData(HeteroData):
    def __init__(self, edge_index, x_src, x_dst, num_nodes_type_1=None, num_nodes_type_2=None):
        super().__init__()
        # Nombres de los tipos de nodos (puedes cambiarlos)
        self.src_node_type = 'patrimonio'
        self.dst_node_type = 'localizacao'
        
        # Define los nodos de cada tipo
        self['patrimonio'].x = x_src
        self['localizacao'].x = x_dst
        
        # Define las aristas entre ellos
        self['patrimonio', 'located_at', 'localizacao'].edge_index = edge_index
        
        # Metadatos adicionales para compatibilidad
        if num_nodes_type_1 is not None:
            self.num_nodes_type_1 = num_nodes_type_1
        else:
            self.num_nodes_type_1 = x_src.size(0)
            
        if num_nodes_type_2 is not None:
            self.num_nodes_type_2 = num_nodes_type_2
        else:
            self.num_nodes_type_2 = x_dst.size(0)

    # Propiedades para compatibilidad con el código existente
    @property
    def num_nodes(self):
        return self.num_nodes_type_1 + self.num_nodes_type_2
        
    @property
    def x(self):
        # Esta propiedad ya no tiene sentido en un grafo bipartito,
        # pero la mantenemos por si alguna parte del código la necesita.
        # Concatenamos las features de ambos tipos
        return torch.cat([self['patrimonio'].x, self['localizacao'].x], dim=0)
    
    @property
    def edge_index(self):
        # Para compatibilidad, devolvemos el edge_index principal
        return self['patrimonio', 'located_at', 'localizacao'].edge_index

# ==============================================================================
# Dataset personalizado para grafos bipartitos
# ==============================================================================
class BipartiteDataset(InMemoryDataset):
    def __init__(self, root, data_path, transform=None, pre_transform=None):
        self.data_path = data_path
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['bipartite_data.pt']

    def process(self):
        # Cargar datos desde el archivo .pt
        raw_data = torch.load(self.data_path)
        
        # Verificar si ya es BipartiteData o convertir desde Data estándar
        if isinstance(raw_data, HeteroData):
            bipartite_data = raw_data
        else:
            # Convertir desde Data estándar a BipartiteData
            # Asumimos que las features están divididas por num_nodes_type_1
            num_nodes_type_1 = raw_data.num_nodes_type_1
            num_nodes_type_2 = raw_data.num_nodes_type_2
            
            x_src = raw_data.x[:num_nodes_type_1]
            x_dst = raw_data.x[num_nodes_type_1:]
            
            bipartite_data = BipartiteData(
                edge_index=raw_data.edge_index,
                x_src=x_src,
                x_dst=x_dst,
                num_nodes_type_1=num_nodes_type_1,
                num_nodes_type_2=num_nodes_type_2
            )
            
            # Preservar metadatos adicionales
            for key, value in raw_data.__dict__.items():
                if key not in ['x', 'edge_index', '_store', '_mapping']:
                    setattr(bipartite_data, key, value)
        
        data_list = [bipartite_data]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

# ==============================================================================
# Modifica la función `get_dataset` para que cargue tus datos bipartitos
# ==============================================================================
def get_dataset(name, root, transform=None, use_hetero=False):
    """
    Load dataset with optional HeteroData format support
    
    Args:
        name: Dataset name
        root: Root directory for data
        transform: Optional data transforms
        use_hetero: Whether to return HeteroData format (default: False for backward compatibility)
    """
    if transform is None:
        transform = Compose([ConvertToFloat(), NormalizeFeatures()])
    
    # --- LÓGICA PARA CARGAR EL DATASET BIPARTITO ---
    if name in ['sismetro_inductive', 'sismetro', 'my-bipartite-dataset']:
        # Ruta al archivo del dataset
        data_path = os.path.join(root, 'processed', f'{name}.pt')
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset file not found: {data_path}")
        
        # Crear directorio temporal para el dataset procesado
        dataset_root = os.path.join(root, 'bipartite_processed', name)
        
        return BipartiteDataset(
            root=dataset_root,
            data_path=data_path,
            transform=transform
        )
    
    # --- LÓGICA ORIGINAL PARA OTROS DATASETS ---
    else:
        raise ValueError(f"Dataset {name} not supported. Use 'sismetro' or 'sismetro_inductive'")

# ==============================================================================
# Función auxiliar para verificar si un dataset es bipartito
# ==============================================================================
def is_bipartite_dataset(data):
    """Verifica si un objeto de datos es bipartito"""
    return isinstance(data, HeteroData)

def get_bipartite_info(data):
    """Extrae información de un grafo bipartito"""
    if not is_bipartite_dataset(data):
        return None
    
    return {
        'node_types': data.node_types,
        'edge_types': data.edge_types,
        'num_nodes_type_1': data.num_nodes_type_1,
        'num_nodes_type_2': data.num_nodes_type_2,
        'src_features_dim': data['patrimonio'].x.size(1),
        'dst_features_dim': data['localizacao'].x.size(1)
    }
