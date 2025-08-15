import torch
from torch_geometric.transforms import Compose
from torch_geometric.utils import dropout_adj, degree
import torch.nn.functional as F

class DropEdges:
    def __init__(self, p):
        self.p = p
    
    def __call__(self, data):
        edge_index, _ = dropout_adj(data.edge_index, p=self.p, 
                                  force_undirected=True, 
                                  num_nodes=data.num_nodes,
                                  training=True)
        data = data.clone()
        data.edge_index = edge_index
        return data

class DropFeatures:
    def __init__(self, p):
        self.p = p
    
    def __call__(self, data):
        data = data.clone()
        if data.x is not None:
            drop_mask = torch.empty((data.x.size(1),), dtype=torch.float32, device=data.x.device).uniform_(0, 1) < self.p
            data.x[:, drop_mask] = 0
        return data

class RandomEdges:
    def __init__(self):
        pass
    
    def __call__(self, data):
        data = data.clone()
        num_nodes = data.num_nodes
        num_edges = data.edge_index.size(1) // 2  # Assuming undirected graph
        
        # Generate random edges
        row = torch.randint(0, num_nodes, (num_edges,))
        col = torch.randint(0, num_nodes, (num_edges,))
        
        # Create bidirectional edges
        edge_index = torch.stack([torch.cat([row, col]), torch.cat([col, row])], dim=0)
        data.edge_index = edge_index
        return data

def compose_transforms(transform_names, **kwargs):
    """Compose multiple transforms"""
    transforms = []
    
    if isinstance(transform_names, str):
        transform_names = [transform_names]
    
    for name in transform_names:
        if name == 'drop-edges':
            transforms.append(DropEdges(kwargs.get('drop_edge_p', 0.2)))
        elif name == 'drop-features':
            transforms.append(DropFeatures(kwargs.get('drop_feat_p', 0.2)))
        elif name == 'random-edges':
            transforms.append(RandomEdges())
    
    return Compose(transforms) if transforms else lambda x: x
