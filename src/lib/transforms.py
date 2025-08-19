import torch
from torch_geometric.transforms import Compose
from torch_geometric.utils import dropout_adj, degree
from torch_geometric.data import HeteroData
import torch.nn.functional as F

class DropEdges:
    def __init__(self, p):
        self.p = p
    
    def __call__(self, data):
        if isinstance(data, HeteroData):
            # Bipartite-aware edge dropping
            data = data.clone()
            for edge_type in data.edge_types:
                edge_index = data[edge_type].edge_index
                edge_index, _ = dropout_adj(edge_index, p=self.p, 
                                          force_undirected=False,  # Bipartite graphs are directed
                                          training=True)
                data[edge_type].edge_index = edge_index
            return data
        else:
            # Original homogeneous behavior
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
        if isinstance(data, HeteroData):
            # Bipartite-aware feature dropping
            for node_type in data.node_types:
                if data[node_type].x is not None:
                    drop_mask = torch.empty((data[node_type].x.size(1),), 
                                          dtype=torch.float32, 
                                          device=data[node_type].x.device).uniform_(0, 1) < self.p
                    data[node_type].x[:, drop_mask] = 0
        else:
            # Original homogeneous behavior
            if data.x is not None:
                drop_mask = torch.empty((data.x.size(1),), dtype=torch.float32, device=data.x.device).uniform_(0, 1) < self.p
                data.x[:, drop_mask] = 0
        return data

class ScrambleFeatures:
    """Scramble node features within each node type (bipartite-aware)"""
    def __init__(self, p=1.0):
        self.p = p  # Probability of scrambling features
    
    def __call__(self, data):
        data = data.clone()
        if isinstance(data, HeteroData):
            # Bipartite-aware feature scrambling
            for node_type in data.node_types:
                if data[node_type].x is not None:
                    x = data[node_type].x
                    if torch.rand(1).item() < self.p:
                        # Randomly permute features within each node type
                        perm_idx = torch.randperm(x.size(0))
                        data[node_type].x = x[perm_idx]
        else:
            # Original homogeneous behavior
            if data.x is not None and torch.rand(1).item() < self.p:
                perm_idx = torch.randperm(data.x.size(0))
                data.x = data.x[perm_idx]
        return data

class AddEdges:
    """Add random edges while respecting bipartite structure"""
    def __init__(self, p=0.1):
        self.p = p  # Proportion of edges to add
    
    def __call__(self, data):
        data = data.clone()
        if isinstance(data, HeteroData):
            # Bipartite-aware edge addition
            for edge_type in data.edge_types:
                src_type, _, dst_type = edge_type
                edge_index = data[edge_type].edge_index
                num_src_nodes = data[src_type].num_nodes
                num_dst_nodes = data[dst_type].num_nodes
                num_existing_edges = edge_index.size(1)
                
                # Calculate number of edges to add
                num_edges_to_add = int(num_existing_edges * self.p)
                
                if num_edges_to_add > 0:
                    # Generate random edges respecting bipartite structure
                    row = torch.randint(0, num_src_nodes, (num_edges_to_add,))
                    col = torch.randint(0, num_dst_nodes, (num_edges_to_add,))
                    new_edges = torch.stack([row, col], dim=0)
                    
                    # Concatenate with existing edges
                    data[edge_type].edge_index = torch.cat([edge_index, new_edges], dim=1)
        else:
            # Original homogeneous behavior (random edges)
            num_nodes = data.num_nodes
            num_existing_edges = data.edge_index.size(1) // 2  # Assuming undirected
            num_edges_to_add = int(num_existing_edges * self.p)
            
            if num_edges_to_add > 0:
                row = torch.randint(0, num_nodes, (num_edges_to_add,))
                col = torch.randint(0, num_nodes, (num_edges_to_add,))
                new_edges = torch.stack([torch.cat([row, col]), torch.cat([col, row])], dim=0)
                data.edge_index = torch.cat([data.edge_index, new_edges], dim=1)
        
        return data

class RandomEdges:
    def __init__(self):
        pass
    
    def __call__(self, data):
        data = data.clone()
        if isinstance(data, HeteroData):
            # For bipartite graphs, replace edges while maintaining structure
            for edge_type in data.edge_types:
                src_type, _, dst_type = edge_type
                num_src_nodes = data[src_type].num_nodes
                num_dst_nodes = data[dst_type].num_nodes
                num_edges = data[edge_type].edge_index.size(1)
                
                # Generate completely random edges
                row = torch.randint(0, num_src_nodes, (num_edges,))
                col = torch.randint(0, num_dst_nodes, (num_edges,))
                data[edge_type].edge_index = torch.stack([row, col], dim=0)
        else:
            # Original homogeneous behavior
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
    """Compose multiple transforms with bipartite support"""
    transforms = []
    
    if isinstance(transform_names, str):
        transform_names = [transform_names]
    
    for name in transform_names:
        if name == 'drop-edges':
            transforms.append(DropEdges(kwargs.get('drop_edge_p', 0.2)))
        elif name == 'drop-features':
            transforms.append(DropFeatures(kwargs.get('drop_feat_p', 0.2)))
        elif name == 'scramble-features':
            transforms.append(ScrambleFeatures(kwargs.get('scramble_feat_p', 1.0)))
        elif name == 'add-edges':
            transforms.append(AddEdges(kwargs.get('add_edge_p', 0.1)))
        elif name == 'random-edges':
            transforms.append(RandomEdges())
    
    return Compose(transforms) if transforms else lambda x: x
