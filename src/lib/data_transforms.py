import torch
import torch.nn.functional as F
from torch_geometric.utils import dropout_adj, add_self_loops
import copy

def drop_edges(data, drop_edge_p):
    """Drop edges randomly from the graph"""
    if drop_edge_p <= 0.0:
        return data
    
    edge_index, _ = dropout_adj(data.edge_index, p=drop_edge_p, 
                               force_undirected=False, num_nodes=data.num_nodes)
    
    new_data = copy.copy(data)
    new_data.edge_index = edge_index.to(data.edge_index.device)
    return new_data

def drop_features(data, drop_feat_p):
    """Drop node features randomly"""
    if drop_feat_p <= 0.0 or data.x is None:
        return data
    
    new_data = copy.copy(data)
    mask = torch.rand(data.x.shape, device=data.x.device) < drop_feat_p
    new_data.x = data.x.clone()
    new_data.x[mask] = 0.0
    return new_data

def random_edges(data):
    """Generate random edges for negative sampling"""
    new_data = copy.copy(data)
    
    # Generate random edges
    num_nodes = data.num_nodes
    num_edges = data.edge_index.size(1)
    device = data.edge_index.device
    
    # Random source and target nodes
    src = torch.randint(0, num_nodes, (num_edges,), device=device)
    dst = torch.randint(0, num_nodes, (num_edges,), device=device)
    
    # Avoid self-loops
    mask = src != dst
    src = src[mask]
    dst = dst[mask]
    
    # Pad if we lost edges due to self-loop removal
    while len(src) < num_edges:
        additional_src = torch.randint(0, num_nodes, (num_edges - len(src),), device=device)
        additional_dst = torch.randint(0, num_nodes, (num_edges - len(src),), device=device)
        additional_mask = additional_src != additional_dst
        src = torch.cat([src, additional_src[additional_mask]])
        dst = torch.cat([dst, additional_dst[additional_mask]])
    
    # Take only the required number of edges
    src = src[:num_edges]
    dst = dst[:num_edges]
    
    new_data.edge_index = torch.stack([src, dst])
    return new_data

def compose_transforms(transform_list, drop_edge_p=0.0, drop_feat_p=0.0):
    """Compose multiple transforms into a single function"""
    def transform_fn(data):
        result = data
        for transform_name in transform_list:
            if transform_name == 'drop-edges':
                result = drop_edges(result, drop_edge_p)
            elif transform_name == 'drop-features':
                result = drop_features(result, drop_feat_p)
            elif transform_name == 'random-edges':
                result = random_edges(result)
        return result
    
    return transform_fn
