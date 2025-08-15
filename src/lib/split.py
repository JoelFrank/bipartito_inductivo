import torch
import numpy as np
import logging
from torch_geometric.utils import negative_sampling

log = logging.getLogger(__name__)

def bipartite_negative_sampling(edge_index, data, num_neg_samples):
    """
    Función de ayuda para el muestreo negativo en grafos bipartitos.
    Garantiza que los enlaces negativos se generen solo entre las dos particiones.
    """
    num_nodes_type_1 = data.num_nodes_type_1
    num_nodes_total = data.num_nodes

    neg_edge_index = negative_sampling(
        edge_index=edge_index,
        num_nodes=(num_nodes_type_1, num_nodes_total - num_nodes_type_1),  # Tupla para muestreo bipartito
        num_neg_samples=num_neg_samples,
        method='sparse'
    )
    # PyG puede devolver índices de destino relativos, así que los ajustamos al rango global
    neg_edge_index[1, :] = neg_edge_index[1, :] % (num_nodes_total - num_nodes_type_1) + num_nodes_type_1
    return neg_edge_index

def generate_neg_edges(pos_edges, num_nodes, num_neg=None):
    """
    Generate negative edges for evaluation.
    """
    if num_neg is None:
        num_neg = pos_edges.size(1)
    
    pos_edge_set = set()
    for i in range(pos_edges.size(1)):
        edge = tuple(sorted([pos_edges[0, i].item(), pos_edges[1, i].item()]))
        pos_edge_set.add(edge)
    
    neg_edges = []
    while len(neg_edges) < num_neg:
        i = np.random.randint(0, num_nodes)
        j = np.random.randint(0, num_nodes)
        if i != j:
            edge = tuple(sorted([i, j]))
            if edge not in pos_edge_set:
                neg_edges.append([i, j])
    
    return torch.tensor(neg_edges, dtype=torch.long).t()

def do_transductive_edge_split(dataset, split_seed=42):
    """
    Perform transductive edge split for bipartite graphs.
    Splits edges into train/val/test sets while maintaining graph connectivity.
    """
    torch.manual_seed(split_seed)
    np.random.seed(split_seed)
    
    data = dataset[0]
    edge_index = data.edge_index
    
    # Check if this is a bipartite graph
    is_bipartite = hasattr(data, 'num_nodes_type_1')
    
    # Remove duplicate edges (keep only one direction)
    edge_set = set()
    unique_edges = []
    
    for i in range(edge_index.size(1)):
        edge = tuple(sorted([edge_index[0, i].item(), edge_index[1, i].item()]))
        if edge not in edge_set:
            edge_set.add(edge)
            unique_edges.append([edge_index[0, i].item(), edge_index[1, i].item()])
    
    unique_edges = torch.tensor(unique_edges, dtype=torch.long).t()
    num_edges = unique_edges.size(1)
    
    # Random permutation
    perm = torch.randperm(num_edges)
    
    # Split ratios
    train_ratio, val_ratio, test_ratio = 0.8, 0.1, 0.1
    train_end = int(train_ratio * num_edges)
    val_end = train_end + int(val_ratio * num_edges)
    
    train_edges = unique_edges[:, perm[:train_end]]
    val_edges = unique_edges[:, perm[train_end:val_end]]
    test_edges = unique_edges[:, perm[val_end:]]
    
    # Generate negative edges - use bipartite-aware sampling if applicable
    if is_bipartite:
        log.info("Grafo bipartito detectado. Usando muestreo negativo bipartito.")
        val_neg_edges = bipartite_negative_sampling(val_edges, data, val_edges.size(1))
        test_neg_edges = bipartite_negative_sampling(test_edges, data, test_edges.size(1))
    else:
        log.info("Grafo estándar detectado. Usando muestreo negativo estándar.")
        val_neg_edges = generate_neg_edges(val_edges, data.num_nodes)
        test_neg_edges = generate_neg_edges(test_edges, data.num_nodes)
    
    return {
        'train': {'edge': train_edges.t()},
        'valid': {'edge': val_edges.t(), 'edge_neg': val_neg_edges.t()},
        'test': {'edge': test_edges.t(), 'edge_neg': test_neg_edges.t()}
    }
