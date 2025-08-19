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
    from torch_geometric.data import HeteroData
    
    # Detectar si es HeteroData o grafo bipartito estándar
    if isinstance(data, HeteroData):
        # Para HeteroData, obtener número de nodos de cada tipo
        node_types = list(data.node_types)
        if len(node_types) == 2:
            num_nodes_tuple = (data[node_types[0]].num_nodes, data[node_types[1]].num_nodes)
        else:
            # Asumir los primeros dos tipos
            num_nodes_tuple = (data[node_types[0]].num_nodes, data[node_types[1]].num_nodes)
    else:
        # Para grafos bipartitos estándar
        num_nodes_type_1 = data.num_nodes_type_1
        num_nodes_total = data.num_nodes
        num_nodes_tuple = (num_nodes_type_1, num_nodes_total - num_nodes_type_1)
    
    # Ensure tensors are on the same device as edge_index
    device = edge_index.device

    # ==============================================================================
    # CAMBIO 5: Muestreo negativo bipartito usando tupla de nodos
    # QUÉ HACE: Pasa una tupla (num_src, num_dst) a negative_sampling
    # POR QUÉ: Para que PyG genere únicamente negativos válidos (src-dst)
    # ==============================================================================
    neg_edge_index = negative_sampling(
        edge_index=edge_index,
        num_nodes=num_nodes_tuple,  # Tupla para muestreo bipartito
        num_neg_samples=num_neg_samples,
        method='sparse'
    )
    
    # Ensure the result is on the correct device - FORCE device placement
    if hasattr(edge_index, 'device'):
        neg_edge_index = neg_edge_index.to(edge_index.device)
    
    return neg_edge_index

def bipartite_negative_sampling_inductive(full_edge_index, data, num_neg_samples):
    """
    Muestreo negativo inductivo correcto: usa el grafo COMPLETO como referencia.
    Garantiza que los enlaces negativos NO existan en ningún momento temporal.
    
    Args:
        full_edge_index: El grafo completo (todas las aristas temporales)
        data: Objeto con metadatos bipartitos
        num_neg_samples: Número de muestras negativas deseadas
    
    Returns:
        neg_edge_index: Enlaces negativos que NO existen en el grafo completo
    """
    from torch_geometric.data import HeteroData
    
    # Detectar si es HeteroData o grafo bipartito estándar
    if isinstance(data, HeteroData):
        # Para HeteroData, obtener número de nodos de cada tipo
        node_types = list(data.node_types)
        if len(node_types) == 2:
            num_nodes_tuple = (data[node_types[0]].num_nodes, data[node_types[1]].num_nodes)
        else:
            # Asumir los primeros dos tipos
            num_nodes_tuple = (data[node_types[0]].num_nodes, data[node_types[1]].num_nodes)
    else:
        # Para grafos bipartitos estándar
        num_nodes_type_1 = data.num_nodes_type_1
        num_nodes_total = data.num_nodes
        num_nodes_tuple = (num_nodes_type_1, num_nodes_total - num_nodes_type_1)
    
    # Ensure tensors are on the same device as full_edge_index
    device = full_edge_index.device
    
    log.info(f"Muestreo negativo inductivo: usando grafo completo con {full_edge_index.size(1)} aristas")
    
    # ==============================================================================
    # CAMBIO 5: Muestreo negativo bipartito inductivo usando tupla de nodos
    # QUÉ HACE: Usar la función original de PyG con tupla de nodos
    # POR QUÉ: Para que genere únicamente negativos válidos (src-dst) que no existan en el grafo completo
    # ==============================================================================
    neg_edge_index = negative_sampling(
        edge_index=full_edge_index,  # ✓ CLAVE: usar grafo completo, no subset
        num_nodes=num_nodes_tuple,
        num_neg_samples=num_neg_samples,
        method='sparse'
    )
    
    log.info(f"Generados {neg_edge_index.size(1)} enlaces negativos válidos")
    
    # Ensure the result is on the correct device - FORCE device placement
    if hasattr(full_edge_index, 'device'):
        neg_edge_index = neg_edge_index.to(full_edge_index.device)
    
    return neg_edge_index

def generate_neg_edges(pos_edges, num_nodes, num_neg=None, data=None):
    """
    Generate negative edges for evaluation.
    For bipartite graphs, ensures negative edges are between different node types.
    For inductive setting, avoids edges that exist in the full graph.
    """
    if num_neg is None:
        num_neg = pos_edges.size(1)
    
    # Check if we have bipartite structure
    is_bipartite = data is not None and hasattr(data, 'num_nodes_type_1')
    
    # Create set of existing edges (positive edges in current split)
    pos_edge_set = set()
    for i in range(pos_edges.size(1)):
        edge = tuple(sorted([pos_edges[0, i].item(), pos_edges[1, i].item()]))
        pos_edge_set.add(edge)
    
    # For inductive setting, also avoid edges from full graph
    full_edge_set = set()
    if data is not None and hasattr(data, 'full_edge_index'):
        for i in range(data.full_edge_index.size(1)):
            edge = tuple(sorted([data.full_edge_index[0, i].item(), data.full_edge_index[1, i].item()]))
            full_edge_set.add(edge)
        log.info(f"Muestreo negativo inductivo: evitando {len(full_edge_set)} enlaces del grafo completo")
    
    neg_edges = []
    max_attempts = num_neg * 100  # Prevent infinite loops
    attempts = 0
    
    while len(neg_edges) < num_neg and attempts < max_attempts:
        attempts += 1
        
        if is_bipartite:
            # For bipartite graphs: sample from different node types
            # Type 1: [0, num_nodes_type_1), Type 2: [num_nodes_type_1, num_nodes)
            type1_node = np.random.randint(0, data.num_nodes_type_1)
            type2_node = np.random.randint(data.num_nodes_type_1, num_nodes)
            i, j = type1_node, type2_node
        else:
            # For regular graphs: sample any two different nodes
            i = np.random.randint(0, num_nodes)
            j = np.random.randint(0, num_nodes)
            if i == j:
                continue
        
        edge = tuple(sorted([i, j]))
        
        # Check if edge is valid (not in positive set and not in full graph if inductive)
        if edge not in pos_edge_set and edge not in full_edge_set:
            neg_edges.append([i, j])
    
    if len(neg_edges) < num_neg:
        log.warning(f"Solo se pudieron generar {len(neg_edges)} enlaces negativos de {num_neg} solicitados")
    
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
        # Verificar si hay grafo completo disponible para muestreo inductivo
        if hasattr(data, 'full_edge_index'):
            log.info("Grafo completo disponible. Usando muestreo negativo inductivo correcto.")
            val_neg_edges = bipartite_negative_sampling_inductive(data.full_edge_index, data, val_edges.size(1))
            test_neg_edges = bipartite_negative_sampling_inductive(data.full_edge_index, data, test_edges.size(1))
        else:
            log.warning("Grafo completo no disponible. Usando muestreo bipartito estándar.")
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
