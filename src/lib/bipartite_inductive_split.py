"""
Implementación correcta del split inductivo para grafos bipartitos.

Esta implementación sigue la lógica del proyecto original:
1. Separar nodos en "observados" (pasado) y "no observados" (futuro)
2. Preentrenar encoder solo con nodos observados
3. Entrenar decoder con enlaces del pasado
4. Evaluar con enlaces que incluyen nodos no observados
"""

import torch
import numpy as np
import logging
from torch_geometric.utils import negative_sampling
from torch_geometric.data import Data

log = logging.getLogger(__name__)

def do_bipartite_inductive_node_split(data, observed_ratio=0.7, split_seed=42):
    """
    Realiza split inductivo basado en nodos para grafos bipartitos.
    
    Args:
        data: Objeto Data con grafo bipartito
        observed_ratio: Fracción de nodos a mantener como "observados" (pasado)
        split_seed: Semilla para reproducibilidad
        
    Returns:
        dict con training_data, inference_data, y edge_bundles
    """
    torch.manual_seed(split_seed)
    np.random.seed(split_seed)
    
    log.info(f"=== SPLIT INDUCTIVO BIPARTITO ===")
    log.info(f"Nodos observados: {observed_ratio*100:.1f}%")
    
    # Verificar que es un grafo bipartito
    if not hasattr(data, 'num_nodes_type_1'):
        raise ValueError("El grafo debe tener atributo 'num_nodes_type_1' para ser bipartito")
    
    num_type1 = data.num_nodes_type_1
    num_type2 = data.num_nodes - num_type1
    
    log.info(f"Grafo original: {num_type1} nodos tipo 1, {num_type2} nodos tipo 2")
    log.info(f"Total aristas: {data.edge_index.size(1)}")
    
    # === PASO 1: SEPARAR NODOS EN OBSERVADOS Y NO OBSERVADOS ===
    
    # Para tipo 1 (patrimônios): seleccionar aleatoriamente los observados
    type1_indices = np.arange(num_type1)
    np.random.shuffle(type1_indices)
    num_observed_type1 = int(observed_ratio * num_type1)
    observed_type1 = set(type1_indices[:num_observed_type1])
    unobserved_type1 = set(type1_indices[num_observed_type1:])
    
    # Para tipo 2 (localizações): seleccionar aleatoriamente los observados
    type2_indices = np.arange(num_type1, data.num_nodes)  # índices globales
    np.random.shuffle(type2_indices)
    num_observed_type2 = int(observed_ratio * num_type2)
    observed_type2 = set(type2_indices[:num_observed_type2])
    unobserved_type2 = set(type2_indices[num_observed_type2:])
    
    # Conjunto completo de nodos observados y no observados
    observed_nodes = observed_type1.union(observed_type2)
    unobserved_nodes = unobserved_type1.union(unobserved_type2)
    
    log.info(f"Nodos observados tipo 1: {len(observed_type1)}/{num_type1}")
    log.info(f"Nodos observados tipo 2: {len(observed_type2)}/{num_type2}")
    log.info(f"Total nodos observados: {len(observed_nodes)}")
    log.info(f"Total nodos no observados: {len(unobserved_nodes)}")
    
    # === PASO 2: CLASIFICAR ARISTAS SEGÚN TIPOS DE ENLACES ===
    
    old_old_edges = []  # Observado-Observado
    old_new_edges = []  # Observado-No observado
    new_new_edges = []  # No observado-No observado
    
    edge_index = data.edge_index
    for i in range(edge_index.size(1)):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        
        src_observed = src in observed_nodes
        dst_observed = dst in observed_nodes
        
        if src_observed and dst_observed:
            old_old_edges.append([src, dst])
        elif src_observed or dst_observed:  # Uno observado, uno no observado
            old_new_edges.append([src, dst])
        else:  # Ambos no observados
            new_new_edges.append([src, dst])
    
    log.info(f"Clasificación de aristas:")
    log.info(f"  Old-Old (Obs-Obs): {len(old_old_edges)}")
    log.info(f"  Old-New (Obs-NoObs): {len(old_new_edges)}")
    log.info(f"  New-New (NoObs-NoObs): {len(new_new_edges)}")
    
    # === PASO 3: CREAR TRAINING_DATA (SOLO NODOS OBSERVADOS) ===
    
    # Mapear nodos observados a nuevos índices
    observed_nodes_list = sorted(list(observed_nodes))
    old_to_new_idx = {old_idx: new_idx for new_idx, old_idx in enumerate(observed_nodes_list)}
    
    # Remapear aristas Old-Old a los nuevos índices
    if old_old_edges:
        training_edges = []
        for src, dst in old_old_edges:
            new_src = old_to_new_idx[src]
            new_dst = old_to_new_idx[dst] 
            training_edges.append([new_src, new_dst])
        
        training_edge_index = torch.tensor(training_edges, dtype=torch.long).t()
    else:
        training_edge_index = torch.empty((2, 0), dtype=torch.long)
    
    # Crear características para nodos observados
    training_x = data.x[observed_nodes_list] if data.x is not None else None
    
    # Calcular num_nodes_type_1 para el grafo de entrenamiento
    training_type1_count = len([idx for idx in observed_nodes_list if idx < num_type1])
    
    training_data = Data(
        x=training_x,
        edge_index=training_edge_index,
        num_nodes=len(observed_nodes_list),
        num_nodes_type_1=training_type1_count,
        num_nodes_type_2=len(observed_nodes_list) - training_type1_count
    )
    
    log.info(f"Training data: {training_data.num_nodes} nodos, {training_data.edge_index.size(1)} aristas")
    log.info(f"  Tipo 1: {training_data.num_nodes_type_1}, Tipo 2: {training_data.num_nodes_type_2}")
    
    # === PASO 4: CREAR INFERENCE_DATA (TODOS LOS NODOS) ===
    
    inference_data = Data(
        x=data.x,
        edge_index=data.edge_index,  # Grafo completo para inference
        num_nodes=data.num_nodes,
        num_nodes_type_1=data.num_nodes_type_1,
        num_nodes_type_2=data.num_nodes - data.num_nodes_type_1
    )
    
    log.info(f"Inference data: {inference_data.num_nodes} nodos, {inference_data.edge_index.size(1)} aristas")
    
    # === PASO 5: CREAR EDGE BUNDLES PARA EVALUACIÓN ===
    
    # Split de aristas Old-Old para train/val (dentro del pasado)
    if old_old_edges:
        old_old_tensor = torch.tensor(old_old_edges, dtype=torch.long).t()
        num_old_old = old_old_tensor.size(1)
        
        # Permutación aleatoria
        perm = torch.randperm(num_old_old)
        train_ratio = 0.8
        train_end = int(train_ratio * num_old_old)
        
        train_edges_global = old_old_tensor[:, perm[:train_end]]
        val_edges_global = old_old_tensor[:, perm[train_end:]]
    else:
        train_edges_global = torch.empty((2, 0), dtype=torch.long)
        val_edges_global = torch.empty((2, 0), dtype=torch.long)
    
    # Test edges: combinación de Old-New y New-New (el futuro inductivo)
    test_edges_list = old_new_edges + new_new_edges
    if test_edges_list:
        test_edges_global = torch.tensor(test_edges_list, dtype=torch.long).t()
    else:
        test_edges_global = torch.empty((2, 0), dtype=torch.long)
    
    log.info(f"Edge splits (índices globales):")
    log.info(f"  Train: {train_edges_global.size(1)} aristas (Old-Old subset)")
    log.info(f"  Val: {val_edges_global.size(1)} aristas (Old-Old subset)")
    log.info(f"  Test: {test_edges_global.size(1)} aristas (Old-New + New-New)")
    
    # === PASO 6: GENERAR ENLACES NEGATIVOS ===
    
    # Para validación: negativos solo entre nodos observados
    val_neg_edges = generate_bipartite_negative_edges(
        val_edges_global, 
        observed_nodes, 
        data,
        val_edges_global.size(1),
        exclude_edges=data.edge_index  # Excluir aristas del grafo completo
    )
    
    # Para test: negativos pueden incluir nodos no observados
    test_neg_edges = generate_bipartite_negative_edges(
        test_edges_global,
        set(range(data.num_nodes)),  # Todos los nodos
        data,
        test_edges_global.size(1),
        exclude_edges=data.edge_index  # Excluir aristas del grafo completo
    )
    
    log.info(f"Enlaces negativos generados:")
    log.info(f"  Val neg: {val_neg_edges.size(1)}")
    log.info(f"  Test neg: {test_neg_edges.size(1)}")
    
    # === CREAR RESULTADO FINAL ===
    
    result = {
        'training_data': training_data,  # Solo nodos observados para preentrenar encoder
        'inference_data': inference_data,  # Todos los nodos para inference
        'train_edge_bundle': train_edges_global,
        'val_edge_bundle': val_edges_global,
        'val_neg_edge_bundle': val_neg_edges,
        'test_edge_bundle': test_edges_global,
        'test_neg_edge_bundle': test_neg_edges,
        'observed_nodes': observed_nodes,
        'unobserved_nodes': unobserved_nodes,
        'old_to_new_mapping': old_to_new_idx
    }
    
    log.info(f"✓ Split inductivo bipartito completado")
    
    return result

def generate_bipartite_negative_edges(pos_edges, allowed_nodes, data, num_neg, exclude_edges=None):
    """
    Genera enlaces negativos para grafos bipartitos respetando las restricciones inductivas.
    
    Args:
        pos_edges: Aristas positivas [2, num_pos]
        allowed_nodes: Conjunto de nodos permitidos para muestreo
        data: Objeto Data con metadatos bipartitos
        num_neg: Número de enlaces negativos a generar
        exclude_edges: Aristas a excluir del muestreo (ej: grafo completo)
    
    Returns:
        neg_edges: Enlaces negativos [2, num_neg]
    """
    
    # Crear conjunto de aristas existentes a evitar
    existing_edges = set()
    
    # Agregar aristas positivas actuales
    if pos_edges.size(1) > 0:
        for i in range(pos_edges.size(1)):
            edge = tuple(sorted([pos_edges[0, i].item(), pos_edges[1, i].item()]))
            existing_edges.add(edge)
    
    # Agregar aristas a excluir (ej: grafo completo)
    if exclude_edges is not None:
        for i in range(exclude_edges.size(1)):
            edge = tuple(sorted([exclude_edges[0, i].item(), exclude_edges[1, i].item()]))
            existing_edges.add(edge)
    
    # Separar nodos permitidos por tipo
    allowed_nodes_list = list(allowed_nodes)
    type1_allowed = [n for n in allowed_nodes_list if n < data.num_nodes_type_1]
    type2_allowed = [n for n in allowed_nodes_list if n >= data.num_nodes_type_1]
    
    log.info(f"Muestreo negativo: {len(type1_allowed)} tipo1, {len(type2_allowed)} tipo2 permitidos")
    
    if len(type1_allowed) == 0 or len(type2_allowed) == 0:
        log.warning("No hay suficientes nodos de ambos tipos para muestreo bipartito")
        return torch.empty((2, 0), dtype=torch.long)
    
    neg_edges = []
    max_attempts = num_neg * 100
    attempts = 0
    
    while len(neg_edges) < num_neg and attempts < max_attempts:
        attempts += 1
        
        # Muestrear un nodo de cada tipo
        type1_node = np.random.choice(type1_allowed)
        type2_node = np.random.choice(type2_allowed)
        
        edge = tuple(sorted([type1_node, type2_node]))
        
        if edge not in existing_edges:
            neg_edges.append([type1_node, type2_node])
            existing_edges.add(edge)  # Evitar duplicados
    
    if len(neg_edges) < num_neg:
        log.warning(f"Solo se generaron {len(neg_edges)} negativos de {num_neg} solicitados")
    
    return torch.tensor(neg_edges, dtype=torch.long).t() if neg_edges else torch.empty((2, 0), dtype=torch.long)
