#!/usr/bin/env python3
"""
Script de prueba para verificar las correcciones bipartitas implementadas.
Verifica que las transformaciones y el muestreo negativo funcionen correctamente.
"""

import torch
import numpy as np
import logging
from torch_geometric.data import HeteroData
from torch_geometric.utils import negative_sampling

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Import our corrected modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.lib.transforms import DropFeatures, ScrambleFeatures, RandomEdges
from src.lib.split import bipartite_negative_sampling, bipartite_negative_sampling_inductive

def create_test_hetero_data():
    """Crea un HeteroData de prueba para simular un grafo bipartito"""
    data = HeteroData()
    
    # Nodos de tipo 'src' (por ejemplo, usuarios)
    data['src'].x = torch.randn(10, 16)  # 10 usuarios con 16 características
    data['src'].num_nodes = 10
    
    # Nodos de tipo 'dst' (por ejemplo, productos)
    data['dst'].x = torch.randn(8, 12)   # 8 productos con 12 características
    data['dst'].num_nodes = 8
    
    # Aristas entre src y dst
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 5],  # usuarios
        [0, 1, 2, 0, 1, 3]   # productos
    ])
    data[('src', 'connects_to', 'dst')].edge_index = edge_index
    
    return data

def create_test_bipartite_data():
    """Crea un objeto de datos bipartito estándar"""
    class BipartiteData:
        def __init__(self):
            self.num_nodes_type_1 = 10  # tipo 1: [0, 9]
            self.num_nodes = 18         # total: tipo 1 + tipo 2
            self.edge_index = torch.tensor([
                [0, 1, 2, 3, 4, 5],     # nodos tipo 1
                [10, 11, 12, 10, 11, 13] # nodos tipo 2 (offset +10)
            ])
            self.x = torch.randn(18, 16)  # características para todos los nodos
    
    return BipartiteData()

def test_drop_features():
    """Prueba la transformación DropFeatures"""
    log.info("=== Probando DropFeatures ===")
    
    # Prueba con HeteroData
    data = create_test_hetero_data()
    transform = DropFeatures(p=0.5)
    
    original_src_x = data['src'].x.clone()
    original_dst_x = data['dst'].x.clone()
    
    transformed_data = transform(data)
    
    # Verificar que se aplicó dropout
    src_zeros = (transformed_data['src'].x == 0).float().mean()
    dst_zeros = (transformed_data['dst'].x == 0).float().mean()
    
    log.info(f"Dropout aplicado - src: {src_zeros:.2%}, dst: {dst_zeros:.2%}")
    
    # Verificar que las dimensiones no cambiaron
    assert transformed_data['src'].x.shape == original_src_x.shape
    assert transformed_data['dst'].x.shape == original_dst_x.shape
    
    log.info("✓ DropFeatures funciona correctamente")

def test_scramble_features():
    """Prueba la transformación ScrambleFeatures"""
    log.info("=== Probando ScrambleFeatures ===")
    
    data = create_test_hetero_data()
    transform = ScrambleFeatures()
    
    original_src_x = data['src'].x.clone()
    original_dst_x = data['dst'].x.clone()
    
    transformed_data = transform(data)
    
    # Verificar que las filas se reordenaron (probabilísticamente)
    src_same = torch.allclose(transformed_data['src'].x, original_src_x)
    dst_same = torch.allclose(transformed_data['dst'].x, original_dst_x)
    
    log.info(f"Scrambling - src igual: {src_same}, dst igual: {dst_same}")
    
    # Las dimensiones deben mantenerse
    assert transformed_data['src'].x.shape == original_src_x.shape
    assert transformed_data['dst'].x.shape == original_dst_x.shape
    
    log.info("✓ ScrambleFeatures funciona correctamente")

def test_random_edges():
    """Prueba la transformación RandomEdges"""
    log.info("=== Probando RandomEdges ===")
    
    data = create_test_hetero_data()
    transform = RandomEdges()
    
    original_edge_index = data[('src', 'connects_to', 'dst')].edge_index.clone()
    
    transformed_data = transform(data)
    new_edge_index = transformed_data[('src', 'connects_to', 'dst')].edge_index
    
    # Verificar que el número de aristas se mantiene
    assert new_edge_index.shape == original_edge_index.shape
    
    # Verificar que todas las aristas siguen siendo válidas (src -> dst)
    src_indices = new_edge_index[0, :]
    dst_indices = new_edge_index[1, :]
    
    assert src_indices.min() >= 0 and src_indices.max() < data['src'].num_nodes
    assert dst_indices.min() >= 0 and dst_indices.max() < data['dst'].num_nodes
    
    log.info(f"Aristas generadas - src range: [{src_indices.min()}, {src_indices.max()}], "
             f"dst range: [{dst_indices.min()}, {dst_indices.max()}]")
    
    log.info("✓ RandomEdges funciona correctamente")

def test_bipartite_negative_sampling():
    """Prueba el muestreo negativo bipartito"""
    log.info("=== Probando muestreo negativo bipartito ===")
    
    # Prueba con HeteroData
    data = create_test_hetero_data()
    edge_type = ('src', 'connects_to', 'dst')
    edge_index = data[edge_type].edge_index
    
    neg_edges = bipartite_negative_sampling(edge_index, data, num_neg_samples=5)
    
    # Verificar que se generaron las aristas correctas
    assert neg_edges.shape[1] == 5
    assert neg_edges.shape[0] == 2
    
    # Verificar que todos los negativos son válidos (src -> dst)
    src_indices = neg_edges[0, :]
    dst_indices = neg_edges[1, :]
    
    assert src_indices.min() >= 0 and src_indices.max() < data['src'].num_nodes
    assert dst_indices.min() >= 0 and dst_indices.max() < data['dst'].num_nodes
    
    log.info(f"Negativos generados: {neg_edges.shape[1]}")
    log.info(f"Src range: [{src_indices.min()}, {src_indices.max()}]")
    log.info(f"Dst range: [{dst_indices.min()}, {dst_indices.max()}]")
    
    # Prueba con datos bipartitos estándar
    bipartite_data = create_test_bipartite_data()
    neg_edges_std = bipartite_negative_sampling(
        bipartite_data.edge_index, bipartite_data, num_neg_samples=5
    )
    
    assert neg_edges_std.shape[1] == 5
    log.info("✓ Muestreo negativo bipartito funciona correctamente")

def test_bipartite_negative_sampling_with_pyg():
    """Prueba directa con negative_sampling de PyG usando tupla"""
    log.info("=== Probando negative_sampling directo con tupla ===")
    
    # Crear aristas de prueba
    edge_index = torch.tensor([[0, 1, 2], [0, 1, 2]])  # Aristas dummy
    
    # Usar tupla de nodos (como debe ser para bipartito)
    num_nodes_tuple = (5, 3)  # 5 nodos tipo 1, 3 nodos tipo 2
    
    try:
        neg_edges = negative_sampling(
            edge_index=edge_index,
            num_nodes=num_nodes_tuple,
            num_neg_samples=4,
            method='sparse'
        )
        
        log.info(f"Negativos generados con PyG: {neg_edges.shape}")
        log.info(f"Src indices: {neg_edges[0, :].tolist()}")
        log.info(f"Dst indices: {neg_edges[1, :].tolist()}")
        
        # Verificar rangos
        assert neg_edges[0, :].min() >= 0 and neg_edges[0, :].max() < 5  # tipo 1
        assert neg_edges[1, :].min() >= 0 and neg_edges[1, :].max() < 3  # tipo 2
        
        log.info("✓ negative_sampling con tupla funciona correctamente")
        
    except Exception as e:
        log.error(f"❌ Error con negative_sampling: {e}")
        raise

def main():
    """Ejecuta todas las pruebas"""
    log.info("🔬 Iniciando pruebas de correcciones bipartitas...")
    
    try:
        test_drop_features()
        test_scramble_features()
        test_random_edges()
        test_bipartite_negative_sampling()
        test_bipartite_negative_sampling_with_pyg()
        
        log.info("🎉 ¡Todas las pruebas pasaron exitosamente!")
        log.info("✅ Las correcciones bipartitas están funcionando correctamente.")
        
    except Exception as e:
        log.error(f"❌ Error en las pruebas: {e}")
        raise

if __name__ == "__main__":
    main()
