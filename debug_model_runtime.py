#!/usr/bin/env python3
"""
Script para debuggear el BipartiteSAGE en tiempo real
Verifica índices de embedding justo antes de que fallen
"""

import torch
import sys
import os
sys.path.append('src')

from lib.models.encoders import BipartiteSAGE

# Cargar datos
print("=== CARGANDO DATOS ===")
data = torch.load("data/processed/sismetro.pt")
print(f"Data loaded: {data.num_nodes} nodes")
print(f"Node features shape: {data.x.shape}")
print(f"Node features min: {data.x.min().item()}, max: {data.x.max().item()}")

# Verificar el número de tipos de embedding
if hasattr(data, 'num_embedding_types'):
    num_embedding_types = data.num_embedding_types
    print(f"num_embedding_types from data: {num_embedding_types}")
else:
    # Calcular basado en los datos
    num_embedding_types = data.x.max().item() + 1
    print(f"num_embedding_types calculated: {num_embedding_types}")

print(f"Expected embedding range: 0 to {num_embedding_types - 1}")

# Crear modelo BipartiteSAGE
print("\n=== CREANDO MODELO ===")
try:
    model = BipartiteSAGE(
        in_channels=64,  # embedding dim
        hidden_channels=128,
        out_channels=64,
        num_layers=2,
        dropout=0.0,
        num_nodes_type_1=data.num_nodes_type_1,
        num_nodes_type_2=data.num_nodes_type_2,
        num_embedding_types=num_embedding_types,
        embedding_dim=64
    )
    print("✅ Modelo creado exitosamente")
    print(f"Embedding layer: {model.embedding_processor.embedding}")
    print(f"Embedding weight shape: {model.embedding_processor.embedding.weight.shape}")
except Exception as e:
    print(f"❌ Error creando modelo: {e}")
    sys.exit(1)

# Test forward pass con datos pequeños
print("\n=== PROBANDO FORWARD PASS ===")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = model.to(device)
data = data.to(device)

# Tomar solo los primeros 100 nodos para debug
subset_nodes = 100
x_subset = data.x[:subset_nodes]
print(f"Testing with subset: {x_subset.shape}")
print(f"Subset min: {x_subset.min().item()}, max: {x_subset.max().item()}")

# Verificar si hay índices fuera de rango
invalid_indices = x_subset >= num_embedding_types
if invalid_indices.any():
    print(f"❌ ENCONTRADOS ÍNDICES INVÁLIDOS:")
    invalid_values = x_subset[invalid_indices]
    print(f"Valores inválidos: {invalid_values.unique()}")
    print(f"Máximo permitido: {num_embedding_types - 1}")
else:
    print("✅ Todos los índices están en rango válido")

# Test embedding lookup
print("\n=== PROBANDO EMBEDDING LOOKUP ===")
try:
    embeddings = model.embedding_processor(x_subset)
    print(f"✅ Embedding lookup exitoso: {embeddings.shape}")
except Exception as e:
    print(f"❌ Error en embedding lookup: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test con edge_index pequeño
print("\n=== PROBANDO CON EDGES ===")
# Crear un edge_index pequeño que use solo los primeros 100 nodos
edge_subset = data.edge_index[:, (data.edge_index[0] < subset_nodes) & (data.edge_index[1] < subset_nodes)]
print(f"Edge subset shape: {edge_subset.shape}")

if edge_subset.shape[1] > 0:
    try:
        # Crear un grafo pequeño para test
        x_test = x_subset
        edge_test = edge_subset
        
        print(f"Test input - x: {x_test.shape}, edge: {edge_test.shape}")
        print(f"Edge min: {edge_test.min().item()}, max: {edge_test.max().item()}")
        
        # Forward pass
        output = model(x_test, edge_test)
        print(f"✅ Forward pass exitoso: {output.shape}")
        
    except Exception as e:
        print(f"❌ Error en forward pass: {e}")
        import traceback
        traceback.print_exc()
else:
    print("⚠️ No hay edges en el subset para probar")

print("\n=== DEBUG COMPLETO ===")
