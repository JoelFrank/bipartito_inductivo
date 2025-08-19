#!/usr/bin/env python3
"""
Script para debuggear índices en TODOS los archivos de dataset
"""

import torch
import os

def check_file_indices(filepath):
    """Verificar índices en un archivo específico"""
    if not os.path.exists(filepath):
        print(f"❌ {filepath} - No existe")
        return
        
    try:
        data = torch.load(filepath)
        print(f"\n🔍 {filepath}")
        
        if hasattr(data, 'x_dict'):
            # HeteroData
            for node_type, x in data.x_dict.items():
                if x.dtype == torch.long:
                    print(f"  {node_type}: min={x.min().item()}, max={x.max().item()}, shape={x.shape}")
                else:
                    print(f"  {node_type}: shape={x.shape}, dtype={x.dtype}")
                    
        elif hasattr(data, 'x'):
            # PyTorch Geometric Data object
            print(f"  Data object: num_nodes={data.num_nodes}, num_edges={data.edge_index.shape[1] if hasattr(data, 'edge_index') else 'N/A'}")
            if hasattr(data, 'x') and data.x is not None:
                x = data.x
                if x.dtype == torch.long:
                    print(f"  x (attributes): min={x.min().item()}, max={x.max().item()}, shape={x.shape}")
                else:
                    print(f"  x (attributes): shape={x.shape}, dtype={x.dtype}")
            if hasattr(data, 'edge_index') and data.edge_index is not None:
                edge_index = data.edge_index
                print(f"  edge_index: min={edge_index.min().item()}, max={edge_index.max().item()}, shape={edge_index.shape}")
                    
        elif isinstance(data, dict):
            # Dictionary
            print(f"  Diccionario con keys: {list(data.keys())}")
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    if value.dtype == torch.long:
                        print(f"    {key}: min={value.min().item()}, max={value.max().item()}, shape={value.shape}")
                    else:
                        print(f"    {key}: shape={value.shape}, dtype={value.dtype}")
                        
        else:
            print(f"  Tipo desconocido: {type(data)}")
            
    except Exception as e:
        print(f"❌ Error cargando {filepath}: {e}")

# Lista de archivos a verificar
files_to_check = [
    "data/processed/sismetro.pt",
    "data/processed/sismetro_inductive_train.pt", 
    "data/processed/sismetro_inductive_val.pt",
    "data/processed/sismetro_inductive_test.pt",
    "data/processed/sismetro_inductive_metadata.pt"
]

print("=== DEBUG DE ÍNDICES EN TODOS LOS ARCHIVOS ===")
for filepath in files_to_check:
    check_file_indices(filepath)

print("\n=== RESUMEN ===")
print("✅ Si todos los archivos tienen max índice ≤ 162, entonces el problema está en otro lado")
print("❌ Si algún archivo tiene índice 163+, ahí está el problema")
