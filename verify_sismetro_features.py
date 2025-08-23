"""
Script para verificar si el dataset Sismetro está usando correctamente 
los atributos "tipo do equipamento" para los nodos de patrimônio.
"""
import torch
import numpy as np

def analyze_sismetro_features():
    print("=== ANÁLISIS DE CARACTERÍSTICAS DEL DATASET SISMETRO ===")
    
    # Cargar dataset
    data_path = 'data/processed/sismetro.pt'
    data = torch.load(data_path)
    
    print(f"Dataset cargado:")
    print(f"  Nodos totales: {data.num_nodes}")
    print(f"  Nodos tipo 1 (patrimônios): {data.num_nodes_type_1}")
    print(f"  Nodos tipo 2 (localizações): {data.num_nodes_type_2}")
    print(f"  Aristas: {data.edge_index.size(1)}")
    print(f"  Características por nodo: {data.x.shape}")
    
    # Analizar las características
    if data.x is not None:
        print(f"\n=== ANÁLISIS DE CARACTERÍSTICAS ===")
        features = data.x
        
        # Separar características por tipo de nodo
        patrimonio_features = features[:data.num_nodes_type_1]  # Primeros N nodos
        localizacao_features = features[data.num_nodes_type_1:]  # Últimos M nodos
        
        print(f"Características patrimônios: {patrimonio_features.shape}")
        print(f"Características localizações: {localizacao_features.shape}")
        
        # Verificar si hay información en las características
        print(f"\n=== ESTADÍSTICAS DE CARACTERÍSTICAS ===")
        print(f"Patrimônios:")
        print(f"  Valores no cero por nodo - Media: {patrimonio_features.sum(dim=1).float().mean():.2f}")
        print(f"  Valores no cero por nodo - Std: {patrimonio_features.sum(dim=1).float().std():.2f}")
        print(f"  Valores no cero por nodo - Min: {patrimonio_features.sum(dim=1).min()}")
        print(f"  Valores no cero por nodo - Max: {patrimonio_features.sum(dim=1).max()}")
        
        print(f"Localizações:")
        print(f"  Valores no cero por nodo - Media: {localizacao_features.sum(dim=1).float().mean():.2f}")
        print(f"  Valores no cero por nodo - Std: {localizacao_features.sum(dim=1).float().std():.2f}")
        print(f"  Valores no cero por nodo - Min: {localizacao_features.sum(dim=1).min()}")
        print(f"  Valores no cero por nodo - Max: {localizacao_features.sum(dim=1).max()}")
        
        # Verificar cuántos nodos tienen características
        patrimonio_with_features = (patrimonio_features.sum(dim=1) > 0).sum().item()
        localizacao_with_features = (localizacao_features.sum(dim=1) > 0).sum().item()
        
        print(f"\n=== COBERTURA DE CARACTERÍSTICAS ===")
        print(f"Patrimônios con características: {patrimonio_with_features}/{data.num_nodes_type_1} ({patrimonio_with_features/data.num_nodes_type_1*100:.1f}%)")
        print(f"Localizações con características: {localizacao_with_features}/{data.num_nodes_type_2} ({localizacao_with_features/data.num_nodes_type_2*100:.1f}%)")
        
        # Mostrar algunos ejemplos
        print(f"\n=== EJEMPLOS DE CARACTERÍSTICAS ===")
        print("Primeros 5 patrimônios:")
        for i in range(min(5, patrimonio_features.size(0))):
            nonzero_indices = patrimonio_features[i].nonzero().flatten()
            print(f"  Patrimônio {i}: {len(nonzero_indices)} características activas")
            if len(nonzero_indices) > 0:
                print(f"    Índices activos: {nonzero_indices.tolist()[:10]}...")  # Mostrar primeros 10
        
        print("Primeras 5 localizações:")
        for i in range(min(5, localizacao_features.size(0))):
            nonzero_indices = localizacao_features[i].nonzero().flatten()
            print(f"  Localização {i}: {len(nonzero_indices)} características activas")
            if len(nonzero_indices) > 0:
                print(f"    Índices activos: {nonzero_indices.tolist()[:10]}...")
        
        # Verificar si el modelo está usando las características
        print(f"\n=== TIPO DE CODIFICACIÓN ===")
        if features.dtype == torch.long and features.size(1) == 1:
            print("✓ USANDO EMBEDDING IDs (recomendado)")
            print("  Las características son índices enteros para embeddings")
            
            # Mostrar distribución de IDs
            unique_ids, counts = torch.unique(features.flatten(), return_counts=True)
            print(f"  IDs únicos: {len(unique_ids)}")
            print(f"  ID más común: {unique_ids[counts.argmax()].item()} (usado {counts.max()} veces)")
            print(f"  Rango de IDs: {unique_ids.min().item()} - {unique_ids.max().item()}")
            
        elif features.dtype == torch.float and features.size(1) > 1:
            print("✓ USANDO ONE-HOT ENCODING")
            print("  Las características son vectores binarios")
            
            # Verificar sparsity
            total_elements = features.numel()
            nonzero_elements = (features > 0).sum().item()
            sparsity = (1 - nonzero_elements/total_elements) * 100
            print(f"  Sparsity: {sparsity:.1f}%")
            
        else:
            print("⚠️ FORMATO DE CARACTERÍSTICAS DESCONOCIDO")
            print(f"  Dtype: {features.dtype}, Shape: {features.shape}")
    else:
        print("❌ NO HAY CARACTERÍSTICAS DE NODOS")
    
    # Verificar si hay atributos adicionales
    print(f"\n=== ATRIBUTOS ADICIONALES ===")
    for attr in dir(data):
        if not attr.startswith('_') and attr not in ['x', 'edge_index', 'num_nodes']:
            value = getattr(data, attr)
            if not callable(value):
                print(f"  {attr}: {type(value)} = {value}")

if __name__ == '__main__':
    analyze_sismetro_features()
