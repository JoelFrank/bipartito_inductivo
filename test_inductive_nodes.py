"""
Script para verificar el dataset Sismetro y probar el nuevo split inductivo basado en nodos.
"""
import torch
import sys
import os

# Agregar el directorio src al path
sys.path.append('src')

from lib.bipartite_inductive_split import do_bipartite_inductive_node_split

def main():
    print("=== VERIFICACIÓN DEL DATASET SISMETRO ===")
    
    # Cargar dataset
    data_path = 'data/processed/sismetro.pt'
    if not os.path.exists(data_path):
        print(f"❌ Dataset no encontrado: {data_path}")
        return
    
    data = torch.load(data_path)
    print(f"✓ Dataset cargado: {data}")
    print(f"✓ Nodos: {data.num_nodes}")
    print(f"✓ Aristas: {data.edge_index.size(1)}")
    
    if hasattr(data, 'num_nodes_type_1'):
        print(f"✓ Nodos tipo 1: {data.num_nodes_type_1}")
        print(f"✓ Nodos tipo 2: {data.num_nodes - data.num_nodes_type_1}")
        print(f"✓ Es grafo bipartito")
    else:
        print(f"❌ No es un grafo bipartito válido")
        return
    
    if hasattr(data, 'x') and data.x is not None:
        print(f"✓ Características de nodos: {data.x.shape}")
    else:
        print(f"⚠️ Sin características de nodos")
    
    print(f"\n=== PROBANDO SPLIT INDUCTIVO BASADO EN NODOS ===")
    
    try:
        # Probar el split inductivo
        result = do_bipartite_inductive_node_split(
            data, 
            observed_ratio=0.7,
            split_seed=42
        )
        
        print(f"✓ Split inductivo exitoso!")
        
        # Verificar resultados
        print(f"\n=== RESUMEN DEL SPLIT ===")
        print(f"Training data (encoder): {result['training_data'].num_nodes} nodos")
        print(f"Inference data (decoder): {result['inference_data'].num_nodes} nodos")
        print(f"Train edges: {result['train_edge_bundle'].size(1)}")
        print(f"Val edges: {result['val_edge_bundle'].size(1)}")
        print(f"Test edges: {result['test_edge_bundle'].size(1)}")
        print(f"Val neg edges: {result['val_neg_edge_bundle'].size(1)}")
        print(f"Test neg edges: {result['test_neg_edge_bundle'].size(1)}")
        print(f"Nodos observados: {len(result['observed_nodes'])}")
        print(f"Nodos no observados: {len(result['unobserved_nodes'])}")
        
        print(f"\n✅ VERIFICACIÓN COMPLETADA - SISTEMA LISTO PARA USAR")
        
    except Exception as e:
        print(f"❌ Error en split inductivo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
