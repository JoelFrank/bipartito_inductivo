#!/usr/bin/env python3
"""
Script de depuración para investigar el desbalance de enlaces positivos/negativos
"""

import sys
import torch
import os
import pandas as pd
from pathlib import Path

def debug_edge_counts():
    print("=== DEPURACIÓN DE CONTEO DE ENLACES ===")
    
    # Método 1: Analizar directamente el CSV generado
    print("\n1. Analizando CSV generado más reciente...")
    
    csv_path = "runs/sismetro_tbgrl_inductive_test/Bipartite_triplet_sismetro_inductive_lr0.008_run_20250816_160708/prod_mlp_test_results.csv"
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        pos_count = len(df[df['label'] == 1])
        neg_count = len(df[df['label'] == 0])
        
        print(f"   - Total filas en CSV: {len(df)}")
        print(f"   - Enlaces positivos: {pos_count}")
        print(f"   - Enlaces negativos: {neg_count}")
        print(f"   - Diferencia: {pos_count - neg_count}")
        
        # Analizar scores para detectar patrones
        pos_scores = df[df['label'] == 1]['score']
        neg_scores = df[df['label'] == 0]['score']
        
        print(f"\n2. Análisis de scores:")
        print(f"   - Score promedio positivos: {pos_scores.mean():.4f}")
        print(f"   - Score promedio negativos: {neg_scores.mean():.4f}")
        print(f"   - Score max positivos: {pos_scores.max():.4f}")
        print(f"   - Score min negativos: {neg_scores.min():.4f}")
        
        # Verificar duplicados en el CSV
        print(f"\n3. Análisis de duplicados en CSV:")
        original_count = len(df)
        df_no_dups = df.drop_duplicates(subset=['u', 'v'])
        clean_count = len(df_no_dups)
        
        print(f"   - Filas originales: {original_count}")
        print(f"   - Filas sin duplicados: {clean_count}")
        print(f"   - Duplicados eliminados: {original_count - clean_count}")
        
        # Análisis detallado de los enlaces
        print(f"\n4. Análisis detallado de enlaces:")
        pos_df = df[df['label'] == 1]
        neg_df = df[df['label'] == 0]
        
        # Verificar nodos únicos
        pos_u_unique = pos_df['u'].nunique()
        pos_v_unique = pos_df['v'].nunique()
        neg_u_unique = neg_df['u'].nunique()
        neg_v_unique = neg_df['v'].nunique()
        
        print(f"   - Positivos - Nodos u únicos: {pos_u_unique}, Nodos v únicos: {pos_v_unique}")
        print(f"   - Negativos - Nodos u únicos: {neg_u_unique}, Nodos v únicos: {neg_v_unique}")
        
        # Verificar solapamiento de nodos entre u y v (bipartito)
        all_u = set(df['u'].unique())
        all_v = set(df['v'].unique())
        overlap = all_u.intersection(all_v)
        
        print(f"   - Solapamiento u-v (debería ser 0 para bipartito): {len(overlap)}")
        if len(overlap) > 0:
            print(f"     Nodos solapados: {list(overlap)[:10]}...")  # Mostrar primeros 10
            
    else:
        print(f"   ❌ CSV no encontrado en: {csv_path}")
    
    # Método 2: Cargar dataset directamente
    print("\n5. Cargando dataset directamente...")
    
    try:
        dataset_path = "data/processed/sismetro_inductive.pt"
        if os.path.exists(dataset_path):
            data = torch.load(dataset_path)
            print(f"   - Nodos totales: {data.num_nodes}")
            if hasattr(data, 'num_nodes_type_1'):
                print(f"   - Nodos tipo 1: {data.num_nodes_type_1}")
                print(f"   - Nodos tipo 2: {data.num_nodes - data.num_nodes_type_1}")
            print(f"   - Enlaces totales: {data.edge_index.size(1)}")
            
            # Verificar splits inductivos
            splits = ['train', 'test', 'valid']
            for split_name in splits:
                split_path = f"data/processed/sismetro_inductive_{split_name}.pt"
                if os.path.exists(split_path):
                    split_data = torch.load(split_path)
                    print(f"   - Split {split_name}: {split_data.edge_index.size(1)} enlaces")
        else:
            print(f"   ❌ Dataset no encontrado en: {dataset_path}")
            
    except Exception as e:
        print(f"   ❌ Error cargando dataset: {e}")

if __name__ == "__main__":
    debug_edge_counts()
