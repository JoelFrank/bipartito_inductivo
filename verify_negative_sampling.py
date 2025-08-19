#!/usr/bin/env python3
"""
Script para verificar que los enlaces negativos NO existen en el grafo completo
"""

import torch
import pandas as pd
import os

def verify_negative_sampling():
    print("=== VERIFICACIÓN DE MUESTREO NEGATIVO INDUCTIVO ===")
    
    # 1. Cargar el grafo completo
    print("\n1. Cargando grafo completo...")
    full_data = torch.load('data/processed/sismetro_inductive.pt')
    full_edges = full_data.edge_index
    
    print(f"   - Grafo completo: {full_edges.size(1):,} enlaces")
    
    # 2. Crear conjunto de todos los enlaces que SÍ existen
    print("\n2. Creando conjunto de enlaces existentes...")
    existing_edges = set()
    
    for i in range(full_edges.size(1)):
        u, v = full_edges[0, i].item(), full_edges[1, i].item()
        # Normalizar dirección (siempre menor, mayor)
        edge = tuple(sorted([u, v]))
        existing_edges.add(edge)
    
    print(f"   - Enlaces únicos existentes: {len(existing_edges):,}")
    
    # 3. Cargar CSV de resultados
    print("\n3. Analizando enlaces negativos del CSV...")
    csv_path = "runs/sismetro_tbgrl_inductive_test/Bipartite_triplet_sismetro_inductive_lr0.008_run_20250816_160708/prod_mlp_test_results.csv"
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        
        # 4. Verificar enlaces negativos
        negative_df = df[df['label'] == 0]
        print(f"   - Enlaces negativos en CSV: {len(negative_df)}")
        
        # Verificar si algún enlace negativo existe en el grafo completo
        violations = 0
        sample_violations = []
        
        for _, row in negative_df.iterrows():
            u, v = int(row['u']), int(row['v'])
            edge = tuple(sorted([u, v]))
            
            if edge in existing_edges:
                violations += 1
                if len(sample_violations) < 5:  # Guardar algunos ejemplos
                    sample_violations.append((u, v, row['score']))
        
        print(f"\n4. RESULTADOS DE VERIFICACIÓN:")
        print(f"   - Enlaces negativos verificados: {len(negative_df)}")
        print(f"   - VIOLACIONES encontradas: {violations}")
        print(f"   - Porcentaje de violaciones: {(violations/len(negative_df)*100):.2f}%")
        
        if violations > 0:
            print(f"\n   ❌ PROBLEMA: {violations} enlaces negativos SÍ existen en el grafo completo!")
            print(f"   Ejemplos de violaciones:")
            for u, v, score in sample_violations:
                print(f"     - Enlace ({u}, {v}) con score {score:.4f}")
        else:
            print(f"\n   ✅ CORRECTO: Ningún enlace negativo existe en el grafo completo")
        
        # 5. Verificar enlaces positivos (deben existir)
        print(f"\n5. Verificando enlaces positivos...")
        positive_df = df[df['label'] == 1]
        pos_violations = 0
        
        for _, row in positive_df.iterrows():
            u, v = int(row['u']), int(row['v'])
            edge = tuple(sorted([u, v]))
            
            if edge not in existing_edges:
                pos_violations += 1
        
        print(f"   - Enlaces positivos: {len(positive_df)}")
        print(f"   - Enlaces positivos que NO existen en grafo: {pos_violations}")
        
        if pos_violations > 0:
            print(f"   ❌ PROBLEMA: {pos_violations} enlaces positivos NO existen en el grafo!")
        else:
            print(f"   ✅ CORRECTO: Todos los enlaces positivos existen en el grafo")
            
    else:
        print(f"   ❌ CSV no encontrado: {csv_path}")

if __name__ == "__main__":
    verify_negative_sampling()
