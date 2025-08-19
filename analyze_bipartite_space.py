#!/usr/bin/env python3
"""
Verificar el espacio de enlaces posibles en grafo bipartito
"""

import torch
import pandas as pd
import os

def analyze_bipartite_space():
    print("=== ANÁLISIS DEL ESPACIO DE ENLACES BIPARTITO ===")
    
    # 1. Cargar datos del grafo
    print("\n1. Analizando estructura bipartita...")
    full_data = torch.load('data/processed/sismetro_inductive.pt')
    
    num_nodes_type_1 = full_data.num_nodes_type_1  # Nodos U
    num_nodes_type_2 = full_data.num_nodes - full_data.num_nodes_type_1  # Nodos V
    
    print(f"   - Nodos tipo U: {num_nodes_type_1:,}")
    print(f"   - Nodos tipo V: {num_nodes_type_2:,}")
    
    # 2. Calcular espacio teórico
    print("\n2. Calculando espacio teórico de enlaces...")
    max_possible_edges = num_nodes_type_1 * num_nodes_type_2
    print(f"   - Enlaces posibles máximos: {max_possible_edges:,}")
    
    # 3. Analizar enlaces existentes
    print("\n3. Analizando enlaces existentes...")
    existing_edges = set()
    full_edges = full_data.edge_index
    
    for i in range(full_edges.size(1)):
        u, v = full_edges[0, i].item(), full_edges[1, i].item()
        # Verificar estructura bipartita
        if u < num_nodes_type_1:  # u es tipo 1
            if v >= num_nodes_type_1:  # v es tipo 2
                edge = (u, v)  # Mantener dirección u->v
            else:
                print(f"   ⚠️  Enlace inválido: ({u}, {v}) - ambos son tipo 1")
                continue
        else:  # u es tipo 2
            if v < num_nodes_type_1:  # v es tipo 1
                edge = (v, u)  # Normalizar a tipo1->tipo2
            else:
                print(f"   ⚠️  Enlace inválido: ({u}, {v}) - ambos son tipo 2")
                continue
        existing_edges.add(edge)
    
    print(f"   - Enlaces únicos existentes: {len(existing_edges):,}")
    print(f"   - Enlaces únicos NO existentes: {max_possible_edges - len(existing_edges):,}")
    print(f"   - Porcentaje de densidad: {(len(existing_edges)/max_possible_edges*100):.4f}%")
    
    # 4. Verificar CSV
    print("\n4. Analizando enlaces del CSV...")
    csv_path = "runs/sismetro_tbgrl_inductive_test/Bipartite_triplet_sismetro_inductive_lr0.008_run_20250816_160708/prod_mlp_test_results.csv"
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        pos_count = len(df[df['label'] == 1])
        neg_count = len(df[df['label'] == 0])
        
        print(f"   - Enlaces positivos en CSV: {pos_count}")
        print(f"   - Enlaces negativos en CSV: {neg_count}")
        print(f"   - Diferencia: {pos_count - neg_count}")
        
        # 5. Verificar si hay suficiente espacio para igualar
        available_negative_space = max_possible_edges - len(existing_edges)
        print(f"\n5. Verificación de espacio disponible:")
        print(f"   - Espacio negativo disponible: {available_negative_space:,}")
        print(f"   - Enlaces positivos requeridos: {pos_count}")
        print(f"   - ¿Hay suficiente espacio? {'✅ SÍ' if available_negative_space >= pos_count else '❌ NO'}")
        
        if available_negative_space >= pos_count:
            print(f"   - Sobra espacio: {available_negative_space - pos_count:,} enlaces")
            print(f"\n   💡 CONCLUSIÓN: SÍ debería ser posible generar {pos_count} enlaces negativos")
            
            if neg_count < pos_count:
                print(f"   ❓ PREGUNTA: ¿Por qué solo se generaron {neg_count} en lugar de {pos_count}?")
        
        # 6. Verificar estructura de enlaces del CSV
        print(f"\n6. Verificando estructura bipartita en CSV...")
        csv_u_nodes = set(df['u'].unique())
        csv_v_nodes = set(df['v'].unique())
        overlap = csv_u_nodes.intersection(csv_v_nodes)
        
        print(f"   - Nodos u únicos en CSV: {len(csv_u_nodes)}")
        print(f"   - Nodos v únicos en CSV: {len(csv_v_nodes)}")
        print(f"   - Solapamiento u-v: {len(overlap)} {'✅' if len(overlap) == 0 else '❌'}")
        
        # Verificar rangos
        u_in_type1 = all(u < num_nodes_type_1 for u in csv_u_nodes)
        v_in_type2 = all(v >= num_nodes_type_1 for v in csv_v_nodes)
        
        print(f"   - Todos los u son tipo 1: {'✅' if u_in_type1 else '❌'}")
        print(f"   - Todos los v son tipo 2: {'✅' if v_in_type2 else '❌'}")

if __name__ == "__main__":
    analyze_bipartite_space()
