import pandas as pd
import torch
from torch_geometric.data import Data
import os
import numpy as np

def create_sismetro_bipartite_graph():
    """
    Crea un grafo bipartito para el dataset Sismetro.
    Nodo u: OBSERVACAO DE PATRIMONIO (con atributo TIPO DE EQUIPAMENTO)
    Nodo v: LOCALIZACAO (sin atributos)
    """
    print(f"--- Creando grafo bipartito Sismetro ---")
    
    # Rutas
    excel_path = r"../data/raw/sismetro/SISMETRO-Exportacao-SS-2021-2024_P1(OK PATRIMONIO).xlsx"
    save_dir = r"../data/processed"
    dataset_name = "sismetro"
    
    # Leer el archivo Excel
    print(f"Cargando datos desde: {excel_path}")
    df = pd.read_excel(excel_path, sheet_name=0)
    print(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    
    # Definir las columnas relevantes
    patrimonio_col = 'OBSERVAÇÃO PATRIMÔNIO'
    localizacao_col = 'LOCALIZAÇÃO'
    tipo_equipamento_col = 'TIPO DO EQUIPAMENTO'
    
    print(f"Columnas utilizadas:")
    print(f"  - Nodo u (Patrimônio): {patrimonio_col}")
    print(f"  - Nodo v (Localização): {localizacao_col}")
    print(f"  - Atributo embedding: {tipo_equipamento_col}")
    
    # Filtrar filas que tengan datos válidos en las columnas principales
    df_clean = df.dropna(subset=[patrimonio_col, localizacao_col])
    print(f"Después de limpiar valores nulos: {df_clean.shape[0]} filas")
    
    # Obtener nodos únicos
    unique_patrimonios = sorted(df_clean[patrimonio_col].unique())
    unique_localizacoes = sorted(df_clean[localizacao_col].unique())
    
    # Crear mapeos de nodos a índices
    patrimonio_to_idx = {node: i for i, node in enumerate(unique_patrimonios)}
    localizacao_to_idx = {node: i for i, node in enumerate(unique_localizacoes)}
    
    num_patrimonios = len(unique_patrimonios)
    num_localizacoes = len(unique_localizacoes)
    
    print(f"Nodos únicos - Patrimônios: {num_patrimonios}")
    print(f"Nodos únicos - Localizações: {num_localizacoes}")
    print(f"Total de nodos: {num_patrimonios + num_localizacoes}")
    
    # Crear las aristas del grafo bipartito
    # Los índices de localización empiezan después de los de patrimonio
    src_nodes = [patrimonio_to_idx[p] for p in df_clean[patrimonio_col]]
    dst_nodes = [localizacao_to_idx[l] + num_patrimonios for l in df_clean[localizacao_col]]
    
    # Crear edge_index (bidireccional para grafo no dirigido)
    edge_index_forward = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
    edge_index_backward = torch.tensor([dst_nodes, src_nodes], dtype=torch.long)
    edge_index = torch.cat([edge_index_forward, edge_index_backward], dim=1)
    
    print(f"Total de aristas originales: {len(src_nodes)}")
    print(f"Tamaño final de edge_index (bidireccional): {edge_index.shape}")
    
    # Crear atributos de nodos para patrimônios (tipo de equipamento)
    # Primero, crear un mapeo de tipos de equipamento
    unique_tipos = sorted(df_clean[tipo_equipamento_col].dropna().unique())
    tipo_to_idx = {tipo: i for i, tipo in enumerate(unique_tipos)}
    
    print(f"Tipos de equipamento únicos: {len(unique_tipos)}")
    
    # Crear atributos para cada nodo de patrimônio
    patrimonio_attrs = torch.zeros(num_patrimonios, len(unique_tipos))
    
    # Para cada patrimônio, encontrar sus tipos de equipamento
    for _, row in df_clean.iterrows():
        patrimonio = row[patrimonio_col]
        tipo_eq = row[tipo_equipamento_col]
        
        if pd.notna(tipo_eq) and patrimonio in patrimonio_to_idx:
            p_idx = patrimonio_to_idx[patrimonio]
            t_idx = tipo_to_idx[tipo_eq]
            patrimonio_attrs[p_idx, t_idx] = 1.0
    
    # Los nodos de localização no tienen atributos específicos
    localizacao_attrs = torch.zeros(num_localizacoes, len(unique_tipos))
    
    # Concatenar atributos de ambos tipos de nodos
    node_features = torch.cat([patrimonio_attrs, localizacao_attrs], dim=0)
    
    print(f"Forma de node_features: {node_features.shape}")
    
    # Crear el objeto Data de PyTorch Geometric
    data = Data(
        x=node_features,
        edge_index=edge_index,
        num_nodes=num_patrimonios + num_localizacoes,
        num_nodes_type_1=num_patrimonios,  # patrimônios
        num_nodes_type_2=num_localizacoes   # localizações
    )
    
    # Guardar el grafo
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    save_path = os.path.join(save_dir, f'{dataset_name}.pt')
    torch.save(data, save_path)
    
    print(f"Grafo bipartito Sismetro guardado en: {save_path}")
    print("Resumen del objeto Data:", data)
    
    # Estadísticas finales
    print("\n=== ESTADÍSTICAS FINALES ===")
    print(f"Nodos totales: {data.num_nodes}")
    print(f"Nodos tipo 1 (Patrimônios): {data.num_nodes_type_1}")
    print(f"Nodos tipo 2 (Localizações): {data.num_nodes_type_2}")
    print(f"Aristas totales: {data.edge_index.shape[1]}")
    print(f"Características por nodo: {data.x.shape[1]}")
    print(f"Tipos de equipamento únicos: {len(unique_tipos)}")
    
    print("--- Proceso finalizado exitosamente ---")
    return data

if __name__ == '__main__':
    create_sismetro_bipartite_graph()
