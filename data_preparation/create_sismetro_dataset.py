import pandas as pd
import torch
from torch_geometric.data import Data
import os
import numpy as np

def explore_sismetro_excel(excel_path):
    """
    Explora el archivo Excel de Sismetro para entender su estructura.
    """
    print(f"--- Explorando archivo Excel: {excel_path} ---")
    
    # Leer el archivo Excel
    try:
        # Primero intentamos leer las hojas disponibles
        xl_file = pd.ExcelFile(excel_path)
        print(f"Hojas disponibles: {xl_file.sheet_names}")
        
        # Leer la primera hoja (o la principal)
        df = pd.read_excel(excel_path, sheet_name=0)
        
        print(f"Forma del dataset: {df.shape}")
        print(f"Columnas: {list(df.columns)}")
        print("\nPrimeras 5 filas:")
        print(df.head())
        
        print("\nInformación sobre valores nulos:")
        print(df.isnull().sum())
        
        # Buscar las columnas relevantes
        print("\nBuscando columnas relevantes...")
        relevant_cols = []
        
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['observacao', 'patrimonio', 'equipamento', 'tipo']):
                print(f"Columna potencial para 'OBSERVACAO DE PATRIMONIO': {col}")
                relevant_cols.append(col)
            elif any(keyword in col_lower for keyword in ['localizacao', 'local', 'endereco', 'lugar']):
                print(f"Columna potencial para 'LOCALIZACAO': {col}")
                relevant_cols.append(col)
        
        if relevant_cols:
            print(f"\nValores únicos en columnas relevantes:")
            for col in relevant_cols:
                print(f"\n{col}: {df[col].nunique()} valores únicos")
                print(f"Ejemplos: {df[col].dropna().unique()[:5]}")
        
        return df
        
    except Exception as e:
        print(f"Error al leer el archivo: {e}")
        return None

def create_sismetro_bipartite_graph(excel_path, save_dir, dataset_name="sismetro"):
    """
    Crea un grafo bipartito para el dataset Sismetro.
    Nodo u: OBSERVACAO DE PATRIMONIO (con atributo TIPO DE EQUIPAMENTO)
    Nodo v: LOCALIZACAO (sin atributos)
    """
    print(f"--- Iniciando la creación del grafo bipartito Sismetro ---")
    
    # Leer el archivo Excel
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
    
    # Crear atributos de nodos para patrimônios (tipo de equipamento) - USANDO EMBEDDINGS
    # Primero, crear un mapeo de tipos de equipamento
    unique_tipos = sorted(df_clean[tipo_equipamento_col].dropna().unique())
    tipo_to_idx = {tipo: i for i, tipo in enumerate(unique_tipos)}
    
    print(f"Tipos de equipamento únicos: {len(unique_tipos)}")
    print("USANDO EMBEDDING IDs en lugar de one-hot encoding")
    
    # Crear atributos para cada nodo de patrimônio - CAMBIO: usar índices en lugar de one-hot
    patrimonio_attrs = torch.full((num_patrimonios, 1), -1, dtype=torch.long)  # -1 para "sin tipo"
    
    # Para cada patrimônio, asignar el índice de su tipo de equipamento principal
    for _, row in df_clean.iterrows():
        patrimonio = row[patrimonio_col]
        tipo_eq = row[tipo_equipamento_col]
        
        if pd.notna(tipo_eq) and patrimonio in patrimonio_to_idx:
            p_idx = patrimonio_to_idx[patrimonio]
            t_idx = tipo_to_idx[tipo_eq]
            patrimonio_attrs[p_idx, 0] = t_idx  # Asignar el índice del tipo
    
    # Los nodos de localização no tienen tipos específicos (usarán índice especial)
    num_embedding_types = len(unique_tipos) + 1 # +1 para índice especial de localizações
    localizacao_attrs = torch.full((num_localizacoes, 1), len(unique_tipos), dtype=torch.long)  # Índice especial para localizações
    
    # Concatenar atributos de ambos tipos de nodos
    node_features = torch.cat([patrimonio_attrs, localizacao_attrs], dim=0)
    
    print(f"Forma de node_features: {node_features.shape}")
    
    # Crear el objeto Data de PyTorch Geometric
    data = Data(
        x=node_features,
        edge_index=edge_index,
        num_nodes=num_patrimonios + num_localizacoes,
        num_nodes_type_1=num_patrimonios,  # patrimônios
        num_nodes_type_2=num_localizacoes,   # localizações
        num_embedding_types=num_embedding_types,  # Para la capa de embedding
        embedding_dim=None  # Se definirá en el modelo
    )
    
    # Guardar mappings adicionales para referencia
    data.patrimonio_to_idx = patrimonio_to_idx
    data.localizacao_to_idx = localizacao_to_idx
    data.tipo_to_idx = tipo_to_idx
    data.unique_patrimonios = unique_patrimonios
    data.unique_localizacoes = unique_localizacoes
    data.unique_tipos = unique_tipos
    
    # Guardar el grafo
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    save_path = os.path.join(save_dir, f'{dataset_name}.pt')
    torch.save(data, save_path)
    
    print(f"Grafo bipartito Sismetro guardado en: {save_path}")
    print("Resumen del objeto Data:", data)
    print("--- Proceso finalizado exitosamente ---")
    
    return data

if __name__ == '__main__':
    excel_path = r"C:\Users\joelf\Documents\non-contrastive-link-prediction-bipartite\data\raw\sismetro\SISMETRO-Exportacao-SS-2021-2024_P1(OK PATRIMONIO).xlsx"
    save_dir = r"C:\Users\joelf\Documents\non-contrastive-link-prediction-bipartite\data\processed"
    
    # Primero explorar el archivo
    print("=== FASE 1: EXPLORACIÓN ===")
    df = explore_sismetro_excel(excel_path)
    
    if df is not None:
        print("\n=== FASE 2: CREACIÓN DEL GRAFO BIPARTITO ===")
        data = create_sismetro_bipartite_graph(excel_path, save_dir, "sismetro")
        
        if data is not None:
            print("\n=== ESTADÍSTICAS FINALES ===")
            print(f"Nodos totales: {data.num_nodes}")
            print(f"Nodos tipo 1 (Patrimônios): {data.num_nodes_type_1}")
            print(f"Nodos tipo 2 (Localizações): {data.num_nodes_type_2}")
            print(f"Aristas totales: {data.edge_index.shape[1]}")
            print(f"Forma de atributos: {data.x.shape}")
            print(f"Tipos de embedding únicos: {data.num_embedding_types}")
            print(f"Tipos de equipamento únicos: {len(data.unique_tipos)}")
            print("NOTA: Usando embedding IDs (enteros) en lugar de one-hot vectors")
            
            # Mostrar algunos ejemplos
            print(f"\nEjemplos de patrimônios: {data.unique_patrimonios[:5]}")
            print(f"Ejemplos de localizações: {data.unique_localizacoes[:5]}")
            print(f"Ejemplos de tipos de equipamento: {data.unique_tipos[:5]}")
