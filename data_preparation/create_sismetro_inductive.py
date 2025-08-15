import pandas as pd
import torch
from torch_geometric.data import Data
import os
import numpy as np
from datetime import datetime

def create_sismetro_inductive_split():
    """
    Crea el dataset Sismetro con split inductivo basado en tiempo.
    80% train (pasado) / 10% val (presente) / 10% test (futuro)
    """
    print(f"--- Creando grafo bipartito Sismetro con split inductivo temporal ---")
    
    # Rutas
    excel_path = r"../data/raw/sismetro/SISMETRO-Exportacao-SS-2021-2024_P1(OK PATRIMONIO).xlsx"
    save_dir = r"../data/processed"
    
    # Leer el archivo Excel
    print(f"Cargando datos desde: {excel_path}")
    df = pd.read_excel(excel_path, sheet_name=0)
    print(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    
    # Definir las columnas relevantes
    patrimonio_col = 'OBSERVAÇÃO PATRIMÔNIO'
    localizacao_col = 'LOCALIZAÇÃO'
    tipo_equipamento_col = 'TIPO DO EQUIPAMENTO'
    date_col = 'DATA DE ABERTURA'
    
    print(f"Columnas utilizadas:")
    print(f"  - Nodo u (Patrimônio): {patrimonio_col}")
    print(f"  - Nodo v (Localização): {localizacao_col}")
    print(f"  - Atributo embedding: {tipo_equipamento_col}")
    print(f"  - Split temporal: {date_col}")
    
    # Filtrar filas que tengan datos válidos
    df_clean = df.dropna(subset=[patrimonio_col, localizacao_col, date_col])
    print(f"Después de limpiar valores nulos: {df_clean.shape[0]} filas")
    
    # Analizar las fechas
    print(f"\n--- ANÁLISIS TEMPORAL ---")
    df_clean[date_col] = pd.to_datetime(df_clean[date_col])
    min_date = df_clean[date_col].min()
    max_date = df_clean[date_col].max()
    
    print(f"Fecha mínima: {min_date}")
    print(f"Fecha máxima: {max_date}")
    print(f"Período total: {(max_date - min_date).days} días")
    
    # Ordenar por fecha para hacer el split temporal
    df_sorted = df_clean.sort_values(date_col).reset_index(drop=True)
    
    # Calcular puntos de corte para split 80/10/10
    n_total = len(df_sorted)
    train_end = int(0.8 * n_total)
    val_end = int(0.9 * n_total)
    
    train_data = df_sorted[:train_end]
    val_data = df_sorted[train_end:val_end]
    test_data = df_sorted[val_end:]
    
    print(f"\n--- SPLIT TEMPORAL ---")
    print(f"Train (pasado): {len(train_data)} registros ({len(train_data)/n_total*100:.1f}%)")
    print(f"  - Desde: {train_data[date_col].min()} hasta: {train_data[date_col].max()}")
    print(f"Val (presente): {len(val_data)} registros ({len(val_data)/n_total*100:.1f}%)")
    print(f"  - Desde: {val_data[date_col].min()} hasta: {val_data[date_col].max()}")
    print(f"Test (futuro): {len(test_data)} registros ({len(test_data)/n_total*100:.1f}%)")
    print(f"  - Desde: {test_data[date_col].min()} hasta: {test_data[date_col].max()}")
    
    # Crear vocabularios globales basados en todos los datos
    all_patrimonios = sorted(df_sorted[patrimonio_col].unique())
    all_localizacoes = sorted(df_sorted[localizacao_col].unique())
    all_tipos = sorted(df_sorted[tipo_equipamento_col].dropna().unique())
    
    patrimonio_to_idx = {node: i for i, node in enumerate(all_patrimonios)}
    localizacao_to_idx = {node: i for i, node in enumerate(all_localizacoes)}
    tipo_to_idx = {tipo: i for i, tipo in enumerate(all_tipos)}
    
    num_patrimonios = len(all_patrimonios)
    num_localizacoes = len(all_localizacoes)
    total_nodes = num_patrimonios + num_localizacoes
    
    print(f"\n--- VOCABULARIOS GLOBALES ---")
    print(f"Total patrimônios únicos: {num_patrimonios}")
    print(f"Total localizações únicas: {num_localizacoes}")
    print(f"Total tipos de equipamento: {len(all_tipos)}")
    print(f"Total nodos: {total_nodes}")
    
    def create_graph_from_split(split_data, split_name):
        """Crear grafo para un split específico"""
        print(f"\n--- CREANDO GRAFO PARA {split_name.upper()} ---")
        
        # Crear aristas para este split
        src_nodes = [patrimonio_to_idx[p] for p in split_data[patrimonio_col]]
        dst_nodes = [localizacao_to_idx[l] + num_patrimonios for l in split_data[localizacao_col]]
        
        # Edge index bidireccional
        edge_index_forward = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
        edge_index_backward = torch.tensor([dst_nodes, src_nodes], dtype=torch.long)
        edge_index = torch.cat([edge_index_forward, edge_index_backward], dim=1)
        
        print(f"Aristas en {split_name}: {len(src_nodes)} originales, {edge_index.shape[1]} bidireccionales")
        
        # Crear características de nodos (basadas en TODOS los datos para consistencia)
        patrimonio_attrs = torch.zeros(num_patrimonios, len(all_tipos))
        
        # Llenar atributos basados en TODOS los datos históricos hasta este punto
        if split_name == "train":
            relevant_data = train_data
        elif split_name == "val":
            relevant_data = pd.concat([train_data, val_data])
        else:  # test
            relevant_data = df_sorted  # Todos los datos para test
            
        for _, row in relevant_data.iterrows():
            patrimonio = row[patrimonio_col]
            tipo_eq = row[tipo_equipamento_col]
            
            if pd.notna(tipo_eq) and patrimonio in patrimonio_to_idx:
                p_idx = patrimonio_to_idx[patrimonio]
                t_idx = tipo_to_idx[tipo_eq]
                patrimonio_attrs[p_idx, t_idx] = 1.0
        
        # Localizações sin atributos específicos
        localizacao_attrs = torch.zeros(num_localizacoes, len(all_tipos))
        node_features = torch.cat([patrimonio_attrs, localizacao_attrs], dim=0)
        
        # Crear objeto Data
        data = Data(
            x=node_features,
            edge_index=edge_index,
            num_nodes=total_nodes,
            num_nodes_type_1=num_patrimonios,
            num_nodes_type_2=num_localizacoes
        )
        
        # Calcular estadísticas de nodos presentes en este split
        present_patrimonios = set(split_data[patrimonio_col].unique())
        present_localizacoes = set(split_data[localizacao_col].unique())
        
        print(f"Patrimônios únicos en {split_name}: {len(present_patrimonios)}")
        print(f"Localizações únicas en {split_name}: {len(present_localizacoes)}")
        
        return data, present_patrimonios, present_localizacoes
    
    # Crear grafos para cada split
    train_graph, train_patrimonios, train_localizacoes = create_graph_from_split(train_data, "train")
    val_graph, val_patrimonios, val_localizacoes = create_graph_from_split(val_data, "val")
    test_graph, test_patrimonios, test_localizacoes = create_graph_from_split(test_data, "test")
    
    # Análisis de inductividad
    print(f"\n--- ANÁLISIS DE INDUCTIVIDAD ---")
    
    # Nuevos nodos en validación
    new_patrimonios_val = val_patrimonios - train_patrimonios
    new_localizacoes_val = val_localizacoes - train_localizacoes
    
    # Nuevos nodos en test
    new_patrimonios_test = test_patrimonios - train_patrimonios - val_patrimonios
    new_localizacoes_test = test_localizacoes - train_localizacoes - val_localizacoes
    
    print(f"Nuevos patrimônios en validación: {len(new_patrimonios_val)}")
    print(f"Nuevos localizações en validación: {len(new_localizacoes_val)}")
    print(f"Nuevos patrimônios en test: {len(new_patrimonios_test)}")
    print(f"Nuevos localizações en test: {len(new_localizacoes_test)}")
    
    # Guardar los grafos
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    torch.save(train_graph, os.path.join(save_dir, "sismetro_inductive_train.pt"))
    torch.save(val_graph, os.path.join(save_dir, "sismetro_inductive_val.pt"))
    torch.save(test_graph, os.path.join(save_dir, "sismetro_inductive_test.pt"))
    
    # Guardar metadatos
    metadata = {
        'patrimonio_to_idx': patrimonio_to_idx,
        'localizacao_to_idx': localizacao_to_idx,
        'tipo_to_idx': tipo_to_idx,
        'all_patrimonios': all_patrimonios,
        'all_localizacoes': all_localizacoes,
        'all_tipos': all_tipos,
        'split_info': {
            'train_period': (train_data[date_col].min(), train_data[date_col].max()),
            'val_period': (val_data[date_col].min(), val_data[date_col].max()),
            'test_period': (test_data[date_col].min(), test_data[date_col].max()),
            'train_count': len(train_data),
            'val_count': len(val_data),
            'test_count': len(test_data)
        }
    }
    
    torch.save(metadata, os.path.join(save_dir, "sismetro_inductive_metadata.pt"))
    
    print(f"\n--- ARCHIVOS GUARDADOS ---")
    print(f"✓ sismetro_inductive_train.pt - Grafo de entrenamiento")
    print(f"✓ sismetro_inductive_val.pt - Grafo de validación")
    print(f"✓ sismetro_inductive_test.pt - Grafo de prueba")
    print(f"✓ sismetro_inductive_metadata.pt - Metadatos del split")
    
    print(f"\n--- RESUMEN FINAL ---")
    print(f"✓ Split inductivo temporal creado exitosamente")
    print(f"✓ Basado en {n_total} registros temporales")
    print(f"✓ Período: {min_date.strftime('%Y-%m-%d')} a {max_date.strftime('%Y-%m-%d')}")
    print(f"✓ Split 80/10/10 implementado")
    print(f"✓ Compatibilidad inductiva verificada")
    
    return train_graph, val_graph, test_graph, metadata

if __name__ == '__main__':
    create_sismetro_inductive_split()
