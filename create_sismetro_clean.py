"""
Crear dataset Sismetro CORRECTO para split inductivo basado en nodos.
Este script deduplica las relaciones y crea un grafo limpio.
"""
import pandas as pd
import torch
from torch_geometric.data import Data
import os
import numpy as np

def create_clean_sismetro_for_inductive():
    """
    Crear dataset Sismetro limpio y correcto para split inductivo basado en nodos.
    """
    print("=== CREANDO DATASET SISMETRO LIMPIO PARA SPLIT INDUCTIVO ===")
    
    # Rutas
    excel_path = "data/raw/sismetro/SISMETRO-Exportacao-SS-2021-2024_P1(OK PATRIMONIO).xlsx"
    save_path = "data/processed/sismetro_clean.pt"
    
    # 1. Cargar datos originales
    print(f"Cargando datos desde: {excel_path}")
    df = pd.read_excel(excel_path, sheet_name=0)
    print(f"Dataset original: {df.shape[0]} filas, {df.shape[1]} columnas")
    
    # 2. Definir columnas
    patrimonio_col = 'OBSERVAÇÃO PATRIMÔNIO'
    localizacao_col = 'LOCALIZAÇÃO'
    tipo_equipamento_col = 'TIPO DO EQUIPAMENTO'
    
    # 3. Limpiar datos básicos
    df_clean = df.dropna(subset=[patrimonio_col, localizacao_col])
    print(f"Después de limpiar nulos: {df_clean.shape[0]} filas")
    
    # 4. DEDUPLICAR RELACIONES (CLAVE PARA SPLIT INDUCTIVO)
    print("\n=== DEDUPLICANDO RELACIONES ===")
    print("Esta es la clave para tener un dataset correcto para split inductivo")
    
    # Mantener la primera ocurrencia de cada relación patrimônio-localização
    relations_df = df_clean[[patrimonio_col, localizacao_col, tipo_equipamento_col]].drop_duplicates(
        subset=[patrimonio_col, localizacao_col], keep='first'
    )
    
    print(f"Relaciones antes de deduplicar: {len(df_clean)}")
    print(f"Relaciones después de deduplicar: {len(relations_df)}")
    print(f"Duplicados eliminados: {len(df_clean) - len(relations_df)}")
    
    # 5. Crear vocabularios
    print(f"\n=== CREANDO VOCABULARIOS ===")
    unique_patrimonios = sorted(relations_df[patrimonio_col].unique())
    unique_localizacoes = sorted(relations_df[localizacao_col].unique())
    unique_tipos = sorted(relations_df[tipo_equipamento_col].dropna().unique())
    
    patrimonio_to_idx = {node: i for i, node in enumerate(unique_patrimonios)}
    localizacao_to_idx = {node: i for i, node in enumerate(unique_localizacoes)}
    tipo_to_idx = {tipo: i for i, tipo in enumerate(unique_tipos)}
    
    num_patrimonios = len(unique_patrimonios)
    num_localizacoes = len(unique_localizacoes)
    total_nodes = num_patrimonios + num_localizacoes
    
    print(f"Patrimônios únicos: {num_patrimonios}")
    print(f"Localizações únicas: {num_localizacoes}")
    print(f"Tipos de equipamento únicos: {len(unique_tipos)}")
    print(f"Total nodos: {total_nodes}")
    
    # 6. Crear aristas LIMPIAS
    print(f"\n=== CREANDO ARISTAS LIMPIAS ===")
    src_nodes = [patrimonio_to_idx[p] for p in relations_df[patrimonio_col]]
    dst_nodes = [localizacao_to_idx[l] + num_patrimonios for l in relations_df[localizacao_col]]
    
    # Edge index bidireccional
    edge_index_forward = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
    edge_index_backward = torch.tensor([dst_nodes, src_nodes], dtype=torch.long)
    edge_index = torch.cat([edge_index_forward, edge_index_backward], dim=1)
    
    print(f"Relaciones únicas: {len(src_nodes)}")
    print(f"Aristas bidireccionales: {edge_index.shape[1]}")
    
    # 7. Crear características para AMBOS tipos de nodos (con ID NEUTRO para localizações)
    print(f"\n=== CREANDO CARACTERÍSTICAS PARA AMBOS TIPOS DE NODOS ===")
    print("Patrimônios: IDs de tipo de equipamento (informativo)")
    print("Localizações: ID NEUTRO (no influye en predicciones)")
    
    # ID NEUTRO para localizações (que no influya en el aprendizaje)
    ID_NEUTRO = len(unique_tipos)  # Siguiente ID disponible después de tipos reales
    total_embedding_ids = len(unique_tipos) + 1  # Tipos + neutro
    
    print(f"📊 CONFIGURACIÓN DE EMBEDDINGS:")
    print(f"  Tipos de equipamento (patrimônios): IDs 0-{len(unique_tipos)-1}")
    print(f"  ID NEUTRO (localizações): ID {ID_NEUTRO}")
    print(f"  Total vocabulary size: {total_embedding_ids}")
    print(f"  ✅ El embedding {ID_NEUTRO} aprenderá a ser NEUTRAL/NO INFORMATIVO")
    
    # Crear características para TODOS los nodos
    node_features = torch.full((total_nodes, 1), ID_NEUTRO, dtype=torch.long)  # Inicializar con ID_NEUTRO
    
    # Patrimônios: asignar IDs reales de tipo de equipamento
    for _, row in relations_df.iterrows():
        patrimonio = row[patrimonio_col]
        tipo_eq = row[tipo_equipamento_col]
        
        if pd.notna(tipo_eq) and patrimonio in patrimonio_to_idx:
            p_idx = patrimonio_to_idx[patrimonio]
            t_idx = tipo_to_idx[tipo_eq]
            node_features[p_idx, 0] = t_idx  # ID real del tipo
    
    # Localizações: mantener ID_NEUTRO (no influye)
    # node_features[num_patrimonios:] ya tiene ID_NEUTRO
    
    print(f"✅ Características creadas para TODOS los nodos: {node_features.shape}")
    print(f"  Patrimônios (0-{num_patrimonios-1}): tipos informativos 0-{len(unique_tipos)-1}")
    print(f"  Localizações ({num_patrimonios}-{total_nodes-1}): ID NEUTRO {ID_NEUTRO}")
    print(f"  Rango total de IDs: {node_features.min().item()} - {node_features.max().item()}")
    print(f"  🎯 El modelo aprenderá que ID {ID_NEUTRO} debe ser NEUTRAL")
    
    # 8. Crear objeto Data
    data = Data(
        x=node_features,  # Solo características para patrimônios
        edge_index=edge_index,
        num_nodes=total_nodes,
        num_nodes_type_1=num_patrimonios,
        num_nodes_type_2=num_localizacoes,
        # Metadatos para referencia
        patrimonio_to_idx=patrimonio_to_idx,
        localizacao_to_idx=localizacao_to_idx,
        tipo_to_idx=tipo_to_idx,
        unique_patrimonios=unique_patrimonios,
        unique_localizacoes=unique_localizacoes,
        unique_tipos=unique_tipos,
        # Información de embeddings
        total_embedding_ids=total_embedding_ids,
        ID_NEUTRO=ID_NEUTRO
    )
    
    # 9. Guardar dataset limpio
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(data, save_path)
    
    print(f"\n=== DATASET LIMPIO GUARDADO ===")
    print(f"Archivo: {save_path}")
    print(f"Grafo final: {data}")
    
    # 10. Verificación
    print(f"\n=== VERIFICACIÓN FINAL ===")
    print(f"✓ Nodos totales: {data.num_nodes}")
    print(f"✓ Patrimônios: {data.num_nodes_type_1}")
    print(f"✓ Localizações: {data.num_nodes_type_2}")
    print(f"✓ Aristas: {data.edge_index.size(1)}")
    print(f"✓ Relaciones únicas: {data.edge_index.size(1) // 2}")
    print(f"✓ Características: {data.x.shape}")
    print(f"✓ Tipos de equipamento: {len(unique_tipos) + 1}")  # +1 para ID_NEUTRO
    
    print(f"\n🎯 DATASET LISTO PARA SPLIT INDUCTIVO CON CARACTERÍSTICAS SIMÉTRICAS")
    print(f"   ✅ Patrimônios: embeddings informativos")
    print(f"   ✅ Localizações: embedding NEUTRO (no influye)")
    
    return data

if __name__ == '__main__':
    create_clean_sismetro_for_inductive()
