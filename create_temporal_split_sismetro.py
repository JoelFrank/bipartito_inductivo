"""
Crear split temporal para SISMETRO siguiendo la metodología correcta:
- Train (80%): Datos más antiguos
- Validation (10%): Datos intermedios  
- Test (10%): Datos más recientes
"""
import pandas as pd
import torch
from torch_geometric.data import Data
import os
import numpy as np
from datetime import datetime

def create_temporal_split_sismetro():
    """
    Crear split temporal correcto para SISMETRO basado en fechas
    """
    print("=== CREANDO SPLIT TEMPORAL PARA SISMETRO ===")
    
    # 1. Cargar datos originales con fechas
    excel_path = "data/raw/sismetro/SISMETRO-Exportacao-SS-2021-2024_P1(OK PATRIMONIO).xlsx"
    print(f"Cargando datos con fechas desde: {excel_path}")
    
    df = pd.read_excel(excel_path, sheet_name=0)
    print(f"Dataset original: {df.shape[0]} filas")
    
    # 2. Identificar columna de fecha
    date_columns = [col for col in df.columns if 'data' in col.lower() or 'date' in col.lower()]
    print(f"Columnas de fecha candidatas: {date_columns}")
    
    if not date_columns:
        print("❌ No se encontraron columnas de fecha. Listando todas las columnas:")
        for i, col in enumerate(df.columns):
            print(f"  {i}: {col}")
        return None
    
    # Usar la primera columna de fecha encontrada
    date_col = date_columns[0]
    print(f"Usando columna de fecha: {date_col}")
    
    # 3. Limpiar y preparar datos
    patrimonio_col = 'OBSERVAÇÃO PATRIMÔNIO'
    localizacao_col = 'LOCALIZAÇÃO'
    tipo_equipamento_col = 'TIPO DO EQUIPAMENTO'
    
    # Limpiar datos básicos
    df_clean = df.dropna(subset=[patrimonio_col, localizacao_col, date_col])
    print(f"Después de limpiar nulos: {df_clean.shape[0]} filas")
    
    # Convertir fecha
    try:
        df_clean[date_col] = pd.to_datetime(df_clean[date_col])
        print(f"✅ Fechas convertidas exitosamente")
        print(f"Rango de fechas: {df_clean[date_col].min()} - {df_clean[date_col].max()}")
    except Exception as e:
        print(f"❌ Error convirtiendo fechas: {e}")
        return None
    
    # 4. Ordenar por fecha y deduplicar
    df_clean = df_clean.sort_values(date_col)
    
    # Deduplicar manteniendo la primera ocurrencia (más antigua)
    relations_df = df_clean.drop_duplicates(
        subset=[patrimonio_col, localizacao_col], keep='first'
    )
    
    print(f"Relaciones después de deduplicar: {len(relations_df)}")
    print(f"Rango temporal final: {relations_df[date_col].min()} - {relations_df[date_col].max()}")
    
    # 5. División temporal
    print(f"\n=== DIVISIÓN TEMPORAL ===")
    total_relations = len(relations_df)
    
    # Calcular índices de corte
    train_end_idx = int(total_relations * 0.8)
    val_end_idx = int(total_relations * 0.9)
    
    # Dividir datos
    train_df = relations_df.iloc[:train_end_idx]
    val_df = relations_df.iloc[train_end_idx:val_end_idx]
    test_df = relations_df.iloc[val_end_idx:]
    
    print(f"📅 DIVISIÓN CRONOLÓGICA:")
    print(f"  Train: {len(train_df)} relaciones ({train_df[date_col].min()} - {train_df[date_col].max()})")
    print(f"  Val:   {len(val_df)} relaciones ({val_df[date_col].min()} - {val_df[date_col].max()})")
    print(f"  Test:  {len(test_df)} relaciones ({test_df[date_col].min()} - {test_df[date_col].max()})")
    
    # 6. Crear vocabularios globales (para consistencia)
    print(f"\n=== CREANDO VOCABULARIOS GLOBALES ===")
    unique_patrimonios = sorted(relations_df[patrimonio_col].unique())
    unique_localizacoes = sorted(relations_df[localizacao_col].unique())
    unique_tipos = sorted(relations_df[tipo_equipamento_col].dropna().unique())
    
    patrimonio_to_idx = {node: i for i, node in enumerate(unique_patrimonios)}
    localizacao_to_idx = {node: i for i, node in enumerate(unique_localizacoes)}
    tipo_to_idx = {tipo: i for i, tipo in enumerate(unique_tipos)}
    
    num_patrimonios = len(unique_patrimonios)
    num_localizacoes = len(unique_localizacoes)
    total_nodes = num_patrimonios + num_localizacoes
    
    # ID NEUTRO para localizações
    ID_NEUTRO = len(unique_tipos)
    total_embedding_ids = len(unique_tipos) + 1
    
    print(f"Patrimônios únicos: {num_patrimonios}")
    print(f"Localizações únicas: {num_localizacoes}")
    print(f"Tipos de equipamento: {len(unique_tipos)}")
    print(f"Total nodos: {total_nodes}")
    print(f"ID NEUTRO: {ID_NEUTRO}")
    
    # 7. Función para crear dataset
    def create_dataset_split(split_df, split_name):
        print(f"\n--- Creando {split_name} dataset ---")
        
        # Crear aristas para este split
        src_nodes = [patrimonio_to_idx[p] for p in split_df[patrimonio_col]]
        dst_nodes = [localizacao_to_idx[l] + num_patrimonios for l in split_df[localizacao_col]]
        
        # Edge index bidireccional
        edge_index_forward = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
        edge_index_backward = torch.tensor([dst_nodes, src_nodes], dtype=torch.long)
        edge_index = torch.cat([edge_index_forward, edge_index_backward], dim=1)
        
        # Características para TODOS los nodos (simetría)
        node_features = torch.full((total_nodes, 1), ID_NEUTRO, dtype=torch.long)
        
        # Patrimônios: IDs reales de tipo
        for _, row in split_df.iterrows():
            patrimonio = row[patrimonio_col]
            tipo_eq = row[tipo_equipamento_col]
            
            if pd.notna(tipo_eq) and patrimonio in patrimonio_to_idx:
                p_idx = patrimonio_to_idx[patrimonio]
                t_idx = tipo_to_idx[tipo_eq]
                node_features[p_idx, 0] = t_idx
        
        # Crear objeto Data
        data = Data(
            x=node_features,
            edge_index=edge_index,
            num_nodes=total_nodes,
            num_nodes_type_1=num_patrimonios,
            num_nodes_type_2=num_localizacoes,
            # Metadatos
            patrimonio_to_idx=patrimonio_to_idx,
            localizacao_to_idx=localizacao_to_idx,
            tipo_to_idx=tipo_to_idx,
            total_embedding_ids=total_embedding_ids,
            ID_NEUTRO=ID_NEUTRO
        )
        
        print(f"  {split_name}: {data.num_nodes} nodos, {data.edge_index.shape[1]} aristas")
        print(f"  Relaciones únicas: {len(split_df)}")
        
        return data
    
    # 8. Crear datasets para cada split
    train_data = create_dataset_split(train_df, "Train")
    val_data = create_dataset_split(val_df, "Validation")
    test_data = create_dataset_split(test_df, "Test")
    
    # 9. Guardar datasets
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = f"{output_dir}/sismetro_clean_inductive_train.pt"
    val_path = f"{output_dir}/sismetro_clean_inductive_val.pt"
    test_path = f"{output_dir}/sismetro_clean_inductive_test.pt"
    
    torch.save(train_data, train_path)
    torch.save(val_data, val_path)
    torch.save(test_data, test_path)
    
    print(f"\n=== DATASETS TEMPORALES GUARDADOS ===")
    print(f"✅ Train: {train_path}")
    print(f"✅ Validation: {val_path}")
    print(f"✅ Test: {test_path}")
    
    # 10. Verificación final
    print(f"\n=== VERIFICACIÓN FINAL ===")
    print(f"📊 METODOLOGÍA TEMPORAL CORRECTA:")
    print(f"  ✅ División cronológica: más antiguo → más reciente")
    print(f"  ✅ Train (80%): {len(train_df)} relaciones")
    print(f"  ✅ Val (10%): {len(val_df)} relaciones")  
    print(f"  ✅ Test (10%): {len(test_df)} relaciones")
    print(f"  ✅ Encoder verá solo: datos de entrenamiento")
    print(f"  ✅ Decoder se evalúa en: datos futuros (test)")
    
    return train_data, val_data, test_data

if __name__ == '__main__':
    create_temporal_split_sismetro()
