"""
Verificar si el dataset sismetro.pt contiene el grafo COMPLETO original
o es una versión ya procesada/cortada.
"""
import pandas as pd
import torch

def verify_sismetro_completeness():
    print("=== VERIFICANDO SI SISMETRO.PT ES EL GRAFO COMPLETO ===")
    
    # 1. Cargar dataset procesado
    data_path = 'data/processed/sismetro.pt'
    data = torch.load(data_path)
    
    print(f"Dataset procesado:")
    print(f"  Nodos: {data.num_nodes}")
    print(f"  Aristas: {data.edge_index.size(1)}")
    print(f"  Aristas únicas (bidireccional): {data.edge_index.size(1) // 2}")
    
    # 2. Cargar datos originales de Excel
    excel_path = "data/raw/sismetro/SISMETRO-Exportacao-SS-2021-2024_P1(OK PATRIMONIO).xlsx"
    
    try:
        df = pd.read_excel(excel_path, sheet_name=0)
        print(f"\nDatos originales de Excel:")
        print(f"  Filas totales: {df.shape[0]}")
        
        # Limpiar datos
        patrimonio_col = 'OBSERVAÇÃO PATRIMÔNIO'
        localizacao_col = 'LOCALIZAÇÃO'
        
        df_clean = df.dropna(subset=[patrimonio_col, localizacao_col])
        print(f"  Filas después de limpiar: {df_clean.shape[0]}")
        
        # Crear relaciones únicas
        unique_relations = df_clean[[patrimonio_col, localizacao_col]].drop_duplicates()
        print(f"  Relaciones únicas: {len(unique_relations)}")
        
        # Comparar
        expected_edges_bidirectional = len(unique_relations) * 2
        actual_edges = data.edge_index.size(1)
        
        print(f"\n=== COMPARACIÓN ===")
        print(f"Esperado (Excel): {expected_edges_bidirectional} aristas bidireccionales")
        print(f"Actual (sismetro.pt): {actual_edges} aristas")
        
        if actual_edges == expected_edges_bidirectional:
            print("✅ EL DATASET CONTIENE EL GRAFO COMPLETO")
            return True
        else:
            print("❌ EL DATASET PARECE ESTAR CORTADO/PROCESADO")
            print(f"   Diferencia: {expected_edges_bidirectional - actual_edges} aristas faltantes")
            return False
            
    except Exception as e:
        print(f"❌ Error al acceder al Excel original: {e}")
        print("⚠️ No se puede verificar completitud")
        return None

if __name__ == '__main__':
    verify_sismetro_completeness()
