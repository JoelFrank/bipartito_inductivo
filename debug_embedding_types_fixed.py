import pandas as pd
import torch

print("=== INVESTIGACIÓN DE TIPOS DE EMBEDDING ===\n")

# Intentar con el archivo SISMETRO
print("Cargando archivo SISMETRO Excel...")
try:
    df_sismetro = pd.read_excel('data/raw/sismetro/SISMETRO-Exportacao-SS-2021-2024_P1(OK PATRIMONIO).xlsx')
    
    # Usar la columna correcta
    tipos_equipamento = df_sismetro['TIPO DO EQUIPAMENTO'].unique()
    print(f"Tipos de equipamento en archivo SISMETRO: {len(tipos_equipamento)}")
    print(f"Primeros 10: {tipos_equipamento[:10]}")
    print()
    
    # Verificar si hay NaN
    nan_count = df_sismetro['TIPO DO EQUIPAMENTO'].isna().sum()
    print(f"Valores NaN: {nan_count}")
    
    # Mostrar todos los tipos únicos
    print("\nTodos los tipos de equipamento:")
    for i, tipo in enumerate(sorted(tipos_equipamento)):
        if pd.notna(tipo):  # Solo mostrar valores no-NaN
            print(f"{i+1}: '{tipo}'")
    
except Exception as e:
    print(f"Error cargando SISMETRO: {e}")

print()

# Cargar el dataset procesado
try:
    data = torch.load('data/processed/sismetro.pt')
    print(f"Tipos de embedding únicos en dataset: {data.num_embedding_types}")
    print()
    
    # Verificar si hay atributos adicionales
    if hasattr(data, 'embedding_mapping'):
        print("Mapping de embeddings:")
        print(data.embedding_mapping)
    
    if hasattr(data, 'x') and data.x is not None:
        print(f"Shape de características x: {data.x.shape}")
        print(f"Tipo de datos x: {data.x.dtype}")
        
    # Análisis de la diferencia
    print("\n=== ANÁLISIS DE LA DIFERENCIA ===")
    tipos_no_nan = len([t for t in tipos_equipamento if pd.notna(t)])
    print(f"Archivo original: {len(tipos_equipamento)} tipos totales")
    print(f"Archivo original (sin NaN): {tipos_no_nan} tipos")
    print(f"Dataset procesado: {data.num_embedding_types} tipos")
    print(f"Diferencia: {data.num_embedding_types - tipos_no_nan}")
    
    if data.num_embedding_types - tipos_no_nan == 1:
        print("Probable causa: Se agregó un token especial (ej: <UNK> para valores desconocidos)")
        
except Exception as e:
    print(f"Error cargando dataset procesado: {e}")
