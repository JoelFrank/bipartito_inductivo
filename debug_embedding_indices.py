import torch
import pandas as pd

print("=== DEBUG DE ÍNDICES DE EMBEDDINGS ===\n")

# Cargar dataset principal
print("1. Cargando dataset principal...")
data_main = torch.load('data/processed/sismetro.pt')
print(f"Dataset principal - Shape x: {data_main.x.shape}")
print(f"Dataset principal - num_embedding_types: {data_main.num_embedding_types}")
print(f"Dataset principal - Min x: {data_main.x.min().item()}")
print(f"Dataset principal - Max x: {data_main.x.max().item()}")
print()

# Cargar datasets inductivos
splits = ['train', 'val', 'test']
for split in splits:
    print(f"2. Cargando dataset inductivo {split}...")
    try:
        data = torch.load(f'data/processed/sismetro_inductive_{split}.pt')
        print(f"  Shape x: {data.x.shape}")
        print(f"  num_embedding_types: {data.num_embedding_types}")
        print(f"  Min x: {data.x.min().item()}")
        print(f"  Max x: {data.x.max().item()}")
        print(f"  num_nodes_type_1: {data.num_nodes_type_1}")
        print(f"  num_nodes_type_2: {data.num_nodes_type_2}")
        
        # Verificar si hay índices fuera de rango
        if data.x.max().item() >= data.num_embedding_types:
            print(f"  ❌ ERROR: Hay índices fuera de rango!")
            print(f"     Max índice: {data.x.max().item()}, pero num_embedding_types: {data.num_embedding_types}")
            
            # Encontrar índices problemáticos
            invalid_mask = data.x.squeeze() >= data.num_embedding_types
            invalid_indices = torch.where(invalid_mask)[0]
            print(f"     Nodos con índices inválidos: {len(invalid_indices)}")
            if len(invalid_indices) > 0:
                print(f"     Primeros 10 nodos problemáticos: {invalid_indices[:10].tolist()}")
                print(f"     Sus valores: {data.x[invalid_indices[:10]].squeeze().tolist()}")
                
                # Ver si son nodos de patrimonio o localización
                patrimonio_mask = invalid_indices < data.num_nodes_type_1
                localizacao_mask = invalid_indices >= data.num_nodes_type_1
                print(f"     Nodos patrimonio problemáticos: {patrimonio_mask.sum().item()}")
                print(f"     Nodos localização problemáticos: {localizacao_mask.sum().item()}")
        else:
            print(f"  ✅ Todos los índices están en rango válido")
        print()
    except Exception as e:
        print(f"  Error cargando {split}: {e}")
        print()

# Verificar el archivo original Excel para contar tipos reales
print("3. Verificando archivo Excel original...")
try:
    df = pd.read_excel('data/raw/sismetro/SISMETRO-Exportacao-SS-2021-2024_P1(OK PATRIMONIO).xlsx')
    tipos_unicos = df['TIPO DO EQUIPAMENTO'].unique()
    tipos_sin_na = df['TIPO DO EQUIPAMENTO'].dropna().unique()
    print(f"  Tipos únicos en Excel (con NaN): {len(tipos_unicos)}")
    print(f"  Tipos únicos en Excel (sin NaN): {len(tipos_sin_na)}")
    print(f"  Primeros 5 tipos: {tipos_sin_na[:5]}")
    print()
except Exception as e:
    print(f"  Error cargando Excel: {e}")
    print()
