import pandas as pd
import torch
import numpy as np

print("=== INVESTIGACIÓN DE TIPOS DE EMBEDDING ===\n")

# Cargar datos originales
df = pd.read_excel('data/raw/sismetro/SISMETRO-Exportacao-SS-2021-2024_P1(OK PATRIMONIO).xlsx')
tipos_equipamento = df['TIPO DE EQUIPAMENTO'].unique()
print(f'Tipos únicos en CSV: {len(tipos_equipamento)}')

# Verificar valores nulos o NaN
nulos = df['TIPO DE EQUIPAMENTO'].isnull().sum()
print(f'Valores nulos: {nulos}')

# Verificar valores vacíos o strings vacías
vacios = (df['TIPO DE EQUIPAMENTO'] == '').sum()
print(f'Valores vacíos: {vacios}')

# Verificar tipos de datos únicos excluyendo NaN
tipos_no_nulos = df['TIPO DE EQUIPAMENTO'].dropna().unique()
print(f'Tipos únicos sin NaN: {len(tipos_no_nulos)}')

# Cargar dataset procesado
data = torch.load('data/processed/sismetro.pt')
print(f'\nNum embedding types en dataset: {data.num_embedding_types}')

# Verificar si existe x_patrimonio
if hasattr(data, 'x_patrimonio'):
    max_index = data.x_patrimonio[:, 0].max().item()
    min_index = data.x_patrimonio[:, 0].min().item()
    unique_indices = torch.unique(data.x_patrimonio[:, 0])
    print(f'Índices de embedding - Min: {min_index}, Max: {max_index}')
    print(f'Número de índices únicos: {len(unique_indices)}')
    print(f'Primeros 10 índices: {unique_indices[:10].tolist()}')
else:
    print('x_patrimonio no encontrado en el dataset')

# Mostrar algunos tipos de equipamento para análisis
print(f'\nPrimeros 20 tipos de equipamento:')
for i, tipo in enumerate(tipos_no_nulos[:20]):
    print(f'{i}: "{tipo}"')

print(f'\nÚltimos 10 tipos de equipamento:')
for i, tipo in enumerate(tipos_no_nulos[-10:], start=len(tipos_no_nulos)-10):
    print(f'{i}: "{tipo}"')
