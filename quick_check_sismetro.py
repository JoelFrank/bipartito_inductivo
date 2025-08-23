"""
Verificación simple del dataset Sismetro para comprobar las características.
"""
import torch

def quick_check():
    print("=== VERIFICACIÓN RÁPIDA DEL DATASET SISMETRO ===")
    
    # Cargar dataset
    data = torch.load('data/processed/sismetro.pt')
    
    print(f"Dataset shape: {data}")
    print(f"Nodos: {data.num_nodes}")
    print(f"Aristas: {data.edge_index.size(1)}")
    
    if hasattr(data, 'x') and data.x is not None:
        print(f"Características: {data.x.shape}")
        print(f"Tipo de datos: {data.x.dtype}")
        
        # Mostrar ejemplos de características
        print(f"\nPrimeros 10 nodos (patrimônios):")
        for i in range(min(10, data.x.size(0))):
            print(f"  Nodo {i}: {data.x[i]}")
        
        print(f"\nÚltimos 10 nodos (localizações):")
        start_idx = max(0, data.x.size(0) - 10)
        for i in range(start_idx, data.x.size(0)):
            print(f"  Nodo {i}: {data.x[i]}")
            
        # Estadísticas
        if data.x.dtype == torch.long:
            unique_vals = torch.unique(data.x)
            print(f"\nValores únicos en características: {unique_vals}")
            print(f"Rango: {unique_vals.min()} - {unique_vals.max()}")
    else:
        print("❌ No hay características de nodos")
    
    # Verificar metadatos bipartitos
    if hasattr(data, 'num_nodes_type_1'):
        print(f"\nTipo 1 (patrimônios): {data.num_nodes_type_1}")
        print(f"Tipo 2 (localizações): {data.num_nodes - data.num_nodes_type_1}")
    
    # Verificar otros atributos
    print(f"\nAtributos disponibles:")
    for attr in dir(data):
        if not attr.startswith('_') and not callable(getattr(data, attr)):
            print(f"  {attr}")

if __name__ == '__main__':
    quick_check()
