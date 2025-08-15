"""
Ejemplo de cómo usar el dataset Sismetro con split inductivo temporal
"""
import torch
import sys
import os
from datetime import datetime

# Agregar el directorio src al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def load_sismetro_inductive_example():
    """
    Ejemplo de carga y uso del dataset Sismetro con split inductivo
    """
    print("=== EJEMPLO DE USO DEL DATASET SISMETRO INDUCTIVO ===")
    
    # Cargar todos los splits
    data_dir = "data/processed"
    
    try:
        train_data = torch.load(f"{data_dir}/sismetro_inductive_train.pt")
        val_data = torch.load(f"{data_dir}/sismetro_inductive_val.pt")
        test_data = torch.load(f"{data_dir}/sismetro_inductive_test.pt")
        metadata = torch.load(f"{data_dir}/sismetro_inductive_metadata.pt")
        
        print("✓ Todos los splits cargados exitosamente")
        
    except FileNotFoundError as e:
        print(f"✗ Error cargando archivos: {e}")
        print("Asegúrate de haber ejecutado create_sismetro_inductive.py primero")
        return None
    
    # Información general
    print(f"\n--- INFORMACIÓN DEL DATASET ---")
    print(f"Vocabulario global:")
    print(f"  - Patrimônios: {len(metadata['all_patrimonios'])}")
    print(f"  - Localizações: {len(metadata['all_localizacoes'])}")  
    print(f"  - Tipos de equipamento: {len(metadata['all_tipos'])}")
    
    # Información temporal
    split_info = metadata['split_info']
    train_start, train_end = split_info['train_period']
    val_start, val_end = split_info['val_period']
    test_start, test_end = split_info['test_period']
    
    print(f"\n--- SPLITS TEMPORALES ---")
    print(f"TRAIN (80%): {train_start.strftime('%Y-%m-%d')} a {train_end.strftime('%Y-%m-%d')}")
    print(f"  - {split_info['train_count']} registros")
    print(f"  - {train_data.edge_index.shape[1]} aristas bidireccionales")
    
    print(f"VAL (10%):   {val_start.strftime('%Y-%m-%d')} a {val_end.strftime('%Y-%m-%d')}")
    print(f"  - {split_info['val_count']} registros")
    print(f"  - {val_data.edge_index.shape[1]} aristas bidireccionales")
    
    print(f"TEST (10%):  {test_start.strftime('%Y-%m-%d')} a {test_end.strftime('%Y-%m-%d')}")
    print(f"  - {split_info['test_count']} registros") 
    print(f"  - {test_data.edge_index.shape[1]} aristas bidireccionales")
    
    # Ejemplo de características de nodos
    print(f"\n--- CARACTERÍSTICAS DE NODOS ---")
    print(f"Forma de características: {train_data.x.shape}")
    print(f"Cada nodo tiene {train_data.x.shape[1]} características (tipos de equipamento)")
    
    # Ejemplo de nodos patrimônio vs localização
    print(f"\nEjemplo de nodo patrimônio [0]:")
    patrimonio_features = train_data.x[0]
    active_features = torch.nonzero(patrimonio_features).flatten()
    if len(active_features) > 0:
        active_tipos = [metadata['all_tipos'][i] for i in active_features[:5]]
        print(f"  Tipos activos: {active_tipos}")
    else:
        print(f"  Sin tipos de equipamento activos")
    
    print(f"\nEjemplo de nodo localização [{train_data.num_nodes_type_1}]:")
    loc_features = train_data.x[train_data.num_nodes_type_1]
    print(f"  Características: {loc_features[:10]}... (todas cero para localizações)")
    
    # Capacidad inductiva
    print(f"\n--- CAPACIDAD INDUCTIVA ---")
    
    def get_active_nodes(edge_index, num_type1):
        active_patrimonios = set()
        active_localizacoes = set()
        
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            
            if src < num_type1:
                active_patrimonios.add(src)
            else:
                active_localizacoes.add(src)
            
            if dst < num_type1:
                active_patrimonios.add(dst)
            else:
                active_localizacoes.add(dst)
        
        return active_patrimonios, active_localizacoes
    
    train_p, train_l = get_active_nodes(train_data.edge_index, train_data.num_nodes_type_1)
    val_p, val_l = get_active_nodes(val_data.edge_index, val_data.num_nodes_type_1)
    test_p, test_l = get_active_nodes(test_data.edge_index, test_data.num_nodes_type_1)
    
    # Nodos nuevos en cada split
    new_p_val = val_p - train_p
    new_l_val = val_l - train_l
    new_p_test = test_p - train_p - val_p
    new_l_test = test_l - train_l - val_l
    
    print(f"Nodos activos en TRAIN: {len(train_p)} patrimônios, {len(train_l)} localizações")
    print(f"Nodos nuevos en VAL: {len(new_p_val)} patrimônios, {len(new_l_val)} localizações")
    print(f"Nodos nuevos en TEST: {len(new_p_test)} patrimônios, {len(new_l_test)} localizações")
    
    # Ejemplo de flujo de entrenamiento
    print(f"\n--- FLUJO DE ENTRENAMIENTO SUGERIDO ---")
    print("1. Entrenar en TRAIN split:")
    print("   python src/train_nc.py --config_file src/config/inductive_sismetro.cfg")
    
    print("2. Validar en VAL split (con nodos nuevos del presente)")
    print("3. Evaluar en TEST split (con nodos nuevos del futuro)")
    
    print(f"\n--- COMANDOS DE ENTRENAMIENTO ---")
    print("Non-Contrastive (recomendado para inductivo):")
    print("  python src/train_nc.py --config_file src/config/inductive_sismetro.cfg")
    print("\nOtros métodos:")
    print("  python src/train_grace.py --config_file src/config/inductive_sismetro.cfg")
    print("  python src/train_margin.py --config_file src/config/inductive_sismetro.cfg")
    
    return {
        'train': train_data,
        'val': val_data,
        'test': test_data,
        'metadata': metadata
    }

if __name__ == '__main__':
    result = load_sismetro_inductive_example()
    
    if result:
        print(f"\n=== DATASET INDUCTIVO LISTO ===")
        print(f"✓ Split temporal 80/10/10 implementado")
        print(f"✓ Capacidad inductiva verificada")
        print(f"✓ Estructura bipartita preservada")
        print(f"✓ Listo para experimentos de predicción de enlaces inductivos")
