"""
Ejemplo de cómo cargar y usar el dataset Sismetro
"""
import torch
import sys
import os

# Agregar el directorio src al path para importar las librerías
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def load_sismetro_example():
    """
    Ejemplo de carga y uso básico del dataset Sismetro
    """
    print("=== EJEMPLO DE USO DEL DATASET SISMETRO ===")
    
    # Cargar el dataset
    data_path = "data/processed/sismetro.pt"
    data = torch.load(data_path)
    
    print(f"Dataset cargado: {data}")
    print(f"Forma de características: {data.x.shape}")
    print(f"Forma de aristas: {data.edge_index.shape}")
    
    # Información del grafo bipartito
    print(f"\n--- ESTRUCTURA DEL GRAFO BIPARTITO ---")
    print(f"Nodos Patrimônio (tipo 1): índices 0 a {data.num_nodes_type_1-1}")
    print(f"Nodos Localização (tipo 2): índices {data.num_nodes_type_1} a {data.num_nodes-1}")
    
    # Ejemplo de cómo separar los nodos por tipo
    patrimonio_nodes = torch.arange(data.num_nodes_type_1)
    localizacao_nodes = torch.arange(data.num_nodes_type_1, data.num_nodes)
    
    print(f"Primeros 5 nodos Patrimônio: {patrimonio_nodes[:5]}")
    print(f"Primeros 5 nodos Localização: {localizacao_nodes[:5]}")
    
    # Ejemplo de características
    print(f"\n--- CARACTERÍSTICAS DE LOS NODOS ---")
    patrimonio_features = data.x[:data.num_nodes_type_1]  # Características de patrimônios
    localizacao_features = data.x[data.num_nodes_type_1:]  # Características de localizações
    
    print(f"Características Patrimônio [0]: {patrimonio_features[0][:10]}... (163 tipos de equipamento)")
    print(f"Características Localização [0]: {localizacao_features[0][:10]}... (sin atributos específicos)")
    
    # Ejemplo de aristas
    print(f"\n--- ESTRUCTURA DE ARISTAS ---")
    print(f"Primeras 5 aristas:")
    for i in range(5):
        src, dst = data.edge_index[0, i].item(), data.edge_index[1, i].item()
        src_type = "Patrimônio" if src < data.num_nodes_type_1 else "Localização"
        dst_type = "Patrimônio" if dst < data.num_nodes_type_1 else "Localização"
        print(f"  Arista {i}: {src} ({src_type}) -> {dst} ({dst_type})")
    
    print(f"\n--- ESTADÍSTICAS RÁPIDAS ---")
    print(f"Densidad del grafo: {data.edge_index.shape[1] / (data.num_nodes * (data.num_nodes - 1)):.6f}")
    print(f"Grado promedio: {data.edge_index.shape[1] / data.num_nodes:.2f}")
    
    return data

def example_training_setup():
    """
    Ejemplo de cómo configurar el entrenamiento
    """
    print(f"\n=== EJEMPLO DE CONFIGURACIÓN DE ENTRENAMIENTO ===")
    print("Para entrenar con el dataset Sismetro, usar:")
    print("python src/train_nc.py --config_file src/config/transductive_sismetro.cfg")
    print("\nO para otros métodos:")
    print("python src/train_grace.py --config_file src/config/transductive_sismetro.cfg")
    print("python src/train_margin.py --config_file src/config/transductive_sismetro.cfg")

if __name__ == '__main__':
    data = load_sismetro_example()
    example_training_setup()
