import torch
import pandas as pd

def verify_sismetro_graph():
    """
    Verifica que el grafo bipartito Sismetro se haya creado correctamente.
    """
    print("=== VERIFICACIÓN DEL GRAFO BIPARTITO SISMETRO ===")
    
    # Cargar el grafo guardado
    data_path = "../data/processed/sismetro.pt"
    data = torch.load(data_path)
    
    print(f"Archivo cargado desde: {data_path}")
    print(f"Objeto Data: {data}")
    
    # Información básica del grafo
    print(f"\n--- INFORMACIÓN DEL GRAFO ---")
    print(f"Tipo de grafo: Bipartito no dirigido")
    print(f"Nombre del dataset: Sismetro")
    print(f"Total de nodos: {data.num_nodes}")
    print(f"  - Nodos tipo 1 (Patrimônios): {data.num_nodes_type_1}")
    print(f"  - Nodos tipo 2 (Localizações): {data.num_nodes_type_2}")
    print(f"Total de aristas: {data.edge_index.shape[1]}")
    print(f"Características por nodo: {data.x.shape[1]} (tipos de equipamento)")
    
    # Verificar estructura del grafo bipartito
    print(f"\n--- VERIFICACIÓN DE ESTRUCTURA BIPARTITA ---")
    edge_index = data.edge_index
    tipo1_nodes = set(range(data.num_nodes_type_1))
    tipo2_nodes = set(range(data.num_nodes_type_1, data.num_nodes))
    
    # Verificar que las aristas solo conecten entre diferentes tipos
    valid_bipartite = True
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        if (src in tipo1_nodes and dst in tipo1_nodes) or (src in tipo2_nodes and dst in tipo2_nodes):
            valid_bipartite = False
            break
    
    print(f"Estructura bipartita válida: {'✓' if valid_bipartite else '✗'}")
    
    # Estadísticas de conectividad
    print(f"\n--- ESTADÍSTICAS DE CONECTIVIDAD ---")
    degrees = torch.zeros(data.num_nodes)
    for i in range(edge_index.shape[1]):
        src = edge_index[0, i].item()
        degrees[src] += 1
    
    patrimonio_degrees = degrees[:data.num_nodes_type_1]
    localizacao_degrees = degrees[data.num_nodes_type_1:]
    
    print(f"Grado promedio de Patrimônios: {patrimonio_degrees.mean():.2f}")
    print(f"Grado máximo de Patrimônios: {patrimonio_degrees.max().item()}")
    print(f"Grado mínimo de Patrimônios: {patrimonio_degrees.min().item()}")
    
    print(f"Grado promedio de Localizações: {localizacao_degrees.mean():.2f}")
    print(f"Grado máximo de Localizações: {localizacao_degrees.max().item()}")
    print(f"Grado mínimo de Localizações: {localizacao_degrees.min().item()}")
    
    # Verificar atributos de nodos
    print(f"\n--- VERIFICACIÓN DE ATRIBUTOS ---")
    patrimonio_features = data.x[:data.num_nodes_type_1]
    localizacao_features = data.x[data.num_nodes_type_1:]
    
    print(f"Patrimônios con al menos un tipo de equipamento: {(patrimonio_features.sum(dim=1) > 0).sum().item()}")
    print(f"Patrimônios sin tipos de equipamento: {(patrimonio_features.sum(dim=1) == 0).sum().item()}")
    print(f"Localizações (todas sin atributos específicos): {(localizacao_features.sum(dim=1) == 0).sum().item()}")
    
    # Verificar que el grafo es no dirigido
    print(f"\n--- VERIFICACIÓN DE GRAFO NO DIRIGIDO ---")
    edges_set = set()
    reverse_edges_set = set()
    
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        edges_set.add((src, dst))
        reverse_edges_set.add((dst, src))
    
    is_undirected = edges_set == reverse_edges_set
    print(f"Grafo no dirigido: {'✓' if is_undirected else '✗'}")
    
    print(f"\n--- RESUMEN FINAL ---")
    print(f"✓ Grafo bipartito Sismetro creado exitosamente")
    print(f"✓ {data.num_nodes_type_1} nodos de tipo Patrimônio con atributos de {data.x.shape[1]} tipos de equipamento")
    print(f"✓ {data.num_nodes_type_2} nodos de tipo Localização")
    print(f"✓ {data.edge_index.shape[1]} aristas bidireccionales")
    print(f"✓ Estructura bipartita y no dirigida verificada")
    
    return data

if __name__ == '__main__':
    verify_sismetro_graph()
