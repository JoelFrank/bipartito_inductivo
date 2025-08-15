import pandas as pd
import torch
from torch_geometric.data import Data
import os
import argparse

def create_bipartite_graph_from_csv(csv_path, save_dir, dataset_name):
    """
    Crea y guarda un grafo bipartito a partir de un archivo CSV.
    El CSV debe contener las interacciones entre dos tipos de nodos.
    """
    print(f"--- Iniciando la creación del grafo para '{dataset_name}' ---")
    
    print(f"Cargando datos desde: {csv_path}")
    df = pd.read_csv(csv_path)
    
    if len(df.columns) < 2:
        raise ValueError("El CSV debe tener al menos dos columnas.")

    node_type_1_col, node_type_2_col = df.columns[0], df.columns[1]
    print(f"Nodos tipo 1: '{node_type_1_col}', Nodos tipo 2: '{node_type_2_col}'")

    unique_nodes_1 = sorted(df[node_type_1_col].unique())
    unique_nodes_2 = sorted(df[node_type_2_col].unique())
    
    mapping_1 = {node_id: i for i, node_id in enumerate(unique_nodes_1)}
    mapping_2 = {node_id: i for i, node_id in enumerate(unique_nodes_2)}
    
    num_nodes_1 = len(unique_nodes_1)
    num_nodes_2 = len(unique_nodes_2)
    
    print(f"Nodos únicos tipo 1: {num_nodes_1}")
    print(f"Nodos únicos tipo 2: {num_nodes_2}")
    print(f"Total de nodos: {num_nodes_1 + num_nodes_2}")

    src = [mapping_1[node] for node in df[node_type_1_col]]
    dst = [mapping_2[node] + num_nodes_1 for node in df[node_type_2_col]]
    
    edge_index_forward = torch.tensor([src, dst], dtype=torch.long)
    edge_index_backward = torch.tensor([dst, src], dtype=torch.long)
    edge_index = torch.cat([edge_index_forward, edge_index_backward], dim=1)
    
    print(f"Total de interacciones: {df.shape[0]}")
    print(f"Tamaño de edge_index: {edge_index.shape}")

    data = Data(
        edge_index=edge_index,
        num_nodes=num_nodes_1 + num_nodes_2,
        num_nodes_type_1=num_nodes_1,
        num_nodes_type_2=num_nodes_2
    )

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    save_path = os.path.join(save_dir, f'{dataset_name}.pt')
    torch.save(data, save_path)
    
    print(f"Grafo guardado exitosamente en: {save_path}")
    print("Resumen del objeto Data:", data)
    print("--- Proceso finalizado ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Procesador de datos para grafos bipartitos.")
    parser.add_argument('--csv', type=str, required=True, help="Ruta al archivo CSV de entrada.")
    parser.add_argument('--name', type=str, required=True, help="Nombre base para el archivo .pt de salida.")
    parser.add_argument('--out_dir', type=str, default='../data/processed', help="Directorio de salida.")
    
    args = parser.parse_args()
    
    create_bipartite_graph_from_csv(
        csv_path=args.csv,
        save_dir=args.out_dir,
        dataset_name=args.name
    )
