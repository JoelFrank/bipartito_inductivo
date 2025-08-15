# Ejemplo de uso del proyecto

import pandas as pd
import numpy as np

# Crear un dataset de ejemplo
def create_example_dataset():
    """Crear un dataset bipartito de ejemplo (usuarios-items)"""
    np.random.seed(42)
    
    # Simular interacciones usuario-item
    n_users = 100
    n_items = 50
    n_interactions = 300
    
    users = np.random.randint(0, n_users, n_interactions)
    items = np.random.randint(0, n_items, n_interactions)
    
    # Crear DataFrame
    df = pd.DataFrame({
        'user_id': users,
        'item_id': items
    })
    
    # Eliminar duplicados
    df = df.drop_duplicates()
    
    return df

if __name__ == "__main__":
    # Crear dataset de ejemplo
    df = create_example_dataset()
    
    # Guardar como CSV
    csv_path = "../data/raw/example_bipartite.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"Dataset de ejemplo creado: {csv_path}")
    print(f"Forma del dataset: {df.shape}")
    print("Primeras filas:")
    print(df.head())
    
    print("\nPara procesar este dataset, ejecuta:")
    print(f"python data_preparation/create_my_dataset.py --csv {csv_path} --name example-bipartite")
