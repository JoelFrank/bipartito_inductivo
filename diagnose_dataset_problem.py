"""
Diagnosticar problemas con el dataset sismetro_clean.pt
"""
import torch
import os

def verify_sismetro_clean():
    """
    Verificar que sismetro_clean.pt existe y tiene la estructura correcta
    """
    print("=== DIAGNÓSTICO SISMETRO_CLEAN.PT ===")
    
    # 1. Verificar existencia del archivo
    clean_path = "data/processed/sismetro_clean.pt"
    original_path = "data/processed/sismetro.pt"
    
    print(f"📁 VERIFICANDO ARCHIVOS:")
    print(f"  sismetro_clean.pt: {'✅ EXISTE' if os.path.exists(clean_path) else '❌ NO EXISTE'}")
    print(f"  sismetro.pt: {'✅ EXISTE' if os.path.exists(original_path) else '❌ NO EXISTE'}")
    
    if not os.path.exists(clean_path):
        print(f"❌ PROBLEMA: El archivo {clean_path} no existe!")
        print(f"🔧 SOLUCIÓN: Ejecutar create_sismetro_clean.py primero")
        return None
    
    # 2. Cargar y analizar dataset clean
    print(f"\n📊 CARGANDO SISMETRO_CLEAN.PT:")
    try:
        data_clean = torch.load(clean_path)
        print(f"✅ Archivo cargado exitosamente")
        print(f"  Tipo: {type(data_clean)}")
        print(f"  Nodos: {data_clean.num_nodes}")
        print(f"  Aristas: {data_clean.edge_index.shape[1]}")
        print(f"  Características: {data_clean.x.shape if hasattr(data_clean, 'x') and data_clean.x is not None else 'None'}")
        print(f"  Patrimônios: {data_clean.num_nodes_type_1 if hasattr(data_clean, 'num_nodes_type_1') else 'No disponible'}")
        print(f"  Localizações: {data_clean.num_nodes_type_2 if hasattr(data_clean, 'num_nodes_type_2') else 'No disponible'}")
        
        # Verificar metadatos de embeddings
        if hasattr(data_clean, 'total_embedding_ids'):
            print(f"  Total embedding IDs: {data_clean.total_embedding_ids}")
            print(f"  ID NEUTRO: {data_clean.ID_NEUTRO}")
        
        return data_clean
        
    except Exception as e:
        print(f"❌ ERROR al cargar: {e}")
        return None

def compare_with_original():
    """
    Comparar con el dataset original para verificar diferencias
    """
    original_path = "data/processed/sismetro.pt"
    
    if not os.path.exists(original_path):
        print(f"⚠️ No se puede comparar: {original_path} no existe")
        return
    
    print(f"\n📊 COMPARANDO CON SISMETRO.PT ORIGINAL:")
    try:
        data_original = torch.load(original_path)
        print(f"✅ Original cargado:")
        print(f"  Nodos: {data_original.num_nodes}")
        print(f"  Aristas: {data_original.edge_index.shape[1]}")
        print(f"  Características: {data_original.x.shape if hasattr(data_original, 'x') and data_original.x is not None else 'None'}")
        
    except Exception as e:
        print(f"❌ ERROR al cargar original: {e}")

def check_data_loading_code():
    """
    Verificar cómo el código de entrenamiento carga los datasets
    """
    print(f"\n🔍 VERIFICANDO CÓDIGO DE CARGA:")
    
    # Verificar si existe el loader para sismetro_clean
    from pathlib import Path
    data_files = list(Path("src/lib").glob("**/*.py"))
    
    for file_path in data_files:
        if "data" in file_path.name.lower():
            print(f"  📄 Archivo relevante: {file_path}")

if __name__ == '__main__':
    # Diagnosticar dataset clean
    data_clean = verify_sismetro_clean()
    
    # Comparar con original
    compare_with_original()
    
    # Verificar código de carga
    check_data_loading_code()
    
    if data_clean is not None:
        print(f"\n✅ SISMETRO_CLEAN.PT está disponible y parece correcto")
        print(f"🔍 El problema puede estar en el código de carga o configuración")
    else:
        print(f"\n❌ PROBLEMA IDENTIFICADO: sismetro_clean.pt no está disponible")
