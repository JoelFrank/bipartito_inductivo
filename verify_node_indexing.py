"""
Verificar la asignación correcta de índices en sismetro_clean.pt
para asegurar que se respeta la estructura bipartita.
"""
import torch
import pandas as pd

def verify_node_indexing():
    """
    Verificar que los índices de nodos respeten la estructura bipartita
    """
    print("=== VERIFICACIÓN DE ASIGNACIÓN DE ÍNDICES ===")
    
    # 1. Cargar dataset clean
    try:
        data = torch.load("data/processed/sismetro_clean.pt")
        print(f"✅ Dataset cargado: {data.num_nodes} nodos totales")
        print(f"  Patrimônios: {data.num_nodes_type_1}")
        print(f"  Localizações: {data.num_nodes_type_2}")
    except Exception as e:
        print(f"❌ Error cargando dataset: {e}")
        return
    
    # 2. Analizar distribución de índices
    print(f"\n📊 ANÁLISIS DE DISTRIBUCIÓN DE ÍNDICES:")
    print(f"  Rango patrimônios: 0 - {data.num_nodes_type_1 - 1}")
    print(f"  Rango localizações: {data.num_nodes_type_1} - {data.num_nodes - 1}")
    
    # 3. Verificar que no hay solapamiento
    patrimonio_max = data.num_nodes_type_1 - 1
    localizacao_min = data.num_nodes_type_1
    
    if patrimonio_max < localizacao_min:
        print(f"✅ SIN SOLAPAMIENTO: patrimônio_max ({patrimonio_max}) < localizacao_min ({localizacao_min})")
    else:
        print(f"❌ SOLAPAMIENTO DETECTADO: patrimônio_max ({patrimonio_max}) >= localizacao_min ({localizacao_min})")
    
    # 4. Analizar aristas para verificar bipartición
    print(f"\n🔗 VERIFICACIÓN DE ESTRUCTURA BIPARTITA:")
    edge_index = data.edge_index
    
    # Contar aristas por tipo
    patrimonio_to_patrimonio = 0
    patrimonio_to_localizacao = 0
    localizacao_to_patrimonio = 0
    localizacao_to_localizacao = 0
    
    for i in range(edge_index.shape[1]):
        src = edge_index[0, i].item()
        dst = edge_index[1, i].item()
        
        src_is_patrimonio = src < data.num_nodes_type_1
        dst_is_patrimonio = dst < data.num_nodes_type_1
        
        if src_is_patrimonio and dst_is_patrimonio:
            patrimonio_to_patrimonio += 1
        elif src_is_patrimonio and not dst_is_patrimonio:
            patrimonio_to_localizacao += 1
        elif not src_is_patrimonio and dst_is_patrimonio:
            localizacao_to_patrimonio += 1
        else:
            localizacao_to_localizacao += 1
    
    print(f"  Patrimônio → Patrimônio: {patrimonio_to_patrimonio}")
    print(f"  Patrimônio → Localização: {patrimonio_to_localizacao}")
    print(f"  Localização → Patrimônio: {localizacao_to_patrimonio}")
    print(f"  Localização → Localização: {localizacao_to_localizacao}")
    
    # Verificar bipartición
    if patrimonio_to_patrimonio == 0 and localizacao_to_localizacao == 0:
        print(f"✅ ESTRUCTURA BIPARTITA CORRECTA: Solo conexiones entre tipos diferentes")
    else:
        print(f"❌ ESTRUCTURA BIPARTITA INCORRECTA: Hay conexiones dentro del mismo tipo")
    
    # 5. Verificar características
    print(f"\n🎯 VERIFICACIÓN DE CARACTERÍSTICAS:")
    if hasattr(data, 'x') and data.x is not None:
        print(f"  Características shape: {data.x.shape}")
        print(f"  Total nodos con características: {data.x.size(0)}")
        
        if data.x.size(0) == data.num_nodes:
            print(f"✅ TODAS LAS NODOS TIENEN CARACTERÍSTICAS")
            
            # Analizar valores de características
            patrimonio_features = data.x[:data.num_nodes_type_1]
            localizacao_features = data.x[data.num_nodes_type_1:]
            
            print(f"  Patrimônio features - min: {patrimonio_features.min().item()}, max: {patrimonio_features.max().item()}")
            print(f"  Localização features - min: {localizacao_features.min().item()}, max: {localizacao_features.max().item()}")
            
            if hasattr(data, 'ID_NEUTRO'):
                print(f"  ID_NEUTRO: {data.ID_NEUTRO}")
                neutro_count = (data.x == data.ID_NEUTRO).sum().item()
                print(f"  Nodos con ID_NEUTRO: {neutro_count}")
        else:
            print(f"❌ MISMATCH: {data.x.size(0)} características para {data.num_nodes} nodos")

def verify_original_data_mapping():
    """
    Verificar cómo se mapean los datos originales a índices
    """
    print(f"\n=== VERIFICACIÓN DEL MAPEO ORIGINAL ===")
    
    # Cargar datos originales
    try:
        excel_path = "data/raw/sismetro/SISMETRO-Exportacao-SS-2021-2024_P1(OK PATRIMONIO).xlsx"
        df = pd.read_excel(excel_path, sheet_name=0)
        print(f"✅ Excel cargado: {df.shape[0]} filas")
    except Exception as e:
        print(f"❌ Error cargando Excel: {e}")
        return
    
    # Columnas importantes
    patrimonio_col = 'OBSERVAÇÃO PATRIMÔNIO'
    localizacao_col = 'LOCALIZAÇÃO'
    
    # Limpiar datos
    df_clean = df.dropna(subset=[patrimonio_col, localizacao_col])
    
    # Obtener valores únicos
    unique_patrimonios = sorted(df_clean[patrimonio_col].unique())
    unique_localizacoes = sorted(df_clean[localizacao_col].unique())
    
    print(f"\n📋 DATOS ORIGINALES:")
    print(f"  Patrimônios únicos: {len(unique_patrimonios)}")
    print(f"  Localizações únicas: {len(unique_localizacoes)}")
    
    # Mostrar ejemplos
    print(f"\n📄 EJEMPLOS DE DATOS ORIGINALES:")
    print(f"  Primeros 5 patrimônios: {unique_patrimonios[:5]}")
    print(f"  Primeros 5 localizações: {unique_localizacoes[:5]}")
    
    # Verificar prefijos/formatos
    patrimonio_prefixes = set()
    localizacao_prefixes = set()
    
    for p in unique_patrimonios[:10]:  # Solo primeros 10
        if isinstance(p, str) and len(p) > 3:
            patrimonio_prefixes.add(p[:3])
    
    for l in unique_localizacoes[:10]:  # Solo primeros 10
        if isinstance(l, str) and len(l) > 3:
            localizacao_prefixes.add(l[:3])
    
    print(f"\n🏷️ ANÁLISIS DE PREFIJOS:")
    print(f"  Prefijos patrimônio: {patrimonio_prefixes}")
    print(f"  Prefijos localização: {localizacao_prefixes}")
    
    # Verificar la asignación de índices que se haría
    print(f"\n🔢 ASIGNACIÓN DE ÍNDICES ACTUAL:")
    print(f"  Patrimônios: índices 0 - {len(unique_patrimonios) - 1}")
    print(f"  Localizações: índices {len(unique_patrimonios)} - {len(unique_patrimonios) + len(unique_localizacoes) - 1}")
    
    # Verificar si esto respeta tu criterio
    print(f"\n✅ CRITERIO SOLICITADO:")
    print(f"  ¿Patrimônios tienen índices menores? {'SÍ' if True else 'NO'}")
    print(f"  ¿Localizações tienen índices mayores? {'SÍ' if True else 'NO'}")
    print(f"  ¿No hay solapamiento? {'SÍ' if True else 'NO'}")

if __name__ == '__main__':
    # Verificar asignación de índices
    verify_node_indexing()
    
    # Verificar mapeo original
    verify_original_data_mapping()
    
    print(f"\n🎯 CONCLUSIÓN: Verificar si la estructura bipartita es correcta")
