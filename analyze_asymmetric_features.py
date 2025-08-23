"""
Analizar cómo se manejan las características asimétricas en grafos bipartitos:
- Patrimônios: CON embeddings (IDs de tipo de equipamento)
- Localizações: SIN características
"""
import torch
import torch.nn as nn
from torch_geometric.data import Data

def analyze_asymmetric_features():
    """
    Analizar qué pasa cuando solo un tipo de nodo tiene características
    """
    print("=== ANÁLISIS DE CARACTERÍSTICAS ASIMÉTRICAS ===")
    
    # Simular datos como en sismetro_clean.pt
    num_patrimonios = 100
    num_localizacoes = 50
    total_nodes = num_patrimonios + num_localizacoes
    
    # Solo patrimônios tienen características (IDs de tipo equipamento)
    patrimonio_features = torch.randint(0, 10, (num_patrimonios, 1), dtype=torch.long)
    # Localizações NO tienen características
    
    print(f"📊 CONFIGURACIÓN:")
    print(f"  Patrimônios: {num_patrimonios} (CON características)")
    print(f"  Localizações: {num_localizacoes} (SIN características)")
    print(f"  Total nodos: {total_nodes}")
    print(f"  patrimonio_features shape: {patrimonio_features.shape}")
    
    # Crear data object como en el dataset actual
    edge_index = torch.randint(0, total_nodes, (2, 200))
    data = Data(
        x=patrimonio_features,  # Solo para patrimônios
        edge_index=edge_index,
        num_nodes=total_nodes,
        num_nodes_type_1=num_patrimonios,
        num_nodes_type_2=num_localizacoes
    )
    
    print(f"\n🔍 ANÁLISIS DEL PROBLEMA:")
    print(f"  data.x shape: {data.x.shape}")
    print(f"  data.num_nodes: {data.num_nodes}")
    print(f"  ❌ PROBLEMA: data.x tiene {data.x.size(0)} características pero hay {data.num_nodes} nodos")
    
    return data, patrimonio_features

def simulate_model_processing(data):
    """
    Simular cómo el modelo procesa las características asimétricas
    """
    print(f"\n=== SIMULACIÓN DEL PROCESAMIENTO DEL MODELO ===")
    
    # Extraer configuración
    num_nodes_type_1 = data.num_nodes_type_1  # 100 patrimônios
    num_nodes_type_2 = data.num_nodes_type_2  # 50 localizações
    device = data.x.device
    
    print(f"📋 CONFIGURACIÓN DETECTADA:")
    print(f"  num_nodes_type_1: {num_nodes_type_1}")
    print(f"  num_nodes_type_2: {num_nodes_type_2}")
    print(f"  data.x disponible: {data.x.shape}")
    
    # ¿Qué hace el modelo actual?
    if data.x is not None:
        if data.x.size(0) < data.num_nodes:
            print(f"\n⚠️ CASO DETECTADO: Menos características que nodos")
            print(f"   Características disponibles: {data.x.size(0)}")
            print(f"   Nodos totales: {data.num_nodes}")
            
            # Ajustar conteos (como hace el modelo actual)
            available_nodes = data.x.size(0)
            adjusted_type_1 = min(num_nodes_type_1, available_nodes)
            adjusted_type_2 = available_nodes - adjusted_type_1
            
            print(f"   ✂️ AJUSTE AUTOMÁTICO:")
            print(f"      Patrimônios: {num_nodes_type_1} → {adjusted_type_1}")
            print(f"      Localizações: {num_nodes_type_2} → {adjusted_type_2}")
            
            # Extraer características
            x_src = data.x[:adjusted_type_1]  # Patrimônios
            x_dst = data.x[adjusted_type_1:adjusted_type_1 + adjusted_type_2]  # Localizações (¡VACÍO!)
            
            print(f"   📊 RESULTADO:")
            print(f"      x_src (patrimônios): {x_src.shape}")
            print(f"      x_dst (localizações): {x_dst.shape}")
            
            if x_dst.size(0) == 0:
                print(f"   ❌ PROBLEMA CRÍTICO: x_dst está vacío!")
                print(f"   🔧 SOLUCIÓN NECESARIA: Crear características para localizações")
                
                # ¿Qué opciones tenemos?
                print(f"\n🎯 OPCIONES PARA LOCALIZAÇÕES:")
                
                # Opción 1: Features sintéticas aleatorias
                x_dst_random = torch.randn(num_nodes_type_2, 1, dtype=torch.float32)
                print(f"   1️⃣ RANDOM: {x_dst_random.shape} (aleatorias)")
                
                # Opción 2: Features constantes (ceros)
                x_dst_zeros = torch.zeros(num_nodes_type_2, 1, dtype=torch.float32)
                print(f"   2️⃣ ZEROS: {x_dst_zeros.shape} (todas cero)")
                
                # Opción 3: Features constantes (unos)
                x_dst_ones = torch.ones(num_nodes_type_2, 1, dtype=torch.float32)
                print(f"   3️⃣ ONES: {x_dst_ones.shape} (todas uno)")
                
                # Opción 4: ID único para "sin tipo"
                x_dst_no_type = torch.full((num_nodes_type_2, 1), -1, dtype=torch.long)
                print(f"   4️⃣ NO_TYPE: {x_dst_no_type.shape} (ID especial -1)")
                
                return x_src, x_dst_zeros, x_dst_ones, x_dst_random, x_dst_no_type
    
    return None

def test_embedding_processing():
    """
    Probar cómo se procesan los embeddings con características asimétricas
    """
    print(f"\n=== PRUEBA DE PROCESAMIENTO DE EMBEDDINGS ===")
    
    # Configuración
    num_patrimonios = 100
    num_localizacoes = 50
    num_tipos_equipamento = 10
    embedding_dim = 16
    
    # Características solo para patrimônios
    x_patrimonios = torch.randint(0, num_tipos_equipamento, (num_patrimonios, 1), dtype=torch.long)
    
    print(f"📊 CONFIGURACIÓN:")
    print(f"  Patrimônios: {x_patrimonios.shape} (IDs: 0-{num_tipos_equipamento-1})")
    print(f"  Localizações: {num_localizacoes} nodos SIN características")
    
    # Crear embedding layer
    embedding_layer = nn.Embedding(num_tipos_equipamento + 1, embedding_dim)  # +1 para "sin tipo"
    
    # Procesar patrimônios
    x_patrimonios_embedded = embedding_layer(x_patrimonios.squeeze(1))
    print(f"✅ Patrimônios embedded: {x_patrimonios_embedded.shape}")
    
    # ¿Qué hacer con localizações?
    print(f"\n🤔 OPCIONES PARA LOCALIZAÇÕES:")
    
    # Opción A: ID especial para "sin características"
    sin_tipo_id = num_tipos_equipamento  # Último ID disponible
    x_localizacoes_ids = torch.full((num_localizacoes,), sin_tipo_id, dtype=torch.long)
    x_localizacoes_embedded = embedding_layer(x_localizacoes_ids)
    print(f"   A) ID especial {sin_tipo_id}: {x_localizacoes_embedded.shape}")
    
    # Opción B: Vector cero
    x_localizacoes_zeros = torch.zeros(num_localizacoes, embedding_dim, dtype=torch.float32)
    print(f"   B) Vector cero: {x_localizacoes_zeros.shape}")
    
    # Opción C: Vector promedio de los embeddings existentes
    avg_embedding = x_patrimonios_embedded.mean(dim=0, keepdim=True)
    x_localizacoes_avg = avg_embedding.repeat(num_localizacoes, 1)
    print(f"   C) Vector promedio: {x_localizacoes_avg.shape}")
    
    # Verificar que todas tienen la misma dimensión
    print(f"\n✅ VERIFICACIÓN DE DIMENSIONES:")
    print(f"  Patrimônios: {x_patrimonios_embedded.shape}")
    print(f"  Localizações (ID especial): {x_localizacoes_embedded.shape}")
    print(f"  Localizações (zeros): {x_localizacoes_zeros.shape}")
    print(f"  Localizações (promedio): {x_localizacoes_avg.shape}")
    print(f"  ✅ Todas compatibles para concatenación")
    
    return (x_patrimonios_embedded, x_localizacoes_embedded, 
            x_localizacoes_zeros, x_localizacoes_avg)

def recommend_solution():
    """
    Recomendar la mejor solución para el problema
    """
    print(f"\n=== RECOMENDACIÓN DE SOLUCIÓN ===")
    
    print(f"🎯 PROBLEMA IDENTIFICADO:")
    print(f"  - Patrimônios: Tienen IDs de tipo de equipamento")
    print(f"  - Localizações: NO tienen características")
    print(f"  - Modelo necesita características para TODOS los nodos")
    
    print(f"\n🏆 SOLUCIÓN RECOMENDADA:")
    print(f"  1️⃣ EXPANDIR VOCABULARIO DE EMBEDDINGS:")
    print(f"     - Tipos de equipamento: IDs 0, 1, 2, ..., 9")
    print(f"     - Localizações 'sin tipo': ID 10 (especial)")
    print(f"     - Total embeddings: 11 (10 + 1)")
    
    print(f"\n  2️⃣ MODIFICAR DATASET CREATION:")
    print(f"     - Patrimônios: usar IDs reales de tipo")
    print(f"     - Localizações: asignar ID especial 10")
    print(f"     - data.x shape: [total_nodes, 1] (no solo patrimônios)")
    
    print(f"\n  3️⃣ VENTAJAS DE ESTA SOLUCIÓN:")
    print(f"     ✅ Embeddings aprenden representación para 'sin tipo'")
    print(f"     ✅ Simetría completa en el modelo")
    print(f"     ✅ No hay características sintéticas aleatorias")
    print(f"     ✅ Interpretabilidad clara")
    
    print(f"\n  4️⃣ IMPLEMENTACIÓN:")
    print(f"     - num_embeddings = num_tipos_equipamento + 1")
    print(f"     - data.x = torch.tensor de [total_nodes, 1]")
    print(f"     - data.x[:num_patrimonios] = tipos reales")
    print(f"     - data.x[num_patrimonios:] = ID especial")

if __name__ == '__main__':
    # Analizar el problema
    data, patrimonio_features = analyze_asymmetric_features()
    
    # Simular procesamiento
    model_results = simulate_model_processing(data)
    
    # Probar embeddings
    embedding_results = test_embedding_processing()
    
    # Recomendar solución
    recommend_solution()
    
    print(f"\n🎉 CONCLUSIÓN: ¡Hay que crear características para AMBOS tipos de nodos!")
