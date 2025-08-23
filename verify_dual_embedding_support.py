"""
Verificar y demostrar cómo el modelo BipartiteSAGE puede soportar
embeddings para AMBOS tipos de nodos en un grafo bipartito.
"""
import torch
import torch.nn as nn
from torch_geometric.data import Data

# Simulación de cómo modificaríamos BipartiteSAGE para soporte dual completo
class BipartiteSAGEDualEmbedding(nn.Module):
    """
    Versión extendida de BipartiteSAGE que soporta embeddings separados
    para cada tipo de nodo en el grafo bipartito.
    """
    def __init__(self, 
                 in_channels_src, in_channels_dst, 
                 hidden_channels, out_channels, 
                 num_layers=2,
                 # Embeddings para nodos tipo 1 (src)
                 num_embeddings_src=None, embedding_dim_src=None,
                 # Embeddings para nodos tipo 2 (dst) 
                 num_embeddings_dst=None, embedding_dim_dst=None):
        super().__init__()
        
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        
        # Configurar embeddings para nodos src (tipo 1)
        self.use_embeddings_src = num_embeddings_src is not None and embedding_dim_src is not None
        if self.use_embeddings_src:
            self.embedding_processor_src = nn.Embedding(num_embeddings_src, embedding_dim_src)
            actual_in_channels_src = embedding_dim_src
        else:
            actual_in_channels_src = in_channels_src
            
        # Configurar embeddings para nodos dst (tipo 2)
        self.use_embeddings_dst = num_embeddings_dst is not None and embedding_dim_dst is not None
        if self.use_embeddings_dst:
            self.embedding_processor_dst = nn.Embedding(num_embeddings_dst, embedding_dim_dst)
            actual_in_channels_dst = embedding_dim_dst
        else:
            actual_in_channels_dst = in_channels_dst
        
        # Proyecciones separadas (ya soportado en el modelo actual)
        self.src_proj = nn.Linear(actual_in_channels_src, hidden_channels)
        self.dst_proj = nn.Linear(actual_in_channels_dst, hidden_channels)
        
        # Resto del modelo igual...
        from torch_geometric.nn import SAGEConv
        from torch_geometric.nn import BatchNorm
        
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            if i == num_layers - 1:
                self.convs.append(SAGEConv(hidden_channels, out_channels))
            else:
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        
        self.norms = nn.ModuleList()
        for i in range(num_layers - 1):
            self.norms.append(BatchNorm(hidden_channels))
    
    def forward(self, data):
        """
        Forward pass con soporte para embeddings duales
        """
        device = data.edge_index.device
        
        # Extraer características según el tipo de data
        if hasattr(data, 'num_nodes_type_1') and hasattr(data, 'num_nodes_type_2'):
            num_nodes_type_1 = data.num_nodes_type_1
            num_nodes_type_2 = data.num_nodes_type_2
            
            if data.x is not None:
                # Caso: ambos tipos tienen características
                x_src_raw = data.x[:num_nodes_type_1]
                x_dst_raw = data.x[num_nodes_type_1:num_nodes_type_1 + num_nodes_type_2]
            elif hasattr(data, 'x_src') and hasattr(data, 'x_dst'):
                # Caso: características separadas por tipo
                x_src_raw = data.x_src
                x_dst_raw = data.x_dst
            else:
                # Caso: crear características sintéticas
                x_src_raw = torch.arange(num_nodes_type_1, device=device).unsqueeze(1)
                x_dst_raw = torch.arange(num_nodes_type_2, device=device).unsqueeze(1)
        else:
            raise ValueError("Datos no soportan estructura bipartita con metadatos")
        
        # Procesar embeddings src
        if self.use_embeddings_src:
            if x_src_raw.dtype == torch.long:
                # IDs de embedding
                x_src = self.embedding_processor_src(x_src_raw.squeeze(1))
            else:
                # Características continuas
                x_src = x_src_raw
        else:
            x_src = x_src_raw.float()
        
        # Procesar embeddings dst
        if self.use_embeddings_dst:
            if x_dst_raw.dtype == torch.long:
                # IDs de embedding
                x_dst = self.embedding_processor_dst(x_dst_raw.squeeze(1))
            else:
                # Características continuas
                x_dst = x_dst_raw
        else:
            x_dst = x_dst_raw.float()
        
        # Proyectar a dimensiones comunes
        x_src = self.src_proj(x_src)
        x_dst = self.dst_proj(x_dst)
        
        # Concatenar y procesar como en el modelo actual
        x_all = torch.cat([x_src, x_dst], dim=0)
        edge_index = data.edge_index
        
        # Aplicar capas SAGE
        for i in range(self.num_layers):
            x_all = self.convs[i](x_all, edge_index)
            if i < self.num_layers - 1:
                x_all = self.norms[i](x_all)
                x_all = torch.relu(x_all)
        
        return x_all

def test_dual_embedding_support():
    """
    Probar el modelo con embeddings para ambos tipos de nodos
    """
    print("=== PROBANDO SOPORTE PARA EMBEDDINGS DUALES ===")
    
    # Configuración del test
    num_patrimonios = 100
    num_localizacoes = 50
    num_edges = 200
    
    # Vocabularios de embeddings
    num_tipos_equipamento = 10  # Para patrimônios
    num_tipos_localizacao = 5   # Para localizações
    
    # Crear datos de prueba
    # Patrimônios: IDs de tipo de equipamento [0, 9]
    x_patrimonios = torch.randint(0, num_tipos_equipamento, (num_patrimonios, 1), dtype=torch.long)
    
    # Localizações: IDs de tipo de localização [0, 4]
    x_localizacoes = torch.randint(0, num_tipos_localizacao, (num_localizacoes, 1), dtype=torch.long)
    
    # Aristas aleatorias
    src_nodes = torch.randint(0, num_patrimonios, (num_edges,))
    dst_nodes = torch.randint(0, num_localizacoes, (num_edges,)) + num_patrimonios
    edge_index = torch.stack([src_nodes, dst_nodes])
    
    # Crear data object con características separadas
    data = Data(
        edge_index=edge_index,
        num_nodes=num_patrimonios + num_localizacoes,
        num_nodes_type_1=num_patrimonios,
        num_nodes_type_2=num_localizacoes,
        x_src=x_patrimonios,
        x_dst=x_localizacoes
    )
    
    print(f"✓ Datos creados:")
    print(f"  Patrimônios: {num_patrimonios} (tipos equipamento: {num_tipos_equipamento})")
    print(f"  Localizações: {num_localizacoes} (tipos localização: {num_tipos_localizacao})")
    print(f"  Aristas: {num_edges}")
    print(f"  x_src shape: {data.x_src.shape}")
    print(f"  x_dst shape: {data.x_dst.shape}")
    
    # Crear modelo con embeddings duales
    model = BipartiteSAGEDualEmbedding(
        in_channels_src=1, in_channels_dst=1,
        hidden_channels=64, out_channels=32,
        num_layers=2,
        # Embeddings para patrimônios (tipos de equipamento)
        num_embeddings_src=num_tipos_equipamento,
        embedding_dim_src=16,
        # Embeddings para localizações (tipos de localização)
        num_embeddings_dst=num_tipos_localizacao,
        embedding_dim_dst=8
    )
    
    print(f"\n✓ Modelo creado:")
    print(f"  Embeddings patrimônios: {num_tipos_equipamento} → {16}")
    print(f"  Embeddings localizações: {num_tipos_localizacao} → {8}")
    print(f"  Hidden channels: 64")
    print(f"  Output channels: 32")
    
    # Forward pass
    with torch.no_grad():
        embeddings = model(data)
    
    print(f"\n✓ Forward pass exitoso:")
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Patrimônios embeddings: {embeddings[:num_patrimonios].shape}")
    print(f"  Localizações embeddings: {embeddings[num_patrimonios:].shape}")
    
    print(f"\n🎯 CONCLUSIÓN: El modelo SÍ puede soportar embeddings para ambos tipos de nodos")
    
    return model, data, embeddings

def compare_with_current_model():
    """
    Comparar con el modelo actual para mostrar qué cambios se necesitan
    """
    print(f"\n=== COMPARACIÓN CON MODELO ACTUAL ===")
    
    print("📋 MODELO ACTUAL (BipartiteSAGE):")
    print("  ✅ Proyecciones separadas: src_proj, dst_proj")
    print("  ✅ Manejo de x_src, x_dst por separado") 
    print("  ⚠️ Un solo EmbeddingProcessor compartido")
    print("  ⚠️ Mismo num_embeddings para ambos tipos")
    print("  ⚠️ Misma embedding_dim para ambos tipos")
    
    print("\n📋 EXTENSIÓN PROPUESTA (BipartiteSAGEDualEmbedding):")
    print("  ✅ Proyecciones separadas: src_proj, dst_proj")
    print("  ✅ Manejo de x_src, x_dst por separado")
    print("  ✅ EmbeddingProcessor separado por tipo: embedding_processor_src, embedding_processor_dst")
    print("  ✅ num_embeddings separado: num_embeddings_src, num_embeddings_dst")
    print("  ✅ embedding_dim separado: embedding_dim_src, embedding_dim_dst")
    
    print("\n🔧 CAMBIOS NECESARIOS EN EL MODELO ACTUAL:")
    print("  1. Agregar parámetros num_embeddings_src/dst, embedding_dim_src/dst")
    print("  2. Crear dos EmbeddingProcessor separados")
    print("  3. Lógica condicional para procesar cada tipo")
    print("  4. Soporte para data.x_src y data.x_dst además de data.x")
    
    print("\n✅ COMPATIBILIDAD HACIA ATRÁS:")
    print("  - Si solo se especifican parámetros src, funciona igual que ahora")
    print("  - Si no se especifican embeddings, usa características continuas")
    print("  - El resto del modelo (proyecciones, SAGE, etc.) no cambia")

if __name__ == '__main__':
    # Probar soporte dual
    model, data, embeddings = test_dual_embedding_support()
    
    # Comparar con modelo actual
    compare_with_current_model()
    
    print(f"\n🎉 ¡El modelo PUEDE soportar embeddings duales con pequeñas modificaciones!")
