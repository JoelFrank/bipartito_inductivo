import torch
import numpy as np
import pandas as pd
import os
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score
from torch_geometric.utils import negative_sampling
from torch_geometric.data import HeteroData  # Importar HeteroData
import logging
from absl import flags

# Import bipartite-aware negative sampling
from .split import bipartite_negative_sampling, bipartite_negative_sampling_inductive

log = logging.getLogger(__name__)
FLAGS = flags.FLAGS

def eval_all(y_pred_pos, y_pred_neg, threshold=0.5):
    """Evaluate link prediction performance with additional classification metrics"""
    y_pred = torch.cat([y_pred_pos, y_pred_neg], dim=0).cpu().numpy()
    y_true = np.concatenate([np.ones(len(y_pred_pos)), np.zeros(len(y_pred_neg))])
    
    # Standard ranking metrics
    auc = roc_auc_score(y_true, y_pred)
    ap = average_precision_score(y_true, y_pred)
    
    # Classification metrics with threshold
    y_pred_binary = (y_pred >= threshold).astype(int)
    precision = precision_score(y_true, y_pred_binary, zero_division=0)
    recall = recall_score(y_true, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)
    
    # Calculate hits@k
    hits_10 = eval_hits_at_k(y_pred_pos, y_pred_neg, k=10)
    hits_50 = eval_hits_at_k(y_pred_pos, y_pred_neg, k=50)
    hits_100 = eval_hits_at_k(y_pred_pos, y_pred_neg, k=100)
    
    return {
        'auc': auc,
        'ap': ap,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'hits@10': hits_10,
        'hits@50': hits_50,
        'hits@100': hits_100
    }

def eval_hits_at_k(y_pred_pos, y_pred_neg, k):
    """Calculate hits@k metric"""
    y_pred_pos = y_pred_pos.cpu().numpy() if torch.is_tensor(y_pred_pos) else y_pred_pos
    y_pred_neg = y_pred_neg.cpu().numpy() if torch.is_tensor(y_pred_neg) else y_pred_neg
    
    # Handle case where k is larger than number of negative edges
    if k > len(y_pred_neg):
        k = len(y_pred_neg)
    
    if k == 0:
        return 0.0
        
    kth_score_in_negative_edges = np.sort(y_pred_neg)[-k]
    hits = np.sum(y_pred_pos > kth_score_in_negative_edges) / len(y_pred_pos)
    return hits

def clean_and_deduplicate_edges(edge_index, scores=None):
    """
    Limpia y deduplica enlaces, manteniendo solo una dirección para grafos no dirigidos
    """
    # Convert to numpy for easier processing
    if torch.is_tensor(edge_index):
        edges = edge_index.cpu().numpy()
    else:
        edges = edge_index
    
    if scores is not None and torch.is_tensor(scores):
        scores = scores.cpu().numpy()
    
    # Create dictionary to track unique edges
    unique_edges = {}
    
    for i, edge in enumerate(edges):
        # Normalize edge direction (always put smaller node first)
        src, dst = int(edge[0]), int(edge[1])
        normalized_edge = tuple(sorted([src, dst]))
        
        # Keep only first occurrence of each edge
        if normalized_edge not in unique_edges:
            # Always store in format (smaller_node, larger_node)
            unique_edges[normalized_edge] = {
                'original_edge': [src, dst],
                'score': scores[i] if scores is not None else None,
                'index': i
            }
    
    # Reconstruct cleaned edges
    clean_edges = []
    clean_scores = []
    
    for normalized_edge, data in unique_edges.items():
        clean_edges.append(data['original_edge'])
        if scores is not None:
            clean_scores.append(data['score'])
    
    # Convert back to tensors
    clean_edge_tensor = torch.tensor(clean_edges, dtype=torch.long)
    clean_score_tensor = torch.tensor(clean_scores, dtype=torch.float32) if scores is not None else None
    
    log.info(f"Limpieza completada: {len(edges)} -> {len(clean_edges)} enlaces únicos")
    
    return clean_edge_tensor, clean_score_tensor

def export_test_results_csv(test_edge, test_edge_neg, test_pos_pred, test_neg_pred, output_dir, model_type):
    """Export test results to CSV format with columns: u, v, label, score"""
    
    # DEBUG: Log de entrada
    log.info(f"=== EXPORTANDO CSV ===")
    log.info(f"Parámetros recibidos:")
    log.info(f"  - test_edge: {type(test_edge)} {test_edge.shape if hasattr(test_edge, 'shape') else 'N/A'}")
    log.info(f"  - test_edge_neg: {type(test_edge_neg)} {test_edge_neg.shape if hasattr(test_edge_neg, 'shape') else 'N/A'}")
    log.info(f"  - test_pos_pred: {type(test_pos_pred)} {test_pos_pred.shape if hasattr(test_pos_pred, 'shape') else 'N/A'}")
    log.info(f"  - test_neg_pred: {type(test_neg_pred)} {test_neg_pred.shape if hasattr(test_neg_pred, 'shape') else 'N/A'}")
    
    # Convert tensors to numpy if needed
    if torch.is_tensor(test_edge):
        test_edge = test_edge.cpu().numpy()
    if torch.is_tensor(test_edge_neg):
        test_edge_neg = test_edge_neg.cpu().numpy()
    if torch.is_tensor(test_pos_pred):
        test_pos_pred = test_pos_pred.cpu().numpy()
    if torch.is_tensor(test_neg_pred):
        test_neg_pred = test_neg_pred.cpu().numpy()
    
    log.info(f"Después de conversión a numpy:")
    log.info(f"  - test_edge: {test_edge.shape}")
    log.info(f"  - test_edge_neg: {test_edge_neg.shape}")
    log.info(f"  - test_pos_pred: {test_pos_pred.shape}")
    log.info(f"  - test_neg_pred: {test_neg_pred.shape}")
    
    # LIMPIEZA: Deduplicar enlaces positivos
    log.info("=== LIMPIANDO ENLACES POSITIVOS ===")
    test_edge_clean, test_pos_pred_clean = clean_and_deduplicate_edges(
        torch.tensor(test_edge), torch.tensor(test_pos_pred)
    )
    test_edge = test_edge_clean.numpy()
    test_pos_pred = test_pos_pred_clean.numpy()
    
    # LIMPIEZA: Deduplicar enlaces negativos  
    log.info("=== LIMPIANDO ENLACES NEGATIVOS ===")
    test_edge_neg_clean, test_neg_pred_clean = clean_and_deduplicate_edges(
        torch.tensor(test_edge_neg), torch.tensor(test_neg_pred)
    )
    test_edge_neg = test_edge_neg_clean.numpy()
    test_neg_pred = test_neg_pred_clean.numpy()
    
    # DEBUG: Investigar duplicados y estructura bipartita
    log.info(f"=== DIAGNÓSTICO DETALLADO ===")
    
    # Verificar estructura bipartita en positive edges
    pos_sources = set(test_edge[:, 0].tolist())
    pos_targets = set(test_edge[:, 1].tolist())
    log.info(f"Positive edges - Sources únicos: {len(pos_sources)}, Targets únicos: {len(pos_targets)}")
    log.info(f"Sources sample: {list(pos_sources)[:10]}")
    log.info(f"Targets sample: {list(pos_targets)[:10]}")
    
    # Verificar estructura bipartita en negative edges  
    neg_sources = set(test_edge_neg[:, 0].tolist())
    neg_targets = set(test_edge_neg[:, 1].tolist())
    log.info(f"Negative edges - Sources únicos: {len(neg_sources)}, Targets únicos: {len(neg_targets)}")
    
    # Verificar solapamiento (no debería haberlo en bipartito)
    source_target_overlap = pos_sources.intersection(pos_targets)
    if source_target_overlap:
        log.error(f"❌ PROBLEMA BIPARTITO: Nodos aparecen como source Y target: {source_target_overlap}")
    else:
        log.info("✓ Estructura bipartita correcta: sin solapamiento source-target")
    
    # Verificar duplicados en edges
    pos_edge_set = set()
    pos_duplicates = 0
    for edge in test_edge:
        edge_tuple = (int(edge[0]), int(edge[1]))
        edge_tuple_rev = (int(edge[1]), int(edge[0]))
        if edge_tuple in pos_edge_set or edge_tuple_rev in pos_edge_set:
            pos_duplicates += 1
        pos_edge_set.add(edge_tuple)
    
    log.info(f"Duplicados en positive edges: {pos_duplicates}")
    
    # Verificar duplicados en predictions
    unique_pos_scores = len(set(test_pos_pred.tolist()))
    unique_neg_scores = len(set(test_neg_pred.tolist()))
    log.info(f"Scores únicos - Positivos: {unique_pos_scores}/{len(test_pos_pred)}, Negativos: {unique_neg_scores}/{len(test_neg_pred)}")
    
    # Create positive samples data
    pos_data = []
    for i, edge in enumerate(test_edge):
        pos_data.append({
            'u': int(edge[0]),
            'v': int(edge[1]),
            'label': 1,
            'score': float(test_pos_pred[i])
        })
    
    # Create negative samples data
    neg_data = []
    for i, edge in enumerate(test_edge_neg):
        neg_data.append({
            'u': int(edge[0]),
            'v': int(edge[1]),
            'label': 0,
            'score': float(test_neg_pred[i])
        })
    
    # Combine and create DataFrame
    all_data = pos_data + neg_data
    df = pd.DataFrame(all_data)
    
    # Sort by score descending for better analysis
    df = df.sort_values('score', ascending=False).reset_index(drop=True)
    
    # Save to CSV
    csv_path = os.path.join(output_dir, f'{model_type}_test_results.csv')
    df.to_csv(csv_path, index=False)
    
    log.info(f"Test results exported to: {csv_path}")
    log.info(f"Total samples: {len(df)} (positive: {len(pos_data)}, negative: {len(neg_data)})")
    
    return csv_path

def do_all_eval(model_name, output_dir, valid_models, dataset, edge_split, 
                embeddings, decoder_zoo, wandb_logger):
    """Perform evaluation on all decoder models with HeteroData support"""
    results = []
    
    device = next(embeddings.parameters()).device
    train_edge = edge_split['train']['edge'].to(device)
    valid_edge = edge_split['valid']['edge'].to(device)
    test_edge = edge_split['test']['edge'].to(device)
    valid_edge_neg = edge_split['valid']['edge_neg'].to(device)
    test_edge_neg = edge_split['test']['edge_neg'].to(device)
    
    # DEBUG: Verificar que los enlaces negativos se cargaron correctamente
    log.info(f"DEBUG - Enlaces cargados en do_all_eval:")
    log.info(f"  - test_edge (positivos): {test_edge.shape}")
    log.info(f"  - test_edge_neg (negativos): {test_edge_neg.shape}")
    log.info(f"  - valid_edge (positivos): {valid_edge.shape}")
    log.info(f"  - valid_edge_neg (negativos): {valid_edge_neg.shape}")
    
    if test_edge_neg.shape[0] == 0:
        log.error("❌ PROBLEMA CRÍTICO: test_edge_neg está vacío!")
        return [], None
    elif test_edge_neg.shape[0] < test_edge.shape[0]:
        log.warning(f"⚠️ PROBLEMA: Muy pocos enlaces negativos: {test_edge_neg.shape[0]} vs {test_edge.shape[0]} positivos")
    
    # DEBUG: Verificar que edge_split contiene datos correctos
    log.info(f"DEBUG edge_split:")
    log.info(f"  train edges: {train_edge.shape}")
    log.info(f"  valid edges: {valid_edge.shape}")
    log.info(f"  test edges: {test_edge.shape}")
    log.info(f"  valid_edge_neg: {valid_edge_neg.shape}")
    log.info(f"  test_edge_neg: {test_edge_neg.shape}")
    
    if test_edge_neg.shape[0] == 0:
        log.error("❌ CRÍTICO: test_edge_neg está vacío en edge_split!")
        return [], None
    
    # Get the original data object for bipartite information
    if hasattr(dataset, 'num_nodes'):
        data = dataset
    else:
        data = dataset[0]
    
    # Check if we're working with HeteroData or bipartite structure
    is_hetero = isinstance(data, HeteroData)
    is_bipartite = is_hetero or hasattr(data, 'num_nodes_type_1')
    
    if is_hetero:
        log.info("HeteroData detectado. Usando embeddings bipartitos.")
        # For HeteroData, we need to map node indices correctly
        num_patrimonio = data['patrimonio'].num_nodes
        num_localizacao = data['localizacao'].num_nodes
    elif is_bipartite:
        log.info("Evaluación bipartita detectada. Usando muestreo negativo bipartito.")
    
    def get_node_embeddings(node_ids, is_src_type=True):
        """Get embeddings for specific node IDs, handling HeteroData mapping"""
        if is_hetero:
            # BipartiteSAGE returns concatenated embeddings [patrimonio_embs, localizacao_embs]
            if hasattr(embeddings, 'module'):
                emb_tensor = embeddings.module.weight
            else:
                emb_tensor = embeddings.weight
            
            # ==============================================================================
            # AÑADIR NORMALIZACIÓN: Escala los vectores para que tengan longitud 1
            # QUÉ HACE: Normaliza los embeddings para estabilizar el entrenamiento
            # POR QUÉ: Evita que magnitudes extremas desestabilicen el decodificador
            # ==============================================================================
            import torch.nn.functional as F
            emb_tensor = F.normalize(emb_tensor, p=2, dim=1)
            # ==============================================================================
            
            if is_src_type:
                # Source nodes (patrimonio) are first
                return emb_tensor[node_ids]
            else:
                # Target nodes (localizacao) come after patrimonio nodes
                return emb_tensor[num_patrimonio + node_ids]
        else:
            # Regular homogeneous embeddings
            # ==============================================================================
            # AÑADIR NORMALIZACIÓN para grafos homogéneos también
            # ==============================================================================
            import torch.nn.functional as F
            if hasattr(embeddings, 'weight'):
                emb_tensor = F.normalize(embeddings.weight, p=2, dim=1)
                return emb_tensor[node_ids]
            else:
                emb = embeddings(node_ids)
                return F.normalize(emb, p=2, dim=1)
            # ==============================================================================
    
    for model_type in valid_models:
        decoder = decoder_zoo.get_model(model_type, embeddings.embedding_dim).to(device)
        
        # ==============================================================================
        # CAMBIO CRÍTICO: Usar el learning rate específico del decodificador
        # QUÉ HACE: Cambia lr=0.01 hardcodeado por FLAGS.link_mlp_lr configurable
        # POR QUÉ: Permite ajuste fino. Un lr alto en el decodificador causa colapso
        # ==============================================================================
        optimizer = torch.optim.Adam(decoder.parameters(), lr=FLAGS.link_mlp_lr)
        criterion = torch.nn.BCEWithLogitsLoss()
        
        for epoch in range(FLAGS.decoder_epochs):  # Use configurable decoder epochs
            decoder.train()
            optimizer.zero_grad()
            
            # Positive edges - handle HeteroData mapping
            if is_hetero:
                pos_embeddings = torch.cat([
                    get_node_embeddings(train_edge[:, 0], is_src_type=True),
                    get_node_embeddings(train_edge[:, 1], is_src_type=False)
                ], dim=1)
            else:
                pos_embeddings = torch.cat([
                    embeddings(train_edge[:, 0]), 
                    embeddings(train_edge[:, 1])
                ], dim=1)
            pos_pred = decoder(pos_embeddings)
            
            # Negative edges (sample random) - CORRECCIÓN: usar muestreo bipartito-consciente
            if is_bipartite:
                # ==============================================================================
                # CAMBIO 4: Muestreo negativo bipartito en el entrenamiento del decodificador
                # QUÉ HACE: Si el grafo es bipartito, le pasa el número de nodos de cada
                # tipo a `negative_sampling`.
                # POR QUÉ: Para que el decodificador aprenda a distinguir aristas positivas
                # de negativas válidas (pares src-dst no conectados).
                # ==============================================================================
                # Verificar si hay grafo completo para muestreo inductivo correcto
                if hasattr(data, 'full_edge_index'):
                    # log.info("Usando muestreo negativo inductivo correcto durante entrenamiento del decoder")
                    if is_hetero:
                        # For HeteroData, extract the edge index from the main edge type
                        edge_type = ('patrimonio', 'located_at', 'localizacao')
                        full_edge_index = data[edge_type].edge_index
                        neg_edge_index = bipartite_negative_sampling_inductive(full_edge_index, data, train_edge.size(0))
                    else:
                        neg_edge_index = bipartite_negative_sampling_inductive(data.full_edge_index, data, train_edge.size(0))
                    neg_edges = neg_edge_index.t().to(device)  # Ensure correct device and format
                else:
                    # log.warning("Grafo completo no disponible. Usando muestreo bipartito estándar en decoder.")
                    if is_hetero:
                        edge_type = ('patrimonio', 'located_at', 'localizacao')
                        edge_index = data[edge_type].edge_index
                        num_nodes_tuple = (data['patrimonio'].num_nodes, data['localizacao'].num_nodes)
                        neg_edge_index = negative_sampling(
                            edge_index=edge_index,
                            num_nodes=num_nodes_tuple,
                            num_neg_samples=train_edge.size(0)
                        )
                    else:
                        neg_edge_index = bipartite_negative_sampling(train_edge.t(), data, train_edge.size(0))
                    neg_edges = neg_edge_index.t().to(device)  # Ensure correct device and format
            else:
                # Para grafos no bipartitos, usar negative_sampling estándar de PyG
                neg_edge_index = negative_sampling(
                    edge_index=train_edge.t(),
                    num_nodes=data.num_nodes,
                    num_neg_samples=train_edge.size(0),
                    method='sparse'
                )
                neg_edges = neg_edge_index.t().to(device)
            
            # Get negative embeddings with proper mapping
            if is_hetero:
                neg_embeddings = torch.cat([
                    get_node_embeddings(neg_edges[:, 0], is_src_type=True),
                    get_node_embeddings(neg_edges[:, 1], is_src_type=False)
                ], dim=1)
            else:
                neg_embeddings = torch.cat([
                    embeddings(neg_edges[:, 0]),
                    embeddings(neg_edges[:, 1])
                ], dim=1)
            neg_pred = decoder(neg_embeddings)
            
            # Loss
            pos_loss = criterion(pos_pred.squeeze(), torch.ones(pos_pred.size(0), device=device))
            neg_loss = criterion(neg_pred.squeeze(), torch.zeros(neg_pred.size(0), device=device))
            loss = pos_loss + neg_loss
            
            loss.backward()
            optimizer.step()
            
            # Log progress cada 10 épocas
            if epoch % 10 == 0:
                print(f"[{model_type.upper()} Decoder] Época {epoch}/{FLAGS.decoder_epochs} - Loss: {loss.item():.6f}")
        
        # Evaluate
        decoder.eval()
        with torch.no_grad():
            # Validation - handle HeteroData mapping
            if is_hetero:
                val_pos_embeddings = torch.cat([
                    get_node_embeddings(valid_edge[:, 0], is_src_type=True),
                    get_node_embeddings(valid_edge[:, 1], is_src_type=False)
                ], dim=1)
                val_neg_embeddings = torch.cat([
                    get_node_embeddings(valid_edge_neg[:, 0], is_src_type=True),
                    get_node_embeddings(valid_edge_neg[:, 1], is_src_type=False)
                ], dim=1)
            else:
                val_pos_embeddings = torch.cat([
                    embeddings(valid_edge[:, 0]),
                    embeddings(valid_edge[:, 1])
                ], dim=1)
                val_neg_embeddings = torch.cat([
                    embeddings(valid_edge_neg[:, 0]),
                    embeddings(valid_edge_neg[:, 1])
                ], dim=1)
            
            val_pos_pred = decoder.predict(val_pos_embeddings).squeeze()
            val_neg_pred = decoder.predict(val_neg_embeddings).squeeze()
            val_results = eval_all(val_pos_pred, val_neg_pred)
            
            # Test - handle HeteroData mapping
            if is_hetero:
                test_pos_embeddings = torch.cat([
                    get_node_embeddings(test_edge[:, 0], is_src_type=True),
                    get_node_embeddings(test_edge[:, 1], is_src_type=False)
                ], dim=1)
                test_neg_embeddings = torch.cat([
                    get_node_embeddings(test_edge_neg[:, 0], is_src_type=True),
                    get_node_embeddings(test_edge_neg[:, 1], is_src_type=False)
                ], dim=1)
            else:
                test_pos_embeddings = torch.cat([
                    embeddings(test_edge[:, 0]),
                    embeddings(test_edge[:, 1])
                ], dim=1)
                test_neg_embeddings = torch.cat([
                    embeddings(test_edge_neg[:, 0]),
                    embeddings(test_edge_neg[:, 1])
                ], dim=1)
            
            test_pos_pred = decoder.predict(test_pos_embeddings).squeeze()
            test_neg_pred = decoder.predict(test_neg_embeddings).squeeze()
            test_results = eval_all(test_pos_pred, test_neg_pred)
            
            # DEBUG: Verificar tamaños antes de exportar CSV
            log.info(f"DEBUG CSV Export:")
            log.info(f"  test_edge shape: {test_edge.shape}")
            log.info(f"  test_edge_neg shape: {test_edge_neg.shape}")
            log.info(f"  test_pos_pred shape: {test_pos_pred.shape}")
            log.info(f"  test_neg_pred shape: {test_neg_pred.shape}")
            
            if test_edge_neg.shape[0] == 0:
                log.error("❌ PROBLEMA: test_edge_neg está vacío!")
            elif test_edge_neg.shape[0] < test_edge.shape[0]:
                log.warning(f"⚠️ PROBLEMA: Pocos enlaces negativos {test_edge_neg.shape[0]} vs {test_edge.shape[0]} positivos")
            
            # DEBUG: Verificar datos antes de exportar CSV
            log.info(f"DEBUG - Datos para CSV:")
            log.info(f"  - test_edge shape: {test_edge.shape}")
            log.info(f"  - test_edge_neg shape: {test_edge_neg.shape}")
            log.info(f"  - test_pos_pred shape: {test_pos_pred.shape}")
            log.info(f"  - test_neg_pred shape: {test_neg_pred.shape}")
            if test_pos_pred.numel() > 0:
                log.info(f"  - test_pos_pred sample: {test_pos_pred[:min(5, test_pos_pred.shape[0])].cpu().numpy()}")
            if test_neg_pred.numel() > 0:
                log.info(f"  - test_neg_pred sample: {test_neg_pred[:min(5, test_neg_pred.shape[0])].cpu().numpy()}")
            
            # Export test results to CSV
            export_test_results_csv(
                test_edge, test_edge_neg, 
                test_pos_pred, test_neg_pred, 
                output_dir, model_type
            )
        
        result = {
            'type': model_type,
            'val': val_results,
            'test': test_results
        }
        results.append(result)
        
        # Log to local logger
        for split, metrics in [('val', val_results), ('test', test_results)]:
            for metric, value in metrics.items():
                wandb_logger.log({f'{model_type}_{split}_{metric}': value})
    
    return results, None
