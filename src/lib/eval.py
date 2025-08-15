import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.utils import negative_sampling
import logging
from absl import flags

# Import bipartite-aware negative sampling
from .split import bipartite_negative_sampling, bipartite_negative_sampling_inductive

log = logging.getLogger(__name__)
FLAGS = flags.FLAGS

def eval_all(y_pred_pos, y_pred_neg):
    """Evaluate link prediction performance"""
    y_pred = torch.cat([y_pred_pos, y_pred_neg], dim=0).cpu().numpy()
    y_true = np.concatenate([np.ones(len(y_pred_pos)), np.zeros(len(y_pred_neg))])
    
    auc = roc_auc_score(y_true, y_pred)
    ap = average_precision_score(y_true, y_pred)
    
    # Calculate hits@k
    hits_10 = eval_hits_at_k(y_pred_pos, y_pred_neg, k=10)
    hits_50 = eval_hits_at_k(y_pred_pos, y_pred_neg, k=50)
    hits_100 = eval_hits_at_k(y_pred_pos, y_pred_neg, k=100)
    
    return {
        'auc': auc,
        'ap': ap,
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

def do_all_eval(model_name, output_dir, valid_models, dataset, edge_split, 
                embeddings, decoder_zoo, wandb_logger):
    """Perform evaluation on all decoder models"""
    results = []
    
    device = next(embeddings.parameters()).device
    train_edge = edge_split['train']['edge'].to(device)
    valid_edge = edge_split['valid']['edge'].to(device)
    test_edge = edge_split['test']['edge'].to(device)
    valid_edge_neg = edge_split['valid']['edge_neg'].to(device)
    test_edge_neg = edge_split['test']['edge_neg'].to(device)
    
    # Get the original data object for bipartite information
    if hasattr(dataset, 'num_nodes'):
        data = dataset
    else:
        data = dataset[0]
    
    is_bipartite = hasattr(data, 'num_nodes_type_1')
    if is_bipartite:
        log.info("Evaluación bipartita detectada. Usando muestreo negativo bipartito.")
    
    for model_type in valid_models:
        decoder = decoder_zoo.get_model(model_type, embeddings.embedding_dim).to(device)
        
        # Train decoder
        optimizer = torch.optim.Adam(decoder.parameters(), lr=0.01)
        criterion = torch.nn.BCEWithLogitsLoss()
        
        for epoch in range(FLAGS.decoder_epochs):  # Use configurable decoder epochs
            decoder.train()
            optimizer.zero_grad()
            
            # Positive edges
            pos_embeddings = torch.cat([
                embeddings(train_edge[:, 0]), 
                embeddings(train_edge[:, 1])
            ], dim=1)
            pos_pred = decoder(pos_embeddings)
            
            # Negative edges (sample random) - CORRECCIÓN: usar muestreo bipartito-consciente
            if is_bipartite:
                # Verificar si hay grafo completo para muestreo inductivo correcto
                if hasattr(data, 'full_edge_index'):
                    log.info("Usando muestreo negativo inductivo correcto durante entrenamiento del decoder")
                    neg_edge_index = bipartite_negative_sampling_inductive(data.full_edge_index, data, train_edge.size(0))
                    neg_edges = neg_edge_index.t().to(device)  # Ensure correct device and format
                else:
                    log.warning("Grafo completo no disponible. Usando muestreo bipartito estándar en decoder.")
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
        
        # Evaluate
        decoder.eval()
        with torch.no_grad():
            # Validation
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
            
            # Test
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
