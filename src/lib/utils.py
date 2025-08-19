import torch
import torch.nn as nn
import copy

class EmbeddingProcessor(nn.Module):
    """
    Procesa características de embedding para nodos.
    Convierte IDs de embedding a vectors densos.
    """
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
    
    def forward(self, x):
        """
        x: tensor de shape (num_nodes, 1) con IDs de embedding
        Retorna: tensor de shape (num_nodes, embedding_dim)
        """
        if x.dim() == 2 and x.size(1) == 1:
            # x contiene IDs de embedding
            # DEBUG: Verificar índices antes del clamp
            x_squeezed = x.squeeze(1)
            max_val = x_squeezed.max().item()
            min_val = x_squeezed.min().item()
            
            if max_val >= self.num_embeddings:
                print(f"❌ ERROR DETECTADO: Índice {max_val} >= num_embeddings {self.num_embeddings}")
                print(f"   Rango de índices en batch: [{min_val}, {max_val}]")
                print(f"   Valores únicos: {torch.unique(x_squeezed).tolist()}")
                print(f"   Num embeddings esperado: {self.num_embeddings} (rango válido: 0-{self.num_embeddings-1})")
            
            # Manejar valores -1 (sin tipo) convirtiéndolos a un índice válido
            x_clean = torch.clamp(x_squeezed, 0, self.num_embeddings - 1)
            return self.embedding(x_clean)
        else:
            # x ya son características dense (compatibilidad)
            return x

@torch.no_grad()
def compute_data_representations_only(encoder, data, device, has_features):
    """Compute node representations using the encoder"""
    encoder.eval()
    if hasattr(data, 'x') and data.x is not None:
        # Data has features
        return encoder(data).detach().cpu()
    else:
        # No features, use node embeddings from encoder
        return encoder(data).detach().cpu()

def print_run_num(run_num):
    """Print current run number"""
    print(f"Starting run {run_num + 1}")

def merge_multirun_results(all_results):
    """Merge results from multiple runs"""
    if not all_results:
        return {}, {}
    
    # Extract metrics from all runs
    metrics = {}
    for results in all_results:
        for result in results:
            for split in ['val', 'test']:
                if split in result:
                    for metric, value in result[split].items():
                        key = f"{split}_{metric}"
                        if key not in metrics:
                            metrics[key] = []
                        metrics[key].append(value)
    
    # Compute mean and std
    agg_results = {}
    to_log = {}
    for key, values in metrics.items():
        mean_val = sum(values) / len(values)
        std_val = (sum((x - mean_val) ** 2 for x in values) / len(values)) ** 0.5
        agg_results[f"{key}_mean"] = mean_val
        agg_results[f"{key}_std"] = std_val
        to_log[f"final_{key}_mean"] = mean_val
        to_log[f"final_{key}_std"] = std_val
    
    return agg_results, to_log
