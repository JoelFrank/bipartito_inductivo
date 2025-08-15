import torch
import copy

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
