import torch
from src.lib.split import bipartite_negative_sampling_inductive

# Test device compatibility
print("=== DEVICE TEST ===")

# Create test data
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create a mock full_edge_index
full_edge_index = torch.randint(0, 100, (2, 200)).to(device)
print(f"full_edge_index device: {full_edge_index.device}")

# Create mock data object
class MockData:
    def __init__(self):
        self.num_nodes_type_1 = 50
        self.num_nodes = 100

data = MockData()

# Test the function
try:
    neg_edges = bipartite_negative_sampling_inductive(full_edge_index, data, 100)
    print(f"neg_edges device: {neg_edges.device}")
    print(f"neg_edges shape: {neg_edges.shape}")
    print("✓ Function works correctly!")
except Exception as e:
    print(f"✗ Error: {e}")
