import torch

print("=== VERIFICANDO DATASETS INDUCTIVOS SISMETRO ===")

# Cargar metadatos
metadata = torch.load('data/processed/sismetro_inductive_metadata.pt')
print('\n=== METADATOS ===')
for key, value in metadata.items():
    if isinstance(value, torch.Tensor):
        print(f'{key}: tensor shape {value.shape}')
    else:
        print(f'{key}: {value}')

# Verificar train data
train_data = torch.load('data/processed/sismetro_inductive_train.pt')
print('\n=== TRAIN DATA ===')
print(f'edge_index shape: {train_data.edge_index.shape}')
print(f'x shape: {train_data.x.shape}')
print(f'Has full_edge_index: {hasattr(train_data, "full_edge_index")}')
if hasattr(train_data, 'full_edge_index'):
    print(f'full_edge_index shape: {train_data.full_edge_index.shape}')

# Verificar val data
val_data = torch.load('data/processed/sismetro_inductive_val.pt')
print('\n=== VAL DATA ===')
print(f'edge_index shape: {val_data.edge_index.shape}')
print(f'x shape: {val_data.x.shape}')
print(f'Has full_edge_index: {hasattr(val_data, "full_edge_index")}')
if hasattr(val_data, 'full_edge_index'):
    print(f'full_edge_index shape: {val_data.full_edge_index.shape}')

# Verificar test data
test_data = torch.load('data/processed/sismetro_inductive_test.pt')
print('\n=== TEST DATA ===')
print(f'edge_index shape: {test_data.edge_index.shape}')
print(f'x shape: {test_data.x.shape}')
print(f'Has full_edge_index: {hasattr(test_data, "full_edge_index")}')
if hasattr(test_data, 'full_edge_index'):
    print(f'full_edge_index shape: {test_data.full_edge_index.shape}')

print('\n=== VERIFICACIÓN COMPLETADA ===')
