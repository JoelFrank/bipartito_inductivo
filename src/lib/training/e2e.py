import os, torch, logging
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from tqdm import tqdm
from ..local_logger import wandb
from torch_geometric.utils import negative_sampling
from ..eval import eval_all
from ..models.decoders import MlpProdDecoder
from absl import flags

FLAGS = flags.FLAGS
log = logging.getLogger(__name__)

def train_epoch(model, predictor, optimizer, training_data, criterion):
    model.train()
    predictor.train()
    optimizer.zero_grad()
    train_edge = training_data.edge_index
    neg_edge_index = negative_sampling(train_edge, num_nodes=training_data.num_nodes, num_neg_samples=train_edge.size(1), method='sparse')
    edge_label_index = torch.cat([train_edge, neg_edge_index], dim=-1)
    edge_label = torch.cat([train_edge.new_ones(train_edge.size(1)), train_edge.new_zeros(neg_edge_index.size(1))], dim=0)
    
    model_out = model(training_data)
    edge_embeddings = model_out[edge_label_index]
    combined = torch.cat([edge_embeddings[0], edge_embeddings[1]], dim=1)
    out = predictor(combined)
    loss = criterion(out.view(-1), edge_label.float())
    loss.backward()
    optimizer.step()
    return loss

@torch.no_grad()
def eval_model(model, predictor, data, eval_edge, eval_edge_neg):
    model.eval()
    predictor.eval()
    n_edges = eval_edge.size(1)
    edge_label_index = torch.cat([eval_edge, eval_edge_neg], dim=-1)
    
    model_out = model(data)
    edge_embeddings = model_out[edge_label_index]
    combined = torch.cat([edge_embeddings[0], edge_embeddings[1]], dim=1)
    out = predictor.predict(combined).view(-1)
    y_pred_pos, y_pred_neg = out[:n_edges], out[n_edges:]
    return eval_all(y_pred_pos, y_pred_neg)

def perform_e2e_transductive_training(model_name, data, edge_split, representation_size, device, input_size, has_features, g_zoo, **kwargs):
    model = g_zoo.get_model(FLAGS.graph_encoder_model, input_size, has_features, data.num_nodes).to(device)
    predictor = MlpProdDecoder(representation_size, hidden_size=FLAGS.link_mlp_hidden_size).to(device)
    optimizer = Adam(list(model.parameters()) + list(predictor.parameters()), lr=FLAGS.lr)
    criterion = BCEWithLogitsLoss()
    
    valid_edge, test_edge = edge_split['valid']['edge'].T.to(device), edge_split['test']['edge'].T.to(device)
    valid_edge_neg, test_edge_neg = edge_split['valid']['edge_neg'].T.to(device), edge_split['test']['edge_neg'].T.to(device)
    
    best_val_hits, best_test_res = -1, None
    for epoch in tqdm(range(1, FLAGS.epochs + 1), desc="E2E Training"):
        loss = train_epoch(model, predictor, optimizer, data, criterion)
        if epoch % 10 == 0:
            val_res = eval_model(model, predictor, data, valid_edge, valid_edge_neg)
            if val_res['hits@50'] > best_val_hits:
                best_val_hits = val_res['hits@50']
                best_test_res = eval_model(model, predictor, data, test_edge, test_edge_neg)
                wandb.log({f'best_e2e_test_{k}': v for k, v in best_test_res.items()})
        wandb.log({'e2e_train_loss': loss.item(), 'epoch': epoch})

    log.info(f"E2E Best Test Results: {best_test_res}")
    results = {'type': 'e2e', 'val': {'hits@50': best_val_hits}, 'test': best_test_res}
    return model, predictor, [results]
