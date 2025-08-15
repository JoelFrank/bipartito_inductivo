import torch, logging
import torch.nn.functional as F
from tqdm import tqdm
from ..local_logger import wandb
from torch_geometric.utils import negative_sampling
from torch.optim import AdamW
from ..eval import eval_all
from ..models.decoders import MlpProdDecoder
from absl import flags

FLAGS = flags.FLAGS
log = logging.getLogger(__name__)

def compute_margin_loss(device, n_nodes, edge_index, model_out):
    pos_dists = F.logsigmoid(F.cosine_similarity(model_out[edge_index[0]], model_out[edge_index[1]])).mean()
    neg_edge_index = negative_sampling(edge_index, n_nodes, edge_index.size(1))
    neg_dists = F.logsigmoid(F.cosine_similarity(model_out[neg_edge_index[0]], model_out[neg_edge_index[1]])).mean()
    return torch.clamp(neg_dists - pos_dists + FLAGS.margin, min=0)

def perform_transductive_margin_training(model_name, data, edge_split, representation_size, device, input_size, has_features, g_zoo, **kwargs):
    model = g_zoo.get_model(FLAGS.graph_encoder_model, input_size, has_features, data.num_nodes).to(device)
    optimizer = AdamW(model.parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)
    
    train_edge = edge_split['train']['edge'].T.to(device)
    valid_edge, test_edge = edge_split['valid']['edge'].T.to(device), edge_split['test']['edge'].T.to(device)
    valid_edge_neg, test_edge_neg = edge_split['valid']['edge_neg'].T.to(device), edge_split['test']['edge_neg'].T.to(device)

    # El predictor aquí solo se usa para la evaluación, no se entrena con el GCN
    predictor = MlpProdDecoder(representation_size, hidden_size=FLAGS.link_mlp_hidden_size).to(device)

    best_val_hits, best_test_res = -1, None
    for epoch in tqdm(range(1, FLAGS.epochs + 1), desc="Margin-GCN Training"):
        model.train()
        optimizer.zero_grad()
        model_out = model(data)
        loss = compute_margin_loss(device, data.num_nodes, train_edge, model_out)
        loss.backward()
        optimizer.step()
        wandb.log({'margin_train_loss': loss.item(), 'epoch': epoch})

        if epoch % 10 == 0:
            # Para evaluar, se necesita un decoder entrenado sobre los embeddings congelados
            # Esta parte se simplifica, ya que el pipeline original lo hace en `do_all_eval`
            pass

    log.info(f"Margin-GCN training finished.")
    # La evaluación real se hace fuera, en do_all_eval. Aquí solo se retorna el encoder
    return model, model(data), None
