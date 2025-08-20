import copy, os, time, torch, logging
from torch.optim import AdamW
from tqdm import tqdm
from ..local_logger import wandb
import torch.nn.functional as F
from ..scheduler import CosineDecayScheduler
from ..utils import compute_data_representations_only
from ..data_transforms import compose_transforms
from ..models import MlpPredictor, TripletBgrl
from absl import flags

FLAGS = flags.FLAGS
log = logging.getLogger(__name__)

def perform_triplet_training(data, output_dir, representation_size, device, input_size: int, has_features: bool, g_zoo, **kwargs):
    transform_1 = compose_transforms(FLAGS.graph_transforms, drop_edge_p=FLAGS.drop_edge_p_1, drop_feat_p=FLAGS.drop_feat_p_1)
    transform_2 = compose_transforms(FLAGS.graph_transforms, drop_edge_p=FLAGS.drop_edge_p_2, drop_feat_p=FLAGS.drop_feat_p_2)
    transform_3 = compose_transforms(FLAGS.negative_transforms) # Corruptions

    encoder = g_zoo.get_model(FLAGS.graph_encoder_model, input_size, has_features, data.num_nodes, data)
    predictor = MlpPredictor(representation_size, representation_size, hidden_size=FLAGS.predictor_hidden_size)
    model = TripletBgrl(encoder, predictor, has_features=has_features).to(device)
    
    optimizer = AdamW(model.trainable_parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)
    lr_scheduler = CosineDecayScheduler(FLAGS.lr, FLAGS.lr_warmup_epochs, FLAGS.epochs)
    mm_scheduler = CosineDecayScheduler(1 - FLAGS.mm, 0, FLAGS.epochs)
    
    for epoch in tqdm(range(1, FLAGS.epochs + 1), desc="T-BGRL Training"):
        model.train()
        lr = lr_scheduler.get(epoch - 1)
        for param_group in optimizer.param_groups: param_group['lr'] = lr
        mm = 1 - mm_scheduler.get(epoch - 1)
        optimizer.zero_grad()
        
        x1, x2, x3 = transform_1(data), transform_2(data), transform_3(data)
        q1, y2 = model(x1, x2)
        q2, y1 = model(x2, x1)
        neg_y = model.forward_target(x3)

        sim1, sim2 = F.cosine_similarity(q1, y2.detach()).mean(), F.cosine_similarity(q2, y1.detach()).mean()
        neg_sim1, neg_sim2 = F.cosine_similarity(q1, neg_y.detach()).mean(), F.cosine_similarity(q2, neg_y.detach()).mean()
        
        loss = FLAGS.neg_lambda * (neg_sim1 + neg_sim2) - (1 - FLAGS.neg_lambda) * (sim1 + sim2)
        loss.backward()
        optimizer.step()
        model.update_target_network(mm)
        # wandb.log({'tbgrl_train_loss': loss.item(), 'epoch': epoch})  # Comentado para logs más limpios
        
        # Log progress cada 10 épocas
        if epoch % 10 == 0:
            print(f"[T-BGRL Encoder] Época {epoch}/{FLAGS.epochs} - Loss: {loss.item():.6f} - LR: {lr:.6f}")
            # Log a wandb solo cada 10 épocas
            wandb.log({'tbgrl_train_loss': loss.item(), 'epoch': epoch})

    encoder = copy.deepcopy(model.online_encoder.eval())
    representations = compute_data_representations_only(encoder, data, device, has_features)
    return encoder, representations, None
