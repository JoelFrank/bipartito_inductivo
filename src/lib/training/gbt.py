import copy, os, torch, logging
from torch.optim import AdamW
from tqdm import tqdm
from ..local_logger import wandb
from ..loss import barlow_twins_loss
from ..utils import compute_data_representations_only
from ..data_transforms import compose_transforms
from ..models import GraphBarlowTwins
from absl import flags

FLAGS = flags.FLAGS
log = logging.getLogger(__name__)

def perform_gbt_training(data, output_dir, device, input_size: int, has_features: bool, g_zoo, **kwargs):
    transform_1 = compose_transforms(FLAGS.graph_transforms, drop_edge_p=FLAGS.drop_edge_p_1, drop_feat_p=FLAGS.drop_feat_p_1)
    transform_2 = compose_transforms(FLAGS.graph_transforms, drop_edge_p=FLAGS.drop_edge_p_2, drop_feat_p=FLAGS.drop_feat_p_2)
    
    encoder = g_zoo.get_model(FLAGS.graph_encoder_model, input_size, has_features, data.num_nodes)
    model = GraphBarlowTwins(encoder, has_features=has_features).to(device)
    optimizer = AdamW(model.trainable_parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)

    for epoch in tqdm(range(1, FLAGS.epochs + 1), desc="GBT Training"):
        model.train()
        optimizer.zero_grad()
        x1, x2 = transform_1(data), transform_2(data)
        y1, y2 = model(x1, x2)
        loss = barlow_twins_loss(y1, y2)
        loss.backward()
        optimizer.step()
        wandb.log({'gbt_train_loss': loss.item(), 'epoch': epoch})

    encoder = copy.deepcopy(model.encoder.eval())
    representations = compute_data_representations_only(encoder, data, device, has_features)
    return encoder, representations, None
