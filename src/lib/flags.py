from absl import flags
from enum import Enum

class ModelGroup(Enum):
    NCL = 'ncl'
    GRACE = 'grace'
    E2E = 'e2e'
    MARGIN = 'margin'

def define_flags(model_group: ModelGroup):
    """Define flags for different model groups"""
    
    # Common flags
    flags.DEFINE_string('dataset', 'my-bipartite-dataset', 'Dataset name')
    flags.DEFINE_string('dataset_dir', '../data', 'Dataset directory')
    flags.DEFINE_string('logdir', '../runs', 'Log directory')
    flags.DEFINE_enum('split_method', 'transductive', ['transductive', 'inductive'], 'Split method')
    flags.DEFINE_integer('split_seed', 42, 'Seed for data splitting')
    flags.DEFINE_boolean('inductive', False, 'Use inductive learning setup')
    flags.DEFINE_boolean('temporal_split', False, 'Use temporal split for inductive learning')
    
    # Training flags
    flags.DEFINE_float('lr', 0.005, 'Learning rate')
    flags.DEFINE_float('link_mlp_lr', 0.001, 'Learning rate for link prediction MLP decoder')
    flags.DEFINE_integer('epochs', 5000, 'Number of epochs')
    flags.DEFINE_integer('lr_warmup_epochs', 500, 'Learning rate warmup epochs')
    flags.DEFINE_integer('decoder_epochs', 100, 'Number of epochs for decoder training')
    flags.DEFINE_float('weight_decay', 1e-5, 'Weight decay')
    flags.DEFINE_integer('num_runs', 1, 'Number of runs')
    
    # Model architecture flags
    flags.DEFINE_enum('graph_encoder_model', 'gcn', ['gcn', 'bipartite_sage'], 'Graph encoder model')
    flags.DEFINE_list('graph_encoder_layer_dims', [128, 128], 'Graph encoder layer dimensions')
    flags.DEFINE_integer('link_mlp_hidden_size', 128, 'Link MLP hidden size')
    flags.DEFINE_integer('predictor_hidden_size', 256, 'Predictor hidden size')
    flags.DEFINE_enum('link_pred_model', 'prod_mlp', ['prod_mlp', 'concat_mlp'], 'Link prediction model')
    flags.DEFINE_boolean('adjust_layer_sizes', False, 'Adjust layer sizes')
    flags.DEFINE_boolean('batch_links', False, 'Batch links')
    flags.DEFINE_boolean('trivial_neg_sampling', False, 'Trivial negative sampling')
    
    # Data augmentation flags
    flags.DEFINE_list('graph_transforms', ['drop-edges', 'drop-features'], 'Graph transforms')
    flags.DEFINE_float('drop_edge_p_1', 0.2, 'Drop edge probability 1')
    flags.DEFINE_float('drop_feat_p_1', 0.2, 'Drop feature probability 1')
    flags.DEFINE_float('drop_edge_p_2', 0.3, 'Drop edge probability 2')
    flags.DEFINE_float('drop_feat_p_2', 0.3, 'Drop feature probability 2')
    
    if model_group == ModelGroup.NCL:
        # NCL specific flags
        flags.DEFINE_enum('base_model', 'bgrl', ['bgrl', 'gbt', 'triplet', 'cca'], 'Base model')
        flags.DEFINE_float('mm', 0.99, 'Momentum for moving average')
        flags.DEFINE_float('neg_lambda', 0.5, 'Negative lambda for triplet loss')
        flags.DEFINE_list('negative_transforms', ['random-edges'], 'Negative transforms')
        flags.DEFINE_float('cca_lambda', 0.01, 'CCA lambda parameter')
    
    elif model_group == ModelGroup.GRACE:
        # GRACE specific flags
        flags.DEFINE_float('tau', 0.5, 'Temperature parameter')
        flags.DEFINE_integer('proj_hidden_dim', 256, 'Projection hidden dimension')
    
    elif model_group == ModelGroup.MARGIN:
        # Margin specific flags
        flags.DEFINE_float('margin', 0.5, 'Margin for triplet loss')
    
    # Convert list flags to proper format
    def process_list_flags():
        import ast
        if hasattr(flags.FLAGS, 'graph_encoder_layer_dims'):
            if isinstance(flags.FLAGS.graph_encoder_layer_dims, list):
                flags.FLAGS.graph_encoder_layer_dims = [int(x) for x in flags.FLAGS.graph_encoder_layer_dims]
