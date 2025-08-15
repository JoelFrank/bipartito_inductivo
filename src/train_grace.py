import logging, os, time, json
from absl import app, flags
import torch
from lib.local_logger import wandb
import torch.nn.functional as F
from lib.logger import setup_logger
from lib.models.decoders import DecoderZoo
from lib.models import EncoderZoo
from lib.eval import do_all_eval
from lib.split import do_transductive_edge_split
from lib.utils import merge_multirun_results, print_run_num, compute_data_representations_only
from lib.data_transforms import compose_transforms
from lib.models.grace import GraceModel, GraceEncoder
import lib.flags as FlagHelper

FLAGS = flags.FLAGS

# Define flags
FlagHelper.define_flags(FlagHelper.ModelGroup.GRACE)
flags.DEFINE_float('tau', 0.5, 'Temperature parameter for GRACE')
flags.DEFINE_integer('proj_hidden_dim', 256, 'Projection hidden dimension')

def get_full_model_name():
    return f"Bipartite_GRACE_{FLAGS.dataset}_lr{FLAGS.lr}"

def train_grace(model, optimizer, x, edge_index, transform_1, transform_2):
    """Train GRACE model for one epoch"""
    model.train()
    optimizer.zero_grad()
    
    # Create two augmented views
    data_1 = torch.clone(torch.zeros(x.size(0), x.size(1)) if x is None else x)
    data_2 = torch.clone(torch.zeros(x.size(0), x.size(1)) if x is None else x)
    
    # Apply transforms (simplified)
    edge_index_1 = edge_index
    edge_index_2 = edge_index
    
    z1 = model(data_1, edge_index_1)
    z2 = model(data_2, edge_index_2)
    
    loss = model.loss(z1, z2)
    loss.backward()
    optimizer.step()
    
    return loss.item()

def main(_):
    # Initialize wandb
    wandb.init(
        project='bipartite-grace-link-prediction',
        config={
            'model_name': get_full_model_name(),
            **FLAGS.flag_values_dict()
        }
    )
    
    if wandb.run is None:
        raise ValueError('Wandb not initialized.')

    model_name_with_id = f'{get_full_model_name()}_{wandb.run.id}'
    OUTPUT_DIR = os.path.join(FLAGS.logdir, model_name_with_id)
    
    # Setup logging
    setup_logger(OUTPUT_DIR, "train_grace_log")
    log = logging.getLogger(__name__)

    # Save configuration
    with open(os.path.join(OUTPUT_DIR, 'config.cfg'), "w") as f:
        f.write(FLAGS.flags_into_string())
    with open(os.path.join(OUTPUT_DIR, 'config.json'), 'w') as f:
        json.dump(FLAGS.flag_values_dict(), f, indent=4)

    # Load dataset
    st_time = time.time()
    dataset_path = os.path.join(FLAGS.dataset_dir, 'processed', f'{FLAGS.dataset}.pt')
    log.info(f"Loading dataset from: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        log.error(f"File not found: {dataset_path}. Run data preparation script first.")
        return
    
    data = torch.load(dataset_path)
    dataset = [data]
    log.info(f'Graph loaded: {data} in {time.time() - st_time:.2f}s')

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f'Using device: {device}')

    # Perform edge split
    if FLAGS.split_method == 'transductive':
        edge_split = do_transductive_edge_split(dataset, split_seed=FLAGS.split_seed)
        data.edge_index = edge_split['train']['edge'].t()
        training_data = data.to(device)
    else:
        log.error("Inductive split not implemented.")
        return

    # Setup model
    has_features = data.x is not None
    input_size = data.x.size(1) if has_features else FLAGS.graph_encoder_layer_dims[0]
    representation_size = FLAGS.graph_encoder_layer_dims[-1]
    
    # Create GRACE encoder
    encoder = GraceEncoder(
        in_channels=input_size,
        out_channels=representation_size,
        activation=F.relu
    )
    
    # Create GRACE model
    model = GraceModel(
        encoder=encoder,
        num_hidden=representation_size,
        num_proj_hidden=FLAGS.proj_hidden_dim,
        tau=FLAGS.tau
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)
    
    # Setup transforms
    transform_1 = compose_transforms(FLAGS.graph_transforms, drop_edge_p=FLAGS.drop_edge_p_1, drop_feat_p=FLAGS.drop_feat_p_1)
    transform_2 = compose_transforms(FLAGS.graph_transforms, drop_edge_p=FLAGS.drop_edge_p_2, drop_feat_p=FLAGS.drop_feat_p_2)
    
    dec_zoo = DecoderZoo(FLAGS)
    all_results = []
    
    for run_num in range(FLAGS.num_runs):
        print_run_num(run_num)
        
        # Training loop
        for epoch in range(1, FLAGS.epochs + 1):
            loss = train_grace(
                model, optimizer, 
                training_data.x, training_data.edge_index,
                transform_1, transform_2
            )
            
            if epoch % 100 == 0:
                wandb.log({'grace_train_loss': loss, 'epoch': epoch})
                log.info(f"Epoch {epoch}, Loss: {loss:.4f}")

        log.info("GRACE training finished. Computing representations...")
        
        # Compute final representations
        model.eval()
        with torch.no_grad():
            if has_features:
                representations = model(training_data.x, training_data.edge_index).cpu()
            else:
                # Create dummy features
                dummy_features = torch.randn(training_data.num_nodes, input_size).to(device)
                representations = model(dummy_features, training_data.edge_index).cpu()
        
        # Create embedding layer
        embeddings = torch.nn.Embedding.from_pretrained(representations, freeze=True)
        
        # Evaluate
        results, _ = do_all_eval(
            get_full_model_name(),
            OUTPUT_DIR,
            [FLAGS.link_pred_model],
            dataset,
            edge_split,
            embeddings,
            dec_zoo,
            wandb
        )
        
        all_results.append(results)

    # Aggregate results
    agg_results, to_log = merge_multirun_results(all_results)
    wandb.log(to_log)
    
    with open(f'{OUTPUT_DIR}/agg_results.json', 'w') as f:
        json.dump(agg_results, f, indent=4)

    log.info(f'--- GRACE TRAINING FINISHED ---\nResults: {agg_results}\nLogs in: {OUTPUT_DIR}')

if __name__ == "__main__":
    app.run(main)
