import logging, os, time, json
from absl import app, flags
import torch
from lib.local_logger import wandb
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from lib.logger import setup_logger
from lib.models.decoders import DecoderZoo
from lib.models import EncoderZoo
from lib.training.e2e import perform_e2e_transductive_training
from lib.split import do_transductive_edge_split
from lib.utils import merge_multirun_results, print_run_num
import lib.flags as FlagHelper

FLAGS = flags.FLAGS

# Define flags
FlagHelper.define_flags(FlagHelper.ModelGroup.E2E)

def get_full_model_name():
    return f"Bipartite_E2E_{FLAGS.dataset}_lr{FLAGS.lr}"

def main(_):
    # Initialize wandb
    wandb.init(
        project='bipartite-e2e-link-prediction',
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
    setup_logger(OUTPUT_DIR, "train_e2e_log")
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

    # Setup model parameters
    has_features = data.x is not None
    input_size = data.x.size(1) if has_features else FLAGS.graph_encoder_layer_dims[0]
    representation_size = FLAGS.graph_encoder_layer_dims[-1]
    
    enc_zoo = EncoderZoo(FLAGS)
    all_results = []
    
    for run_num in range(FLAGS.num_runs):
        print_run_num(run_num)
        
        # Train E2E model
        encoder, predictor, results = perform_e2e_transductive_training(
            get_full_model_name(),
            training_data,
            edge_split,
            representation_size,
            device,
            input_size,
            has_features,
            enc_zoo
        )
        
        all_results.extend(results)

    # Aggregate results
    agg_results, to_log = merge_multirun_results(all_results)
    wandb.log(to_log)
    
    with open(f'{OUTPUT_DIR}/agg_results.json', 'w') as f:
        json.dump(agg_results, f, indent=4)

    log.info(f'--- E2E TRAINING FINISHED ---\nResults: {agg_results}\nLogs in: {OUTPUT_DIR}')

if __name__ == "__main__":
    app.run(main)
