import logging, os, time, json, sys
from absl import app, flags
import torch
from lib.local_logger import wandb
from lib.logger import setup_logger
from lib.models.decoders import DecoderZoo
from lib.models import EncoderZoo
from lib.eval import do_all_eval
from lib.training import perform_bgrl_training, perform_cca_ssg_training, perform_gbt_training, perform_triplet_training
from lib.split import do_transductive_edge_split
from lib.utils import merge_multirun_results, print_run_num
import lib.flags as FlagHelper

FLAGS = flags.FLAGS

# Define all flags
FlagHelper.define_flags(FlagHelper.ModelGroup.NCL)

# Additional flag for config file
flags.DEFINE_string('config_file', None, 'Path to configuration file')

def process_config_file():
    """Process config file before main execution"""
    config_processed = False
    if len(sys.argv) > 1:
        # Look for --config_file in command line args
        for i, arg in enumerate(sys.argv):
            if arg == '--config_file' and i + 1 < len(sys.argv):
                config_path = sys.argv[i + 1]
                if os.path.exists(config_path) and not config_processed:
                    # Read config file and insert flags into sys.argv
                    with open(config_path, 'r') as f:
                        config_flags = []
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith('#'):
                                if '=' in line:
                                    key, value = line.split('=', 1)
                                    if not key.startswith('--'):
                                        key = '--' + key
                                    config_flags.extend([key, value])
                                elif line.startswith('--'):
                                    parts = line.split(' ', 1)
                                    if len(parts) == 2:
                                        config_flags.extend(parts)
                                    else:
                                        config_flags.append(line)
                    
                    # Insert config flags before the config_file argument, remove duplicates
                    sys.argv = [sys.argv[0]] + config_flags + sys.argv[i+2:]  # Skip config_file args
                    config_processed = True
                break

# Process config file before flag parsing
process_config_file()

def get_full_model_name():
    return f"Bipartite_{FLAGS.base_model}_{FLAGS.dataset}_lr{FLAGS.lr}"

def main(_):
    # Process list flags to convert strings to integers
    if isinstance(FLAGS.graph_encoder_layer_dims, list):
        FLAGS.graph_encoder_layer_dims = [int(x) for x in FLAGS.graph_encoder_layer_dims]
    if isinstance(FLAGS.negative_transforms, str):
        FLAGS.negative_transforms = [FLAGS.negative_transforms]
    
    # Initialize wandb
    wandb.init(
        project='bipartite-ncl-link-prediction', 
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
    setup_logger(OUTPUT_DIR, "train_nc_log")
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
    log.info(f'Graph loaded: {data.num_nodes} nodes, {data.edge_index.shape[1]} edges in {time.time() - st_time:.2f}s')

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f'Using device: {device}')

    # Perform edge split
    if FLAGS.split_method == 'transductive':
        edge_split = do_transductive_edge_split(dataset, split_seed=FLAGS.split_seed)
        # Assign training edges to data object for encoder training
        data.edge_index = edge_split['train']['edge'].t()
        training_data = data.to(device)
    elif FLAGS.split_method == 'inductive':
        log.info("Loading inductive split datasets...")
        # Load inductive split datasets
        train_path = os.path.join(FLAGS.dataset_dir, 'processed', f'{FLAGS.dataset}_inductive_train.pt')
        val_path = os.path.join(FLAGS.dataset_dir, 'processed', f'{FLAGS.dataset}_inductive_val.pt')
        test_path = os.path.join(FLAGS.dataset_dir, 'processed', f'{FLAGS.dataset}_inductive_test.pt')
        
        if not all(os.path.exists(path) for path in [train_path, val_path, test_path]):
            log.error(f"Inductive split files not found. Required: {train_path}, {val_path}, {test_path}")
            return
        
        train_data = torch.load(train_path)
        val_data = torch.load(val_path)
        test_data = torch.load(test_path)
        
        # LIMPIEZA: Deduplicar enlaces en los datos cargados
        log.info("Limpiando duplicados en datos inductivos...")
        
        def clean_data_edges(data, name):
            original_count = data.edge_index.shape[1]
            # Convert to [num_edges, 2] format for cleaning
            edges_2d = data.edge_index.t()
            
            # Remove duplicates
            edge_set = set()
            clean_edges = []
            
            for edge in edges_2d:
                edge_tuple = tuple(sorted([edge[0].item(), edge[1].item()]))
                if edge_tuple not in edge_set:
                    edge_set.add(edge_tuple)
                    # Always store in consistent direction (smaller -> larger for undirected)
                    clean_edges.append([edge[0].item(), edge[1].item()])
            
            # Update data object
            if clean_edges:
                data.edge_index = torch.tensor(clean_edges).t().long()
                log.info(f"  {name}: {original_count} -> {len(clean_edges)} enlaces únicos")
            else:
                log.error(f"  ❌ {name}: No se encontraron enlaces válidos")
        
        clean_data_edges(train_data, "Train")
        clean_data_edges(val_data, "Val") 
        clean_data_edges(test_data, "Test")
        
        # Cargar metadatos para obtener el grafo completo (muestreo negativo inductivo)
        metadata_path = os.path.join(FLAGS.dataset_dir, 'processed', f'{FLAGS.dataset}_metadata.pt')
        if os.path.exists(metadata_path):
            metadata = torch.load(metadata_path)
            if 'full_edge_index' in metadata:
                # Agregar el grafo completo a cada split para muestreo negativo correcto
                train_data.full_edge_index = metadata['full_edge_index']
                val_data.full_edge_index = metadata['full_edge_index']
                test_data.full_edge_index = metadata['full_edge_index']
                log.info(f"✓ Grafo completo cargado desde metadatos: {metadata['full_edge_index'].size(1)} aristas")
            else:
                log.warning("Grafo completo no encontrado en metadatos. Usando muestreo estándar.")
        else:
            log.warning(f"Archivo de metadatos no encontrado: {metadata_path}")
        
        log.info(f"Train data: {train_data.num_nodes} nodes, {train_data.edge_index.shape[1]} edges")
        log.info(f"Val data: {val_data.num_nodes} nodes, {val_data.edge_index.shape[1]} edges")
        log.info(f"Test data: {test_data.num_nodes} nodes, {test_data.edge_index.shape[1]} edges")
        
        # For inductive learning, we use the train_data for training
        training_data = train_data.to(device)
        
        # Move full_edge_index to the correct device if it exists
        if hasattr(training_data, 'full_edge_index'):
            training_data.full_edge_index = training_data.full_edge_index.to(device)
            log.info(f"✓ full_edge_index moved to device: {device}")
        if hasattr(val_data, 'full_edge_index'):
            val_data.full_edge_index = val_data.full_edge_index.to(device)
        if hasattr(test_data, 'full_edge_index'):
            test_data.full_edge_index = test_data.full_edge_index.to(device)
        
        # Create edge split structure for compatibility
        edge_split = {
            'train': {'edge': train_data.edge_index.t()},
            'valid': {'edge': val_data.edge_index.t()}, 
            'test': {'edge': test_data.edge_index.t()}
        }
        
        # Generate negative edges for evaluation
        log.info("Generando enlaces negativos para evaluación...")
        from lib.split import generate_neg_edges
        
        log.info(f"Val data: {val_data.edge_index.shape[1]} enlaces positivos")
        # CORRECCIÓN: usar edge_index original (formato [2, num_edges]) para generate_neg_edges
        edge_split['valid']['edge_neg'] = generate_neg_edges(
            val_data.edge_index, val_data.num_nodes, 
            num_neg=val_data.edge_index.shape[1], data=val_data
        ).t()  # Transponer resultado para mantener consistencia con edge_split format [num_edges, 2]
        log.info(f"✓ Val negativos generados usando full_edge_index: {edge_split['valid']['edge_neg'].shape}")
        
        log.info(f"Test data: {test_data.edge_index.shape[1]} enlaces positivos")
        # CORRECCIÓN: usar edge_index original (formato [2, num_edges]) para generate_neg_edges
        edge_split['test']['edge_neg'] = generate_neg_edges(
            test_data.edge_index, test_data.num_nodes, 
            num_neg=test_data.edge_index.shape[1], data=test_data
        ).t()  # Transponer resultado para mantener consistencia con edge_split format [num_edges, 2]
        log.info(f"✓ Test negativos generados usando full_edge_index: {edge_split['test']['edge_neg'].shape}")
        
        # Diagnóstico adicional
        log.info(f"Diagnóstico de enlaces negativos:")
        log.info(f"  - Positivos val: {edge_split['valid']['edge'].shape}")
        log.info(f"  - Negativos val: {edge_split['valid']['edge_neg'].shape}")
        log.info(f"  - Positivos test: {edge_split['test']['edge'].shape}")
        log.info(f"  - Negativos test: {edge_split['test']['edge_neg'].shape}")
        log.info(f"  - Full graph edges: {val_data.full_edge_index.shape[1]}")
        
        # Verificar que los negativos no estén vacíos
        if edge_split['test']['edge_neg'].shape[0] == 0:
            log.error("❌ No se generaron enlaces negativos para test!")
        elif edge_split['test']['edge_neg'].shape[0] < edge_split['test']['edge'].shape[0]:
            log.warning(f"⚠️ Pocos enlaces negativos: {edge_split['test']['edge_neg'].shape[0]} vs {edge_split['test']['edge'].shape[0]} positivos")
        
        # Update data to use training data structure
        data = train_data
    else:
        log.error(f"Unknown split method: {FLAGS.split_method}")
        return

    # Determine input size and features
    has_features = data.x is not None
    input_size = data.x.size(1) if has_features else FLAGS.graph_encoder_layer_dims[0]
    representation_size = FLAGS.graph_encoder_layer_dims[-1]
    
    # Initialize model factories
    enc_zoo = EncoderZoo(FLAGS)
    dec_zoo = DecoderZoo(FLAGS)

    # Training loop for multiple runs
    all_results = []
    for run_num in range(FLAGS.num_runs):
        print_run_num(run_num)
        
        # Select training function based on base model
        training_function_map = {
            'bgrl': perform_bgrl_training,
            'cca': perform_cca_ssg_training,
            'gbt': perform_gbt_training,
            'triplet': perform_triplet_training,
        }
        
        train_func = training_function_map.get(FLAGS.base_model)
        if not train_func:
            raise NotImplementedError(f"Base model not supported: {FLAGS.base_model}")
        
        # Train encoder
        encoder, representations, _ = train_func(
            data=training_data,
            output_dir=OUTPUT_DIR,
            representation_size=representation_size,
            device=device,
            input_size=input_size,
            has_features=has_features,
            g_zoo=enc_zoo
        )

        log.info("Encoder training finished. Evaluating decoder...")
        
        if FLAGS.split_method == 'inductive':
            # For inductive evaluation, generate fresh embeddings using the trained encoder
            # All data splits already contain all nodes (2798), so we can use any of them
            # We use train_data structure but the encoder was trained on its edge pattern
            log.info("Generando embeddings para evaluación inductiva...")
            
            encoder.eval()
            with torch.no_grad():
                # Generate embeddings for all nodes using train_data structure
                # (all splits have the same nodes, only edges differ)
                all_representations = encoder(training_data)
            
            embeddings = torch.nn.Embedding.from_pretrained(all_representations, freeze=True)
            log.info(f"✓ Embeddings generados para {all_representations.size(0)} nodos")
        else:
            # Transductive case: use the representations from training
            embeddings = torch.nn.Embedding.from_pretrained(representations, freeze=True)
        
        # Evaluate decoder
        if FLAGS.split_method == 'inductive':
            # For inductive evaluation, we need to evaluate on val and test sets
            # Pass train_data which contains the correct bipartite metadata and full_edge_index
            results, _ = do_all_eval(
                get_full_model_name(),
                OUTPUT_DIR,
                [FLAGS.link_pred_model],
                train_data,  # Use training data which has bipartite metadata and full_edge_index
                edge_split,
                embeddings,
                dec_zoo,
                wandb
            )
        else:
            # Transductive evaluation
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

    # Aggregate results across runs
    agg_results, to_log = merge_multirun_results(all_results)
    wandb.log(to_log)
    
    # Save aggregated results
    with open(f'{OUTPUT_DIR}/agg_results.json', 'w') as f:
        json.dump(agg_results, f, indent=4)

    log.info(f'--- TRAINING FINISHED ---\nResults: {agg_results}\nLogs in: {OUTPUT_DIR}')

if __name__ == "__main__":
    process_config_file()
    app.run(main)
