"""This module does blah blah."""
import json

SEED = 12
NUM_THREADS = 8
WRITE_IMG_SUMMARY_EVERY = 100

with open('config.json', 'r') as f:
    CONFIG_JSON = json.load(f)

CONFIG = CONFIG_JSON[CONFIG_JSON['active_config']]

BASE_PATH = CONFIG['base_path']
DATA_PATH = CONFIG['data_path']
CHKPTS_PATH = CONFIG['chkpts_path']
GRAPHS_PATH = CONFIG['graphs_path']
OUTPUT_PATH = CONFIG['output_path']
FULL_OUTPUT_PATH = BASE_PATH + OUTPUT_PATH

DIM_X = CONFIG['output']['dimX']
DIM_Y = CONFIG['output']['dimY']
DIM_Z = CONFIG['output']['dimZ']

BATCH_SIZE = CONFIG['input']['batch_size']
Z_NOISE_DIM = CONFIG['input']['z_noise_dim']

NUM_EPOCHS = CONFIG['training']['num_epochs']
D_LEARNING_RATE = CONFIG['training']['d_learning_rate']
G_LEARNING_RATE = CONFIG['training']['g_learning_rate']
BETA1 = CONFIG['training']['beta1']

SAVE_MODEL_EVERY = CONFIG['save_model_every']
SAVE_EXAMPLE_EVERY = CONFIG['save_example_every']
PRINT_INFO_EVERY = CONFIG['print_info_every']
