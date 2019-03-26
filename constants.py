import json

SEED = 1122
NUM_THREADS = 4

BASE_PATH = '/home/ruben/Master/'
DATA_PATH = 'img_align_celeba/'
CHKPTS_PATH = 'checkpoints/'
GRAPHS_PATH = 'graphs/'
OUTPUT_PATH = 'output/'

with open('config.json', 'r') as f:
    config = json.load(f)

CONFIG = config['default_config']

DIM_X = CONFIG['output']['dimX']
DIM_Y = CONFIG['output']['dimY'] 
DIM_Z = CONFIG['output']['dimZ'] 

BATCH_SIZE = CONFIG['input']['batch_size']
Z_NOISE_DIM = CONFIG['input']['batch_size'] 

NUM_EPOCHS = CONFIG['training']['num_epochs']
LEARNING_RATE = CONFIG['training']['learning_rate']
BETA1 = CONFIG['training']['beta1']

SAVE_MODEL_EVERY = CONFIG['save_model_every']
SAVE_EXAMPLE_EVERY = CONFIG['save_example_every']
PRINT_INFO_EVERY = CONFIG['print_info_every']