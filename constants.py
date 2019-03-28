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

config = config['default_config']

DIM_X = config['output']['dimX']
DIM_Y = config['output']['dimY'] 
DIM_Z = config['output']['dimZ'] 

BATCH_SIZE = config['input']['batch_size']
Z_NOISE_DIM = config['input']['batch_size'] 

NUM_EPOCHS = config['training']['num_epochs']
LEARNING_RATE = config['training']['learning_rate']
BETA1 = config['training']['beta1']

SAVE_MODEL_EVERY = config['save_model_every']
SAVE_EXAMPLE_EVERY = config['save_example_every']
PRINT_INFO_EVERY = config['print_info_every']