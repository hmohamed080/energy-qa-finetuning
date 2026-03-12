# Configurations for Energy QA Finetuning

# Model configuration
MODEL_NAME = 'bert-base-uncased'

# Training hyperparameters
LEARNING_RATE = 2e-5
BATCH_SIZE = 32
EPOCHS = 3

# Paths
TRAINING_DATA_PATH = 'data/training_data.json'
EVALUATION_DATA_PATH = 'data/evaluation_data.json'
OUTPUT_DIR = 'output/'

