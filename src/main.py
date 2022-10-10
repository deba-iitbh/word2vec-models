import torch
from train import train

# Pre-processing vars
DATA_DIR = "../data/Wiki"
DEVICE = torch.device("cuda")

# Model Vars
NUM_EPOCHS = 10
LR = 0.001
BATCH_SIZE = 1000
CONTEXT_SIZE = 3
EMBEDDING_DIM = 100

# Saving vars
MODEL_DIR = "../model"

# skipgram without neg-subsampling
TYPE = "skipgram"
NEGATIVE_SAMPLES = 0

train(
    DATA_DIR,
    CONTEXT_SIZE,
    TYPE,
    BATCH_SIZE,
    EMBEDDING_DIM,
    DEVICE,
    NEGATIVE_SAMPLES,
    NUM_EPOCHS,
    LR,
    MODEL_DIR,
)

# skipgram with neg-subsampling
TYPE = "skipgram"
NEGATIVE_SAMPLES = 10
train(
    DATA_DIR,
    CONTEXT_SIZE,
    TYPE,
    BATCH_SIZE,
    EMBEDDING_DIM,
    DEVICE,
    NEGATIVE_SAMPLES,
    NUM_EPOCHS,
    LR,
    MODEL_DIR,
)

# CBOW without neg-subsampling
TYPE = "cbow"
NEGATIVE_SAMPLES = 0

train(
    DATA_DIR,
    CONTEXT_SIZE,
    TYPE,
    BATCH_SIZE,
    EMBEDDING_DIM,
    DEVICE,
    NEGATIVE_SAMPLES,
    NUM_EPOCHS,
    LR,
    MODEL_DIR,
)

# CBOW with neg-subsampling
TYPE = "cbow"
NEGATIVE_SAMPLES = 10

train(
    DATA_DIR,
    CONTEXT_SIZE,
    TYPE,
    BATCH_SIZE,
    EMBEDDING_DIM,
    DEVICE,
    NEGATIVE_SAMPLES,
    NUM_EPOCHS,
    LR,
    MODEL_DIR,
)

# glove without neg-subsampling
TYPE = "glove"
NEGATIVE_SAMPLES = 0

train(
    DATA_DIR,
    CONTEXT_SIZE,
    TYPE,
    BATCH_SIZE,
    EMBEDDING_DIM,
    DEVICE,
    NEGATIVE_SAMPLES,
    NUM_EPOCHS,
    LR,
    MODEL_DIR,
)

# glove with neg-subsampling
TYPE = "glove"
NEGATIVE_SAMPLES = 10

train(
    DATA_DIR,
    CONTEXT_SIZE,
    TYPE,
    BATCH_SIZE,
    EMBEDDING_DIM,
    DEVICE,
    NEGATIVE_SAMPLES,
    NUM_EPOCHS,
    LR,
    MODEL_DIR,
)
