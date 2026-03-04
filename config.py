import os
import random
import numpy as np
import torch

# TALC paths
BASE_DIR = "/work/TALC/ensf617_2026w/garbage_data"
OUT_DIR = os.path.join(os.getcwd(), "outputs")

TRAIN_DIR = os.path.join(BASE_DIR, "CVPR_2024_dataset_Train")
VAL_DIR   = os.path.join(BASE_DIR, "CVPR_2024_dataset_Val")
TEST_DIR  = os.path.join(BASE_DIR, "CVPR_2024_dataset_Test")

# Get class names from subfolder names in the training directory
CLASS_NAMES = sorted([d for d in os.listdir(TRAIN_DIR)
                      if os.path.isdir(os.path.join(TRAIN_DIR, d))])

# Training hyperparameters
BATCH_SIZE = 32
LABEL_SMOOTHING = 0.1
EARLY_STOPPING_PATIENCE = 5

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Progressive fine-tuning stages configuration
PROGRESSIVE_STAGES = [
    {"n_blocks": 2,  "epochs": 4,  "lr_backbone": 0.0,   "lr_head": 2e-4}, # Stage 0: Head Only
    {"n_blocks": 4,  "epochs": 4,  "lr_backbone": 1e-5,  "lr_head": 1e-4}, # Stage 1: Partial
    {"n_blocks": 6,  "epochs": 4,  "lr_backbone": 5e-6,  "lr_head": 5e-5}, # Stage 2: Deep
    {"n_blocks": 8,  "epochs": 6,  "lr_backbone": 1e-6,  "lr_head": 1e-5}, # Stage 3: Full
]

# W&B Configuration
WANDB_PROJECT = "garbage-classification-multimodal"
WANDB_ENTITY = None  
WANDB_MODE = "online" 