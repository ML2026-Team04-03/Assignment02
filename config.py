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

# Always align class order to folder names
CLASS_NAMES = sorted([d for d in os.listdir(TRAIN_DIR)
                      if os.path.isdir(os.path.join(TRAIN_DIR, d))])

# Reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)