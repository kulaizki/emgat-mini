import os
from pathlib import Path
import warnings
import random 
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import image, plotting, datasets, signal
from nilearn.maskers import NiftiMapsMasker
from nilearn.connectome import ConnectivityMeasure
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, Dataset, DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
import torch.nn.functional as F

import matplotlib.pyplot as plt
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    plt.style.use('ggplot')

# --- Core Configuration ---
FMRIPREP_DIR = Path("./data/ds004496/derivatives/fmriprep")
BIDS_RAW_DIR = Path("./data/ds004496/raw")
OUTPUT_DIR = Path("./output")
SYNSET_WORDS_FILE = Path("./data/ds004496/synset_words.txt") 

# --- Atlas Configuration ---
ATLAS_DIM = 64
ATLAS_NAME = f"difumo{ATLAS_DIM}"
ATLAS_RESOLUTION_MM = 2

# --- DiFuMo Labels/Coordinates CSV Configuration ---
DIFUMO_LABELS_FILE = Path("./labels_64_dictionary.csv") 

# Name of the column containing the component index (0-based or 1-based)
DIFUMO_LABEL_INDEX_COL = "Component"

# Name of the column containing the human-readable region name
DIFUMO_LABEL_NAME_COL = "Difumo_names"

# Names of columns containing MNI coordinates.
DIFUMO_COORD_X_COL = 'x_mni'
DIFUMO_COORD_Y_COL = 'y_mni'
DIFUMO_COORD_Z_COL = 'z_mni'
# --- End DiFuMo CSV Configuration ---

# --- Data Selection (Run-Level Analysis) ---
SUBJECTS = ['01', '02']
SESSIONS = ['imagenet01', 'imagenet02', 'imagenet03', 'imagenet04'] 
TASK_NAME = 'imagenet'
N_RUNS_PER_SESSION = 10 # Number of runs per session to process

# --- fMRI Processing Parameters ---
TR = 2.0 # Repetition Time in seconds
HEMODYNAMIC_DELAY_TR = 2 # Not used in run-level analysis, but kept for reference
CONFOUNDS_TO_USE = [
    'trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z',
    'trans_x_derivative1', 'trans_y_derivative1', 'trans_z_derivative1',
    'rot_x_derivative1', 'rot_y_derivative1', 'rot_z_derivative1',
    'csf', 'white_matter',
    'cosine00', 'cosine01', 'cosine02', 'cosine03',
]
FD_THRESHOLD = 0.5 
CLEAN_SIGNAL_PARAMS = {
    't_r': TR,
    'high_pass': 0.01, # Hz
    'low_pass': None, # Hz
    'detrend': True,
    'standardize': 'zscore_sample'
}

# --- Connectivity & Graph Parameters (Run-Level Analysis) ---
# MIN_VOLUMES_PER_CATEGORY = 32 # Not needed for run-level
CONNECTIVITY_KIND = 'correlation'
FISHER_Z_TRANSFORM = True # Apply Fisher transform to run-level correlations

# --- Model & Training Parameters ---
GAT_HIDDEN_CHANNELS = 32
GAT_HEADS = 8
GAT_DROPOUT = 0.3
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
N_EPOCHS = 300

# --- Explainability Parameters ---
# Ensure this is an integer if GNNExplainer is used later
GNN_EXPLAINER_EPOCHS = 100

# --- Miscellaneous ---
RANDOM_SEED = 42
NILEARN_CACHE_DIR = OUTPUT_DIR / 'nilearn_cache'

# --- Setup Functions ---
def setup_environment(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # print(f"Random seed set to: {seed}") # Moved print to main

def ensure_directories():
    """Create output and cache directories if they don't exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    NILEARN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # print(f"Ensured output directory exists: {OUTPUT_DIR}") # Moved print to main
    # print(f"Ensured Nilearn cache directory exists: {NILEARN_CACHE_DIR}") # Moved print to main

# --- Run Setup on Import ---
setup_environment(RANDOM_SEED)
ensure_directories()

# print("Configuration loaded and environment set up.") # Moved print to main
# print("-" * 30) # Moved print to main