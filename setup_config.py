import os
import glob
import re
import pickle
from pathlib import Path
import warnings

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
from torch_geometric.explain import GNNExplainer
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
DIFUMO_LABELS_FILE = Path("./labels_64_dictionary.csv") 

# --- Atlas Configuration ---
ATLAS_DIM = 64
ATLAS_NAME = f"difumo{ATLAS_DIM}"
ATLAS_RESOLUTION_MM = 2 

# --- Data Selection ---
SUBJECTS = ['01', '02']
SESSIONS = ['imagenet01', 'imagenet02'] 
TASK_NAME = 'imagenet' 
N_RUNS_PER_SESSION = 10 # Number of runs per session to process

# --- fMRI Processing Parameters ---
TR = 2.0 # Repetition Time in seconds 
HEMODYNAMIC_DELAY_TR = 2 # Delay in TRs (e.g., 4s delay / 2s TR = 2 TRs)
CONFOUNDS_TO_USE = [ 
    'trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z',
    'trans_x_derivative1', 'trans_y_derivative1', 'trans_z_derivative1',
    'rot_x_derivative1', 'rot_y_derivative1', 'rot_z_derivative1',
    'csf', 'white_matter',
    'cosine00', 'cosine01', 'cosine02', 'cosine03', 
]
FD_THRESHOLD = 0.5 # Framewise displacement threshold (if used for scrubbing/analysis)
CLEAN_SIGNAL_PARAMS = {
    't_r': TR,
    'high_pass': 0.01, # Hz
    'low_pass': None, # Hz (e.g., 0.1)
    'detrend': True,
    'standardize': 'zscore_sample' # Standardize after cleaning
}

# --- Category Definition ---
POTENTIAL_CATEGORIES = {
    "Mammals": ['dog', 'cat', 'monkey', 'elephant', 'bear', 'lion', 'zebra', 'mammal', 'primate'],
    "Birds": ['bird', 'eagle', 'owl', 'parrot', 'penguin', 'swan'],
    "Tools": ['hammer', 'screwdriver', 'wrench', 'pliers', 'tool', 'saw'],
    "Vehicles": ['car', 'truck', 'bus', 'airplane', 'train', 'vehicle', 'motorcycle', 'boat'],
    "Food": ['apple', 'banana', 'pizza', 'bread', 'food', 'fruit', 'vegetable', 'cake', 'mushroom'],
    "Man-made Structures": ['house', 'building', 'bridge', 'church', 'castle', 'structure', 'tower', 'skyscraper']
}
MIN_SYNSETS_PER_CATEGORY = 5 # Minimum number of unique synsets mapping to a category to keep it
N_CATEGORIES_TO_USE = 6 # Max number of categories to use if many meet the threshold
UNCATEGORIZED_NAME = "Other" # Name for items not fitting into selected categories

# --- Connectivity & Graph Parameters ---
MIN_VOLUMES_PER_CATEGORY = ATLAS_DIM # Heuristic threshold for connectivity calculation
CONNECTIVITY_KIND = 'correlation' # or 'partial correlation', 'tangent'
FISHER_Z_TRANSFORM = True # Apply Fisher transform to correlations for GAT edge weights

# --- Model & Training Parameters ---
GAT_HIDDEN_CHANNELS = 16 # Number of features in the hidden layer
GAT_HEADS = 4 # Number of attention heads
GAT_DROPOUT = 0.6 # Dropout rate in GAT layer
LEARNING_RATE = 0.005
WEIGHT_DECAY = 5e-4 # L2 regularization for Adam optimizer
N_EPOCHS = 150 # Number of training epochs (adjust based on convergence)

# --- Explainability Parameters ---
GNN_EXPLAINER_EPOCHS = 100

# --- Miscellaneous ---
RANDOM_SEED = 42 
NILEARN_CACHE_DIR = OUTPUT_DIR / 'nilearn_cache' 

# --- Setup ---
def setup_environment(seed):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to: {seed}")

def ensure_directories():
    """Create output and cache directories if they don't exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    NILEARN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Ensured output directory exists: {OUTPUT_DIR}")
    print(f"Ensured Nilearn cache directory exists: {NILEARN_CACHE_DIR}")

setup_environment(RANDOM_SEED)
ensure_directories()

print("Configuration loaded and environment set up.")
print("-" * 30) 