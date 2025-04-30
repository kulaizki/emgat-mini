
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

# ==============================================================
# !!! ADDED/UPDATED SECTION for DiFuMo CSV Configuration !!!
# ==============================================================
# Path to the CSV file containing labels and potentially coordinates
DIFUMO_LABELS_FILE = Path("./labels_64_dictionary.csv") # Ensure this path is correct

# --- Column Names in DIFUMO_LABELS_FILE (!!! ADJUST THESE !!!) ---
# Set these based on the *actual* header names in your CSV file.

# Name of the column containing the component index (e.g., 0-63 or 1-64)
# Example: If your column is named 'Component_ID', use that.
# Set to None if your CSV doesn't have an index column or you don't need to sort/index by it.
DIFUMO_LABEL_INDEX_COL = "Component" # <-- MODIFY THIS if your column name is different, or set to None

# Name of the column containing the human-readable region name
# Example: If your column is named 'RegionName', use that.
DIFUMO_LABEL_NAME_COL = "Difumo_names" # <-- MODIFY THIS if your column name is different

# Names of columns containing MNI coordinates.
# Set these ONLY if you want to potentially load coordinates from this CSV
# as a fallback if the nilearn method fails.
# Set to None if your CSV doesn't have coordinate columns.
DIFUMO_COORD_X_COL = "MNI_X" # <-- MODIFY THIS if your column name is different, or set to None
DIFUMO_COORD_Y_COL = "MNI_Y" # <-- MODIFY THIS if your column name is different, or set to None
DIFUMO_COORD_Z_COL = "MNI_Z" # <-- MODIFY THIS if your column name is different, or set to None
# ==============================================================
# !!! END OF SECTION TO ADD/UPDATE !!!
# ==============================================================


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
    'cosine00', 'cosine01', 'cosine02', 'cosine03', # Example high-pass filter confounds from fmriprep
    # Add other confounds like 'global_signal' or FD if desired, e.g., 'framewise_displacement'
]
FD_THRESHOLD = 0.5 # Framewise displacement threshold (not currently used in cleaning, but defined)
CLEAN_SIGNAL_PARAMS = {
    't_r': TR,
    'high_pass': 0.01, # Hz
    'low_pass': None, # Hz (e.g., 0.1) - Set to None if no low-pass filtering desired
    'detrend': True,
    'standardize': 'zscore_sample' # Standardize after cleaning
}

# --- Category Definition ---
# Keywords are matched case-insensitively against the *first* label from synset_words.txt
POTENTIAL_CATEGORIES = {
    "Mammals": ['dog', 'cat', 'monkey', 'elephant', 'bear', 'lion', 'zebra', 'mammal', 'primate', 'fox', 'wolf', 'horse', 'pig', 'sheep', 'cow', 'hippo', 'panda', 'koala', 'kangaroo', 'badger', 'otter', 'skunk'],
    "Birds": ['bird', 'eagle', 'owl', 'parrot', 'penguin', 'swan', 'cock', 'hen', 'finch', 'robin', 'jay', 'magpie', 'kite', 'vulture'],
    "Tools": ['hammer', 'screwdriver', 'wrench', 'pliers', 'tool', 'saw', 'chisel', 'hatchet', 'drill', 'vise'],
    "Vehicles": ['car', 'truck', 'bus', 'airplane', 'train', 'vehicle', 'motorcycle', 'boat', 'bicycle', 'scooter', 'van', 'taxi'],
    "Food": ['apple', 'banana', 'pizza', 'bread', 'food', 'fruit', 'vegetable', 'cake', 'mushroom', 'orange', 'lemon', 'pineapple', 'strawberry', 'fig', 'hotdog', 'pretzel', 'bagel', 'burrito'],
    "Man-made Structures": ['house', 'building', 'bridge', 'church', 'castle', 'structure', 'tower', 'skyscraper', 'monastery', 'library', 'palace', 'dam', 'barn', 'stupa']
}
MIN_SYNSETS_PER_CATEGORY = 5 # Minimum number of unique synsets mapping to a category to keep it
N_CATEGORIES_TO_USE = 6 # Max number of categories to use if many meet the threshold
UNCATEGORIZED_NAME = "Other" # Name for items not fitting into selected categories

# --- Connectivity & Graph Parameters ---
# Heuristic threshold for connectivity calculation (consider lowering for PoC if needed)
MIN_VOLUMES_PER_CATEGORY = ATLAS_DIM # Set equal to number of nodes (e.g., 64)
# Alternative: Set a fixed minimum like 30, but acknowledge lower reliability
# MIN_VOLUMES_PER_CATEGORY = 30

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
GNN_EXPLAINER_EPOCHS = 100 # Ensure this is an integer

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

def ensure_directories():
    """Create output and cache directories if they don't exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    NILEARN_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# --- Run Setup on Import ---
# These will run automatically whenever setup_config is imported
setup_environment(RANDOM_SEED)
ensure_directories()