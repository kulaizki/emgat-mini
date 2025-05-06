import pickle
import numpy as np
import torch
from torch_geometric.data import Data
from pathlib import Path

# --- Safe loading registration (remains important) ---
from torch.serialization import add_safe_globals
safe_classes_to_register = [Data]
try:
    from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
    safe_classes_to_register.extend([DataEdgeAttr, DataTensorAttr])
except ImportError: pass
try:
    from torch_geometric.data.storage import EdgeStorage, GlobalStorage, NodeStorage
    safe_classes_to_register.extend([EdgeStorage, GlobalStorage, NodeStorage])
except ImportError: pass
# --- End Registration Setup ---

# --- Import configuration variables ---
from setup_config import (
    OUTPUT_DIR, ATLAS_DIM, FISHER_Z_TRANSFORM
)

# --- Register Safe Globals (actual registration) ---
try:
    unique_safe_classes = list(set(safe_classes_to_register))
    add_safe_globals(unique_safe_classes)
    print(f"Registered {len(unique_safe_classes)} PyTorch Geometric classes globally for safe loading.")
except NameError:
     print("Warning: add_safe_globals not defined (likely older PyTorch).")
except Exception as e:
     print(f"Warning: Could not register PyG safe globals at module level: {e}")
# --- End Register ---


# --- Graph Construction & Node Features (Trial-Level) ---
def create_pyg_graph_list(trial_data_dict):
    """
    Creates a list of PyG Data objects from TRIAL-LEVEL connectivity matrices and category labels.
    Node features are identity matrix. Edge attributes are flattened connectivity.
    Graph label (y) is the category index.
    """
    print("--- Creating PyG Graph Objects (Trial-Level) ---")
    if not trial_data_dict:
        print("  ERROR: Missing trial-level data dictionary. Cannot proceed.")
        return None, None, None

    # Load category info if import failed (requires the file to exist)
    local_categories = CATEGORIES
    local_num_categories = NUM_CATEGORIES
    if local_categories is None or local_num_categories is None:
        category_map_path = OUTPUT_DIR / "manual_synset_category_map.pkl"
        print(f"Attempting to load category info from {category_map_path}")
        if category_map_path.exists():
            try:
                with open(category_map_path, 'rb') as f:
                    loaded_map = pickle.load(f)
                local_categories = loaded_map['categories']
                local_num_categories = len(local_categories)
                print(f"  Successfully loaded {local_num_categories} categories: {local_categories}")
            except Exception as e:
                print(f"  ERROR loading category map: {e}. Cannot proceed without category info.")
                return None, None, None
        else:
            print(f"  ERROR: Category map file not found at {category_map_path}. Cannot proceed.")
            return None, None, None

    pyg_data_list = []
    num_nodes = ATLAS_DIM

    # Define node features (identity matrix)
    node_features = torch.eye(num_nodes, dtype=torch.float)

    # Define edge index for a fully connected graph (excluding self-loops)
    adj = torch.ones(num_nodes, num_nodes)
    adj.fill_diagonal_(0)
    edge_index = adj.nonzero(as_tuple=False).t().contiguous()

    print(f"  Using fully connected graph structure with {edge_index.shape[1]} edges.")
    print(f"  Task: Trial Category Classification ({local_num_categories} classes)")

    created_count = 0
    skipped_count = 0
    epsilon = 1e-6 # For Fisher Z stability

    for sub, sessions in trial_data_dict.items():
        for ses, trials in sessions.items():
            for trial_info in trials:
                trial_id = trial_info.get('trial_id')
                conn_matrix = trial_info.get('connectivity')
                category_label_index = trial_info.get('category_label_index')
                category_name = trial_info.get('category_name', 'Unknown')

                if conn_matrix is None or category_label_index is None:
                    print(f"    Skipping trial {trial_id} for {sub}/{ses}: Missing connectivity or category index.")
                    skipped_count += 1
                    continue

                # --- Process matrix and create graph_data ---
                try:
                    # Apply Fisher Z transform if specified
                    if FISHER_Z_TRANSFORM:
                        processed_matrix = np.arctanh(np.clip(conn_matrix, -1 + epsilon, 1 - epsilon))
                    else:
                        processed_matrix = conn_matrix

                    # Ensure matrix dimensions are correct
                    if processed_matrix.shape != (num_nodes, num_nodes):
                        print(f"    ERROR: Matrix dimensions ({processed_matrix.shape}) don't match ATLAS_DIM ({num_nodes}) for trial {trial_id} ({sub}/{ses}). Skipping.")
                        skipped_count += 1
                        continue

                    # Extract edge attributes (flattened upper/lower triangle values corresponding to edge_index)
                    # Note: This assumes edge_index covers all pairs (i, j) where i != j
                    edge_attr_values = processed_matrix[edge_index[0], edge_index[1]]
                    edge_attr = torch.tensor(edge_attr_values, dtype=torch.float).unsqueeze(1)

                    # Ensure edge_attr shape matches edge_index
                    if edge_attr.shape[0] != edge_index.shape[1]:
                         print(f"    ERROR: Mismatch between edge_attr length ({edge_attr.shape[0]}) and edge_index columns ({edge_index.shape[1]}) for trial {trial_id}. Skipping.")
                         skipped_count += 1
                         continue

                    # Create PyG Data object
                    graph_data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
                    graph_data.y = torch.tensor([category_label_index], dtype=torch.long)

                    # Add metadata
                    graph_data.subject = sub
                    graph_data.session = ses
                    graph_data.trial_id = trial_id
                    graph_data.category_name = category_name

                    pyg_data_list.append(graph_data)
                    created_count += 1
                except Exception as e:
                    print(f"    ERROR processing trial {trial_id} for {sub}/{ses}. Skipping. Error: {e}")
                    skipped_count += 1
                    continue
                # --- End process matrix ---

    total_trials = sum(len(trials) for sessions in trial_data_dict.values() for trials in sessions.values())
    print(f"  Created {created_count} graph Data objects. Skipped {skipped_count} trials out of {total_trials} total.")
    if not pyg_data_list:
        print("  ERROR: No graphs were created. Check trial data extraction and processing.")
        return None, None, None

    return pyg_data_list, local_num_categories, local_categories

# --- Save/Load Trial-Level Dataset ---
def save_pyg_trial_dataset(pyg_data_list, num_categories, category_names, root_dir):
    """
    Saves the list of trial-level PyG Data objects and dataset info.
    """
    print("--- Saving PyG Trial-Level Graph List & Info ---")
    if not pyg_data_list:
        print("  ERROR: No graph data provided to save.")
        return False # Indicate failure

    dataset_dir = Path(root_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    graph_list_path = dataset_dir / 'trial_level_graph_list.pt'
    info_path = dataset_dir / "trial_level_dataset_info.pkl"

    # Save graph list
    try:
        # Ensure safe globals registered before saving
        try: add_safe_globals(unique_safe_classes)
        except Exception: pass
        torch.save(pyg_data_list, graph_list_path)
        print(f"  Successfully saved list of {len(pyg_data_list)} trial graphs to: {graph_list_path}")
    except Exception as e:
        print(f"  ERROR saving graph list: {e}")
        if graph_list_path.exists():
             try: graph_list_path.unlink(); print(f"  Removed potentially corrupted file: {graph_list_path}")
             except OSError as oe: print(f"  Error removing file: {oe}")
        return False # Indicate failure on graph save error

    # Save dataset info
    dataset_info = {'num_categories': num_categories, 'category_names': category_names}
    try:
        with open(info_path, 'wb') as f:
            pickle.dump(dataset_info, f)
        print(f"  Saved trial-level dataset info to: {info_path}")
    except Exception as e:
        print(f"  ERROR saving dataset info: {e}")
        # Don't necessarily fail the whole process if info saving fails, but warn
        return True # Still return True as graphs were saved

    return True # Indicate success


# --- Main Execution Function (Updated for Trial-Level) ---
if __name__ == "__main__":
    print("Executing Graph Construction Pipeline (Trial-Level - Direct Test)...")

    # --- Dummy Data Setup (Trial Level) ---
    print("WARNING: Running script directly. Using dummy trial data for testing.")
    # Use dummy categories if import failed
    if CATEGORIES is None:
         print("Using dummy categories for test.")
         DUMMY_CATEGORIES = ['Vehicle', 'Animal', 'Tool', 'Food', 'Unknown']
         DUMMY_NUM_CATEGORIES = len(DUMMY_CATEGORIES)
    else:
         DUMMY_CATEGORIES = CATEGORIES
         DUMMY_NUM_CATEGORIES = NUM_CATEGORIES

    dummy_trial_data = {
        '01': {
            'imagenet01': [
                {'trial_id': 1, 'connectivity': np.random.rand(ATLAS_DIM, ATLAS_DIM), 'category_label_index': 0, 'category_name': DUMMY_CATEGORIES[0]},
                {'trial_id': 2, 'connectivity': np.random.rand(ATLAS_DIM, ATLAS_DIM), 'category_label_index': 1, 'category_name': DUMMY_CATEGORIES[1]},
                {'trial_id': 3, 'connectivity': np.random.rand(ATLAS_DIM, ATLAS_DIM), 'category_label_index': 0, 'category_name': DUMMY_CATEGORIES[0]}
            ],
            'imagenet02': [
                {'trial_id': 4, 'connectivity': np.random.rand(ATLAS_DIM, ATLAS_DIM), 'category_label_index': 2, 'category_name': DUMMY_CATEGORIES[2]},
                {'trial_id': 5, 'connectivity': None, 'category_label_index': 3, 'category_name': DUMMY_CATEGORIES[3]} # Test missing conn
            ]
        },
        '02': {
            'imagenet01': [
                {'trial_id': 6, 'connectivity': np.random.rand(ATLAS_DIM, ATLAS_DIM), 'category_label_index': 1, 'category_name': DUMMY_CATEGORIES[1]},
                {'trial_id': 7, 'connectivity': np.random.rand(ATLAS_DIM, ATLAS_DIM), 'category_label_index': 4, 'category_name': DUMMY_CATEGORIES[4]},
                {'trial_id': 8, 'connectivity': np.random.rand(ATLAS_DIM, ATLAS_DIM), 'category_label_index': None, 'category_name': 'Unknown'} # Test missing label
            ]
        }
    }
    # Symmetrize dummy matrices
    for sub, sessions in dummy_trial_data.items():
        for ses, trials in sessions.items():
            for trial in trials:
                if trial['connectivity'] is not None:
                     trial['connectivity'] = (trial['connectivity'] + trial['connectivity'].T) / 2
                     np.fill_diagonal(trial['connectivity'], 1) 

    TEST_OUTPUT_DIR = Path("./output_test_gc_trial") 
    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Using dummy data. Output directory: {TEST_OUTPUT_DIR}")
    # --- End Dummy Data ---

    # --- Execute Trial-Level Functions ---
    graph_list, num_categories_out, category_names_out = create_pyg_graph_list(dummy_trial_data)

    if graph_list:
        dataset_storage_path = TEST_OUTPUT_DIR / 'pyg_trial_level_dataset'
        success = save_pyg_trial_dataset(graph_list, num_categories_out, category_names_out, dataset_storage_path)
        if success:
             print("\nGraph Construction Standalone Test Complete (Trial-Level).")
             print(f"Number of Categories: {num_categories_out}")
             print(f"Category Names: {category_names_out}")
             print(f"Saved {len(graph_list)} graphs.")
        else:
             print("\nGraph Construction Standalone Test Partially Failed (Trial-Level) during dataset saving.")
    else:
        print("\nGraph Construction Standalone Test Failed (Trial-Level) during graph creation.")