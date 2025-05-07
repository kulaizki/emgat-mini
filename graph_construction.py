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
    OUTPUT_DIR, ATLAS_DIM, FISHER_Z_TRANSFORM, SUBJECTS
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


# --- Graph Construction & Node Features (Run-Level Subject Classification) ---
def create_pyg_graph_list(run_connectivity_matrices):
    """
    Creates a list of PyG Data objects from RUN-LEVEL connectivity matrices.
    Node features are an identity matrix. Edge attributes are from the connectivity matrix.
    Graph label (y) is the subject index.
    """
    print("--- Creating PyG Graph Objects (Run-Level Subject Classification) ---")
    if not run_connectivity_matrices:
        print("  ERROR: Missing run-level connectivity matrices. Cannot proceed.")
        return None, None, None

    pyg_data_list = []
    num_nodes = ATLAS_DIM

    # Define node features (identity matrix for all graphs)
    node_features = torch.eye(num_nodes, dtype=torch.float)

    # Define edge index for a fully connected graph (excluding self-loops)
    # This is consistent for all graphs if they have the same number of nodes
    adj = torch.ones(num_nodes, num_nodes)
    adj.fill_diagonal_(0) # No self-loops
    edge_index = adj.nonzero(as_tuple=False).t().contiguous()

    print(f"  Using fully connected graph structure with {edge_index.shape[1]} edges for {num_nodes} nodes.")

    # Determine subjects and assign integer labels
    # Using SUBJECTS from setup_config directly as the source of truth for subjects
    subject_ids = SUBJECTS
    num_subjects = len(subject_ids)
    subject_to_index = {sub_id: i for i, sub_id in enumerate(subject_ids)}

    if num_subjects == 0:
        print("  ERROR: No subjects defined in SUBJECTS list from setup_config. Cannot create subject labels.")
        return None, None, None
    print(f"  Task: Run-Level Subject Classification ({num_subjects} subjects: {subject_ids})")

    created_count = 0
    skipped_count = 0
    epsilon = 1e-6 # For Fisher Z stability

    for sub_id, sessions in run_connectivity_matrices.items():
        if sub_id not in subject_to_index:
            print(f"    Warning: Subject ID {sub_id} from data not in configured SUBJECTS list. Skipping.")
            skipped_count += (len(run_dict) for ses_dict in sessions.values() for run_dict in ses_dict.values()) # Estimate skips
            continue

        subject_label_index = subject_to_index[sub_id]

        for ses_id, runs in sessions.items():
            for run_id, conn_matrix in runs.items():
                if conn_matrix is None:
                    print(f"    Skipping run {run_id} for {sub_id}/{ses_id}: Missing connectivity matrix.")
                    skipped_count += 1
                    continue

                try:
                    if FISHER_Z_TRANSFORM:
                        processed_matrix = np.arctanh(np.clip(conn_matrix, -1 + epsilon, 1 - epsilon))
                    else:
                        processed_matrix = conn_matrix

                    if processed_matrix.shape != (num_nodes, num_nodes):
                        print(f"    ERROR: Matrix dimensions ({processed_matrix.shape}) don't match ATLAS_DIM ({num_nodes}) for run {run_id} ({sub_id}/{ses_id}). Skipping.")
                        skipped_count += 1
                        continue

                    edge_attr_values = processed_matrix[edge_index[0], edge_index[1]]
                    edge_attr = torch.tensor(edge_attr_values, dtype=torch.float).unsqueeze(1)

                    if edge_attr.shape[0] != edge_index.shape[1]:
                         print(f"    ERROR: Mismatch between edge_attr length ({edge_attr.shape[0]}) and edge_index columns ({edge_index.shape[1]}) for run {run_id}. Skipping.")
                         skipped_count += 1
                         continue

                    graph_data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
                    graph_data.y = torch.tensor([subject_label_index], dtype=torch.long)

                    # Add metadata
                    graph_data.subject_id = sub_id
                    graph_data.session_id = ses_id
                    graph_data.run_id = run_id
                    # graph_data.category_name = "N/A" # No category for run-level subject classification

                    pyg_data_list.append(graph_data)
                    created_count += 1
                except Exception as e:
                    print(f"    ERROR processing run {run_id} for {sub_id}/{ses_id}. Skipping. Error: {e}")
                    skipped_count += 1
                    continue

    total_runs_expected = sum(len(runs) for sessions in run_connectivity_matrices.values() for runs in sessions.values())
    print(f"  Created {created_count} graph Data objects. Skipped {skipped_count} runs out of {total_runs_expected} total processed from input.")
    if not pyg_data_list:
        print("  ERROR: No graphs were created. Check run connectivity data and SUBJECTS configuration.")
        return None, None, None

    return pyg_data_list, num_subjects, subject_ids # Return num_subjects and subject_ids as class_names

# --- Save/Load Run-Level Dataset ---
def save_pyg_run_level_dataset(pyg_data_list, num_subjects, subject_ids, root_dir):
    """
    Saves the list of run-level PyG Data objects and dataset info.
    """
    print("--- Saving PyG Run-Level Graph List & Info ---")
    if not pyg_data_list:
        print("  ERROR: No graph data provided to save.")
        return False

    dataset_dir = Path(root_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    graph_list_path = dataset_dir / 'run_level_graph_list.pt' # Updated filename
    info_path = dataset_dir / "run_level_dataset_info.pkl"   # Updated filename

    try:
        try: add_safe_globals(unique_safe_classes)
        except Exception: pass
        torch.save(pyg_data_list, graph_list_path)
        print(f"  Successfully saved list of {len(pyg_data_list)} run-level graphs to: {graph_list_path}")
    except Exception as e:
        print(f"  ERROR saving graph list: {e}")
        if graph_list_path.exists():
             try: graph_list_path.unlink(); print(f"  Removed potentially corrupted file: {graph_list_path}")
             except OSError as oe: print(f"  Error removing file: {oe}")
        return False

    dataset_info = {'num_classes': num_subjects, 'class_names': subject_ids} # Updated info
    try:
        with open(info_path, 'wb') as f:
            pickle.dump(dataset_info, f)
        print(f"  Saved run-level dataset info to: {info_path}")
    except Exception as e:
        print(f"  ERROR saving dataset info: {e}")
        return True # Still return True if graphs were saved, but info failed

    return True


# --- Main Execution Function (Updated for Run-Level Subject Classification) ---
if __name__ == "__main__":
    print("Executing Graph Construction Pipeline (Run-Level Subject Classification - Direct Test)...")

    # --- Dummy Data Setup (Run-Level) ---
    print("WARNING: Running script directly. Using dummy run-level data for testing.")
    
    # Use SUBJECTS from config, or define dummy if not available for direct test
    if not SUBJECTS:
        DUMMY_SUBJECTS = ['sub-01', 'sub-02']
        print(f"  Using dummy subjects for test: {DUMMY_SUBJECTS}")
    else:
        DUMMY_SUBJECTS = SUBJECTS
        print(f"  Using SUBJECTS from config: {DUMMY_SUBJECTS}")


    dummy_run_connectivity_matrices = {
        DUMMY_SUBJECTS[0]: {
            'ses-01': {
                'run-01': np.random.rand(ATLAS_DIM, ATLAS_DIM),
                'run-02': np.random.rand(ATLAS_DIM, ATLAS_DIM)
            }
        },
        DUMMY_SUBJECTS[1]: {
            'ses-01': {
                'run-01': np.random.rand(ATLAS_DIM, ATLAS_DIM),
                'run-02': None # Test missing matrix
            }
        }
    }
    if len(DUMMY_SUBJECTS) > 2: # Add more dummy data if more subjects
        dummy_run_connectivity_matrices[DUMMY_SUBJECTS[2]] = {
            'ses-01': {'run-01': np.random.rand(ATLAS_DIM, ATLAS_DIM)}
        }


    # Symmetrize dummy matrices
    for sub_id, sessions in dummy_run_connectivity_matrices.items():
        for ses_id, runs in sessions.items():
            for run_id, matrix in runs.items():
                if matrix is not None:
                     runs[run_id] = (matrix + matrix.T) / 2
                     np.fill_diagonal(runs[run_id], 0) # Typically no self-loops in FC

    TEST_OUTPUT_DIR = Path("./output_test_gc_run_level")
    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Using dummy data. Output directory: {TEST_OUTPUT_DIR}")
    # --- End Dummy Data ---

    # Temporarily override SUBJECTS for the test if it was empty
    original_subjects_config = list(SUBJECTS) # Save original state
    if not SUBJECTS:
        SUBJECTS.extend(DUMMY_SUBJECTS)


    # --- Execute Run-Level Functions ---
    # The create_pyg_graph_list now relies on SUBJECTS from setup_config
    graph_list, num_classes_out, class_names_out = create_pyg_graph_list(dummy_run_connectivity_matrices)

    # Restore SUBJECTS if it was modified for the test
    if not original_subjects_config and SUBJECTS:
        SUBJECTS.clear()
        SUBJECTS.extend(original_subjects_config)


    if graph_list:
        dataset_storage_path = TEST_OUTPUT_DIR / 'pyg_run_level_dataset'
        success = save_pyg_run_level_dataset(graph_list, num_classes_out, class_names_out, dataset_storage_path)
        if success:
             print("Graph Construction Standalone Test Complete (Run-Level).")
             print(f"Number of Classes (Subjects): {num_classes_out}")
             print(f"Class Names (Subject IDs): {class_names_out}")
             print(f"Saved {len(graph_list)} graphs.")
        else:
             print("Graph Construction Standalone Test Partially Failed (Run-Level) during dataset saving.")
    else:
        print("Graph Construction Standalone Test Failed (Run-Level) during graph creation.")