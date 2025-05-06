# Filename: main.py
# Location: <workspace_root>/main.py
# Description: Main script to run the entire PoC workflow using
#              RUN-LEVEL connectivity and SUBJECT classification.
# Changes:
#  - Updated workflow for run-level analysis (removed category steps).
#  - Correctly assigns output from compute_run_level_connectivity.
#  - Saves and loads run_level_dataset_info.pkl.
#  - Loads run_level_graph_list.pt directly (bypasses Dataset object issues).
#  - Instantiates model with correct class count (2).
#  - Passes graph_list and dataset_info to downstream functions.

import time
import torch
import pickle
from pathlib import Path

# Register PyTorch Geometric classes as safe globals for serialization
# (Best effort, might depend on PyG version)
try:
    from torch.serialization import add_safe_globals
    from torch_geometric.data import Data # Base class
    # Add other common classes, suppress errors if not found
    safe_classes_to_register = [Data]
    try:
        from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
        safe_classes_to_register.extend([DataEdgeAttr, DataTensorAttr])
    except ImportError: pass
    try:
        from torch_geometric.data.storage import EdgeStorage, GlobalStorage, NodeStorage
        safe_classes_to_register.extend([EdgeStorage, GlobalStorage, NodeStorage])
    except ImportError: pass
    add_safe_globals(list(set(safe_classes_to_register)))
    print("Attempted registration of PyTorch Geometric classes as safe globals.")
except ImportError:
    print("Warning: torch_geometric.data not fully found, skipping some safe global registrations.")
except AttributeError:
     print("Warning: add_safe_globals not available, skipping safe global registration.")
except Exception as e:
    print(f"Warning: Could not register PyG safe globals: {e}")


# Import functions from other PoC modules
from setup_config import OUTPUT_DIR, RANDOM_SEED, SUBJECTS, SESSIONS, N_RUNS_PER_SESSION, setup_environment
from data_preparation import (
    verify_data_location,
    perform_qc_check,
    # create_stimulus_category_mapping, # Removed
    get_atlas_masker,
    extract_roi_time_series,
    clean_extracted_signals,
    compute_run_level_connectivity # Use the correct function name
)
from graph_construction import (
    create_pyg_graph_list,
    create_pyg_dataset # This now just saves the list
    # ConnectivityDataset # No longer needed here
)
from model_definition import get_model
from training import train_model, plot_training_history
from baseline import run_svm_baseline # Assumes baseline.py is updated
from explainability import analyze_attention, run_gnn_explainer # Assumes explainability.py is updated
from visualization import run_visualization # Assumes visualization.py is updated
from evaluation import evaluate_model # Assumes evaluation.py is updated
from robustness import run_robustness_check # Assumes robustness.py is updated

def main():
    """Runs the full PoC pipeline sequentially using RUN-LEVEL analysis."""
    start_time = time.time()
    print("="*50)
    print("=== Starting EmGAT Mini PoC Workflow (Run-Level Analysis) ===")
    print("="*50)
    # Setup environment (directories ensured by setup_config import)
    # setup_environment(RANDOM_SEED) # Called automatically on import
    print(f"Output directory: {OUTPUT_DIR}")

    # --- Variables ---
    file_info = None
    run_connectivity_matrices = None
    graph_list = None   # Will hold the loaded list of graphs
    dataset_info = None # Will hold {'num_classes': 2, 'class_names': ['Sub 01', 'Sub 02']}
    actual_num_classes = None
    trained_model = None
    history = None
    attention_results = None
    gnn_results = None

    # --- Stage 1: Data Preparation ---
    print("\n--- Stage 1: Data Preparation ---")
    try:
        file_info = verify_data_location() # Verifies fmriprep files
        perform_qc_check(file_info)
        # Removed stimulus category mapping
        atlas_masker = get_atlas_masker()
        raw_ts_data = extract_roi_time_series(file_info, atlas_masker)
        cleaned_ts_data = clean_extracted_signals(raw_ts_data, file_info)
        # Compute connectivity per run and ASSIGN the result
        run_connectivity_matrices = compute_run_level_connectivity(cleaned_ts_data)
        if run_connectivity_matrices is None: raise ValueError("Run-level connectivity computation failed or produced no matrices.")
        print("--- Data Preparation Stage Complete ---")
    except Exception as e:
        print(f"\n*** ERROR in Data Preparation Stage: {e} ***")
        print("Pipeline halted.")
        return

    # --- Stage 2: Graph Construction ---
    print("\n--- Stage 2: Graph Construction ---")
    try:
        # Create list returns list, num_classes (2), class_names (subjects)
        graph_list_created, num_classes, class_names = create_pyg_graph_list(
            run_connectivity_matrices
        )
        if graph_list_created is None: raise ValueError("Graph list creation failed.")
        if num_classes is None or num_classes < 2:
             raise ValueError(f"Insufficient number of classes ({num_classes}) identified for subject classification.")

        # Store class info for later stages
        dataset_info = {'num_classes': num_classes, 'class_names': class_names}
        info_path = OUTPUT_DIR / "run_level_dataset_info.pkl"
        try:
            with open(info_path, 'wb') as f: pickle.dump(dataset_info, f)
            print(f"Saved run-level dataset info to: {info_path}")
        except Exception as e:
            # Treat failure to save info as potentially critical
            raise IOError(f"Failed to save dataset info file: {e}")

        # Save the graph list directly using the function from graph_construction
        dataset_storage_path = OUTPUT_DIR / 'pyg_run_level_dataset'
        saved_list_check = create_pyg_dataset(graph_list_created, dataset_storage_path) # Saves the list
        if saved_list_check is None: raise ValueError("Saving graph list failed.")

        # --- Load the saved graph list ---
        list_save_path = dataset_storage_path / 'run_level_graph_list.pt'
        print(f"Loading graph list from: {list_save_path}")
        try:
             # Ensure safe globals are registered if needed before load
             graph_list = torch.load(list_save_path) # Default weights_only=True might fail
             print(f"Successfully loaded graph list ({len(graph_list)} graphs) using default torch.load.")
        except RuntimeError as re:
            if "Unsupported global" in str(re) or "Weights only load failed" in str(re):
                 print(f"  Warning: torch.load failed with default weights_only=True ({re}).")
                 print(f"  Attempting load with weights_only=False (ensure file source is trusted).")
                 try:
                     graph_list = torch.load(list_save_path, weights_only=False)
                     print(f"  Successfully loaded graph list using weights_only=False.")
                 except Exception as e_fallback:
                      print(f"  ERROR: Failed to load graph list even with weights_only=False: {e_fallback}")
                      raise # Re-raise error if fallback fails
            else: raise re # Re-raise other runtime errors
        except Exception as e:
             print(f"  ERROR: Unexpected error loading graph list: {e}")
             raise # Re-raise other errors

        # Validate loaded list
        expected_graphs = len(SUBJECTS) * len(SESSIONS) * N_RUNS_PER_SESSION
        if graph_list is None or len(graph_list) != expected_graphs:
             print(f"Loaded graph list is invalid or incomplete (loaded {len(graph_list) if graph_list else 0} graphs, expected {expected_graphs}).")
             # Decide whether to raise error or just warn
             raise ValueError("Loaded graph list validation failed.")

        print("--- Graph Construction Stage Complete ---")

    except Exception as e:
        print(f"\n*** ERROR in Graph Construction Stage: {e} ***")
        print("Pipeline halted.")
        return

    # --- Check if necessary data exists before proceeding ---
    if graph_list is None or dataset_info is None:
         print("\n*** ERROR: Crucial data (graph_list or dataset_info) missing after graph construction. Pipeline halted. ***")
         return
    actual_num_classes = dataset_info['num_classes'] # Get final class count here

    # --- Stage 3: GAT Model Training ---
    print("\n--- Stage 3: GAT Model Training ---")
    try:
        # Instantiate model using the ACTUAL number of classes (2)
        model_instance = get_model(num_categories=actual_num_classes)
        if model_instance is None: raise ValueError("Failed to instantiate model.")

        # Train the model on the loaded graph_list
        trained_model, history = train_model(model_instance, graph_list)
        if trained_model is None: raise ValueError("Model training failed.")
        plot_training_history(history)
        print("--- Model Training Stage Complete ---")
    except Exception as e:
        print(f"\n*** ERROR in Model Training Stage: {e} ***")
        print("Pipeline halted.")
        return

    # --- Stage 4: Baseline Model (Optional) ---
    # Assumes baseline.py is updated for run-level matrices and dataset_info
    print("\n--- Stage 4: Baseline SVM (Optional) ---")
    try:
        run_svm_baseline(run_connectivity_matrices, dataset_info)
        print("--- Baseline SVM Stage Complete (or skipped/failed) ---")
    except Exception as e:
        print(f"\n* Warning: Baseline SVM failed: {e} *")

    # --- Stage 5: Explainability ---
    # Assumes explainability.py is updated for graph_list and dataset_info
    # Also assumes GNNExplainer error related to epochs/config is fixed
    print("\n--- Stage 5: Explainability Analysis ---")
    attention_results = None
    gnn_results = None
    try:
        attention_results = analyze_attention(trained_model, graph_list, dataset_info)
        gnn_results = run_gnn_explainer(trained_model, graph_list, dataset_info)
        print("--- Explainability Analysis Stage Complete (or skipped/failed) ---")
    except Exception as e:
        print(f"\n* Warning: Explainability analysis failed: {e} *")

    # --- Stage 6: Visualization ---
    # Assumes visualization.py is updated and coordinate loading works
    print("\n--- Stage 6: Visualization ---")
    try:
        run_visualization(attention_results, gnn_results)
        print("--- Visualization Stage Complete (or skipped/failed) ---")
    except Exception as e:
        print(f"\n* Warning: Visualization failed: {e} *")

    # --- Stage 7: Performance Evaluation ---
    # Assumes evaluation.py is updated for graph_list and dataset_info
    print("\n--- Stage 7: Performance Evaluation ---")
    try:
        evaluate_model(trained_model, graph_list, dataset_info)
        print("--- Performance Evaluation Stage Complete (or skipped/failed) ---")
    except Exception as e:
        print(f"\n* Warning: Performance evaluation failed: {e} *")

    # --- Stage 8: Robustness Check (Optional) ---
    # Assumes robustness.py is updated for graph_list, dataset_info, num_classes
    print("\n--- Stage 8: Robustness Check (Optional) ---")
    try:
        run_robustness_check(graph_list, dataset_info, actual_num_classes)
        print("--- Robustness Check Stage Complete (or skipped/failed) ---")
    except Exception as e:
        print(f"\n* Warning: Robustness check failed: {e} *")

    # --- Completion ---
    end_time = time.time()
    print("\n" + "="*50)
    print(f"=== EmGAT Mini PoC Workflow Completed (Run-Level Analysis) ===")
    print(f"Total execution time: {time.strftime('%H:%M:%S', time.gmtime(end_time - start_time))}")
    print(f"Outputs saved in: {OUTPUT_DIR}")
    print("="*50)

if __name__ == "__main__":
    main()