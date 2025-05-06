# Filename: robustness.py
# Location: <workspace_root>/robustness.py
# Description: Optional Step 12 - Checks stability of GNNExplainer results across
#              different training runs (seeds) for TRIAL-LEVEL category classification.

import torch
import numpy as np
import pickle
import random
from pathlib import Path

# Import relevant functions from other scripts
from setup_config import OUTPUT_DIR, setup_environment, RANDOM_SEED
from model_definition import get_model
from training import train_model
from explainability import run_gnn_explainer

N_ROBUSTNESS_RUNS = 3 # Number of different seeds to try (including the original)

def run_robustness_check(graph_list, dataset_info):
    """
    Performs robustness check by training/explaining with multiple seeds for
    trial-level category classification.

    Args:
        graph_list (list): List of PyG Data objects (trial-level graphs).
        dataset_info (dict): Dictionary containing 'num_categories' and 'category_names'.

    Returns:
        dict: Dictionary containing explanation results keyed by seed, or None.
    """
    print(f"--- Running Robustness Check (Trial-Level) with {N_ROBUSTNESS_RUNS} seeds ---")
    if not graph_list or not dataset_info:
        print("  ERROR: Graph list or dataset info not provided. Cannot run check.")
        return None

    try:
        actual_num_categories = dataset_info['num_categories']
        if actual_num_categories < 1:
            print(f"  ERROR: Invalid 'num_categories' ({actual_num_categories}) in dataset_info. Need >= 1.")
            return None
    except KeyError:
        print("  ERROR: 'num_categories' not found in dataset_info.")
        return None

    original_seed = RANDOM_SEED
    additional_seeds = []
    while len(additional_seeds) < N_ROBUSTNESS_RUNS - 1:
        new_seed = random.randint(0, 10000)
        if new_seed != original_seed and new_seed not in additional_seeds:
             additional_seeds.append(new_seed)
    seeds_to_run = [original_seed] + additional_seeds

    all_explanation_results = {} # Store results keyed by seed

    for i, seed in enumerate(seeds_to_run):
        print(f"\n--- Running Seed {i+1}/{N_ROBUSTNESS_RUNS} ({seed}) ---")

        setup_environment(seed)

        model = get_model(num_categories=actual_num_categories)
        if model is None:
             print(f"  ERROR: Failed to get model for seed {seed}. Skipping run.")
             continue

        print("  Re-training model for current seed...")
        trained_model, _ = train_model(model, graph_list)
        if trained_model is None:
            print(f"  ERROR: Training failed for seed {seed}. Skipping explanation.")
            continue

        print("  Running GNNExplainer for current seed...")
        explanation_results = run_gnn_explainer(trained_model, graph_list, dataset_info)
        if explanation_results:
            all_explanation_results[seed] = explanation_results
        else:
            print(f"  Warning: GNNExplainer failed or returned no results for seed {seed}.")
            all_explanation_results[seed] = None

    print("\n--- Robustness Check Complete ---")
    robustness_save_path = OUTPUT_DIR / "robustness_check_results_trial_level.pkl"
    try:
        with open(robustness_save_path, 'wb') as f:
            pickle.dump(all_explanation_results, f)
        print(f"  Saved robustness check results to: {robustness_save_path}")
    except Exception as e:
        print(f"  Warning: Could not save robustness results: {e}")

    compare_explanations(all_explanation_results)

    return all_explanation_results

def compare_explanations(all_explanations):
    """Provides a basic qualitative comparison of edge masks across seeds."""
    print("\n--- Comparing Explanations Across Seeds (Qualitative) ---")
    if not all_explanations:
         print("  No explanation results dictionary provided.")
         return

    valid_seed_results = {seed: res for seed, res in all_explanations.items() if res is not None and isinstance(res, dict)}

    if len(valid_seed_results) < 2:
        print(f"  Need results from at least 2 seeds for comparison. Found {len(valid_seed_results)} valid sets.")
        return

    seeds = list(valid_seed_results.keys())
    first_seed_results = valid_seed_results[seeds[0]]

    first_graph_id_to_compare = None
    base_edge_mask = None

    # Iterate through graphs in the first seed's results to find a valid one
    for graph_id, explanation in first_seed_results.items():
        if isinstance(explanation, dict) and explanation.get('edge_mask') is not None:
            first_graph_id_to_compare = graph_id
            base_edge_mask = explanation['edge_mask']
            break

    if base_edge_mask is None or first_graph_id_to_compare is None:
         print(f"  No valid explanation with edge_mask found for any graph in the first seed ({seeds[0]}) results. Cannot compare.")
         return

    print(f"  Comparing edge masks for graph: {first_graph_id_to_compare} (using seed {seeds[0]} as baseline)")

    correlations = {}
    for seed in seeds[1:]:
        current_seed_explanations = valid_seed_results.get(seed, {})
        current_explanation = current_seed_explanations.get(first_graph_id_to_compare)

        if isinstance(current_explanation, dict) and current_explanation.get('edge_mask') is not None:
            current_edge_mask = current_explanation['edge_mask']
            if base_edge_mask.shape == current_edge_mask.shape:
                try:
                    mask1 = np.nan_to_num(base_edge_mask.flatten())
                    mask2 = np.nan_to_num(current_edge_mask.flatten())
                    if np.std(mask1) > 1e-6 and np.std(mask2) > 1e-6:
                        corr = np.corrcoef(mask1, mask2)[0, 1]
                    else:
                        corr = np.nan
                    correlations[seed] = corr
                    print(f"    Seed {seed} vs Seed {seeds[0]}: Edge mask correlation = {corr:.3f}")
                except Exception as e:
                     print(f"    Seed {seed}: Error calculating correlation for graph {first_graph_id_to_compare}: {e}")
                     correlations[seed] = np.nan
            else:
                print(f"    Seed {seed}: Cannot compare (mask shape mismatch for graph {first_graph_id_to_compare}).")
                correlations[seed] = np.nan
        else:
             print(f"    Seed {seed}: Explanation or edge_mask missing for graph {first_graph_id_to_compare}.")
             correlations[seed] = np.nan

    corr_values = [c for c in correlations.values() if not np.isnan(c)]
    if corr_values:
         avg_corr = np.mean(corr_values)
         print(f"\n  Average correlation with first seed ({seeds[0]} for graph {first_graph_id_to_compare}): {avg_corr:.3f} ({len(corr_values)} comparisons)")
    else:
         print(f"\n  Could not compute average correlation for graph {first_graph_id_to_compare} (no valid comparisons).")
    print("  (Note: High correlation suggests similar important edges identified across runs)")


# --- Main Execution Function (Updated for Trial-Level) ---
if __name__ == "__main__":
    print("Executing Robustness Check Pipeline (Trial-Level)...")

    # --- Load trial-level graph list and dataset info ---
    dataset_storage_path = OUTPUT_DIR / 'pyg_trial_level_dataset'
    list_save_path = dataset_storage_path / 'trial_level_graph_list.pt'
    info_path = OUTPUT_DIR / "trial_level_dataset_info.pkl"

    graph_list = None
    dataset_info = None

    try:
        print(f"Loading graph list from {list_save_path}...")
        try: graph_list = torch.load(list_save_path)
        except RuntimeError: graph_list = torch.load(list_save_path, weights_only=False)
        print(f"Loaded graph list with {len(graph_list)} graphs.")

        with open(info_path, 'rb') as f:
             dataset_info = pickle.load(f)
             print(f"Loaded dataset info. Categories: {dataset_info.get('num_categories')}, Names: {dataset_info.get('category_names')}")

    except FileNotFoundError as e:
        print(f"ERROR: Cannot find required input file: {e.filename}. Please run previous steps.")
        exit(1)
    except Exception as e:
        print(f"ERROR loading files: {e}")
        exit(1)

    if not graph_list or not dataset_info or 'num_categories' not in dataset_info:
        print("Exiting due to missing or incomplete data.")
        exit(1)

    robustness_results = run_robustness_check(graph_list, dataset_info)

    if robustness_results:
        print("\nRobustness Check Pipeline Complete (Trial-Level).")
    else:
        print("\nRobustness Check Pipeline Failed or produced no results (Trial-Level).")