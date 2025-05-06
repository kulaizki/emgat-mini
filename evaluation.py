import torch
import pickle
import numpy as np
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
from sklearn.exceptions import UndefinedMetricWarning
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from setup_config import OUTPUT_DIR
from model_definition import get_model

def evaluate_model(model, graph_list, dataset_info):
    """
    Evaluates the trained model on the trial-level graph list and generates metrics
    for category classification.

    Args:
        model (torch.nn.Module): The trained GAT model instance.
        graph_list (list): The list of PyG Data objects (trial-level graphs).
        dataset_info (dict): Dictionary containing 'num_categories'
                             and 'category_names'.

    Returns:
        dict or None: Dictionary containing evaluation results, or None on error.
    """
    print("--- Evaluating Trained GAT Model (Trial-Level) ---")
    if not graph_list or not dataset_info:
        print("  ERROR: Graph list or dataset info not provided. Cannot evaluate.")
        return None

    # --- Use category info from dataset_info ---
    try:
        category_names = dataset_info['category_names']
        num_categories = dataset_info['num_categories']
        if num_categories != len(category_names) or num_categories < 1:
             raise ValueError("Invalid category info in dataset_info.")
        report_labels = list(range(num_categories))
    except KeyError as e:
        print(f"  ERROR: Missing expected key {e} in dataset_info dictionary.")
        print("         Ensure graph_construction.py saved 'trial_level_dataset_info.pkl' correctly.")
        return None
    except ValueError as ve:
         print(f"  ERROR: {ve}")
         return None
    # --- End Use category info ---

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval() 

    all_preds = []
    all_labels = []
    all_graph_ids = []

    print(f"  Evaluating on {len(graph_list)} trial graphs using {num_categories} categories: {category_names}...")
    with torch.no_grad():
        for i, graph in enumerate(graph_list):
            graph = graph.to(device)
            # Construct graph ID using available attributes
            try:
                graph_id = f"Trial_{graph.trial_id}_Sub_{graph.subject}_Ses_{graph.session}_Cat_{graph.category_name}"
            except AttributeError:
                graph_id = f"Graph_{i}" # Fallback ID

            try:
                # Forward pass
                output = model(graph.x, graph.edge_index, graph.edge_attr, graph.batch if hasattr(graph, 'batch') else None)
                logits = output[0] if isinstance(output, tuple) else output

                if logits.shape[1] != num_categories:
                     print(f"\nFATAL ERROR: Model output size {logits.shape[1]} does not match expected categories {num_categories} from dataset info. Mismatched model loaded?")
                     return None # Stop evaluation

                predicted_class = logits.argmax(dim=1).item()
                true_label = graph.y.item() # Should be the category label index

                all_preds.append(predicted_class)
                all_labels.append(true_label)
                all_graph_ids.append(graph_id)
            except Exception as e:
                 print(f"\nERROR during model prediction for {graph_id}: {e}")
                 return None # Stop evaluation if prediction fails

    # --- Calculate metrics using category info ---
    if not all_labels:
        print("  ERROR: No labels collected for evaluation. Check graph list.")
        return None

    accuracy = accuracy_score(all_labels, all_preds)
    print(f"\n  Overall Accuracy: {accuracy:.4f}")

    # Classification Report
    print("\n  Classification Report:")
    # Check labels found vs expected
    unique_labels_in_data = sorted(np.unique(all_labels).tolist())
    if set(unique_labels_in_data) != set(report_labels):
        print(f"  Warning: Unique labels found in data {unique_labels_in_data} do not exactly match expected labels {report_labels}. Check data/categories.")
        # Use labels found in data for report if mismatch, but log warning
        report_labels_used = unique_labels_in_data
        try:
             # Get names only for labels actually present
             report_target_names_used = [category_names[i] for i in report_labels_used if i < len(category_names)]
        except IndexError:
             print("  ERROR: Cannot get target names for labels found in data. Using numerical labels.")
             report_target_names_used = [str(l) for l in report_labels_used]
    else:
        report_labels_used = report_labels
        report_target_names_used = category_names

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        report_str = classification_report(
            all_labels,
            all_preds,
            labels=report_labels_used,
            target_names=report_target_names_used,
            zero_division=0,
            digits=3
        )
        try:
             report_dict = classification_report(
                 all_labels,
                 all_preds,
                 labels=report_labels_used,
                 target_names=report_target_names_used,
                 zero_division=0,
                 output_dict=True
             )
        except ValueError as e:
             print(f"  Warning: Could not generate classification report dictionary: {e}")
             report_dict = {}

    print(report_str)

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds, labels=report_labels)
    # Use category_names for labeling the axes
    cm_df = pd.DataFrame(cm, index=category_names, columns=category_names)
    print("\n  Confusion Matrix:")
    print(cm_df)

    # Plot Confusion Matrix
    plt.figure(figsize=(max(6, num_categories * 1.2), max(4, num_categories * 0.8))) # Adjust size
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues",
                xticklabels=category_names, yticklabels=category_names)
    plt.ylabel('Actual Category')
    plt.xlabel('Predicted Category')
    plt.title(f'Confusion Matrix (Trial Data, N={len(graph_list)})')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    cm_save_path = OUTPUT_DIR / "confusion_matrix_trial_level.png" # Updated filename
    try:
        plt.savefig(cm_save_path)
        print(f"\n  Confusion matrix plot saved to: {cm_save_path.name}")
    except Exception as e:
        print(f"  Warning: Failed to save confusion matrix plot: {e}")
    plt.close() # Close the plot figure

    # --- End use category info ---

    evaluation_results = {
        'accuracy': accuracy,
        'classification_report_str': report_str,
        'classification_report_dict': report_dict,
        'confusion_matrix_df': cm_df,
        'predictions': all_preds,
        'true_labels': all_labels,
        'graph_ids': all_graph_ids,
        'num_categories': num_categories, # Store info used
        'category_names': category_names # Store info used
    }

    # Save evaluation results
    eval_save_path = OUTPUT_DIR / "evaluation_results_trial_level.pkl" # Updated filename
    try:
        with open(eval_save_path, 'wb') as f:
            pickle.dump(evaluation_results, f)
        print(f"  Saved evaluation results to: {eval_save_path}")
    except Exception as e:
        print(f"  Warning: Could not save evaluation results: {e}")

    return evaluation_results

# --- Main Execution Function (Updated for Trial-Level) ---
if __name__ == "__main__":
    print("Executing Model Evaluation Pipeline (Trial-Level)...")

    # --- Load trial-level graph list, dataset info, and model ---
    dataset_storage_path = OUTPUT_DIR / 'pyg_trial_level_dataset' # Updated path
    list_save_path = dataset_storage_path / 'trial_level_graph_list.pt' # Updated filename
    info_path = OUTPUT_DIR / "trial_level_dataset_info.pkl" # Updated filename
    model_path = OUTPUT_DIR / "gat_model_trial_level.pt" # Updated filename

    graph_list = None
    dataset_info = None
    actual_num_categories = None

    try:
        # Load the list of graph objects
        print(f"Loading graph list from {list_save_path}...")
        try:
             graph_list = torch.load(list_save_path)
        except RuntimeError as re:
            if "Unsupported global" in str(re) or "Weights only load failed" in str(re):
                 print(f"  Warning: torch.load failed with default weights_only=True. Retrying with weights_only=False.")
                 graph_list = torch.load(list_save_path, weights_only=False)
            else: raise re
        print(f"Loaded graph list with {len(graph_list)} graphs.")

        # Load the dataset info dictionary
        with open(info_path, 'rb') as f:
             dataset_info = pickle.load(f)
             actual_num_categories = dataset_info['num_categories']
             actual_category_names = dataset_info['category_names']
             print(f"Loaded dataset info. Categories: {actual_num_categories} ({actual_category_names})")

        # Instantiate model with the actual number of categories
        model = get_model(num_categories=actual_num_categories)
        # Load the trained model state
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Loaded dataset info and trained model.")

    except FileNotFoundError as fnf:
        print(f"ERROR: Required file not found: {fnf.filename}. Cannot proceed.")
        print("Ensure previous steps (graph construction, training) ran successfully.")
        exit(1)
    except Exception as e:
        print(f"ERROR loading data or model: {e}")
        exit(1)

    # --- Perform Evaluation ---
    if model and graph_list and dataset_info:
        evaluation_results = evaluate_model(model, graph_list, dataset_info)
        if evaluation_results:
            print("\nModel Evaluation Complete (Trial-Level).")
            print(f"Accuracy: {evaluation_results['accuracy']:.4f}")
        else:
            print("\nModel Evaluation Failed (Trial-Level).")
    else:
        print("\nSkipping evaluation due to loading errors.")