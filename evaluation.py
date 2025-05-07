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
    Evaluates the trained model on the run-level graph list and generates metrics
    for subject classification.

    Args:
        model (torch.nn.Module): The trained GAT model instance.
        graph_list (list): The list of PyG Data objects (run-level graphs).
        dataset_info (dict): Dictionary containing 'num_classes' (number of subjects)
                             and 'class_names' (subject IDs).

    Returns:
        dict or None: Dictionary containing evaluation results, or None on error.
    """
    print("--- Evaluating Trained GAT Model (Run-Level Subject Classification) ---")
    if not graph_list or not dataset_info:
        print("  ERROR: Graph list or dataset info not provided. Cannot evaluate.")
        return None

    # --- Use class info from dataset_info ---
    try:
        class_names = dataset_info['class_names'] # e.g., subject IDs
        num_classes = dataset_info['num_classes'] # e.g., number of subjects
        if num_classes != len(class_names) or num_classes < 1:
             raise ValueError("Invalid class info in dataset_info.")
        report_labels = list(range(num_classes))
    except KeyError as e:
        print(f"  ERROR: Missing expected key {e} in dataset_info dictionary.")
        print("         Ensure graph_construction.py saved 'run_level_dataset_info.pkl' correctly.")
        return None
    except ValueError as ve:
         print(f"  ERROR: {ve}")
         return None
    # --- End Use class info ---

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval() 

    all_preds = []
    all_labels = []
    all_graph_ids = []

    print(f"  Evaluating on {len(graph_list)} run graphs using {num_classes} classes (subjects): {class_names}...")
    with torch.no_grad():
        for i, graph in enumerate(graph_list):
            graph = graph.to(device)
            # Construct graph ID using available attributes
            try:
                # graph_id = f"Trial_{graph.trial_id}_Sub_{graph.subject}_Ses_{graph.session}_Cat_{graph.category_name}"
                graph_id = f"Run_{graph.run_id}_Sub_{graph.subject_id}_Ses_{graph.session_id}"
            except AttributeError:
                graph_id = f"Graph_{i}" # Fallback ID

            try:
                # Forward pass
                output = model(graph.x, graph.edge_index, graph.edge_attr, graph.batch if hasattr(graph, 'batch') else None)
                logits = output[0] if isinstance(output, tuple) else output

                if logits.shape[1] != num_classes:
                     print(f"\nFATAL ERROR: Model output size {logits.shape[1]} does not match expected classes {num_classes} from dataset info. Mismatched model loaded?")
                     return None # Stop evaluation

                predicted_class = logits.argmax(dim=1).item()
                true_label = graph.y.item() # Should be the subject index

                all_preds.append(predicted_class)
                all_labels.append(true_label)
                all_graph_ids.append(graph_id)
            except Exception as e:
                 print(f"\nERROR during model prediction for {graph_id}: {e}")
                 return None # Stop evaluation if prediction fails

    # --- Calculate metrics using class info ---
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
        print(f"  Warning: Unique labels found in data {unique_labels_in_data} do not exactly match expected labels {report_labels}. Check data/subject labels.")
        # Use labels found in data for report if mismatch, but log warning
        report_labels_used = unique_labels_in_data
        try:
             # Get names only for labels actually present
             report_target_names_used = [class_names[i] for i in report_labels_used if i < len(class_names)]
        except IndexError:
             print("  ERROR: Cannot get target names for labels found in data. Using numerical labels.")
             report_target_names_used = [str(l) for l in report_labels_used]
    else:
        report_labels_used = report_labels
        report_target_names_used = class_names

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
    # Use class_names (subject IDs) for labeling the axes
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    print("\n  Confusion Matrix:")
    print(cm_df)

    # Plot Confusion Matrix
    plt.figure(figsize=(max(6, num_classes * 1.2), max(4, num_classes * 0.8))) # Adjust size
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual Subject')
    plt.xlabel('Predicted Subject')
    plt.title(f'Confusion Matrix (Run-Level Data, N={len(graph_list)})')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    cm_save_path = OUTPUT_DIR / "confusion_matrix_run_level.png" # Updated filename
    try:
        plt.savefig(cm_save_path)
        print(f"\n  Confusion matrix plot saved to: {cm_save_path.name}")
    except Exception as e:
        print(f"  Warning: Failed to save confusion matrix plot: {e}")
    plt.close() # Close the plot figure

    # --- End use class info ---

    evaluation_results = {
        'accuracy': accuracy,
        'classification_report_str': report_str,
        'classification_report_dict': report_dict,
        'confusion_matrix_df': cm_df,
        'predictions': all_preds,
        'true_labels': all_labels,
        'graph_ids': all_graph_ids,
        'num_classes': num_classes, # Store info used (number of subjects)
        'class_names': class_names # Store info used (subject IDs)
    }

    # Save evaluation results
    eval_save_path = OUTPUT_DIR / "evaluation_results_run_level.pkl" # Updated filename
    try:
        with open(eval_save_path, 'wb') as f:
            pickle.dump(evaluation_results, f)
        print(f"  Saved evaluation results to: {eval_save_path}")
    except Exception as e:
        print(f"  Warning: Could not save evaluation results: {e}")

    return evaluation_results

# --- Main Execution Function (Updated for Run-Level) ---
if __name__ == "__main__":
    print("Executing Model Evaluation Pipeline (Run-Level)...")

    # --- Load run-level graph list, dataset info, and model ---
    dataset_storage_path = OUTPUT_DIR / 'pyg_run_level_dataset' # Path for run-level data
    list_save_path = dataset_storage_path / 'run_level_graph_list.pt' # Filename for run-level graph list
    info_path = OUTPUT_DIR / "run_level_dataset_info.pkl" # Filename for run-level info
    model_path = OUTPUT_DIR / "gat_model_run_level.pt" # Filename for run-level model

    graph_list = None
    dataset_info = None
    actual_num_classes = None # Will store number of subjects

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
             actual_num_classes = dataset_info['num_classes'] # num_subjects
             actual_class_names = dataset_info['class_names'] # subject_ids
             print(f"Loaded dataset info. Classes (Subjects): {actual_num_classes} ({actual_class_names})")

        # Instantiate model with the actual number of classes (subjects)
        model = get_model(num_classes=actual_num_classes)
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
            print("\nModel Evaluation Complete (Run-Level).")
            print(f"Accuracy: {evaluation_results['accuracy']:.4f}")
        else:
            print("\nModel Evaluation Failed (Run-Level).")
    else:
        print("\nSkipping evaluation due to loading errors.")