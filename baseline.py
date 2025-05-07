import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.exceptions import UndefinedMetricWarning
import warnings
from pathlib import Path

from setup_config import (
    OUTPUT_DIR, RANDOM_SEED # Added RANDOM_SEED for consistency
)

def run_svm_baseline(run_connectivity_matrices, dataset_info):
    """
    Trains and evaluates an SVM on flattened run-level connectivity matrices
    for subject classification.

    Args:
        run_connectivity_matrices (dict): Nested dict {sub: {ses: {run: matrix}}} from data_prep.
        dataset_info (dict): Dictionary containing 'num_classes' (number of subjects)
                             and 'class_names' (subject IDs).

    Returns:
        tuple: (accuracy, report_string) or (None, None) on failure.
    """
    print("--- Running Baseline SVM Classifier (Run-Level Subject Classification) ---")
    if not run_connectivity_matrices or not dataset_info:
        print("  ERROR: Missing run connectivity matrices or dataset info. Cannot run baseline.")
        return None, None

    if 'num_classes' not in dataset_info or 'class_names' not in dataset_info:
        print("  ERROR: Dataset info incomplete. Missing 'num_classes' or 'class_names'.")
        return None, None

    num_classes = dataset_info['num_classes']
    class_names = dataset_info['class_names'] # These are subject IDs
    report_labels = list(range(num_classes))
    subject_to_index = {sub_id: i for i, sub_id in enumerate(class_names)}

    features = []
    labels = []

    # Prepare feature matrix (X) and label vector (y) from run-level data
    print("  Preparing features and labels from run data...")
    skipped_runs = 0
    processed_runs = 0
    for sub_id, sessions in run_connectivity_matrices.items():
        if sub_id not in subject_to_index:
            print(f"    Warning: Subject ID {sub_id} from data not in configured class_names (subject IDs). Skipping their runs.")
            skipped_runs += sum(len(runs) for runs in sessions.values())
            continue
        
        subject_label_index = subject_to_index[sub_id]

        for ses_id, runs in sessions.items():
            for run_id, matrix in runs.items():
                processed_runs +=1
                if matrix is not None:
                    flat_features = matrix.flatten()
                    features.append(flat_features)
                    labels.append(subject_label_index) # Use subject index as label
                else:
                    print(f"    Skipping run {sub_id}/{ses_id}/{run_id} due to missing connectivity matrix.")
                    skipped_runs += 1

    if skipped_runs > 0:
        print(f"    Skipped {skipped_runs} runs out of {processed_runs} processed runs due to missing data or unknown subject.")

    if not features:
        print("  ERROR: No valid feature vectors created for baseline. Check run connectivity data.")
        return None, None

    X = np.array(features)
    y = np.array(labels)

    print(f"  Baseline feature matrix shape: {X.shape}")
    print(f"  Baseline label vector shape: {y.shape}")
    unique_labels, counts = np.unique(y, return_counts=True)
    print(f"  Unique labels found: {unique_labels} with counts: {counts}")

    if X.shape[0] < 2 or len(unique_labels) < 1:
        print(f"  ERROR: Need at least 2 samples and at least 1 class represented to train SVM.")
        return None, None

    # Train SVM (multi-class)
    # Linear kernel is often a good baseline
    svm = SVC(kernel='linear', C=1.0, class_weight='balanced', decision_function_shape='ovr', random_state=RANDOM_SEED)
    print(f"  Training SVM ({svm.kernel} kernel, OVR)...")
    try:
        svm.fit(X, y)
    except ValueError as ve:
        if "The number of classes has to be greater than one" in str(ve):
            print(f"  ERROR: Only one class found in the data ({unique_labels}). Cannot train multi-class SVM.")
            return None, None
        else:
            print(f"  ERROR during SVM fitting: {ve}")
            return None, None
    except Exception as e:
        print(f"  ERROR during SVM fitting: {e}")
        return None, None

    # Evaluate on the same data (no train/test split here for simplicity)
    print("  Evaluating SVM on training data...")
    y_pred = svm.predict(X)

    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)

    # Classification Report
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        # Check labels found vs expected
        labels_present = sorted(np.unique(y).tolist())
        # Use class_names (subject_ids) for target names
        target_names_present = [class_names[i] for i in labels_present if i < len(class_names)]

        report = classification_report(y, y, # Using y vs y to get report structure for present classes
                                       labels=labels_present,
                                       target_names=target_names_present,
                                       zero_division=0)
        # Now generate the actual report
        report = classification_report(y, y_pred,
                                       labels=labels_present,
                                       target_names=target_names_present,
                                       zero_division=0)

    print("--- Baseline SVM Results ---")
    print(f"  Accuracy: {accuracy:.4f}")
    print("  Classification Report (for subjects present in data):")
    print(report)

    baseline_results = {
        'accuracy': accuracy,
        'report': report,
    }

    # Save results
    baseline_results_path = OUTPUT_DIR / "baseline_svm_run_level_results.pkl" # Updated filename
    try:
        with open(baseline_results_path, 'wb') as f:
            pickle.dump(baseline_results, f)
        print(f"  Saved baseline results to: {baseline_results_path}")
    except Exception as e:
        print(f"  Warning: Could not save baseline results: {e}")

    return accuracy, report


# --- Main Execution Function (Updated for Trial-Level) ---
if __name__ == "__main__":
    print("Executing Baseline SVM Pipeline (Run-Level)...")

    # Load run-level data dictionary and dataset info
    run_data_path = OUTPUT_DIR / "run_level_connectivity_matrices.pkl" 
    info_path = OUTPUT_DIR / "run_level_dataset_info.pkl" 

    run_connectivity_matrices_loaded = None
    dataset_info_loaded = None

    try:
        with open(run_data_path, 'rb') as f:
            run_connectivity_matrices_loaded = pickle.load(f)
        with open(info_path, 'rb') as f:
             dataset_info_loaded = pickle.load(f)
        print("Loaded run-level connectivity matrices and dataset info.")

    except FileNotFoundError as e:
        print(f"ERROR: Cannot find required input file: {e.filename}")
        print("Please run data_preparation.py and graph_construction.py first.")
        exit(1)
    except Exception as e:
        print(f"ERROR loading input files: {e}")
        exit(1)

    # Run baseline
    svm_accuracy, svm_report = run_svm_baseline(run_connectivity_matrices_loaded, dataset_info_loaded)

    if svm_accuracy is not None:
        print("\nBaseline SVM Pipeline Complete.")
    else:
        print("\nBaseline SVM Pipeline Failed.")