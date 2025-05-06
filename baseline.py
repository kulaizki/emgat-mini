import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.exceptions import UndefinedMetricWarning
import warnings
from pathlib import Path

# Import configuration variables
from setup_config import (
    OUTPUT_DIR, RANDOM_SEED # Added RANDOM_SEED for consistency
)

def run_svm_baseline(trial_data_dict, dataset_info):
    """
    Trains and evaluates an SVM on flattened trial-level connectivity matrices
    for visual object category classification.

    Args:
        trial_data_dict (dict): Nested dict {sub: {ses: [trial_info]}} from data_prep.
        dataset_info (dict): Dictionary containing 'num_categories'
                             and 'category_names'.

    Returns:
        tuple: (accuracy, report_string) or (None, None) on failure.
    """
    print("--- Running Baseline SVM Classifier (Trial-Level Category Classification) ---")
    if not trial_data_dict or not dataset_info:
        print("  ERROR: Missing trial data dictionary or dataset info. Cannot run baseline.")
        return None, None

    if 'num_categories' not in dataset_info or 'category_names' not in dataset_info:
        print("  ERROR: Dataset info incomplete. Missing 'num_categories' or 'category_names'.")
        return None, None

    num_categories = dataset_info['num_categories']
    category_names = dataset_info['category_names']
    report_labels = list(range(num_categories))

    features = []
    labels = []

    # Prepare feature matrix (X) and label vector (y) from trial-level data
    print("  Preparing features and labels from trial data...")
    skipped_trials = 0
    for sub, sessions in trial_data_dict.items():
        for ses, trials in sessions.items():
            for trial_info in trials:
                matrix = trial_info.get('connectivity')
                category_index = trial_info.get('category_label_index')

                if matrix is not None and category_index is not None:
                    # Use the full flattened matrix
                    flat_features = matrix.flatten()
                    features.append(flat_features)
                    labels.append(category_index) # Use category index as label
                else:
                    skipped_trials += 1

    if skipped_trials > 0:
        print(f"    Skipped {skipped_trials} trials due to missing connectivity or category index.")

    if not features:
        print("  ERROR: No valid feature vectors created for baseline. Check trial data.")
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
        target_names_present = [category_names[i] for i in labels_present if i < len(category_names)]

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
    print("  Classification Report (for classes present in data):")
    print(report)

    baseline_results = {
        'accuracy': accuracy,
        'report': report,
    }

    # Save results
    baseline_results_path = OUTPUT_DIR / "baseline_svm_trial_level_results.pkl" # Updated filename
    try:
        with open(baseline_results_path, 'wb') as f:
            pickle.dump(baseline_results, f)
        print(f"  Saved baseline results to: {baseline_results_path}")
    except Exception as e:
        print(f"  Warning: Could not save baseline results: {e}")

    return accuracy, report


# --- Main Execution Function (Updated for Trial-Level) ---
if __name__ == "__main__":
    print("Executing Baseline SVM Pipeline (Trial-Level)...")

    # Load trial-level data dictionary and dataset info
    trial_data_path = OUTPUT_DIR / "trial_level_data.pkl" # Updated path
    info_path = OUTPUT_DIR / "trial_level_dataset_info.pkl" # Updated path

    trial_data_dict = None
    dataset_info = None

    try:
        with open(trial_data_path, 'rb') as f:
            trial_data_dict = pickle.load(f)
        with open(info_path, 'rb') as f:
             dataset_info = pickle.load(f)
        print("Loaded trial-level data dictionary and dataset info.")

    except FileNotFoundError as e:
        print(f"ERROR: Cannot find required input file: {e.filename}")
        print("Please run data_preparation.py and graph_construction.py first.")
        exit(1)
    except Exception as e:
        print(f"ERROR loading input files: {e}")
        exit(1)

    # Run baseline
    svm_accuracy, svm_report = run_svm_baseline(trial_data_dict, dataset_info)

    if svm_accuracy is not None:
        print("\nBaseline SVM Pipeline Complete.")
    else:
        print("\nBaseline SVM Pipeline Failed.")