# Filename: data_preparation.py
# Location: <workspace_root>/data_preparation.py
# Description: Data verification, QC, ROI extraction, cleaning, and
#              RUN-LEVEL functional connectivity computation.
# Changes:
#  - Removed create_stimulus_category_mapping function.
#  - Renamed and modified aggregate_and_compute_connectivity to
#    compute_run_level_connectivity.
#  - No longer requires events.tsv for connectivity step.

import warnings
import pickle
import re # Keep for potential future use, though not used now
import pandas as pd
import numpy as np
import nibabel as nib
from pathlib import Path
from nilearn import image, plotting, datasets, signal
from nilearn.maskers import NiftiMapsMasker, NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
import matplotlib.pyplot as plt # Import for plotting close
import seaborn as sns # Optional, for potentially nicer heatmaps

# Import configuration variables
from setup_config import (
    FMRIPREP_DIR, BIDS_RAW_DIR, OUTPUT_DIR, SYNSET_WORDS_FILE, DIFUMO_LABELS_FILE,
    ATLAS_DIM, ATLAS_NAME, ATLAS_RESOLUTION_MM, SUBJECTS, SESSIONS, TASK_NAME,
    N_RUNS_PER_SESSION, TR, HEMODYNAMIC_DELAY_TR, CONFOUNDS_TO_USE,
    CLEAN_SIGNAL_PARAMS, CONNECTIVITY_KIND, NILEARN_CACHE_DIR,
    # Removed category-specific config imports
    # Potential coordinate/label config imports remain if needed for visualization link
    DIFUMO_LABEL_INDEX_COL, DIFUMO_LABEL_NAME_COL,
    DIFUMO_COORD_X_COL, DIFUMO_COORD_Y_COL, DIFUMO_COORD_Z_COL
)

def verify_data_location():
    """Locates and verifies required input files for the PoC."""
    print("--- Verifying Data Locations ---")
    required_files_info = {}
    all_files_found = True
    warnings_found = False

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for sub in SUBJECTS:
        required_files_info[sub] = {}
        for ses in SESSIONS:
            required_files_info[sub][ses] = {}
            # Use N_RUNS_PER_SESSION from config
            for run in range(1, N_RUNS_PER_SESSION + 1):
                run_id = f"{run:02d}"
                run_key = f"run-{run_id}"
                # Initialize only fmriprep dict, raw events not needed for run-level FC
                required_files_info[sub][ses][run_key] = {'fmriprep': {}}

                fmriprep_base = FMRIPREP_DIR / f"sub-{sub}" / f"ses-{ses}" / "func"
                # Raw base path removed as events.tsv is not required for this analysis path
                # raw_base = BIDS_RAW_DIR / f"sub-{sub}" / f"ses-{ses}" / "func"
                file_prefix = f"sub-{sub}_ses-{ses}_task-{TASK_NAME}_run-{run_id}"

                fmriprep_files = {
                    'bold': fmriprep_base / f"{file_prefix}_space-T1w_desc-preproc_bold.nii.gz",
                    'confounds_tsv': fmriprep_base / f"{file_prefix}_desc-confounds_timeseries.tsv",
                    'confounds_json': fmriprep_base / f"{file_prefix}_desc-confounds_timeseries.json"
                }
                # raw_files dict removed

                for key, path in fmriprep_files.items():
                    if path.exists():
                        required_files_info[sub][ses][run_key]['fmriprep'][key] = path
                    else:
                        # All fmriprep files are essential for this approach
                        print(f"  ERROR: Missing essential file: {path}")
                        required_files_info[sub][ses][run_key]['fmriprep'][key] = None
                        all_files_found = False

                # Raw files check removed
                # for key, path in raw_files.items(): ...

    # Check non-run-specific files needed for later stages (optional here)
    if not SYNSET_WORDS_FILE.exists():
         print(f"  INFO: Synset words file not found: {SYNSET_WORDS_FILE} (Not needed for run-level analysis).")
         # warnings_found = True # Not necessarily a warning for this workflow
    if not DIFUMO_LABELS_FILE.exists():
         print(f"  WARNING: Missing DiFuMo labels/coords file: {DIFUMO_LABELS_FILE}. Visualization may be affected.")
         warnings_found = True

    if not all_files_found:
        raise FileNotFoundError("Essential fMRIPrep input files are missing. Cannot proceed.")
    elif warnings_found:
        print("Warnings encountered (e.g., missing DiFuMo labels file). Check logs.")
    else:
        print("All required files verified successfully.")

    return required_files_info

def perform_qc_check(required_files_info):
    """(Optional) Loads one BOLD file and plots its mean image using 'inferno' colormap."""
    print("--- Performing Basic QC Check ---")
    try:
        # Select the first available run for QC check
        example_sub, example_ses, example_run = None, None, None
        example_bold_path = None
        for sub in SUBJECTS:
            for ses in SESSIONS:
                # Check runs based on config
                for run in range(1, N_RUNS_PER_SESSION + 1):
                    run_key = f"run-{run:02d}"
                    if run_key in required_files_info[sub][ses]:
                        bold_path_candidate = required_files_info[sub][ses][run_key]['fmriprep'].get('bold')
                        if bold_path_candidate:
                            example_sub, example_ses, example_run = sub, ses, run_key
                            example_bold_path = bold_path_candidate
                            break
                if example_bold_path: break
            if example_bold_path: break

        if example_bold_path:
            print(f"  Loading: {example_bold_path.name}")
            # Use copy_header=True to avoid FutureWarning and preserve metadata
            mean_func_img = image.mean_img(str(example_bold_path), copy_header=True)
            qc_plot_path = OUTPUT_DIR / f"qc_mean_func_sub-{example_sub}_ses-{example_ses}_{example_run}.png"
            plotting.plot_epi(mean_func_img, title="Mean Functional Image (QC)",
                              output_file=str(qc_plot_path), display_mode='ortho',
                              cmap='inferno')
            plt.close() # Close the plot figure using matplotlib
            print(f"  Saved QC plot to: {qc_plot_path}")
        else:
             print("  Skipping QC check: No BOLD file found in info for any specified subject/session/run.")

    except Exception as e:
        print(f"  WARNING: QC check failed: {e}")

# Removed create_stimulus_category_mapping function

# --- Step 3: ROI Time Series Extraction (Unchanged) ---
def get_atlas_masker():
    """Fetches the DiFuMo atlas and initializes a NiftiMapsMasker."""
    print(f"--- Initializing Atlas Masker (DiFuMo {ATLAS_DIM}) ---")
    try:
        difumo_cache_path = NILEARN_CACHE_DIR / 'difumo'
        difumo_cache_path.mkdir(parents=True, exist_ok=True)
        atlas_data = datasets.fetch_atlas_difumo(
            dimension=ATLAS_DIM,
            resolution_mm=ATLAS_RESOLUTION_MM,
            data_dir=str(difumo_cache_path)
        )
        atlas_filename = atlas_data.maps
        print(f"  Using atlas map: {atlas_filename}")

        masker_cache_path = NILEARN_CACHE_DIR / 'nilearn_cache'
        masker_cache_path.mkdir(parents=True, exist_ok=True)
        masker = NiftiMapsMasker(
            maps_img=atlas_filename,
            standardize="zscore_sample",
            memory=str(masker_cache_path),
            memory_level=1,
            verbose=0
        )
        print("  NiftiMapsMasker initialized.")
        return masker
    except Exception as e:
        print(f"  ERROR fetching/loading atlas or initializing masker: {e}")
        raise

def extract_roi_time_series(required_files_info, masker):
    """Extracts ROI time series for all specified runs using the provided masker."""
    print("--- Extracting ROI Time Series ---")
    extracted_ts = {sub: {ses: {} for ses in SESSIONS} for sub in SUBJECTS}

    for sub in SUBJECTS:
        for ses in SESSIONS:
             # Use N_RUNS_PER_SESSION from config
            for run in range(1, N_RUNS_PER_SESSION + 1):
                run_id = f"{run:02d}"
                run_key = f"run-{run_id}"
                # Check if run info exists (might not if verify failed partially)
                if run_key not in required_files_info[sub][ses]:
                    print(f"    Skipping {sub}/{ses}/{run_key}: Info not found.")
                    extracted_ts[sub][ses][run_key] = None
                    continue

                bold_path = required_files_info[sub][ses][run_key]['fmriprep'].get('bold')
                if bold_path:
                    print(f"    Processing: {bold_path.name}")
                    try:
                        time_series = masker.fit_transform(str(bold_path))
                        if np.isnan(time_series).any():
                           print(f"      Warning: NaNs detected in extracted time series for {bold_path.name}. Check BOLD data/mask overlap.")
                        extracted_ts[sub][ses][run_key] = time_series
                    except Exception as e:
                        print(f"      ERROR during extraction for {bold_path.name}: {e}")
                        extracted_ts[sub][ses][run_key] = None
                else:
                    print(f"    Skipping {sub}/{ses}/{run_key}: Missing BOLD file path in info.")
                    extracted_ts[sub][ses][run_key] = None

    print("  ROI time series extraction complete.")
    return extracted_ts

# --- Step 4: Signal Cleaning (Unchanged Logic, but acts on all runs) ---
def clean_extracted_signals(extracted_ts, required_files_info):
    """Cleans the extracted time series using specified confounds."""
    print("--- Cleaning Extracted Time Series ---")
    cleaned_ts = {sub: {ses: {} for ses in SESSIONS} for sub in SUBJECTS}

    for sub in SUBJECTS:
        for ses in SESSIONS:
             # Use N_RUNS_PER_SESSION from config
            for run in range(1, N_RUNS_PER_SESSION + 1):
                run_id = f"{run:02d}"
                run_key = f"run-{run_id}"
                # Check if run info and extracted TS exist
                if run_key not in extracted_ts[sub][ses] or extracted_ts[sub][ses][run_key] is None:
                     print(f"    Skipping cleaning for {sub}/{ses}/{run_key}: Missing raw TS data.")
                     cleaned_ts[sub][ses][run_key] = None
                     continue

                raw_ts = extracted_ts[sub][ses][run_key]
                confounds_path = required_files_info[sub][ses][run_key]['fmriprep'].get('confounds_tsv')

                if confounds_path:
                    print(f"    Cleaning: sub-{sub}_ses-{ses}_{run_key}")
                    try:
                        confounds_df = pd.read_csv(confounds_path, sep='\t')

                        if np.isnan(raw_ts).any():
                            print(f"      ERROR: NaNs found in raw time series for {run_key} before cleaning. Skipping run.")
                            cleaned_ts[sub][ses][run_key] = None
                            continue

                        available_confounds = [c for c in CONFOUNDS_TO_USE if c in confounds_df.columns]
                        missing_confounds = [c for c in CONFOUNDS_TO_USE if c not in confounds_df.columns]
                        if missing_confounds:
                            print(f"      Warning: Missing requested confounds in {confounds_path.name}: {missing_confounds}")

                        if not available_confounds:
                            print(f"      Warning: No specified confounds found in {confounds_path.name}. Cleaning without confounds.")
                            confounds_matrix = None
                        else:
                            confounds_matrix = confounds_df[available_confounds].values
                            if np.isnan(confounds_matrix).any():
                                print("      Imputing NaNs in confounds matrix with column means.")
                                col_mean = np.nanmean(confounds_matrix, axis=0)
                                col_mean = np.nan_to_num(col_mean)
                                inds = np.where(np.isnan(confounds_matrix))
                                confounds_matrix[inds] = np.take(col_mean, inds[1])
                                if np.isnan(confounds_matrix).any():
                                    print(f"      ERROR: NaNs remain in confounds after imputation for {run_key}. Skipping cleaning.")
                                    cleaned_ts[sub][ses][run_key] = None
                                    continue

                        if confounds_matrix is not None and confounds_matrix.shape[0] != raw_ts.shape[0]:
                            print(f"      ERROR: Mismatch between TS length ({raw_ts.shape[0]}) and confounds length ({confounds_matrix.shape[0]}) for {run_key}. Skipping.")
                            cleaned_ts[sub][ses][run_key] = None
                            continue

                        cleaned_signal = signal.clean(
                            raw_ts,
                            confounds=confounds_matrix,
                            **CLEAN_SIGNAL_PARAMS
                        )

                        if np.isnan(cleaned_signal).any():
                            print(f"      ERROR: NaNs detected after cleaning for {run_key}. Skipping run.")
                            cleaned_ts[sub][ses][run_key] = None
                        else:
                            cleaned_ts[sub][ses][run_key] = cleaned_signal

                    except Exception as e:
                        print(f"      ERROR during cleaning for {sub}/{ses}/{run_key}: {e}")
                        cleaned_ts[sub][ses][run_key] = None
                else:
                    print(f"    Skipping cleaning for {sub}/{ses}/{run_key}: Missing confounds file path.")
                    cleaned_ts[sub][ses][run_key] = None

    print("  Signal cleaning complete.")
    # Optional: Save cleaned data dictionary
    # cleaned_data_path = OUTPUT_DIR / "cleaned_timeseries_runlevel.pkl"
    # try:
    #     with open(cleaned_data_path, 'wb') as f:
    #         pickle.dump(cleaned_ts, f)
    #     print(f"  Saved dictionary of cleaned time series to: {cleaned_data_path}")
    # except Exception as e:
    #     print(f"  ERROR saving cleaned time series dictionary: {e}")

    return cleaned_ts

# --- Helper: Plot and Save Connectivity Matrices ---
def save_connectivity_matrix_plots(run_connectivity_matrices, output_dir):
    """Saves computed run-level connectivity matrices as heatmap plots."""
    plot_dir = output_dir / "connectivity_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    print(f"--- Saving Connectivity Matrix Plots to: {plot_dir} ---")
    saved_count = 0
    skipped_count = 0

    if not run_connectivity_matrices:
        print("  No connectivity matrices provided to plot.")
        return

    for sub, sessions in run_connectivity_matrices.items():
        for ses, runs in sessions.items():
            for run_key, matrix in runs.items():
                if matrix is not None and isinstance(matrix, np.ndarray):
                    try:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        # Use seaborn heatmap if available and preferred, else matplotlib imshow
                        if 'sns' in globals():
                            sns.heatmap(matrix, ax=ax, cmap='viridis', square=True,
                                        xticklabels=False, yticklabels=False, 
                                        cbar_kws={"shrink": .5})
                        else:
                            im = ax.imshow(matrix, cmap='viridis', interpolation='nearest')
                            plt.colorbar(im, ax=ax, shrink=0.5)

                        title = f"Connectivity: Sub {sub}, Ses {ses}, Run {run_key}"
                        ax.set_title(title, fontsize=10)
                        
                        filename = f"connectivity_sub-{sub}_ses-{ses}_{run_key}.png"
                        save_path = plot_dir / filename
                        plt.savefig(save_path, dpi=150, bbox_inches='tight')
                        plt.close(fig)
                        saved_count += 1
                    except Exception as e:
                        print(f"    ERROR plotting/saving matrix for {sub}/{ses}/{run_key}: {e}")
                        skipped_count += 1
                else:
                    skipped_count += 1
                    
    print(f"  Saved {saved_count} plots. Skipped {skipped_count} matrices (missing or invalid).")

# --- Step 5: Renamed and Modified Function ---
def compute_run_level_connectivity(cleaned_ts_data):
    """
    Computes correlation matrices for each individual run.
    Returns a nested dictionary: {subject: {session: {run: matrix}}}
    Also saves plots of the matrices.
    """
    print("--- Computing Run-Level Connectivity Matrices (Step 5b) ---")
    if not cleaned_ts_data:
        print("  ERROR: Missing cleaned time series data.")
        return None

    conn_measure = ConnectivityMeasure(kind='correlation', vectorize=False, discard_diagonal=False)
    run_connectivity_matrices = {}
    computed_count = 0
    error_count = 0

    for sub, sessions_data in cleaned_ts_data.items():
        run_connectivity_matrices[sub] = {}
        for ses, runs_data in sessions_data.items():
            run_connectivity_matrices[sub][ses] = {}
            for run_key, run_ts in runs_data.items():
                if run_ts is not None and isinstance(run_ts, np.ndarray) and run_ts.shape[0] > 1 and run_ts.shape[1] > 1:
                    try:
                        # Compute correlation matrix for the single run's time series
                        conn_matrix = conn_measure.fit_transform([run_ts])[0] # Input needs to be a list
                        run_connectivity_matrices[sub][ses][run_key] = conn_matrix
                        computed_count += 1
                    except Exception as e:
                        print(f"    ERROR computing connectivity for {sub}/{ses}/{run_key}: {e}")
                        run_connectivity_matrices[sub][ses][run_key] = None
                        error_count += 1
                else:
                    print(f"    Skipping connectivity for {sub}/{ses}/{run_key}: Invalid time series data (shape: {run_ts.shape if run_ts is not None else 'None'}).")
                    run_connectivity_matrices[sub][ses][run_key] = None
                    error_count += 1

    total_runs = sum(len(runs) for sessions in cleaned_ts_data.values() for runs in sessions.values())
    print(f"  Computed {computed_count} run-level connectivity matrices. Encountered {error_count} errors/skips out of {total_runs} total runs.")

    if computed_count == 0:
        print("  ERROR: No connectivity matrices were successfully computed.")
        return None

    # --- Save Plots of Computed Matrices --- 
    save_connectivity_matrix_plots(run_connectivity_matrices, OUTPUT_DIR)
    # --- End Save Plots --- 

    return run_connectivity_matrices

# --- Main Execution Function (Example/Testing) ---
if __name__ == "__main__":
    print("Executing Data Preparation Pipeline (Run-Level Connectivity)...")
    try:
        # Step 1
        file_info = verify_data_location()
        perform_qc_check(file_info)

        # Step 2 (Category mapping skipped)

        # Step 3
        atlas_masker = get_atlas_masker()
        raw_ts_data = extract_roi_time_series(file_info, atlas_masker)

        # Step 4
        cleaned_ts_data = clean_extracted_signals(raw_ts_data, file_info)

        # Step 5 (Run-level connectivity)
        conn_matrices = compute_run_level_connectivity(cleaned_ts_data)

        if conn_matrices is None:
            print("\nPipeline stopped due to issues in connectivity computation.")
            exit()

        print(f"\nData Preparation Pipeline Complete.")

    except FileNotFoundError as e:
         print(f"\nPipeline Halted: {e}")
    except Exception as e:
         print(f"\nPipeline Halted Unexpectedly: {e}")