import warnings
import pickle
import re
import pandas as pd
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
from nilearn import image, plotting, datasets, signal
from nilearn.maskers import NiftiMapsMasker
from nilearn.connectome import ConnectivityMeasure

from setup_config import (
    FMRIPREP_DIR, BIDS_RAW_DIR, OUTPUT_DIR, SYNSET_WORDS_FILE, DIFUMO_LABELS_FILE,
    ATLAS_DIM, ATLAS_NAME, ATLAS_RESOLUTION_MM, SUBJECTS, SESSIONS, TASK_NAME, 
    N_RUNS_PER_SESSION, TR, HEMODYNAMIC_DELAY_TR, CONFOUNDS_TO_USE,
    CLEAN_SIGNAL_PARAMS, POTENTIAL_CATEGORIES, MIN_SYNSETS_PER_CATEGORY,
    N_CATEGORIES_TO_USE, UNCATEGORIZED_NAME, MIN_VOLUMES_PER_CATEGORY,
    CONNECTIVITY_KIND, NILEARN_CACHE_DIR
)

def verify_data_location():
    """Locates and verifies required input files for the PoC."""
    print("--- Verifying Data Locations ---")
    required_files_info = {}
    all_files_found = True
    warnings_found = False

    for sub in SUBJECTS:
        required_files_info[sub] = {}
        for ses in SESSIONS:
            required_files_info[sub][ses] = {}
            for run in range(1, N_RUNS_PER_SESSION + 1):
                run_id = f"{run:02d}"
                run_key = f"run-{run_id}"
                required_files_info[sub][ses][run_key] = {'fmriprep': {}, 'raw': {}}

                fmriprep_base = FMRIPREP_DIR / f"sub-{sub}" / f"ses-{ses}" / "func"
                raw_base = BIDS_RAW_DIR / f"sub-{sub}" / f"ses-{ses}" / "func"
                file_prefix = f"sub-{sub}_ses-{ses}_task-{TASK_NAME}_run-{run_id}"

                fmriprep_files = {
                    'bold': fmriprep_base / f"{file_prefix}_space-T1w_desc-preproc_bold.nii.gz",
                    'confounds_tsv': fmriprep_base / f"{file_prefix}_desc-confounds_timeseries.tsv",
                    'confounds_json': fmriprep_base / f"{file_prefix}_desc-confounds_timeseries.json"
                }
                raw_files = {'events_tsv': raw_base / f"{file_prefix}_events.tsv"}

                for key, path in fmriprep_files.items():
                    if path.exists():
                        required_files_info[sub][ses][run_key]['fmriprep'][key] = path
                    else:
                        print(f"  ERROR: Missing essential file: {path}")
                        required_files_info[sub][ses][run_key]['fmriprep'][key] = None
                        all_files_found = False

                for key, path in raw_files.items():
                    if path.exists():
                        required_files_info[sub][ses][run_key]['raw'][key] = path
                    else:
                        print(f"  WARNING: Missing non-essential file (events): {path}")
                        required_files_info[sub][ses][run_key]['raw'][key] = None
                        warnings_found = True # Events are needed, so this is a problem

    if not SYNSET_WORDS_FILE.exists():
        print(f"  ERROR: Missing essential file: {SYNSET_WORDS_FILE}")
        all_files_found = False
    if not DIFUMO_LABELS_FILE.exists():
         print(f"  WARNING: Missing DiFuMo labels file: {DIFUMO_LABELS_FILE}")
         warnings_found = True 

    if not all_files_found:
        raise FileNotFoundError("Essential input files are missing. Cannot proceed.")
    elif warnings_found:
        print("Warnings encountered (e.g., missing events or labels files). Check logs.")
    else:
        print("All required files verified successfully.")

    return required_files_info

def perform_qc_check(required_files_info):
    """(Optional) Loads one BOLD file and plots its mean image."""
    print("--- Performing Basic QC Check ---")
    try:
        example_sub, example_ses, example_run = SUBJECTS[0], SESSIONS[0], 'run-01'
        example_bold_path = required_files_info[example_sub][example_ses][example_run]['fmriprep'].get('bold')

        if example_bold_path:
            print(f"  Loading: {example_bold_path.name}")
            mean_func_img = image.mean_img(str(example_bold_path), copy_header=True)
            qc_plot_path = OUTPUT_DIR / f"qc_mean_func_sub-{example_sub}_ses-{example_ses}_{example_run}.png"
            plotting.plot_epi(mean_func_img, title="Mean Functional Image (QC)",
                              output_file=str(qc_plot_path), display_mode='ortho', cmap='inferno')
            plt.close()
            print(f"  Saved QC plot to: {qc_plot_path}")
        else:
             print("  Skipping QC check: Example BOLD file not found in info.")

    except Exception as e:
        print(f"  WARNING: QC check failed: {e}")

def create_stimulus_category_mapping(required_files_info):
    """Creates and saves a mapping from stimulus files/Synsets to broad categories."""
    print("--- Creating Stimulus Category Mapping ---")
    all_stim_files = set()
    synset_pattern = re.compile(r"^(n\d{8})")
    stim_to_synset = {}

    # Collect unique stimulus filenames and map to synset
    print("  Reading event files...")
    for sub in SUBJECTS:
        for ses in SESSIONS:
            for run_key, run_info in required_files_info[sub][ses].items():
                event_path = run_info['raw'].get('events_tsv')
                if event_path:
                    try:
                        events_df = pd.read_csv(event_path, sep='\t')
                        if 'stim_file' in events_df.columns:
                            unique_stims_run = events_df['stim_file'].dropna().unique()
                            all_stim_files.update(unique_stims_run)
                            for stim_file in unique_stims_run:
                                match = synset_pattern.match(Path(stim_file).stem)
                                if match:
                                    stim_to_synset[stim_file] = match.group(1)
                    except Exception as e:
                        print(f"    Warning: Error reading {event_path.name}: {e}")

    unique_synsets = set(stim_to_synset.values())
    print(f"  Found {len(all_stim_files)} unique stimuli, mapping to {len(unique_synsets)} unique synsets.")

    # Read synset labels
    synset_to_label = {}
    try:
        with open(SYNSET_WORDS_FILE, 'r') as f:
            for line in f:
                parts = line.strip().split(' ', 1)  # Split on first space only
                if len(parts) == 2:
                    synset_id, label_text = parts
                    synset_to_label[synset_id] = label_text.split(',')[0].strip()
        print(f"  Read {len(synset_to_label)} labels from {SYNSET_WORDS_FILE.name}.")
    except Exception as e:
        print(f"  ERROR reading synset labels: {e}. Cannot map to categories.")
        return None

    # Define broad categories based on keywords in labels
    synset_to_broad_category = {}
    category_counts = {}

    def get_broad_category(synset_id, label_map, category_definitions):
        label = label_map.get(synset_id, "").lower()
        if not label:
            return UNCATEGORIZED_NAME
        for category, keywords in category_definitions.items():
            if any(keyword in label for keyword in keywords):
                return category
        return UNCATEGORIZED_NAME

    for synset_id in unique_synsets:
        category = get_broad_category(synset_id, synset_to_label, POTENTIAL_CATEGORIES)
        synset_to_broad_category[synset_id] = category
        category_counts[category] = category_counts.get(category, 0) + 1

    valid_categories = {
        cat: count for cat, count in category_counts.items()
        if count >= MIN_SYNSETS_PER_CATEGORY and cat != UNCATEGORIZED_NAME
    }

    if not valid_categories:
        print("  ERROR: No categories met the minimum synset count threshold.")
        return None

    if len(valid_categories) > N_CATEGORIES_TO_USE:
        sorted_cats = sorted(valid_categories.items(), key=lambda item: item[1], reverse=True)
        final_category_names = [cat for cat, count in sorted_cats[:N_CATEGORIES_TO_USE]]
    else:
        final_category_names = list(valid_categories.keys())

    print(f"  Selected Broad Categories: {final_category_names}")
    # print("  Synsets per category (initial):", category_counts)
    print("  Synsets per selected category:", {cat: valid_categories[cat] for cat in final_category_names})

    # Create final mapping and save
    final_stim_to_category = {}
    final_synset_to_category = {}
    for stim, synset in stim_to_synset.items():
        broad_cat = synset_to_broad_category.get(synset)
        if broad_cat in final_category_names:
            final_stim_to_category[stim] = broad_cat
            final_synset_to_category[synset] = broad_cat

    mapping_data = {
        'stim_file_to_category': final_stim_to_category,
        'synset_to_final_category': final_synset_to_category,
        'category_list': final_category_names,
        'category_to_int': {name: i for i, name in enumerate(final_category_names)}
    }

    mapping_file_path = OUTPUT_DIR / "stimulus_category_mapping.pkl"
    try:
        with open(mapping_file_path, 'wb') as f:
            pickle.dump(mapping_data, f)
        print(f"  Saved stimulus mapping ({len(final_stim_to_category)} stimuli mapped) to: {mapping_file_path}")
        return mapping_data
    except Exception as e:
        print(f"  ERROR saving mapping file: {e}")
        return None

# ROI Time Series Extraction

def get_atlas_masker():
    """Fetches the DiFuMo atlas and initializes a NiftiMapsMasker."""
    print(f"--- Initializing Atlas Masker (DiFuMo {ATLAS_DIM}) ---")
    try:
        atlas_data = datasets.fetch_atlas_difumo(
            dimension=ATLAS_DIM,
            resolution_mm=ATLAS_RESOLUTION_MM,
            data_dir=str(NILEARN_CACHE_DIR / 'difumo') 
        )
        atlas_filename = atlas_data.maps
        print(f"  Using atlas map: {atlas_filename}")

        masker = NiftiMapsMasker(
            maps_img=atlas_filename,
            standardize="zscore_sample", 
            memory=str(NILEARN_CACHE_DIR), 
            memory_level=1,
            verbose=0
        )
        print("  NiftiMapsMasker initialized.")
        return masker

    except Exception as e:
        print(f"  ERROR fetching/loading atlas or initializing masker: {e}")
        raise # Atlas is critical

def extract_roi_time_series(required_files_info, masker):
    """Extracts ROI time series for all specified runs using the provided masker."""
    print("--- Extracting ROI Time Series ---")
    extracted_ts = {sub: {ses: {} for ses in SESSIONS} for sub in SUBJECTS}

    for sub in SUBJECTS:
        for ses in SESSIONS:
            for run_key, run_info in required_files_info[sub][ses].items():
                bold_path = run_info['fmriprep'].get('bold')
                if bold_path:
                    print(f"    Processing: {bold_path.name}")
                    try:
                        time_series = masker.fit_transform(str(bold_path))
                        if np.isnan(time_series).any():
                           print(f"      Warning: NaNs detected in extracted time series for {bold_path.name}. Check BOLD data.")
                        extracted_ts[sub][ses][run_key] = time_series
                    except Exception as e:
                        print(f"      ERROR during extraction for {bold_path.name}: {e}")
                        extracted_ts[sub][ses][run_key] = None
                else:
                    print(f"    Skipping {sub}/{ses}/{run_key}: Missing BOLD file.")
                    extracted_ts[sub][ses][run_key] = None

    print("  ROI time series extraction complete.")
    return extracted_ts

# Signal Cleaning

def clean_extracted_signals(extracted_ts, required_files_info):
    """Cleans the extracted time series using specified confounds."""
    print("--- Cleaning Extracted Time Series ---")
    cleaned_ts = {sub: {ses: {} for ses in SESSIONS} for sub in SUBJECTS}

    for sub in SUBJECTS:
        for ses in SESSIONS:
            for run_key, raw_ts in extracted_ts[sub][ses].items():
                confounds_path = required_files_info[sub][ses][run_key]['fmriprep'].get('confounds_tsv')

                if raw_ts is not None and confounds_path:
                    print(f"    Cleaning: sub-{sub}_ses-{ses}_{run_key}")
                    try:
                        confounds_df = pd.read_csv(confounds_path, sep='\t')

                        # Check for NaNs in raw TS *before* cleaning
                        if np.isnan(raw_ts).any():
                            print(f"      ERROR: NaNs found in raw time series for {run_key} before cleaning. Skipping run.")
                            cleaned_ts[sub][ses][run_key] = None
                            continue

                        # Select configured confounds, warn if some are missing
                        available_confounds = [c for c in CONFOUNDS_TO_USE if c in confounds_df.columns]
                        missing_confounds = [c for c in CONFOUNDS_TO_USE if c not in confounds_df.columns]
                        if missing_confounds:
                            print(f"      Warning: Missing requested confounds in {confounds_path.name}: {missing_confounds}")

                        if not available_confounds:
                            print(f"      Warning: No specified confounds found in {confounds_path.name}. Cleaning without confounds.")
                            confounds_matrix = None
                        else:
                            confounds_matrix = confounds_df[available_confounds].values

                            # Impute NaNs in confounds matrix (e.g., FD often has NaN in first volume)
                            if np.isnan(confounds_matrix).any():
                                print("      Imputing NaNs in confounds matrix with column means.")
                                col_mean = np.nanmean(confounds_matrix, axis=0)
                                # Handle cases where entire columns might be NaN
                                col_mean = np.nan_to_num(col_mean) 
                                inds = np.where(np.isnan(confounds_matrix))
                                confounds_matrix[inds] = np.take(col_mean, inds[1])
                                if np.isnan(confounds_matrix).any(): # Still NaNs? Problem.
                                    print(f"      ERROR: NaNs remain in confounds after imputation for {run_key}. Skipping cleaning.")
                                    cleaned_ts[sub][ses][run_key] = None
                                    continue
                        
                        # Ensure confound matrix length matches time series length
                        if confounds_matrix is not None and confounds_matrix.shape[0] != raw_ts.shape[0]:
                            print(f"      ERROR: Mismatch between TS length ({raw_ts.shape[0]}) and confounds length ({confounds_matrix.shape[0]}) for {run_key}. Skipping.")
                            cleaned_ts[sub][ses][run_key] = None
                            continue

                        # Perform cleaning
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
                    if raw_ts is None:
                        print(f"    Skipping {sub}/{ses}/{run_key}: Missing raw TS data.")
                    elif confounds_path is None:
                        print(f"    Skipping {sub}/{ses}/{run_key}: Missing confounds file path.")
                    cleaned_ts[sub][ses][run_key] = None

    print("  Signal cleaning complete.")
    return cleaned_ts

# --- Step 5: Aggregation & Connectivity ---

def aggregate_and_compute_connectivity(cleaned_ts, required_files_info, mapping_data):
    """Aggregates cleaned TS by category and computes connectivity matrices."""
    print("--- Aggregating by Category & Computing Connectivity ---")
    if mapping_data is None:
        print("  ERROR: Category mapping data is missing. Cannot proceed.")
        return None

    stim_file_to_category = mapping_data['stim_file_to_category']
    final_category_names = mapping_data['category_list']

    connectivity_matrices = {sub: {} for sub in SUBJECTS}
    connectome_measure = ConnectivityMeasure(kind=CONNECTIVITY_KIND, vectorize=False, discard_diagonal=False)

    for sub in SUBJECTS:
        print(f"  Processing Subject: {sub}")
        for category in final_category_names:
            print(f"    Category: {category}")
            category_ts_segments = []
            total_volumes_in_category = 0

            for ses in SESSIONS:
                for run_key, run_cleaned_ts in cleaned_ts[sub][ses].items():
                    events_path = required_files_info[sub][ses][run_key]['raw'].get('events_tsv')

                    if run_cleaned_ts is not None and events_path:
                        try:
                            events_df = pd.read_csv(events_path, sep='\t')
                            if 'stim_file' not in events_df.columns or 'onset' not in events_df.columns or 'duration' not in events_df.columns:
                                print(f"      Warning: Missing required columns (stim_file, onset, duration) in {events_path.name}. Skipping run.")
                                continue

                            events_df['category'] = events_df['stim_file'].map(stim_file_to_category)
                            category_events = events_df[events_df['category'] == category].copy()
                            category_events.dropna(subset=['onset', 'duration'], inplace=True)

                            if not category_events.empty:
                                n_volumes_run = run_cleaned_ts.shape[0]
                                for _, trial in category_events.iterrows():
                                    onset = float(trial['onset'])
                                    duration = float(trial['duration'])

                                    start_vol = int(np.floor(onset / TR) + HEMODYNAMIC_DELAY_TR)
                                    end_vol = int(np.ceil((onset + duration) / TR) + HEMODYNAMIC_DELAY_TR) # Use ceil for end

                                    # Clamp indices to valid range
                                    start_vol = max(0, start_vol)
                                    end_vol = min(n_volumes_run, end_vol)

                                    if end_vol > start_vol:
                                        segment = run_cleaned_ts[start_vol:end_vol, :]
                                        if segment.shape[0] > 0:
                                            category_ts_segments.append(segment)
                                            total_volumes_in_category += segment.shape[0]

                        except Exception as e:
                            print(f"      Warning: Error processing events/ts for {sub}/{ses}/{run_key}: {e}")

            # Concatenate and compute connectivity
            if category_ts_segments:
                concatenated_ts = np.concatenate(category_ts_segments, axis=0)
                print(f"      Total volumes for {category}: {concatenated_ts.shape[0]}")

                if concatenated_ts.shape[0] >= MIN_VOLUMES_PER_CATEGORY:
                    try:
                        conn_matrix = connectome_measure.fit_transform([concatenated_ts])[0]
                        # Check for NaNs in connectivity matrix
                        if np.isnan(conn_matrix).any():
                            print(f"      ERROR: NaNs found in connectivity matrix for {sub}/{category}. Skipping.")
                            connectivity_matrices[sub][category] = None
                        else:
                            connectivity_matrices[sub][category] = conn_matrix
                            matrix_filename = OUTPUT_DIR / f"sub-{sub}_{category}_connectivity_{CONNECTIVITY_KIND}.npy"
                            np.save(matrix_filename, conn_matrix)
                            print(f"        Saved connectivity matrix to {matrix_filename.name}")

                    except Exception as e:
                        print(f"      ERROR computing/saving connectivity for {sub}/{category}: {e}")
                        connectivity_matrices[sub][category] = None
                else:
                    print(f"      Skipping connectivity for {category}: Insufficient volumes ({concatenated_ts.shape[0]} < {MIN_VOLUMES_PER_CATEGORY}).")
                    connectivity_matrices[sub][category] = None
            else:
                print(f"      Skipping connectivity for {category}: No valid time series segments found.")
                connectivity_matrices[sub][category] = None

    print("  Connectivity computation complete.")
    # Save the computed matrices dictionary
    conn_dict_path = OUTPUT_DIR / "connectivity_matrices.pkl"
    try:
        with open(conn_dict_path, 'wb') as f:
            pickle.dump(connectivity_matrices, f)
        print(f"  Saved dictionary of connectivity matrices to: {conn_dict_path}")
    except Exception as e:
        print(f"  ERROR saving connectivity dictionary: {e}")

    return connectivity_matrices

# Main Execution Function
if __name__ == "__main__":
    print("Executing Data Preparation Pipeline...")

    # Step 1
    file_info = verify_data_location()
    perform_qc_check(file_info)

    # Step 2
    mapping = create_stimulus_category_mapping(file_info)
    if mapping is None:
        print("\nPipeline stopped due to issues in category mapping.")
        exit()

    # Step 3
    atlas_masker = get_atlas_masker()
    raw_ts_data = extract_roi_time_series(file_info, atlas_masker)

    # Step 4
    cleaned_ts_data = clean_extracted_signals(raw_ts_data, file_info)

    # Step 5
    conn_matrices = aggregate_and_compute_connectivity(cleaned_ts_data, file_info, mapping)

    if conn_matrices is None:
        print("\nPipeline stopped due to issues in connectivity computation.")
        exit()
    
    # Count how many matrices were successfully computed
    success_count = sum(1 for sub_data in conn_matrices.values() 
                      for matrix in sub_data.values() if matrix is not None)
    print(f"\nData Preparation Pipeline Complete. Successfully computed {success_count} connectivity matrices.") 