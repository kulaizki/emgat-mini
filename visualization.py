import pickle
import numpy as np
import pandas as pd
from nilearn import plotting, datasets, image
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
import os
from dotenv import load_dotenv
from google import genai

from setup_config import (
    OUTPUT_DIR, ATLAS_DIM, ATLAS_RESOLUTION_MM, NILEARN_CACHE_DIR,
    DIFUMO_LABELS_FILE,
    DIFUMO_LABEL_INDEX_COL,
    DIFUMO_LABEL_NAME_COL,
    DIFUMO_COORD_X_COL,
    DIFUMO_COORD_Y_COL,
    DIFUMO_COORD_Z_COL
)

# --- Gemini API Configuration ---
def configure_gemini():
    """Loads API key from .env and creates the Gemini client."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("\n*** WARNING: GEMINI_API_KEY not found in .env file. ROI interpretation will be skipped. ***")
        print("Please create a .env file in the project root with your key: GEMINI_API_KEY='YOUR_API_KEY'")
        return None
    try:
        client = genai.Client(api_key=api_key)
        print("  Gemini API client created successfully.")
        return client
    except Exception as e:
        print(f"  ERROR creating Gemini API client: {e}")
        return None

def interpret_rois_with_gemini(roi_names, client, model_name="gemini-1.5-pro-latest"):
    """Uses Gemini API Client (client.models.generate_content) for interpretations."""
    if client is None:
        return "Gemini API client not available. Skipping interpretation."
    if not roi_names:
        return "No ROI names provided for interpretation."

    print(f"\n--- Requesting Gemini Interpretation for {len(roi_names)} ROIs... ---")
    try:
        prompt_parts = [
            "For each of the following brain ROIs (Region of Interest) based on the DiFuMo atlas, provide:",
            "1. A concise summary of its primary known function(s).",
            "2. A brief explanation of its potential relevance to visual processing or connectivity, especially if it's not a primary visual area.",
            "\nFormat the output clearly for each ROI listed below:",
        ]
        for name in roi_names:
            prompt_parts.append(f"- {name}")
        prompt = "\n".join(prompt_parts)

        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]

        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )

        # Process response using response.text
        if hasattr(response, 'text') and response.text:
            print("  Interpretation received from Gemini.")
            return response.text
        elif hasattr(response, 'prompt_feedback') and response.prompt_feedback and hasattr(response.prompt_feedback, 'block_reason') and response.prompt_feedback.block_reason:
             reason = response.prompt_feedback.block_reason
             print(f"  ERROR: Gemini request blocked due to {reason}.")
             return f"Interpretation request blocked by safety filters ({reason})."
        else:
             print("  ERROR: Received an empty or unexpected response from Gemini.")
             return "No interpretation received from Gemini (empty response)."

    except Exception as e:
        print(f"  ERROR calling Gemini API: {e}")
        return f"Failed to get interpretation due to API error: {e}"

def load_difumo_metadata():
    """
    Loads DiFuMo coordinates and labels, preferring direct access from fetch_atlas_difumo,
    then trying find_parcellation_cut_coords, and finally falling back to CSV.
    """
    print("--- Loading DiFuMo Atlas Metadata ---")
    coords = None
    labels = None
    atlas_data = None
    labels_df = None

    # --- Fetch Atlas Data ---
    try:
        difumo_cache_path = NILEARN_CACHE_DIR / 'difumo'
        difumo_cache_path.mkdir(parents=True, exist_ok=True)
        atlas_data = datasets.fetch_atlas_difumo(
            dimension=ATLAS_DIM,
            resolution_mm=ATLAS_RESOLUTION_MM,
            data_dir=str(difumo_cache_path)
        )
        print(f"  Fetched DiFuMo atlas data ({ATLAS_DIM} dimensions, {ATLAS_RESOLUTION_MM}mm).")
    except Exception as e:
        print(f"  ERROR: Failed to fetch DiFuMo atlas data: {e}")

    # --- Attempt 1: Direct Coordinate Access ---
    if atlas_data and hasattr(atlas_data, 'region_coords') and atlas_data.region_coords is not None:
        try:
            coords_arr = np.array(atlas_data.region_coords)
            if coords_arr.shape == (ATLAS_DIM, 3):
                coords = coords_arr
                print(f"  Successfully loaded coordinates ({coords.shape}) directly from fetched atlas data.")
            else:
                print(f"  Warning: Coordinates from atlas_data.region_coords have unexpected shape {coords_arr.shape}. Expected ({ATLAS_DIM}, 3).")
        except Exception as e:
            print(f"  Warning: Failed to process coordinates from atlas_data.region_coords: {e}")
            coords = None
    elif atlas_data and hasattr(atlas_data, 'labels') and isinstance(atlas_data.labels, list) and len(atlas_data.labels) > 0:
         try:
             coords_list = [ (float(d.get('x', 0)), float(d.get('y', 0)), float(d.get('z', 0))) for d in atlas_data.labels ]
             if len(coords_list) == ATLAS_DIM:
                  coords_arr = np.array(coords_list)
                  if coords_arr.shape == (ATLAS_DIM, 3):
                      coords = coords_arr
                      print(f"  Successfully loaded coordinates ({coords.shape}) from atlas_data.labels.")
                  else:
                       print(f"  Warning: Coordinates processed from atlas_data.labels have unexpected shape {coords_arr.shape}.")
             else:
                  print(f"  Warning: Length of atlas_data.labels ({len(coords_list)}) does not match ATLAS_DIM ({ATLAS_DIM}).")
         except Exception as e:
              print(f"  Warning: Failed to process coordinates from atlas_data.labels: {e}")
              coords = None

    # --- Attempt 2: find_parcellation_cut_coords ---
    if coords is None and atlas_data and hasattr(atlas_data, 'maps') and atlas_data.maps is not None:
        print("  Attempting coordinate extraction using find_parcellation_cut_coords...")
        try:
            maps_4d_img = image.load_img(atlas_data.maps)
            max_prob_map_data = np.argmax(maps_4d_img.get_fdata(), axis=-1)
            max_prob_img = image.new_img_like(maps_4d_img, max_prob_map_data + 1, affine=maps_4d_img.affine)

            if max_prob_img.get_fdata().ndim == 3:
                 coords_nifti = plotting.find_parcellation_cut_coords(max_prob_img)
                 if coords_nifti is not None and coords_nifti.shape[1] == 3:
                     if coords_nifti.shape[0] == ATLAS_DIM:
                         coords = coords_nifti
                         print(f"  Successfully loaded coordinates ({coords.shape}) using find_parcellation_cut_coords on max prob map.")
                     elif coords_nifti.shape[0] == ATLAS_DIM - 1: # Handle the off-by-one case
                         print(f"  Warning: find_parcellation_cut_coords returned {coords_nifti.shape[0]} coordinates (expected {ATLAS_DIM}). Padding with one [0,0,0] entry.")
                         coords = np.vstack([coords_nifti, np.zeros((1, 3))])
                         print(f"  Successfully loaded and padded coordinates ({coords.shape}).")
                     else:
                          print(f"  Warning: find_parcellation_cut_coords returned unexpected shape: {coords_nifti.shape}. Expected ({ATLAS_DIM} or {ATLAS_DIM-1}, 3).")
                 else:
                     print(f"  Warning: find_parcellation_cut_coords returned None or incorrect column count ({coords_nifti.shape if coords_nifti is not None else 'None'}).")
            else:
                 print("  Warning: Derived max probability map is not 3D. Cannot use find_parcellation_cut_coords.")
        except Exception as e:
            print(f"  Warning: Failed during find_parcellation_cut_coords approach: {e}")
            coords = None

    # --- Attempt 3: Load from CSV ---
    if coords is None and DIFUMO_LABELS_FILE and DIFUMO_COORD_X_COL and DIFUMO_COORD_Y_COL and DIFUMO_COORD_Z_COL:
        print(f"  Attempting to load coordinates from CSV: {DIFUMO_LABELS_FILE}")
        try:
            labels_df = pd.read_csv(DIFUMO_LABELS_FILE)
            coord_cols = [DIFUMO_COORD_X_COL, DIFUMO_COORD_Y_COL, DIFUMO_COORD_Z_COL]
            if all(col in labels_df.columns for col in coord_cols):
                if DIFUMO_LABEL_INDEX_COL and DIFUMO_LABEL_INDEX_COL in labels_df.columns:
                     if pd.api.types.is_numeric_dtype(labels_df[DIFUMO_LABEL_INDEX_COL]) and labels_df[DIFUMO_LABEL_INDEX_COL].iloc[0] == 1:
                          labels_df[DIFUMO_LABEL_INDEX_COL] = labels_df[DIFUMO_LABEL_INDEX_COL] - 1
                     labels_df = labels_df.sort_values(by=DIFUMO_LABEL_INDEX_COL).set_index(DIFUMO_LABEL_INDEX_COL)
                     labels_df = labels_df.reindex(range(ATLAS_DIM))
                else:
                     if len(labels_df) < ATLAS_DIM:
                           padding = pd.DataFrame(index=range(len(labels_df), ATLAS_DIM), columns=labels_df.columns)
                           labels_df = pd.concat([labels_df, padding], ignore_index=False)

                if len(labels_df) >= ATLAS_DIM:
                     coords_df = labels_df.loc[range(ATLAS_DIM), coord_cols]
                     if coords_df.isnull().values.any():
                           print(f"    Warning: NaNs found in CSV coordinate columns. Filling with 0.")
                           coords_df = coords_df.fillna(0)
                     coords = coords_df.to_numpy()
                     if coords.shape == (ATLAS_DIM, 3):
                          print(f"  Successfully loaded coordinates ({coords.shape}) from CSV.")
                     else:
                          print(f"  ERROR: Coordinate shape from CSV is incorrect ({coords.shape}).")
                          coords = None
                else:
                     print(f"  ERROR: CSV processing resulted in fewer rows ({len(labels_df)}) than ATLAS_DIM.")
                     coords = None
            else:
                print(f"  Coordinate columns not found in CSV.")
        except FileNotFoundError:
            print(f"  ERROR: Labels file not found at {DIFUMO_LABELS_FILE}.")
        except Exception as e:
            print(f"  ERROR loading coordinates from CSV {DIFUMO_LABELS_FILE}: {e}")
            coords = None

    if coords is None:
         print("  ERROR: Failed to obtain coordinates using all methods. Visualization will be limited.")

    # --- Load Labels ---
    labels = [f"Node {i}" for i in range(ATLAS_DIM)] # Default
    labels_loaded_source = "default"

    # Attempt 1: Direct from atlas_data.labels
    if atlas_data and hasattr(atlas_data, 'labels') and isinstance(atlas_data.labels, list) and len(atlas_data.labels) > 0:
        try:
            if isinstance(atlas_data.labels[0], str):
                 if len(atlas_data.labels) == ATLAS_DIM:
                     labels = list(atlas_data.labels)
                     labels_loaded_source = "atlas_data.labels (list)"
                 else:
                      print(f"  Warning: Length of atlas_data.labels list ({len(atlas_data.labels)}) != ATLAS_DIM.")
            elif isinstance(atlas_data.labels[0], dict) and 'name' in atlas_data.labels[0]:
                 labels_list = [d.get('name', f'Node {i}') for i, d in enumerate(atlas_data.labels)]
                 if len(labels_list) == ATLAS_DIM:
                      labels = labels_list
                      labels_loaded_source = "atlas_data.labels (dicts)"
                 else:
                      print(f"  Warning: Length of atlas_data.labels dict list ({len(labels_list)}) != ATLAS_DIM.")
            else:
                 print("  Warning: atlas_data.labels format not recognized (expected list of str or list of dict with 'name').")

        except Exception as e:
            print(f"  Warning: Error processing labels from atlas_data.labels: {e}")
            labels = [f"Node {i}" for i in range(ATLAS_DIM)] # Reset to default
            labels_loaded_source = "default"

    # Attempt 2: Load from CSV
    if (labels_loaded_source == "default" or len(labels) != ATLAS_DIM) and DIFUMO_LABELS_FILE and DIFUMO_LABEL_NAME_COL:
        print(f"  Attempting to load labels from CSV: {DIFUMO_LABELS_FILE}")
        try:
            if labels_df is None: # Load if not already loaded for coords
                labels_df = pd.read_csv(DIFUMO_LABELS_FILE)
                if DIFUMO_LABEL_INDEX_COL and DIFUMO_LABEL_INDEX_COL in labels_df.columns:
                    if pd.api.types.is_numeric_dtype(labels_df[DIFUMO_LABEL_INDEX_COL]) and labels_df[DIFUMO_LABEL_INDEX_COL].iloc[0] == 1:
                        labels_df[DIFUMO_LABEL_INDEX_COL] = labels_df[DIFUMO_LABEL_INDEX_COL] - 1
                    labels_df = labels_df.sort_values(by=DIFUMO_LABEL_INDEX_COL).set_index(DIFUMO_LABEL_INDEX_COL)
                    labels_df = labels_df.reindex(range(ATLAS_DIM)) # Ensure it has ATLAS_DIM rows
                else:
                    # If no index col, assume order is correct but pad if needed
                    if len(labels_df) < ATLAS_DIM:
                           padding = pd.DataFrame(index=range(len(labels_df), ATLAS_DIM), columns=labels_df.columns)
                           labels_df = pd.concat([labels_df, padding], ignore_index=False)
                           labels_df.index = range(ATLAS_DIM) # Make sure index is 0 to ATLAS_DIM-1

            if DIFUMO_LABEL_NAME_COL in labels_df.columns and len(labels_df) >= ATLAS_DIM:
                labels_series = labels_df.loc[range(ATLAS_DIM), DIFUMO_LABEL_NAME_COL]
                if labels_series.isnull().any():
                     print("    Warning: NaNs found in label column. Filling with 'Unknown Node'.")
                     labels_series = labels_series.fillna(f'Unknown Node')
                labels = labels_series.tolist()
                labels_loaded_source = "csv"
                print(f"  Successfully loaded labels ({len(labels)}) from CSV.")
            elif DIFUMO_LABEL_NAME_COL not in labels_df.columns:
                 print("   Label name column not found in CSV.")
            else: # len(labels_df) < ATLAS_DIM
                 print(f"   CSV processing resulted in fewer rows ({len(labels_df)}) than ATLAS_DIM ({ATLAS_DIM}) for labels.")

        except FileNotFoundError:
            print(f"  ERROR: Labels file not found at {DIFUMO_LABELS_FILE}.")
        except Exception as e:
            print(f"  ERROR loading labels from CSV {DIFUMO_LABELS_FILE}: {e}")
            if labels_loaded_source != "default" and len(labels) != ATLAS_DIM: # Revert to default if CSV failed badly
                 labels = [f"Node {i}" for i in range(ATLAS_DIM)]
                 labels_loaded_source = "default"

    print(f"--- Finished loading metadata. Using labels from: {labels_loaded_source} ---")
    if coords is None:
         print("  WARNING: Coordinates are missing. Connectome/Node plots will be disabled.")
    elif coords.shape[0] != ATLAS_DIM:
         print(f"  WARNING: Final coordinate array shape {coords.shape} does not match ATLAS_DIM {ATLAS_DIM}. Plots may be incorrect.")
    if len(labels) != ATLAS_DIM:
         print(f"  WARNING: Final labels list length ({len(labels)}) does not match ATLAS_DIM {ATLAS_DIM}. Plot labels may be incorrect.")
    return coords, labels

# --- Helper: Calculate and Print Top ROIs ---
def print_top_connected_rois(graph_id, edge_index, edge_mask, node_labels, title_prefix, top_n=5):
    """Calculates node importance by summing absolute edge mask values and prints top N ROIs.
    DEPRECATED: Use calculate_node_importance and print_overall_top_rois instead.
    """
    print(f"--- Top {top_n} ROIs for {graph_id} ({title_prefix}) ---")
    if edge_mask is None:
        print("  Skipping: Edge mask is None.")
        return
    if edge_index is None or edge_index.shape[1] != len(edge_mask):
        print("  Skipping: Edge index mismatch or missing.")
        return
    if node_labels is None:
        print("  Skipping: Node labels missing.")
        return

    num_nodes = len(node_labels)
    node_importance = np.zeros(num_nodes)
    edge_mask_clean = np.nan_to_num(np.abs(edge_mask)) # Use absolute, cleaned values

    try:
        # Sum absolute edge weights for each node (both incoming and outgoing)
        # Ensure indices are within bounds before using add.at
        valid_indices = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
        if not np.all(valid_indices):
             print("    Warning: Some edge indices are out of bounds for node importance calc. Skipping those edges.")
             edge_index_valid = edge_index[:, valid_indices]
             edge_mask_clean_valid = edge_mask_clean[valid_indices]
        else:
             edge_index_valid = edge_index
             edge_mask_clean_valid = edge_mask_clean
        
        # Use the valid edges/masks
        np.add.at(node_importance, edge_index_valid[0], edge_mask_clean_valid)
        np.add.at(node_importance, edge_index_valid[1], edge_mask_clean_valid)

    except IndexError as ie:
         print(f"    ERROR during node importance calculation (IndexError): {ie}. Indices might be out of bounds ({edge_index.max()} vs {num_nodes}). Skipping.")
         return
    except Exception as e:
        print(f"    ERROR calculating node importance: {e}. Skipping.")
        return

    if node_importance.sum() < 1e-9: # Check if all importances are effectively zero
        print("  All calculated node importances are zero. No top ROIs to display.")
        return

    # Get top N indices
    # Use argpartition for efficiency if only top N are needed, then sort those N
    top_indices_unsorted = np.argpartition(node_importance, -top_n)[-top_n:]
    # Sort these top N indices by their importance values (descending)
    top_indices = top_indices_unsorted[np.argsort(node_importance[top_indices_unsorted])][::-1]

    print(f"  Top {top_n} ROIs by summed absolute edge importance:")
    for i, idx in enumerate(top_indices):
        if 0 <= idx < len(node_labels):
            label = node_labels[idx]
            score = node_importance[idx]
            print(f"    {i+1}. Index {idx}: '{label}' (Score: {score:.4f})")
        else:
            print(f"    {i+1}. Index {idx}: Error - Index out of bounds for labels.")

def calculate_node_importance(edge_index, edge_mask, num_nodes):
    """Calculates node importance scores by summing absolute edge mask values."""
    if edge_mask is None or edge_index is None or edge_index.shape[1] != len(edge_mask):
        return None # Cannot calculate

    node_importance = np.zeros(num_nodes)
    edge_mask_clean = np.nan_to_num(np.abs(edge_mask)) # Use absolute, cleaned values

    try:
        valid_indices = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
        if not np.all(valid_indices):
            print("    Warning: Some edge indices are out of bounds for node importance calc. Skipping those edges.")
            edge_index_valid = edge_index[:, valid_indices]
            edge_mask_clean_valid = edge_mask_clean[valid_indices]
        else:
            edge_index_valid = edge_index
            edge_mask_clean_valid = edge_mask_clean
        
        np.add.at(node_importance, edge_index_valid[0], edge_mask_clean_valid)
        np.add.at(node_importance, edge_index_valid[1], edge_mask_clean_valid)
        return node_importance
    except Exception as e:
        print(f"    ERROR calculating node importance: {e}")
        return None

def print_overall_top_rois(aggregated_importance, node_labels, title_prefix, client, top_n=5):
    """
    Prints the top N ROIs, saves details and interpretation to Markdown for GNNExplainer,
    and returns top indices/scores for GNNExplainer.
    """
    print(f"\n--- Overall Top {top_n} ROIs ({title_prefix}) ---")
    if aggregated_importance is None or node_labels is None:
        print("  Skipping: Missing aggregated scores or labels.")
        return None, None # Return None for indices and scores

    num_nodes = len(node_labels)
    if len(aggregated_importance) != num_nodes:
        print(f"  Skipping: Aggregated scores length ({len(aggregated_importance)}) doesn't match labels length ({num_nodes}).")
        return None, None # Return None for indices and scores

    if aggregated_importance.sum() < 1e-9:
        print("  All aggregated node importances are zero.")
        return None, None # Return None for indices and scores

    top_roi_names = []
    top_indices = []
    top_scores = []
    markdown_output_lines = []
    is_gnn_explainer = "GNNExplainer" in title_prefix

    try:
        # Find top N indices and their scores
        top_indices_unsorted = np.argpartition(aggregated_importance, -top_n)[-top_n:]
        sorted_order = np.argsort(aggregated_importance[top_indices_unsorted])[::-1]
        top_indices = top_indices_unsorted[sorted_order]
        top_scores = aggregated_importance[top_indices]

        print(f"  Top {top_n} ROIs by aggregated summed absolute edge importance:")
        if is_gnn_explainer:
             markdown_output_lines.append(f"# Top {top_n} GNNExplainer ROIs (Aggregated)")
             markdown_output_lines.append("Based on aggregated summed absolute edge importance scores across all graphs.")
             markdown_output_lines.append("\n**Top ROIs:**\n")

        for i, idx in enumerate(top_indices):
            if 0 <= idx < len(node_labels):
                label = node_labels[idx]
                score = top_scores[i] # Use the sorted score
                print(f"    {i+1}. Index {idx}: '{label}' (Score: {score:.4f})")
                top_roi_names.append(label)
                if is_gnn_explainer:
                     markdown_output_lines.append(f"{i+1}. **{label}** (Index: {idx}, Score: {score:.4f})")
            else:
                print(f"    {i+1}. Index {idx}: Error - Index out of bounds for labels.")
                top_roi_names.append(f"Invalid Index {idx}")
                if is_gnn_explainer:
                     markdown_output_lines.append(f"{i+1}. **Invalid Index {idx}** (Score: N/A)")

        # --- Call Gemini & Append Interpretation to Markdown (only for GNNExplainer) ---
        if client and is_gnn_explainer:
             interpretation = interpret_rois_with_gemini(top_roi_names, client)
             print("\n--- Gemini Interpretation of Top 5 GNNExplainer ROIs ---")
             print(interpretation) # Still print to console for quick view
             markdown_output_lines.append("\n## Gemini Interpretation\n")
             markdown_output_lines.append(interpretation)
        elif not client and is_gnn_explainer:
             print("\n(Gemini interpretation skipped: API client not available)")
             markdown_output_lines.append("\n*Gemini interpretation skipped: API client not available.*")

        # --- Save Markdown File (only for GNNExplainer) ---
        if is_gnn_explainer:
            md_save_path = OUTPUT_DIR / "top_5_gnn_interpretation.md"
            try:
                with open(md_save_path, 'w') as f:
                    f.write("\n".join(markdown_output_lines))
                print(f"  Saved Top 5 GNNExplainer ROIs and interpretation to: {md_save_path.name}")
            except Exception as e_write:
                print(f"  ERROR writing markdown file {md_save_path}: {e_write}")

        # Return indices and scores only if it's GNNExplainer for the new plot
        if is_gnn_explainer:
             return top_indices, top_scores
        else:
             return None, None # Return None for non-GNNExplainer calls

    except Exception as e:
        print(f"  ERROR determining or printing overall top ROIs: {e}")
        return None, None # Return None on error

# --- Plotting Functions (remain largely the same, rely on valid coords/labels input) ---

def plot_explanation_connectome(graph_id, edge_index, edge_mask, coords, node_labels=None, threshold=0.6, top_n=50, title_prefix="GNNExplainer"):
    """Plots the connectome with edges weighted by explanation mask."""
    print(f"  Plotting connectome for {graph_id}...")
    if coords is None:
        print("    Skipping connectome plot: Missing coordinates.")
        return
    if edge_mask is None:
        print(f"    Skipping connectome plot for {graph_id}: Edge mask is None.")
        return
    if edge_index is None or edge_index.shape[1] != len(edge_mask):
         print(f"    Skipping connectome plot for {graph_id}: Edge index shape mismatch or missing.")
         return

    # Create adjacency matrix from edge mask
    num_nodes = coords.shape[0]
    adj_matrix = np.zeros((num_nodes, num_nodes))
    # Normalize mask only if max is non-zero, handle potential NaN/Inf
    edge_mask_clean = np.nan_to_num(edge_mask)
    max_mask = np.abs(edge_mask_clean).max() # Use abs for normalization range
    if max_mask > 1e-9: # Avoid division by zero or near-zero
        # Normalize to range [-1, 1] if mask has negative values, else [0, 1]
        if edge_mask_clean.min() < 0:
             edge_mask_norm = edge_mask_clean / max_mask
        else:
             edge_mask_norm = edge_mask_clean / max_mask
    else:
        edge_mask_norm = edge_mask_clean # Keep as is if max is zero

    kept_edges = 0
    edge_weights_for_plot = []
    for i in range(edge_index.shape[1]):
        # Ensure indices are within bounds
        u, v = edge_index[0, i], edge_index[1, i]
        if 0 <= u < num_nodes and 0 <= v < num_nodes:
            # Use absolute value for thresholding, keep original sign for weight
            weight_norm = edge_mask_norm[i]
            weight_orig = edge_mask_clean[i] # Keep original (possibly signed) value for matrix
            if abs(weight_norm) >= threshold: # Threshold absolute normalized value
                adj_matrix[u, v] = weight_orig # Store original value in matrix
                adj_matrix[v, u] = weight_orig # Assume symmetric importance for connectome plot
                edge_weights_for_plot.append(abs(weight_norm)) # Store absolute norm weight for thresholding top_n
                kept_edges += 1
        else:
             print(f"    Warning: Edge index ({u}, {v}) out of bounds for {num_nodes} nodes.")

    if kept_edges == 0:
        print(f"    Warning: No edges passed the absolute threshold ({threshold}) for {graph_id}. Skipping connectome plot.")
        return

    # Select only the top N edges if too many pass the threshold, based on absolute normalized weight
    if top_n is not None and kept_edges > top_n:
        print(f"    Reducing displayed edges from {kept_edges} to top {top_n} based on abs value.")
        # Find the threshold value corresponding to the Nth largest *absolute normalized* edge weight
        if edge_weights_for_plot:
            abs_threshold_val = np.sort(np.abs(edge_weights_for_plot))[-top_n] # Nth largest abs value
            # Keep original signs but zero out edges below threshold
            adj_matrix[np.abs(adj_matrix) < abs_threshold_val] = 0
        else:
            # Should not happen if kept_edges > 0, but defensive check
             print("    Warning: No edge weights found for top_n thresholding.")

    try:
        fig_title = f"{title_prefix}: {graph_id} (|Thr|={threshold:.1f}, Top {top_n})"
        save_path = OUTPUT_DIR / f"{title_prefix}_{graph_id}_connectome.png"

        # Remove node_kwargs and use direct parameters
        plotting.plot_connectome(
            adj_matrix,
            coords,
            edge_vmin= -max_mask if edge_mask_clean.min() < 0 else 0, # Adjust vmin if negative values exist
            edge_vmax= max_mask,
            node_color='black',  # Direct parameter instead of node_kwargs
            node_size=15,        # Direct parameter instead of node_kwargs
            edge_kwargs={'linewidth': 1.5, 'cmap': 'viridis'},
            display_mode='lzry',
            colorbar=True,
            title=fig_title,
            output_file=str(save_path),
            annotate=True
        )
        plt.close()
        print(f"    Saved connectome plot to: {save_path.name}")

    except Exception as e:
        print(f"    ERROR plotting connectome for {graph_id}: {e}")

def plot_node_importance(graph_id, node_mask, coords, node_labels=None, threshold=0.5, title_prefix="GNNExplainer"):
    """Plots important nodes on a brain map based on node mask."""
    print(f"  Plotting node importance for {graph_id}...")
    if coords is None:
        print("    Skipping node plot: Missing coordinates.")
        return
    if node_mask is None or node_mask.size == 0 or node_mask.size != coords.shape[0]:
        print(f"    Skipping node plot for {graph_id}: Node mask is None, empty, or size mismatch ({node_mask.size if node_mask is not None else 'None'} vs {coords.shape[0]} nodes).")
        return

    # Normalize node mask only if max is non-zero
    node_mask_clean = np.nan_to_num(node_mask)
    max_mask = np.abs(node_mask_clean).max()
    if max_mask > 1e-9:
        node_mask_norm = node_mask_clean / max_mask # Normalize to range [-1, 1] or [0, 1]
    else:
        node_mask_norm = node_mask_clean

    # Use absolute value for thresholding
    important_nodes_idx = np.where(np.abs(node_mask_norm) >= threshold)[0]

    if len(important_nodes_idx) == 0:
        print(f"    Warning: No nodes passed the absolute threshold ({threshold}) for {graph_id}. Skipping node plot.")
        return

    important_coords = coords[important_nodes_idx]
    important_weights = node_mask_norm[important_nodes_idx] # Keep original (normalized) sign/value for plotting

    # Get labels for important nodes only (but don't pass to plot_markers)
    marker_labels_list = None
    if node_labels:
        try:
            marker_labels_list = [node_labels[i] for i in important_nodes_idx]
        except IndexError:
             print("    Warning: Index error retrieving node labels. Using default node indices.")
             marker_labels_list = [f"Node {i}" for i in important_nodes_idx]
        except Exception as e:
             print(f"    Warning: Error retrieving node labels: {e}. Using default node indices.")
             marker_labels_list = [f"Node {i}" for i in important_nodes_idx]

    try:
        fig_title = f"{title_prefix} Node Importance: {graph_id} (|Thr|={threshold:.1f})"
        save_path = OUTPUT_DIR / f"{title_prefix}_{graph_id}_nodes.png"

        plotting.plot_markers(
            node_values=important_weights,  # Use original normalized values for color potentially
            node_coords=important_coords,
            # node_labels removed as it's causing errors with current nilearn version
            node_size=np.abs(important_weights) * 30 + 5,  # Scale size by *absolute* importance
            node_cmap='viridis',
            # Adjust vmin/vmax based on whether mask has negative values
            node_vmin= -1.0 if node_mask_clean.min() < 0 else 0.0,
            node_vmax=1.0,
            title=fig_title,
            colorbar=True,
            display_mode='lzry',
            output_file=str(save_path),
            annotate=True
        )
        plt.close()
        print(f"    Saved node importance plot to: {save_path.name}")

    except Exception as e:
        print(f"    ERROR plotting node importance for {graph_id}: {e}")

# --- New Function: Plot Only Top Aggregated Nodes ---
def plot_top_aggregated_nodes(top_indices, top_scores, coords, node_labels, title_prefix, output_filename="top_5_aggregated_nodes.png"):
    """Plots only the specified top nodes, sized and colored by their aggregated scores."""
    print(f"\n--- Plotting Top {len(top_indices)} Aggregated Nodes ({title_prefix}) ---")
    if top_indices is None or top_scores is None or len(top_indices) == 0:
        print("  Skipping top nodes plot: No indices or scores provided.")
        return
    if coords is None:
        print("  Skipping top nodes plot: Missing coordinates.")
        return
    if node_labels is None:
        print("  Warning: Missing node labels for top nodes plot.")
        labels_for_plot = [f"Node {idx}" for idx in top_indices]
    else:
        try:
            labels_for_plot = [node_labels[i] for i in top_indices]
        except IndexError:
             print("    Warning: Index error retrieving node labels for top nodes plot. Using default indices.")
             labels_for_plot = [f"Node {i}" for i in top_indices]
        except Exception as e:
             print(f"    Warning: Error retrieving node labels for top nodes plot: {e}. Using default indices.")
             labels_for_plot = [f"Node {i}" for i in top_indices]

    top_coords = coords[top_indices]

    # Normalize scores for better visualization scaling (e.g., size, color intensity)
    # Handle case where all scores might be zero (already checked before calling, but safety)
    min_score = top_scores.min()
    max_score = top_scores.max()
    if max_score > 1e-9:
        # Simple min-max scaling to 0-1 for color/size base
        scores_norm = (top_scores - min_score) / (max_score - min_score + 1e-9) # Add epsilon for stability
        node_sizes = scores_norm * 45 + 5 # Scale size: min 5, max 50
        node_values = top_scores # Use original scores for color map range
        vmin, vmax = min_score, max_score
    else:
        node_sizes = 15 # Default size if scores are zero/tiny
        node_values = top_scores
        vmin, vmax = 0, 1 # Default range

    try:
        fig_title = f"Top {len(top_indices)} {title_prefix} ROIs (Aggregated)"
        save_path = OUTPUT_DIR / output_filename

        # First create the brain markers plot
        brain_fig = plotting.plot_markers(
            node_values=node_values,
            node_coords=top_coords,
            # node_labels removed to match current nilearn API
            node_size=node_sizes,
            node_cmap='viridis', # Or choose another cmap
            node_vmin=vmin,
            node_vmax=vmax,
            title=fig_title,
            colorbar=True,
            display_mode='lzry',
            # Don't save yet - we'll add a legend first
            annotate=True # Show labels
        )

        # Create a custom figure with both brain views and ROI names
        plt.figure(figsize=(15, 10))
        
        # Create a larger figure for our combined plot
        fig, axes = plt.subplots(2, 1, figsize=(12, 14), gridspec_kw={'height_ratios': [4, 1]})
        
        # Close the previous brain plot window as we'll recreate it
        plt.close(brain_fig)
        
        # Replot the brain in the top subplot
        brain_fig = plotting.plot_markers(
            node_values=node_values,
            node_coords=top_coords,
            node_size=node_sizes,
            node_cmap='viridis',
            node_vmin=vmin,
            node_vmax=vmax,
            title=fig_title,
            colorbar=True,
            display_mode='lzry',
            axes=axes[0],
            annotate=True
        )
        
        # Create a table in the bottom subplot with ROI information
        axes[1].axis('off')  # Hide axes
        
        # Create a table with ROI names and their scores
        table_data = []
        for i, (idx, score) in enumerate(zip(top_indices, top_scores)):
            roi_name = labels_for_plot[i]
            table_data.append([f"{i+1}", f"{idx}", f"{roi_name}", f"{score:.4f}"])
        
        table = axes[1].table(
            cellText=table_data,
            colLabels=["Rank", "ROI Index", "ROI Name", "Importance Score"],
            loc='center',
            cellLoc='center',
            colWidths=[0.1, 0.1, 0.6, 0.2]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)  # Adjust table scale for better visibility
        
        # Add a title for the table
        axes[1].set_title("Top ROIs Details", pad=20)
        
        plt.tight_layout()
        
        # Save the combined figure
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved top aggregated nodes plot with ROI labels to: {save_path.name}")

    except Exception as e:
        print(f"  ERROR plotting top aggregated nodes: {e}")
        # Attempt fallback to original method without labels
        try:
            save_path = OUTPUT_DIR / output_filename
            brain_fig = plotting.plot_markers(
                node_values=node_values,
                node_coords=top_coords,
                node_size=node_sizes,
                node_cmap='viridis',
                node_vmin=vmin,
                node_vmax=vmax,
                title=fig_title,
                colorbar=True,
                display_mode='lzry',
                output_file=str(save_path),
                annotate=True
            )
            plt.close(brain_fig)
            print(f"  Used fallback method - saved basic top aggregated nodes plot to: {save_path.name}")
        except Exception as fallback_e:
            print(f"  ERROR in fallback plot: {fallback_e}")

# --- Main Execution Function ---
def run_visualization(attention_results, gnn_results):
    """
    Visualizes attention weights and GNNExplainer results for run-level subject classification.
    Generates connectome plots, node importance plots for a subset of runs/graphs,
    and identifies top ROIs based on average GNNExplainer importance across visualized graphs.
    Queries Gemini for interpretation of the top ROIs.
    """
    print("--- Running Visualization Pipeline (Run-Level Subject Classification) ---")

    # --- Configure Gemini API ---
    print("--- Configuring External APIs ---")
    gemini_client = configure_gemini()
    # --- End Configure API ---

    # Load coordinates and labels
    difumo_coords, difumo_labels = load_difumo_metadata()
    if difumo_coords is None:
        print("  FATAL ERROR: Could not load coordinates. Most visualizations will fail.")
        # return # Exit if coords are essential
    if difumo_labels is None or len(difumo_labels) != ATLAS_DIM:
         print("  Warning: Labels failed to load correctly. Top ROI names might be incorrect.")
         if difumo_labels is None:
              difumo_labels = [f"Node {i}" for i in range(ATLAS_DIM)]

    num_nodes = len(difumo_labels)
    aggregated_gnn_importance = np.zeros(num_nodes)
    aggregated_attn_importance = np.zeros(num_nodes)
    gnn_graphs_processed = 0
    attn_graphs_processed = 0

    # Visualize GNNExplainer results and aggregate importance
    if gnn_results:
        print("\nVisualizing GNNExplainer Results & Aggregating Importance...")
        for graph_id, result in gnn_results.items():
            if result and result.get('edge_mask') is not None: # Check if explanation & mask exist
                plot_explanation_connectome(
                    graph_id,
                    result['edge_index'],
                    result['edge_mask'], 
                    difumo_coords,
                    node_labels=difumo_labels,
                    threshold=0.6, 
                    top_n=50,      
                    title_prefix="GNNExplainer"
                )
                
                # Calculate importance for this graph
                node_imp = calculate_node_importance(result['edge_index'], result['edge_mask'], num_nodes)
                if node_imp is not None:
                    aggregated_gnn_importance += node_imp
                    gnn_graphs_processed += 1
                
                if result.get('node_feat_mask') is not None: # Check if node mask exists
                    plot_node_importance(
                        graph_id,
                        result['node_feat_mask'], 
                        difumo_coords,
                        node_labels=difumo_labels,
                        threshold=0.5, 
                        title_prefix="GNNExplainer"
                    )
            else:
                 print(f"  Skipping visualization/aggregation for {graph_id}: GNNExplainer result invalid or missing mask.")
    else:
        print("  No GNNExplainer results found to visualize.")

    # Visualize Attention results and aggregate importance
    if attention_results:
        print("\nVisualizing Attention Analysis Results & Aggregating Importance...")
        for graph_id, result in attention_results.items():
             if result and result.get('avg_attention') is not None: # Check if result and avg attention exist
                plot_explanation_connectome(
                    graph_id,
                    result['edge_index'],
                    result['avg_attention'], 
                    difumo_coords,
                    node_labels=difumo_labels,
                    threshold=0.5, 
                    top_n=50,
                    title_prefix="AvgAttention"
                )

                # Calculate importance for this graph
                node_imp = calculate_node_importance(result['edge_index'], result['avg_attention'], num_nodes)
                if node_imp is not None:
                    aggregated_attn_importance += node_imp
                    attn_graphs_processed += 1
                    
                # Optionally derive and plot node importance from attention
                try:
                     node_importance = np.zeros(ATLAS_DIM)
                     edge_idx = result['edge_index']
                     avg_attn = result['avg_attention']
                     # Check if shapes are compatible before using add.at
                     if edge_idx.max() < ATLAS_DIM and len(avg_attn) == edge_idx.shape[1]:
                         np.add.at(node_importance, edge_idx[0], np.abs(avg_attn)) # Sum absolute outgoing
                         np.add.at(node_importance, edge_idx[1], np.abs(avg_attn)) # Sum absolute incoming
                         if node_importance.max() > 0: # Normalize if needed
                              node_importance = node_importance / node_importance.max()
                         plot_node_importance(
                              graph_id,
                              node_importance,
                              difumo_coords,
                              node_labels=difumo_labels,
                              threshold=0.6, # Adjust threshold for summed importance
                              title_prefix="SummedAttention"
                         )
                     else:
                          print(f"    Warning: Cannot compute summed node attention for {graph_id} due to index/shape mismatch.")
                except Exception as e_node_attn:
                     print(f"    Warning: Could not derive/plot node importance from attention for {graph_id}: {e_node_attn}")

             else:
                  print(f"  Skipping visualization/aggregation for {graph_id}: Attention result invalid or missing average attention.")
    else:
        print("  No Attention Analysis results found to visualize.")

    # --- Print Overall Top ROIs & Get Interpretation ---
    top_gnn_indices = None
    top_gnn_scores = None
    if gnn_graphs_processed > 0:
        # Pass the client object, get back indices/scores
        top_gnn_indices, top_gnn_scores = print_overall_top_rois(
            aggregated_gnn_importance, difumo_labels, "GNNExplainer (Aggregated)", gemini_client, top_n=5
        )
        # Plot the dedicated top 5 GNN figure if we got indices
        if top_gnn_indices is not None:
            plot_top_aggregated_nodes(
                top_indices=top_gnn_indices,
                top_scores=top_gnn_scores,
                coords=difumo_coords,
                node_labels=difumo_labels,
                title_prefix="GNNExplainer",
                output_filename="top_5_gnn_aggregated_nodes.png"
            )
    else:
        print("\nNo GNNExplainer graphs processed for overall importance calculation.")

    if attn_graphs_processed > 0:
        # Pass None for client, don't expect indices/scores back for plotting this one
        print_overall_top_rois(aggregated_attn_importance, difumo_labels, "Attention (Aggregated)", None, top_n=5)
    else:
        print("\nNo Attention graphs processed for overall importance calculation.")

# --- Main Execution Block (Example) ---
if __name__ == "__main__":
    print("Executing Visualization Pipeline...")

    # Load results from explainability step
    attention_path = OUTPUT_DIR / "attention_analysis.pkl"
    gnn_explainer_path = OUTPUT_DIR / "gnnexplainer_results.pkl"

    attn_res = None
    gnn_res = None

    try:
        if attention_path.exists():
            with open(attention_path, 'rb') as f:
                attn_res = pickle.load(f)
            print(f"Loaded attention results from {attention_path}")
        else:
            print(f"Attention results file not found: {attention_path}")

        if gnn_explainer_path.exists():
            with open(gnn_explainer_path, 'rb') as f:
                gnn_res = pickle.load(f)
            print(f"Loaded GNNExplainer results from {gnn_explainer_path}")
        else:
             print(f"GNNExplainer results file not found: {gnn_explainer_path}")

    except Exception as e:
        print(f"ERROR loading explainability results: {e}")
        # Decide whether to proceed if only one type of result is available

    if attn_res is None and gnn_res is None:
        print("No explainability results found to visualize. Exiting.")
    else:
        # Ensure output dir exists before plotting
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        run_visualization(attn_res, gnn_res)
        print("\nVisualization Pipeline Complete.")