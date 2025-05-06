import pickle
from pathlib import Path
import os

# --- Step 1: Imports and Setup ---
print("--- Initializing Visualization Verification Script ---")

# Assuming visualization.py and setup_config.py are in the same directory or accessible
try:
    from visualization import run_visualization, OUTPUT_DIR
    print(f"Imported run_visualization function. Using OUTPUT_DIR: {OUTPUT_DIR}")
except ImportError as e:
    print(f"Error importing from visualization.py: {e}")
    print("Please ensure visualization.py is in the correct path and defines OUTPUT_DIR.")
    exit()
except NameError:
    # This might happen if visualization.py doesn't define OUTPUT_DIR directly
    try:
        from setup_config import OUTPUT_DIR
        from visualization import run_visualization
        print(f"Imported run_visualization. Using OUTPUT_DIR from setup_config: {OUTPUT_DIR}")
    except (ImportError, NameError) as e2:
        print(f"Error importing OUTPUT_DIR from setup_config or run_visualization: {e2}")
        print("Please ensure OUTPUT_DIR is defined in either visualization.py or setup_config.py.")
        exit()

# --- Step 2: Load Explainability Results ---
print("\n--- Loading Explainability Results ---")
attention_path = OUTPUT_DIR / "attention_analysis.pkl"
gnn_explainer_path = OUTPUT_DIR / "gnnexplainer_results.pkl"

attn_res = None
gnn_res = None
results_loaded = False

print(f"Looking for results in: {OUTPUT_DIR}")

try:
    if attention_path.exists():
        with open(attention_path, 'rb') as f:
            attn_res = pickle.load(f)
        print(f"- Loaded attention results from {attention_path.name}")
        results_loaded = True
    else:
        print(f"- Attention results file not found: {attention_path}")

    if gnn_explainer_path.exists():
        with open(gnn_explainer_path, 'rb') as f:
            gnn_res = pickle.load(f)
        print(f"- Loaded GNNExplainer results from {gnn_explainer_path.name}")
        results_loaded = True
    else:
         print(f"- GNNExplainer results file not found: {gnn_explainer_path}")

except Exception as e:
    print(f"ERROR loading explainability results: {e}")

if not results_loaded:
    print("\nNo explainability results found to visualize. Exiting script.")
    exit()
else:
    print("- Results loaded successfully.")

# --- Step 3: Run Visualization Pipeline ---
print("\n--- Running Visualization Pipeline ---")
pipeline_executed = False
try:
    # Ensure output dir exists before plotting
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Ensured output directory exists: {OUTPUT_DIR}")
    run_visualization(attn_res, gnn_res)
    print("Visualization Pipeline Complete.")
    pipeline_executed = True
except Exception as e:
    print(f"An error occurred during visualization: {e}")
    # You might want to print traceback here for debugging
    # import traceback
    # traceback.print_exc()

# --- Step 4: Verify and Report Output Files ---
print("\n--- Verifying Output Files ---")

top_5_plot_path = OUTPUT_DIR / "top_5_gnn_aggregated_nodes.png"
interpretation_md_path = OUTPUT_DIR / "top_5_gnn_interpretation.md"

if not pipeline_executed:
    print("Skipping output verification as visualization pipeline did not complete successfully.")
else:
    # Check for plot
    print("\nChecking for Top 5 GNN ROI Plot...")
    if top_5_plot_path.exists():
        print(f"SUCCESS: Plot file found at: {top_5_plot_path}")
        print("(Open this file manually to view the image)")
    else:
        print(f"FAILURE: Plot file NOT found at {top_5_plot_path}")

    # Check for Markdown interpretation
    print("\nChecking for Top 5 GNN Interpretation File...")
    if interpretation_md_path.exists():
        print(f"SUCCESS: Interpretation file found at: {interpretation_md_path}")
        try:
            print("\n--- Content of Interpretation File: ---")
            with open(interpretation_md_path, 'r') as f:
                md_content = f.read()
            print(md_content)
            print("--- End of File Content ---")
        except Exception as e:
            print(f"Error reading interpretation file {interpretation_md_path}: {e}")
    else:
        print(f"FAILURE: Interpretation file NOT found at {interpretation_md_path}")

print("\n--- Verification Script Finished ---") 