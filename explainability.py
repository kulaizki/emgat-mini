import torch
import numpy as np
import pickle

try:
    from torch_geometric.explain import GNNExplainer as GNNExplainerAlgo, Explainer
except ImportError:
    try:
        from torch_geometric.nn import GNNExplainer as GNNExplainerAlgo 
        Explainer = None 
        print("Warning: Using legacy GNNExplainer. New Explainer API not found.")
    except ImportError:
        print("ERROR: Could not import GNNExplainer. Check PyTorch Geometric installation.")
        GNNExplainerAlgo = None
        Explainer = None

from pathlib import Path
from torch_geometric.loader import DataLoader

from setup_config import (
    OUTPUT_DIR, GAT_HEADS, GNN_EXPLAINER_EPOCHS 
)

from model_definition import get_model

# --- Model Wrapper (Essential for Explainer API) ---
class ModelWrapper(torch.nn.Module):
    def __init__(self, model_to_wrap):
        super().__init__()
        self.model = model_to_wrap
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        output = self.model(x, edge_index, edge_attr, batch)
        return output[0] if isinstance(output, tuple) else output

def analyze_attention(model, graph_list, dataset_info):
    """
    Extracts and analyzes attention weights from the trained GAT model
    for each trial graph in the provided list.

    Args:
        model (torch.nn.Module): The trained GAT model instance.
        graph_list (list): List of PyG Data objects (trial-level graphs).
        dataset_info (dict): Dictionary containing 'num_categories' and 'category_names'.

    Returns:
        dict: Dictionary containing attention results keyed by graph ID, or None.
    """
    print("--- Analyzing GAT Attention Weights (Trial-Level) ---")
    if not graph_list or not dataset_info:
        print("  ERROR: Graph list or dataset info not provided. Cannot analyze attention.")
        return None

    category_names = dataset_info.get('category_names', [f'Category {i}' for i in range(dataset_info.get('num_categories',0))])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    attention_results = {}
    print(f"  Extracting attention for {len(graph_list)} trial graphs...")

    with torch.no_grad():
        for i, graph in enumerate(graph_list):
            graph = graph.to(device)
            try:
                graph_id = f"Trial_{graph.trial_id}_Sub_{graph.subject}_Ses_{graph.session}_Cat_{graph.category_name}"
            except AttributeError:
                 graph_id = f"Graph_{i}"

            try:
                output = model(graph.x, graph.edge_index, graph.edge_attr, graph.batch if hasattr(graph, 'batch') else None)
                if isinstance(output, tuple) and len(output) == 2 and isinstance(output[1], tuple) and len(output[1]) == 2:
                     logits, (edge_index_att, alpha) = output
                else:
                     print(f"  Warning: Model forward pass for {graph_id} did not return expected (logits, (edge_index, alpha)) tuple. Cannot extract attention.")
                     attention_results[graph_id] = None
                     continue

                predicted_class_idx = logits.argmax(dim=1).item()
                predicted_class_name = category_names[predicted_class_idx] if predicted_class_idx < len(category_names) else "Unknown"
                true_label_idx = graph.y.item()
                true_class_name = category_names[true_label_idx] if true_label_idx < len(category_names) else "Unknown"

                avg_attention = alpha.mean(dim=1).cpu().numpy()

                attention_results[graph_id] = {
                    'edge_index': edge_index_att.cpu().numpy(),
                    'attention_weights': alpha.cpu().numpy(),
                    'avg_attention': avg_attention,
                    'true_label': true_label_idx,
                    'predicted_label': predicted_class_idx,
                    'true_class_name': true_class_name,
                    'predicted_class_name': predicted_class_name,
                    'subject': graph.subject if hasattr(graph, 'subject') else 'N/A',
                    'session': graph.session if hasattr(graph, 'session') else 'N/A',
                    'trial_id': graph.trial_id if hasattr(graph, 'trial_id') else 'N/A'
                }
            except Exception as e:
                 print(f"    ERROR during forward pass/attention extraction for {graph_id}: {e}")
                 attention_results[graph_id] = None

    successful_extractions = sum(1 for res in attention_results.values() if res is not None)
    print(f"  Extracted attention for {successful_extractions} graphs.")

    attention_save_path = OUTPUT_DIR / "attention_analysis_trial_level.pkl"
    try:
        with open(attention_save_path, 'wb') as f:
            pickle.dump(attention_results, f)
        print(f"  Saved attention analysis results to: {attention_save_path}")
    except Exception as e:
        print(f"  Warning: Could not save attention results: {e}")

    return attention_results

# Helper function to create a batch of the same graph
def create_graph_batch(graph, batch_size):
    if batch_size <= 0: return graph # No batching if batch_size is invalid
    graphs = [graph.clone() for _ in range(batch_size)]
    loader = DataLoader(graphs, batch_size=batch_size)
    return next(iter(loader))

def run_gnn_explainer(model, graph_list, dataset_info):
    """
    Runs GNNExplainer for trial-level category classification.
    Uses the newer Explainer API if available, otherwise falls back to legacy GNNExplainer.

    Args:
        model (torch.nn.Module): The trained GAT model instance.
        graph_list (list): List of PyG Data objects (trial-level graphs).
        dataset_info (dict): Dictionary containing 'num_categories' and 'category_names'.

    Returns:
        dict: Dictionary containing explanation results keyed by graph ID, or None.
    """
    print("--- Running GNNExplainer (Trial-Level) ---")
    if GNNExplainerAlgo is None and Explainer is None:
        print("  ERROR: GNNExplainer algorithm not imported. Skipping.")
        return None
    if not graph_list or not dataset_info:
        print("  ERROR: Graph list or dataset info not provided.")
        return None

    num_categories = dataset_info['num_categories']
    category_names = dataset_info['category_names']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    model_for_explainer = ModelWrapper(model).to(device)

    explainer_instance = None
    if Explainer is not None: # Prefer new API
        print("  Using new Explainer API with GNNExplainer algorithm.")
        try:
            gnn_algo = GNNExplainerAlgo(epochs=GNN_EXPLAINER_EPOCHS)
            explainer_instance = Explainer(
                model=model_for_explainer,
                algorithm=gnn_algo,
                explanation_type='model',
                model_config=dict(
                    mode='multiclass_classification', # Correct mode
                    task_level='graph',
                    return_type='raw'
                ),
                node_mask_type='object',
                edge_mask_type='object'
            )
            print(f"  New Explainer configured (epochs={GNN_EXPLAINER_EPOCHS}).")
        except Exception as e:
            print(f"  ERROR Configuring new Explainer API: {e}. Fallback may not be available.")
            return None
    elif GNNExplainerAlgo is not None: # Fallback to legacy GNNExplainer
        print("  Using legacy GNNExplainer API.")
        try:
            # Legacy GNNExplainer is instantiated per graph usually, or reconfigured.
            # For simplicity, we'll create it once and assume it can handle multiple calls
            # or we handle it inside the loop if necessary.
            explainer_instance = GNNExplainerAlgo(
                model_for_explainer, 
                epochs=GNN_EXPLAINER_EPOCHS,
                # Add other relevant parameters for legacy version if needed, e.g., log, return_type
            )
            print(f"  Legacy GNNExplainer configured (epochs={GNN_EXPLAINER_EPOCHS}).")
        except Exception as e:
            print(f"  ERROR Configuring legacy GNNExplainer: {e}.")
            return None
    else:
        # Should have been caught earlier, but as a safeguard.
        print("  ERROR: No GNNExplainer implementation available.")
        return None

    explanation_results = {}
    # For BatchNorm, a batch size > 1 is needed. For GNNExplainer, often small batch helps.
    # Let's use a small batch size if BatchNorm is present, otherwise 1.
    has_batchnorm = any(isinstance(module, torch.nn.BatchNorm1d) for module in model.modules())
    effective_batch_size = 4 if has_batchnorm else 1 
    if effective_batch_size > 1 : print(f"  Using batch size {effective_batch_size} for GNNExplainer due to BatchNorm.")

    print(f"  Generating explanations for {len(graph_list)} trial graphs...")

    for i, graph_original in enumerate(graph_list):
        graph = graph_original.to(device)
        try:
            graph_id = f"Trial_{graph.trial_id}_Sub_{graph.subject}_Ses_{graph.session}_Cat_{graph.category_name}"
        except AttributeError:
            graph_id = f"Graph_{i}"

        try:
            model_for_explainer.eval() # Ensure model is in eval mode
            with torch.no_grad():
                 logits = model_for_explainer(graph.x, graph.edge_index, graph.edge_attr, graph.batch)
                 prediction_idx = logits.argmax().item()

            # Prepare graph for explainer (potentially batching)
            graph_to_explain = graph
            if effective_batch_size > 1:
                graph_to_explain = create_graph_batch(graph, effective_batch_size).to(device)
            
            # Ensure model is in eval again right before explaining
            model_for_explainer.eval()

            if Explainer is not None: # New API
                explanation_obj = explainer_instance(
                    x=graph_to_explain.x,
                    edge_index=graph_to_explain.edge_index,
                    edge_attr=graph_to_explain.edge_attr,
                    batch=graph_to_explain.batch,
                    target=torch.tensor([prediction_idx]).to(device) if graph_to_explain.batch is not None else None # Target for graph-level
                )
                node_mask = explanation_obj.node_mask.cpu().numpy() if hasattr(explanation_obj, 'node_mask') and explanation_obj.node_mask is not None else None
                edge_mask = explanation_obj.edge_mask.cpu().numpy() if hasattr(explanation_obj, 'edge_mask') and explanation_obj.edge_mask is not None else None
                # Extract for the first graph if batched
                if effective_batch_size > 1 and node_mask is not None: node_mask = node_mask[:graph.num_nodes]
                if effective_batch_size > 1 and edge_mask is not None: edge_mask = edge_mask[:graph.num_edges]
            
            else: 
                node_feat_mask, edge_mask_legacy = explainer_instance.explain_graph(
                    graph_to_explain.x, 
                    graph_to_explain.edge_index, 
                    edge_attr=graph_to_explain.edge_attr, 
                    batch=graph_to_explain.batch if hasattr(graph_to_explain, 'batch') else None
                )
                node_mask = node_feat_mask.cpu().numpy() if node_feat_mask is not None else None
                edge_mask = edge_mask_legacy.cpu().numpy() if edge_mask_legacy is not None else None
                if effective_batch_size > 1 and node_mask is not None: node_mask = node_mask[:graph.num_nodes]
                if effective_batch_size > 1 and edge_mask is not None: edge_mask = edge_mask[:graph.num_edges]

            true_label_idx = graph.y.item()
            explanation_results[graph_id] = {
                'edge_index': graph.edge_index.cpu().numpy(),
                'node_feat_mask': node_mask, # May represent feature importance or node mask
                'edge_mask': edge_mask,
                'true_label': true_label_idx,
                'predicted_label': prediction_idx,
                'true_class_name': category_names[true_label_idx] if true_label_idx < len(category_names) else 'Unknown',
                'predicted_class_name': category_names[prediction_idx] if prediction_idx < len(category_names) else 'Unknown',
                'subject': graph.subject if hasattr(graph, 'subject') else 'N/A',
                'session': graph.session if hasattr(graph, 'session') else 'N/A',
                'trial_id': graph.trial_id if hasattr(graph, 'trial_id') else 'N/A'
            }
        except Exception as e:
            print(f"    ERROR during GNNExplainer for {graph_id}: {e}")
            explanation_results[graph_id] = None

    successful_explanations = sum(1 for res in explanation_results.values() if res is not None)
    print(f"  Generated GNNExplanations for {successful_explanations} graphs.")

    gnnexplainer_save_path = OUTPUT_DIR / "gnnexplainer_results_trial_level.pkl"
    try:
        with open(gnnexplainer_save_path, 'wb') as f:
            pickle.dump(explanation_results, f)
        print(f"  Saved GNNExplainer results to: {gnnexplainer_save_path}")
    except Exception as e:
        print(f"  Warning: Could not save GNNExplainer results: {e}")

    return explanation_results

# --- Main Execution Function ---
if __name__ == "__main__":
    print("Executing Explainability Pipeline (Trial-Level)...")

    # --- Load trial-level graph list, dataset info, and model ---
    dataset_storage_path = OUTPUT_DIR / 'pyg_trial_level_dataset'
    list_save_path = dataset_storage_path / 'trial_level_graph_list.pt'
    info_path = OUTPUT_DIR / "trial_level_dataset_info.pkl"
    model_path = OUTPUT_DIR / "gat_model_trial_level.pt"

    graph_list = None
    dataset_info = None
    model_state_dict = None
    actual_num_categories = None

    try:
        print(f"Loading graph list from {list_save_path}...")
        try: graph_list = torch.load(list_save_path)
        except RuntimeError: graph_list = torch.load(list_save_path, weights_only=False)
        print(f"Loaded graph list with {len(graph_list)} graphs.")

        with open(info_path, 'rb') as f:
             dataset_info = pickle.load(f)
             actual_num_categories = dataset_info['num_categories']
             print(f"Loaded dataset info. Categories: {actual_num_categories}")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_state_dict = torch.load(model_path, map_location=device)
        print(f"Loaded trained model state from {model_path}")

    except FileNotFoundError as e:
        print(f"ERROR: Required file not found: {e.filename}. Cannot proceed.")
        print("Ensure previous steps (graph construction, training) ran successfully.")
        exit(1)
    except Exception as e:
        print(f"ERROR loading data or model: {e}")
        exit(1)

    # --- Instantiate and Load Model ---
    model = get_model(num_categories=actual_num_categories)
    model.load_state_dict(model_state_dict)
    model.eval() # Set to eval mode

    # --- Run Explainability Methods ---
    if model and graph_list and dataset_info:
        print("\nRunning Attention Analysis...")
        attention_results = analyze_attention(model, graph_list, dataset_info)
        if attention_results:
            print("Attention Analysis completed.")

        print("\nRunning GNNExplainer...")
        gnn_results = run_gnn_explainer(model, graph_list, dataset_info)
        if gnn_results:
            print("GNNExplainer completed.")

        print("\nExplainability Pipeline Complete (Trial-Level).")
    else:
        print("\nSkipping explainability due to loading errors or missing components.")