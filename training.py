# Filename: training.py
# Location: <workspace_root>/training.py
# Description: Implements the GAT model training loop (Step 10) and plots history
#              for the RUN-LEVEL subject classification task.
# Changes:
#  - Loads graph list saved by graph_construction.py instead of Dataset object.
#  - Loads run_level_dataset_info.pkl to get actual class count (2).
#  - Instantiates model with correct number of classes.
#  - train_model accepts and iterates through the graph list.

import torch
import torch.optim as optim
import torch.nn.functional as F # Often needed, though maybe not explicitly here
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pickle
from pathlib import Path # Added
import numpy as np
from torch_geometric.loader import DataLoader # Import DataLoader

# Import configuration and model definition
from setup_config import (
    OUTPUT_DIR, LEARNING_RATE, WEIGHT_DECAY, N_EPOCHS, RANDOM_SEED # Add RANDOM_SEED
)
# Removed graph_construction import as we load list directly
# from graph_construction import ConnectivityDataset
from model_definition import get_model


def train_model(model, graph_list, batch_size=8): # Added batch_size argument
    """
    Trains the GAT model using DataLoader for batching.

    Args:
        model (torch.nn.Module): The GAT model instance.
        graph_list (list): The full list of PyG Data objects.
        batch_size (int): The size of mini-batches for training.

    Returns:
        tuple: (trained_model, training_history_dict) or (None, {}) on error.
    """
    print("--- Starting Model Training ---")
    if not graph_list:
        print("  ERROR: Graph list not provided or empty. Cannot train.")
        return None, {}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Using device: {device}")
    model.to(device)

    # --- Verify Model Output Size vs Data Labels ---
    try:
        # Updated to access the last layer of the model's output_mlp
        model_output_classes = model.output_mlp[-1].out_features
        # Check against maximum label value + 1 in the dataset list
        max_label_in_data = -1
        if graph_list: # Ensure list is not empty
             max_label_in_data = max(data.y.item() for data in graph_list if hasattr(data, 'y') and data.y is not None)
        expected_classes_from_data = max_label_in_data + 1

        if model_output_classes != expected_classes_from_data:
             # This check might still trigger if model is 2-class but data has only 1 class (unlikely but possible)
             print(f"  WARNING: Model output layer size ({model_output_classes}) might not match "
                   f"expected classes based on dataset labels ({expected_classes_from_data}).")
        else:
             print(f"  Model output size ({model_output_classes}) matches dataset labels.")
    except AttributeError as ae:
        print(f"  Warning: Could not verify model output layer size: {ae}")
    except Exception as e:
         print(f"  Warning: Error verifying model/dataset class consistency: {e}")
    # --- End Verification ---

    # --- Data Splitting and DataLoader --- 
    print(f"  Creating DataLoaders with batch size: {batch_size}")
    np.random.seed(RANDOM_SEED)  # Use seed from config
    indices = np.random.permutation(len(graph_list))
    train_size = int(0.8 * len(graph_list))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_graphs = [graph_list[i] for i in train_indices]
    val_graphs = [graph_list[i] for i in val_indices]

    # Create DataLoaders
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False) # No shuffle for validation
    print(f"  Data split: {len(train_graphs)} training graphs, {len(val_graphs)} validation graphs")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # Add learning rate scheduler for better convergence - removed verbose parameter for compatibility
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5,
                                                    patience=20)
    criterion = torch.nn.CrossEntropyLoss()

    history = {'epoch': [], 'train_loss': [], 'train_accuracy': [],
               'val_loss': [], 'val_accuracy': []}

    # Setup early stopping
    best_val_acc = 0.0
    patience = 30  # Number of epochs to wait for improvement
    patience_counter = 0
    best_model_state = None

    print(f"  Training for up to {N_EPOCHS} epochs...")
    print(f"  Using early stopping with patience={patience}")

    for epoch in range(N_EPOCHS):
        # --- Training Phase ---
        model.train()
        train_total_loss = 0
        train_correct_preds = 0
        train_total_samples = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            try:
                # Forward pass - expects batch object
                out_logits, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

                target = batch.y
                # Ensure target has correct shape [batch_size]
                if target.dim() > 1: target = target.squeeze()

                # Validate shapes and targets
                if out_logits.shape[0] != target.shape[0]:
                    print(f"\nSKIP BATCH: Mismatch logits ({out_logits.shape[0]}) vs target ({target.shape[0]})")
                    continue
                if target.max().item() >= out_logits.shape[1]:
                    print(f"\nSKIP BATCH: Target label {target.max().item()} >= output size {out_logits.shape[1]}")
                    continue

                # Handle potential NaN in input or output
                if torch.isnan(batch.x).any() or torch.isnan(batch.edge_attr).any():
                    print("\nSKIP BATCH: NaN detected in input data.")
                    continue

                loss = criterion(out_logits, target)

                if torch.isnan(loss):
                    print(f"\nSKIP BATCH: NaN loss detected. Logits max: {out_logits.max()}, min: {out_logits.min()}")
                    continue
                    
                loss.backward()
                optimizer.step()

                train_total_loss += loss.item() * batch.num_graphs # Accumulate loss weighted by batch size
                preds = out_logits.argmax(dim=1)
                train_correct_preds += (preds == target).sum().item()
                train_total_samples += batch.num_graphs
                
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print("\nERROR: CUDA out of memory. Try reducing batch_size.")
                    return None, history
                elif "Expected more than 1 value per channel" in str(e):
                     print(f"\nSKIP BATCH: BatchNorm1d error (likely batch size 1). Batch size: {batch.num_graphs}")
                     continue # Skip this batch if it has size 1
                else:
                    print(f"\nERROR during training batch: {e}")
                    continue # Skip problematic batch
            except Exception as e:
                print(f"\nERROR during training batch processing: {e}")
                continue
        
        # --- Validation Phase ---
        model.eval()
        val_total_loss = 0
        val_correct_preds = 0
        val_total_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                
                try:
                    out_logits, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                    target = batch.y
                    if target.dim() > 1: target = target.squeeze()

                    if out_logits.shape[0] != target.shape[0]: continue
                    if target.max().item() >= out_logits.shape[1]: continue
                    if torch.isnan(batch.x).any() or torch.isnan(batch.edge_attr).any(): continue

                    loss = criterion(out_logits, target)
                    
                    if torch.isnan(loss): continue # Skip batches with NaN loss

                    val_total_loss += loss.item() * batch.num_graphs
                    preds = out_logits.argmax(dim=1)
                    val_correct_preds += (preds == target).sum().item()
                    val_total_samples += batch.num_graphs
                    
                except Exception as e:
                    print(f"\nWarning: Error during validation batch: {e}")
                    continue
        
        # Calculate metrics
        train_avg_loss = train_total_loss / train_total_samples if train_total_samples > 0 else 0
        train_accuracy = train_correct_preds / train_total_samples if train_total_samples > 0 else 0
        
        val_avg_loss = val_total_loss / val_total_samples if val_total_samples > 0 else 0
        val_accuracy = val_correct_preds / val_total_samples if val_total_samples > 0 else 0

        history['epoch'].append(epoch)
        history['train_loss'].append(train_avg_loss)
        history['train_accuracy'].append(train_accuracy)
        history['val_loss'].append(val_avg_loss)
        history['val_accuracy'].append(val_accuracy)
        
        scheduler.step(val_accuracy)
        
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:03d}/{N_EPOCHS} | Train Loss: {train_avg_loss:.4f} | "
                  f"Train Acc: {train_accuracy:.4f} | Val Loss: {val_avg_loss:.4f} | "
                  f"Val Acc: {val_accuracy:.4f}")
        
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}. Best val accuracy: {best_val_acc:.4f}")
            break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"  Restored best model with validation accuracy: {best_val_acc:.4f}")

    print("--- Training Complete ---")
    # --- Updated Save Paths for Trial-Level ---
    model_save_path = OUTPUT_DIR / "gat_model_trial_level.pt"
    history_save_path = OUTPUT_DIR / "training_history_trial_level.pkl"
    # ---

    try:
        torch.save(model.state_dict(), model_save_path)
        print(f"  Trained model saved to: {model_save_path}")
    except Exception as e:
        print(f"  ERROR saving model: {e}")
        return None, history

    try:
        with open(history_save_path, 'wb') as f:
            pickle.dump(history, f)
        print(f"  Training history saved to: {history_save_path}")
    except Exception as e:
        print(f"  Warning: Failed to save training history: {e}")

    return model, history

def plot_training_history(history):
    """Plots the training loss and accuracy over epochs."""
    print("--- Plotting Training History ---")
    if not history or not history.get('epoch'): # Check if history exists and has epochs
        print("  No valid training history to plot.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Plot losses
    ax1.set_title('Training and Validation Loss')
    ax1.plot(history['epoch'], history['train_loss'], 'b-', label='Training Loss')
    if 'val_loss' in history:
        ax1.plot(history['epoch'], history['val_loss'], 'r-', label='Validation Loss')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    ax1.legend()
    
    # Plot accuracies
    ax2.set_title('Training and Validation Accuracy')
    ax2.plot(history['epoch'], history['train_accuracy'], 'b-', label='Training Accuracy')
    if 'val_accuracy' in history:
        ax2.plot(history['epoch'], history['val_accuracy'], 'r-', label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0, 1.05)
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()

    plot_save_path = OUTPUT_DIR / "training_history_runlevel.png"
    try:
        plt.savefig(plot_save_path)
        print(f"  Training history plot saved to: {plot_save_path}")
    except Exception as e:
        print(f"  Warning: Failed to save training plot: {e}")
    
    plt.close(fig) # Close plot figure


# --- Main Execution Function (Example/Testing) ---
if __name__ == "__main__":
    print("Executing Model Training Pipeline (Run-Level)...")

    # --- Load the graph list and dataset info ---
    dataset_storage_path = OUTPUT_DIR / 'pyg_run_level_dataset'
    list_save_path = dataset_storage_path / 'run_level_graph_list.pt'
    info_path = OUTPUT_DIR / "run_level_dataset_info.pkl"

    graph_list = None
    dataset_info = None
    actual_num_classes = None

    try:
        # Load the list of graph objects
        print(f"Loading graph list from {list_save_path}...")
        # Use weights_only=False as fallback if needed
        try:
             graph_list = torch.load(list_save_path)
        except RuntimeError as re:
            if "Unsupported global" in str(re) or "Weights only load failed" in str(re):
                 print(f"  Warning: torch.load failed with default weights_only=True. Retrying with weights_only=False.")
                 graph_list = torch.load(list_save_path, weights_only=False)
            else: raise re # Reraise other runtime errors
        print(f"Loaded graph list with {len(graph_list)} graphs.")

        # Load the dataset info dictionary
        with open(info_path, 'rb') as f:
             dataset_info = pickle.load(f)
             actual_num_classes = dataset_info['num_classes']
             print(f"Loaded dataset info. Actual classes: {actual_num_classes}")

    except FileNotFoundError as e:
        print(f"ERROR: Cannot find required input file: {e}")
        print("Please run graph_construction.py (modified for run-level) first.")
        exit()
    except Exception as e:
        print(f"ERROR loading dataset list or info: {e}")
        exit()

    # Basic validation
    if not graph_list or not dataset_info or actual_num_classes is None:
         print("ERROR: Failed to load valid graph list or dataset info.")
         exit()

    # Instantiate the model using the ACTUAL number of classes
    model = get_model(num_categories=actual_num_classes)
    if model is None:
        print("ERROR: Failed to instantiate model. Exiting.")
        exit()

    # Train the model using the loaded graph list
    trained_model, history = train_model(model, graph_list)

    # Plot results if training completed successfully
    if history and trained_model is not None:
        plot_training_history(history)
        print("\nModel Training Pipeline Complete.")
    else:
        print("\nModel Training Pipeline Failed or did not produce results.")