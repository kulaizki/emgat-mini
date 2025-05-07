import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

# Import configuration variables
from setup_config import (
    ATLAS_DIM, GAT_HIDDEN_CHANNELS, GAT_HEADS, GAT_DROPOUT
)

class GATClassifier(nn.Module):
    """Graph Attention Network model for graph classification."""
    def __init__(self, num_node_features, num_classes):
        super().__init__()
        self.num_node_features = num_node_features
        self.num_classes = num_classes # Represents number of subjects for run-level classification

        # First GAT Layer
        self.conv1 = GATConv(num_node_features, 
                             GAT_HIDDEN_CHANNELS, 
                             heads=GAT_HEADS, 
                             dropout=GAT_DROPOUT, 
                             edge_dim=1,
                             add_self_loops=False,
                             concat=True)
        
        # Add a second GAT layer for more capacity
        self.conv2 = GATConv(GAT_HIDDEN_CHANNELS * GAT_HEADS,
                             GAT_HIDDEN_CHANNELS,
                             heads=GAT_HEADS,
                             dropout=GAT_DROPOUT,
                             edge_dim=1,
                             add_self_loops=False,
                             concat=True)
        
        # Activation functions
        self.act1 = nn.ELU()
        self.act2 = nn.ELU()
        
        # Dropout layers for regularization
        self.dropout1 = nn.Dropout(GAT_DROPOUT)
        self.dropout2 = nn.Dropout(GAT_DROPOUT)
        
        # Global Pooling
        self.pool = global_mean_pool

        # Output MLP for better classification
        self.output_mlp = nn.Sequential(
            nn.Linear(GAT_HIDDEN_CHANNELS * GAT_HEADS, GAT_HIDDEN_CHANNELS * 2),
            nn.BatchNorm1d(GAT_HIDDEN_CHANNELS * 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(GAT_HIDDEN_CHANNELS * 2, num_classes)
        )

    def forward(self, x, edge_index, edge_attr, batch=None):
        """Defines the forward pass of the model.
        
        Args:
            x (Tensor): Node feature matrix [num_nodes, num_node_features].
            edge_index (LongTensor): Graph connectivity [2, num_edges].
            edge_attr (Tensor): Edge feature matrix [num_edges, edge_dim].
            batch (LongTensor, optional): Batch vector [num_nodes], assigns each node to a graph.
                                        Defaults to None for single graphs.

        Returns:
            Tensor: Raw classification logits [batch_size, num_classes].
            Tuple (Optional): Contains attention weights if return_attention_weights=True in GATConv.
                              (edge_index_att, attention_weights)
        """
        # First GAT layer with attention weights
        x_conv1, (edge_index_att, alpha) = self.conv1(x, edge_index, edge_attr=edge_attr, return_attention_weights=True)
        x1 = self.act1(x_conv1)
        x1 = self.dropout1(x1)
        
        # Second GAT layer
        x_conv2, _ = self.conv2(x1, edge_index, edge_attr=edge_attr, return_attention_weights=False)
        x2 = self.act2(x_conv2)
        x2 = self.dropout2(x2)

        # Global pooling
        graph_embedding = self.pool(x2, batch)

        # Final classification through MLP
        out_logits = self.output_mlp(graph_embedding)
        
        # Return logits and attention weights from first layer
        return out_logits, (edge_index_att, alpha)

# --- Function to Instantiate Model ---
def get_model(num_classes): # Renamed parameter for clarity in run-level context
    """Instantiates the GAT model based on configuration."""
    print("--- Instantiating GAT Model ---")
    num_node_features = ATLAS_DIM # Based on identity features
    model = GATClassifier(num_node_features=num_node_features, num_classes=num_classes)
    print(f"  Model Instantiated:")
    print(f"    Input node features: {num_node_features}")
    print(f"    Hidden channels: {GAT_HIDDEN_CHANNELS}")
    print(f"    Attention heads: {GAT_HEADS}")
    print(f"    Output classes (e.g., subjects): {num_classes}")
    # print(model) # Optionally print the full model structure
    return model

# --- Main Execution Function (Example) ---
if __name__ == "__main__":
    example_num_subjects = 2
    model = get_model(example_num_subjects)
    print("\nModel Definition Script Complete.") 