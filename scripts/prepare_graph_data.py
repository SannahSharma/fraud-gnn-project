"""
prepare_graph_data.py

Purpose: Convert the NetworkX transaction graph (built by build_graph.py) into
a PyTorch Geometric `Data` object -- the format GraphSAGE needs to train.

Steps:
1. Assign each account (node) numeric FEATURES based on its transaction
   behavior -- never based on the fraud label itself, to avoid "leaking"
   the answer into the input.
2. Assign each account a LABEL: was it ever involved in at least one
   fraudulent transaction? This is what GraphSAGE will learn to predict.
3. Convert the graph's structure into the edge_index format PyTorch
   Geometric expects.
4. Split nodes into train/test sets.
5. Save the resulting Data object to disk.
"""

import pickle
import numpy as np
import torch
from torch_geometric.data import Data

# --- Step 1: Load the graph we built earlier ---
print("Loading transaction graph...")
with open("data/transaction_graph.pkl", "rb") as f:
    G = pickle.load(f)

print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

# --- Step 2: Give every node a fixed integer index ---
# PyTorch Geometric works with plain integer node IDs (0, 1, 2, ...), not
# strings like "C1231006815". We build a lookup table to convert between them.
nodes = list(G.nodes())
node_to_idx = {node: i for i, node in enumerate(nodes)}
num_nodes = len(nodes)

# --- Step 3: Build per-node features from transaction behavior ---
# For each account: how many transactions did it send/receive, how much
# money on average did it send/receive, and is it a merchant account?
# None of these use the isFraud label -- no leakage.
sent_amounts = {node: [] for node in nodes}
received_amounts = {node: [] for node in nodes}

for u, v, data in G.edges(data=True):
    sent_amounts[u].append(data["amount"])
    received_amounts[v].append(data["amount"])

features = np.zeros((num_nodes, 5), dtype=np.float32)

for node in nodes:
    idx = node_to_idx[node]
    sent = sent_amounts[node]
    received = received_amounts[node]

    features[idx, 0] = len(sent)                                          # out-degree
    features[idx, 1] = len(received)                                      # in-degree
    features[idx, 2] = sum(sent) / len(sent) if sent else 0.0             # avg sent amount
    features[idx, 3] = sum(received) / len(received) if received else 0.0 # avg received amount
    features[idx, 4] = 1.0 if node.startswith("M") else 0.0               # is merchant?

# --- Step 4: Build the node label -- was this account ever part of fraud? ---
labels = np.zeros(num_nodes, dtype=np.int64)
for u, v, data in G.edges(data=True):
    if data["isFraud"] == 1:
        labels[node_to_idx[u]] = 1
        labels[node_to_idx[v]] = 1

print(f"Nodes labeled as fraud-associated: {labels.sum()} out of {num_nodes} "
      f"({(labels.sum() / num_nodes) * 100:.2f}%)")

# --- Step 5: Build edge_index -- PyTorch Geometric's format for graph structure ---
# A 2-row tensor: row 0 = source node indices, row 1 = target node indices.
# Edge i goes from edge_index[0, i] to edge_index[1, i].
edge_list = [(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()]
edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

# --- Step 6: Split nodes into train/test sets (80% train, 20% test) ---
# We split by NODE (account), since GraphSAGE here predicts a property of accounts.
np.random.seed(42)
indices = np.random.permutation(num_nodes)
split_point = int(num_nodes * 0.8)
train_idx = indices[:split_point]
test_idx = indices[split_point:]

train_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)
train_mask[train_idx] = True
test_mask[test_idx] = True

# --- Step 7: Package everything into a PyTorch Geometric Data object ---
data = Data(
    x=torch.tensor(features, dtype=torch.float),
    edge_index=edge_index,
    y=torch.tensor(labels, dtype=torch.long),
    train_mask=train_mask,
    test_mask=test_mask,
)

print("\nPyTorch Geometric Data object:")
print(data)

# --- Step 8: Save it, along with the node_to_idx lookup (needed later to map
# predictions back to real account IDs, e.g. for the dashboard) ---
with open("data/graph_data.pkl", "wb") as f:
    pickle.dump({"data": data, "node_to_idx": node_to_idx}, f)

print("\nSaved PyTorch Geometric data to data/graph_data.pkl")