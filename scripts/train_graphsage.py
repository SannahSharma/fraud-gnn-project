"""
train_graphsage.py

Purpose: Define and train a GraphSAGE model to classify accounts (nodes) as
fraud-associated or not, based on transaction graph structure and behavior.

We also save the model's learned NODE EMBEDDINGS -- the numeric "fingerprint"
GraphSAGE computes for each account by looking at its neighborhood. On Day 2,
we'll feed these embeddings into XGBoost alongside transaction-level features,
combining "does this account look suspicious in its network" with
"does this specific transaction look suspicious on its own."
"""

import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from sklearn.metrics import average_precision_score

# --- Step 1: Load the PyTorch Geometric data we prepared earlier ---
print("Loading graph data...")
with open("data/graph_data.pkl", "rb") as f:
    saved = pickle.load(f)

data = saved["data"]
node_to_idx = saved["node_to_idx"]

# --- Step 2: Normalize features ---
# Raw features (transaction counts, average amounts) have very different
# scales. Neural networks train much more reliably when inputs are on a
# similar scale, so we log-transform (compresses large values) then
# standardize (subtract mean, divide by std) using ONLY training nodes'
# statistics -- using test nodes here would leak information.
x = data.x.clone()
x[:, :4] = torch.log1p(x[:, :4])  # log1p(x) = log(1 + x), safely handles zeros

train_mean = x[data.train_mask].mean(dim=0)
train_std = x[data.train_mask].std(dim=0) + 1e-6  # avoid divide-by-zero
x = (x - train_mean) / train_std
data.x = x

print(f"Feature shape after normalization: {data.x.shape}")


# --- Step 3: Define the GraphSAGE model ---
class GraphSAGE(nn.Module):
    """
    A 2-layer GraphSAGE network. Each SAGEConv layer aggregates a node's
    neighbors' features and combines that with the node's own features.
    Two stacked layers mean each node "sees" its neighbors' neighbors --
    a 2-hop neighborhood -- which is how ring-like fraud patterns become
    visible in the learned representation.
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        embeddings = F.relu(self.conv2(x, edge_index))
        out = self.classifier(embeddings)
        return out, embeddings


model = GraphSAGE(in_channels=data.x.shape[1], hidden_channels=64, out_channels=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# --- Step 4: Handle class imbalance in the loss function ---
# Only ~2.9% of nodes are fraud-associated. Without correction, the model
# could get low loss by just predicting "not fraud" for everyone. We fix
# this by weighting the loss so mistakes on fraud nodes count more.
num_fraud = (data.y[data.train_mask] == 1).sum().item()
num_legit = (data.y[data.train_mask] == 0).sum().item()
class_weights = torch.tensor([1.0, num_legit / num_fraud], dtype=torch.float)
print(f"Class weights (legit, fraud): {class_weights.tolist()}")

# --- Step 5: Training loop ---
print("\nTraining GraphSAGE...")
model.train()
for epoch in range(1, 101):
    optimizer.zero_grad()
    out, _ = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask], weight=class_weights)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f}")

# --- Step 6: Evaluate on the held-out test nodes ---
model.eval()
with torch.no_grad():
    out, embeddings = model(data.x, data.edge_index)
    probs = F.softmax(out, dim=1)[:, 1]  # predicted probability of being fraud-associated

test_probs = probs[data.test_mask].numpy()
test_labels = data.y[data.test_mask].numpy()

pr_auc = average_precision_score(test_labels, test_probs)
print(f"\nTest set Precision-Recall AUC: {pr_auc:.4f}")
print("(Random guessing on this imbalanced data would score close to "
      f"{test_labels.mean():.4f} -- our score should be well above that.)")

# --- Step 7: Save the trained model and embeddings for every node ---
# We save embeddings for ALL nodes (not just test) because Day 2's XGBoost
# model will need embeddings for every account it encounters, regardless of
# which "split" that account happened to fall into here.
torch.save(model.state_dict(), "data/graphsage_model.pt")

with open("data/node_embeddings.pkl", "wb") as f:
    pickle.dump({
        "embeddings": embeddings.numpy(),
        "node_to_idx": node_to_idx,
    }, f)

print("\nSaved trained model to data/graphsage_model.pt")
print("Saved node embeddings to data/node_embeddings.pkl")