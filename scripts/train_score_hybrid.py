"""
train_score_hybrid.py

Purpose: Try a leaner, more interpretable hybrid feature -- instead of
concatenating GraphSAGE's raw 64-dim embeddings (which, in the previous
experiment, slightly HURT performance: 0.8770 vs 0.8832 tabular-only,
likely due to added dimensionality diluting the tree model's splits) --
use GraphSAGE's own predicted fraud-risk PROBABILITY per account. This is
a single, interpretable number per account (0 = looks nothing like known
fraud accounts, 1 = looks exactly like them), matching the project's
original framing of "combine the GNN's output with XGBoost."

Steps:
1. Reload the trained GraphSAGE model and compute a fraud-risk score for
   every account (re-using the exact same feature normalization as training).
2. Rebuild the hardened tabular features (same as prepare_hardened_features.py).
3. Add just 2 new columns: the sender's and receiver's risk scores.
4. Train and evaluate tabular-only (for a clean side-by-side) and this new
   lean hybrid model, using the SAME train/test split settings as before for
   a fair comparison.
5. Save the risk scores (reusable later for the dashboard) and update
   data/results.json.
"""

import pickle
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score

# ============================================================
# PART 1: Compute per-account fraud-risk scores from GraphSAGE
# ============================================================
print("Loading graph data...")
with open("data/graph_data.pkl", "rb") as f:
    saved = pickle.load(f)

graph_data = saved["data"]
node_to_idx = saved["node_to_idx"]

# Re-apply the EXACT same normalization used during training, or the model
# will see features on a different scale than it learned and produce garbage.
x = graph_data.x.clone()
x[:, :4] = torch.log1p(x[:, :4])
train_mean = x[graph_data.train_mask].mean(dim=0)
train_std = x[graph_data.train_mask].std(dim=0) + 1e-6
x = (x - train_mean) / train_std


class GraphSAGE(nn.Module):
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


model = GraphSAGE(in_channels=x.shape[1], hidden_channels=64, out_channels=2)
model.load_state_dict(torch.load("data/graphsage_model.pt"))
model.eval()

with torch.no_grad():
    out, _ = model(x, graph_data.edge_index)
    node_risk_scores = F.softmax(out, dim=1)[:, 1].numpy()

print(f"Computed risk scores for {len(node_risk_scores)} accounts "
      f"(mean={node_risk_scores.mean():.4f}, max={node_risk_scores.max():.4f})")

with open("data/node_scores.pkl", "wb") as f:
    pickle.dump({"node_scores": node_risk_scores, "node_to_idx": node_to_idx}, f)
print("Saved per-account risk scores to data/node_scores.pkl")

# ============================================================
# PART 2: Rebuild hardened tabular features + add risk scores
# ============================================================
print("\nLoading sampled transactions...")
df = pd.read_csv("data/paysim_sample.csv")

type_dummies = pd.get_dummies(df["type"], prefix="type")
tabular_features = pd.concat([
    df[["step", "amount", "oldbalanceDest", "newbalanceDest"]],
    type_dummies,
], axis=1).reset_index(drop=True)

orig_risk = np.array([node_risk_scores[node_to_idx[o]] for o in df["nameOrig"]])
dest_risk = np.array([node_risk_scores[node_to_idx[d]] for d in df["nameDest"]])

score_hybrid_features = tabular_features.copy()
score_hybrid_features["orig_risk_score"] = orig_risk
score_hybrid_features["dest_risk_score"] = dest_risk

y = df["isFraud"].reset_index(drop=True)

print(f"Tabular-only shape: {tabular_features.shape}")
print(f"Score-hybrid shape: {score_hybrid_features.shape}")

# Same split settings as before -- keeps this comparable to earlier hardened results.
(X_tab_train, X_tab_test,
 X_hyb_train, X_hyb_test,
 y_train, y_test) = train_test_split(
    tabular_features, score_hybrid_features, y,
    test_size=0.2, random_state=42, stratify=y
)

# ============================================================
# PART 3: Train and compare
# ============================================================
num_legit = (y_train == 0).sum()
num_fraud = (y_train == 1).sum()
scale_pos_weight = num_legit / num_fraud


def train_and_evaluate(X_train, X_test, name):
    clf = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42,
    )
    clf.fit(X_train, y_train)
    test_probs = clf.predict_proba(X_test)[:, 1]
    pr_auc = average_precision_score(y_test, test_probs)
    print(f"{name} Precision-Recall AUC: {pr_auc:.4f}")
    return clf, pr_auc

print("\nTraining tabular-only (for a clean side-by-side)...")
tabular_model, tabular_pr_auc = train_and_evaluate(X_tab_train, X_tab_test, "Tabular-only")

print("\nTraining score-based hybrid (tabular + 2 risk-score columns)...")
hybrid_model, hybrid_pr_auc = train_and_evaluate(X_hyb_train, X_hyb_test, "Score-hybrid")

improvement = ((hybrid_pr_auc - tabular_pr_auc) / tabular_pr_auc) * 100
print("\n=== Score-Hybrid Results Summary ===")
print(f"Tabular-only PR-AUC:  {tabular_pr_auc:.4f}")
print(f"Score-hybrid PR-AUC:  {hybrid_pr_auc:.4f}")
print(f"Improvement:          {improvement:+.1f}%")

with open("data/xgboost_score_hybrid_model.pkl", "wb") as f:
    pickle.dump(hybrid_model, f)

with open("data/results.json", "r") as f:
    results = json.load(f)

results.update({
    "score_hybrid_tabular_pr_auc": tabular_pr_auc,
    "score_hybrid_pr_auc": hybrid_pr_auc,
    "score_hybrid_improvement_pct": improvement,
})

with open("data/results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nUpdated data/results.json with score-hybrid results")