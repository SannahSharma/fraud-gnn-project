"""
prepare_hybrid_features.py

Purpose: Build the feature table for a HYBRID fraud model, combining:
  1. Tabular features -- facts about this specific transaction (amount, type,
     account balances before/after).
  2. Graph features -- the GraphSAGE embeddings for the sender and receiver
     accounts, learned from the transaction graph on Day 1.

This is the technical core of the project: proving that adding graph
awareness to a normal tabular model catches things a tabular-only model
misses. To prove that, we build BOTH a tabular-only feature set and a
hybrid (tabular + graph) feature set, so we can train two XGBoost models
and directly compare them later.
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# --- Step 1: Load the sampled transaction table ---
print("Loading sampled transactions...")
df = pd.read_csv("data/paysim_sample.csv")
print(f"Loaded {len(df)} transactions")

# --- Step 2: Load the node embeddings from GraphSAGE training ---
print("Loading node embeddings...")
with open("data/node_embeddings.pkl", "rb") as f:
    saved = pickle.load(f)
embeddings = saved["embeddings"]       # shape: [num_nodes, embedding_dim]
node_to_idx = saved["node_to_idx"]
embedding_dim = embeddings.shape[1]
print(f"Embeddings shape: {embeddings.shape}")

# --- Step 3: Build tabular features ---
# One-hot encode the transaction "type" (PAYMENT, TRANSFER, CASH_OUT, etc.)
# since XGBoost needs numeric input, not text categories.
type_dummies = pd.get_dummies(df["type"], prefix="type")

tabular_features = pd.concat([
    df[["step", "amount", "oldbalanceOrg", "newbalanceOrig",
        "oldbalanceDest", "newbalanceDest"]],
    type_dummies,
], axis=1)

print(f"Tabular feature columns: {list(tabular_features.columns)}")

# --- Step 4: Look up graph embeddings for each transaction's sender/receiver ---
# Every account in this sample was a node in our graph, so every lookup
# should succeed. We concatenate the sender's embedding and receiver's
# embedding side by side -- this lets XGBoost see "does the sender's
# neighborhood look suspicious" AND "does the receiver's neighborhood
# look suspicious" as separate signals.
orig_embeddings = np.zeros((len(df), embedding_dim), dtype=np.float32)
dest_embeddings = np.zeros((len(df), embedding_dim), dtype=np.float32)

for i, (orig, dest) in enumerate(zip(df["nameOrig"], df["nameDest"])):
    orig_embeddings[i] = embeddings[node_to_idx[orig]]
    dest_embeddings[i] = embeddings[node_to_idx[dest]]

orig_cols = [f"orig_emb_{i}" for i in range(embedding_dim)]
dest_cols = [f"dest_emb_{i}" for i in range(embedding_dim)]

graph_features = pd.concat([
    pd.DataFrame(orig_embeddings, columns=orig_cols),
    pd.DataFrame(dest_embeddings, columns=dest_cols),
], axis=1)

# --- Step 5: Assemble tabular-only and hybrid feature sets ---
X_tabular = tabular_features.reset_index(drop=True)
X_hybrid = pd.concat([tabular_features.reset_index(drop=True),
                       graph_features.reset_index(drop=True)], axis=1)
y = df["isFraud"].reset_index(drop=True)

print(f"\nTabular-only feature shape: {X_tabular.shape}")
print(f"Hybrid feature shape: {X_hybrid.shape}")

# --- Step 6: Train/test split ---
# stratify=y keeps the same fraud percentage in both train and test sets --
# important since fraud is so rare; a plain random split could accidentally
# put very few (or zero) fraud cases into the test set.
(X_tab_train, X_tab_test,
 X_hyb_train, X_hyb_test,
 y_train, y_test) = train_test_split(
    X_tabular, X_hybrid, y,
    test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set: {len(y_train)} rows ({y_train.sum()} fraud)")
print(f"Test set: {len(y_test)} rows ({y_test.sum()} fraud)")

# --- Step 7: Save everything for the training script ---
with open("data/hybrid_features.pkl", "wb") as f:
    pickle.dump({
        "X_tab_train": X_tab_train, "X_tab_test": X_tab_test,
        "X_hyb_train": X_hyb_train, "X_hyb_test": X_hyb_test,
        "y_train": y_train, "y_test": y_test,
    }, f)

print("\nSaved feature sets to data/hybrid_features.pkl")