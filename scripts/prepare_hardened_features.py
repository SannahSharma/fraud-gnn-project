"""
prepare_hardened_features.py

Purpose: Build a HARDER, more realistic tabular feature set for a fairer
tabular-vs-hybrid comparison.

Why: feature importance analysis showed the original tabular model relied
overwhelmingly (~68% of its decision weight) on newbalanceOrig and
oldbalanceOrg. This is a known artifact of the PaySim simulator: when a
transaction is fraudulent, the origin account's balance is often driven to
exactly 0 as a side effect of how fraud is simulated -- not because real
fraudsters reliably drain accounts to exactly zero. A model exploiting this
quirk gets a suspiciously perfect score without learning anything that would
generalize to real fraud detection.

To get a meaningful comparison, we drop the two most artifact-prone columns
(oldbalanceOrg, newbalanceOrig) and rebuild both a "hardened" tabular-only
feature set and a "hardened" hybrid (tabular + graph) feature set.
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# --- Step 1: Load the sampled transaction table ---
print("Loading sampled transactions...")
df = pd.read_csv("data/paysim_sample.csv")

# --- Step 2: Load the node embeddings from GraphSAGE training ---
print("Loading node embeddings...")
with open("data/node_embeddings.pkl", "rb") as f:
    saved = pickle.load(f)
embeddings = saved["embeddings"]
node_to_idx = saved["node_to_idx"]
embedding_dim = embeddings.shape[1]

# --- Step 3: Build HARDENED tabular features ---
# Dropping oldbalanceOrg and newbalanceOrig -- the two columns that were
# dominating the original model's decisions due to a simulator artifact.
# We keep destination balances since their importance was already low,
# suggesting they aren't driving the same shortcut.
type_dummies = pd.get_dummies(df["type"], prefix="type")

tabular_features = pd.concat([
    df[["step", "amount", "oldbalanceDest", "newbalanceDest"]],
    type_dummies,
], axis=1)

print(f"Hardened tabular feature columns: {list(tabular_features.columns)}")

# --- Step 4: Look up graph embeddings for each transaction's sender/receiver ---
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

# --- Step 5: Assemble hardened tabular-only and hardened hybrid feature sets ---
X_tabular = tabular_features.reset_index(drop=True)
X_hybrid = pd.concat([tabular_features.reset_index(drop=True),
                       graph_features.reset_index(drop=True)], axis=1)
y = df["isFraud"].reset_index(drop=True)

print(f"\nHardened tabular-only feature shape: {X_tabular.shape}")
print(f"Hardened hybrid feature shape: {X_hybrid.shape}")

# --- Step 6: Train/test split (same seed as before, for consistency) ---
(X_tab_train, X_tab_test,
 X_hyb_train, X_hyb_test,
 y_train, y_test) = train_test_split(
    X_tabular, X_hybrid, y,
    test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set: {len(y_train)} rows ({y_train.sum()} fraud)")
print(f"Test set: {len(y_test)} rows ({y_test.sum()} fraud)")

# --- Step 7: Save ---
with open("data/hardened_features.pkl", "wb") as f:
    pickle.dump({
        "X_tab_train": X_tab_train, "X_tab_test": X_tab_test,
        "X_hyb_train": X_hyb_train, "X_hyb_test": X_hyb_test,
        "y_train": y_train, "y_test": y_test,
    }, f)

print("\nSaved hardened feature sets to data/hardened_features.pkl")