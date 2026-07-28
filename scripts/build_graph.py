"""
build_graph.py

Purpose:
1. Load the full PaySim dataset.
2. Keep ALL fraud transactions + a random sample of legitimate transactions,
   so the graph stays small enough to build and train on quickly, without
   exhausting RAM.
3. Build a directed graph: accounts are nodes, transactions are edges.
4. Save both the sampled transaction table and the graph to disk, so later
   steps (GraphSAGE training, XGBoost) can reuse them without redoing this work.
"""

import pandas as pd
import networkx as nx
import pickle

# --- Step 1: Load the full dataset ---
print("Loading full dataset...")
df = pd.read_csv("data/paysim.csv")
print(f"Full dataset: {len(df)} transactions")

# --- Step 2: Split into fraud and non-fraud ---
fraud_df = df[df["isFraud"] == 1]
legit_df = df[df["isFraud"] == 0]

print(f"Fraud transactions: {len(fraud_df)}")
print(f"Legit transactions: {len(legit_df)}")

# --- Step 3: Sample legitimate transactions ---
# We keep ALL fraud (it's rare and valuable -- we never want to throw any away),
# but only a random sample of legitimate transactions. Without this, the graph
# would have millions of nodes/edges and be too slow/heavy for a laptop.
SAMPLE_SIZE = 300_000
legit_sample = legit_df.sample(n=SAMPLE_SIZE, random_state=42)
# random_state=42 makes the "random" sample reproducible -- running this
# script again will pick the exact same rows, which matters for consistent results.

# --- Step 4: Combine fraud + sample, then shuffle the rows ---
sampled_df = pd.concat([fraud_df, legit_sample]).sample(frac=1, random_state=42).reset_index(drop=True)
print(f"\nSampled dataset: {len(sampled_df)} transactions")
print(f"Fraud percentage in sample: {(sampled_df['isFraud'].sum() / len(sampled_df)) * 100:.2f}%")

# Save this sampled table -- we'll reuse it for the XGBoost model on Day 2,
# so we don't have to reload and re-sample the full 6M-row file every time.
sampled_df.to_csv("data/paysim_sample.csv", index=False)
print("Saved sampled data to data/paysim_sample.csv")

# --- Step 5: Build the graph ---
# DIRECTED graph because money flows FROM one account TO another -- direction
# matters (e.g. spotting money moving in a circle through several accounts).
# Nodes = accounts (both senders "nameOrig" and receivers "nameDest").
# Edges = individual transactions, carrying attributes like amount and fraud label.
print("\nBuilding graph... (this may take 1-2 minutes)")
G = nx.DiGraph()

for _, row in sampled_df.iterrows():
    G.add_edge(
        row["nameOrig"],
        row["nameDest"],
        amount=row["amount"],
        step=row["step"],
        type=row["type"],
        isFraud=row["isFraud"],
    )

print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# How many distinct accounts touched at least one fraudulent transaction?
# This gives us a sense of how many "suspicious neighborhoods" exist.
fraud_nodes = set()
for u, v, data in G.edges(data=True):
    if data["isFraud"] == 1:
        fraud_nodes.add(u)
        fraud_nodes.add(v)

print(f"Nodes involved in at least one fraud transaction: {len(fraud_nodes)}")

# --- Step 6: Save the graph to disk ---
# Pickling saves the entire Python graph object to a file so we can reload it
# instantly later, instead of rebuilding it from scratch every time.
with open("data/transaction_graph.pkl", "wb") as f:
    pickle.dump(G, f)

print("\nSaved graph to data/transaction_graph.pkl")