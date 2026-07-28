"""
detect_communities.py

Purpose: Run Louvain community detection over the transaction graph to find
CLUSTERS of connected accounts -- the "fraud ring" signal a flat,
per-transaction tabular model cannot represent, since it only ever scores
one row at a time.

We restrict this to the TRANSFER/CASH_OUT subgraph (the only transaction
types where fraud occurs in PaySim) to avoid merchant "hub" accounts merging
unrelated parts of the graph together.

For each community, we compute fraud density (% of accounts fraud-linked).
Communities well above the overall average are flagged as likely rings.
We report "ring-detection recall": what fraction of ALL fraud-linked
accounts fall inside a flagged community.
"""

import pickle
import json
import networkx as nx

# --- Step 1: Load the transaction graph ---
print("Loading transaction graph...")
with open("data/transaction_graph.pkl", "rb") as f:
    G = pickle.load(f)

# --- Step 2: Load node fraud labels ---
with open("data/graph_data.pkl", "rb") as f:
    saved = pickle.load(f)

graph_data = saved["data"]
node_to_idx = saved["node_to_idx"]
labels = graph_data.y.numpy()  # 1 = fraud-associated, 0 = not
total_fraud_nodes = int(labels.sum())
overall_fraud_rate = labels.mean()
print(f"Overall fraud-associated rate: {overall_fraud_rate * 100:.2f}%")

# --- Step 3: Build the fraud-relevant (TRANSFER/CASH_OUT only) subgraph ---
fraud_relevant_edges = [
    (u, v) for u, v, data in G.edges(data=True)
    if data["type"] in ("TRANSFER", "CASH_OUT")
]
G_sub = G.edge_subgraph(fraud_relevant_edges).copy()
print(f"Fraud-relevant subgraph: {G_sub.number_of_nodes()} nodes, "
      f"{G_sub.number_of_edges()} edges")

# Sanity check: confirm no fraud-linked account got excluded by this filter.
fraud_nodes_in_sub = sum(1 for n in G_sub.nodes() if labels[node_to_idx[n]] == 1)
print(f"Fraud-linked accounts covered by subgraph: {fraud_nodes_in_sub} / {total_fraud_nodes}")

# --- Step 4: Convert to undirected and run Louvain ---
# Louvain groups nodes to maximize "modularity" -- how much more densely
# connected nodes within a group are, versus what random chance would predict.
G_undirected = G_sub.to_undirected()
print("\nRunning Louvain community detection... (may take a minute)")
communities = nx.algorithms.community.louvain_communities(G_undirected, seed=42)
print(f"Found {len(communities)} communities")

# --- Step 5: Compute fraud density per community ---
community_stats = []
for community in communities:
    node_indices = [node_to_idx[node] for node in community]
    fraud_count = int(labels[node_indices].sum())
    size = len(community)
    community_stats.append({
        "nodes": community,
        "size": size,
        "fraud_count": fraud_count,
        "fraud_density": fraud_count / size,
    })

# Only consider communities of size >= 3 -- pairs/singletons are too small
# to call a "ring."
sizeable = [c for c in community_stats if c["size"] >= 3]
sizeable.sort(key=lambda c: c["fraud_density"], reverse=True)

print("\nTop 10 most suspicious communities (by fraud density):")
for i, c in enumerate(sizeable[:10]):
    print(f"  #{i+1}: size={c['size']:4d}, fraud accounts={c['fraud_count']:3d}, "
          f"fraud density={c['fraud_density']*100:.1f}%")

# --- Step 6: Flag communities as likely fraud rings ---
# Flag any community whose fraud density is at least 5x the overall average --
# a judgment call, documented here for transparency.
FLAG_THRESHOLD = overall_fraud_rate * 5
flagged = [c for c in community_stats if c["size"] >= 3 and c["fraud_density"] >= FLAG_THRESHOLD]

flagged_fraud_nodes = sum(c["fraud_count"] for c in flagged)
ring_detection_recall = flagged_fraud_nodes / total_fraud_nodes

print(f"\nFlag threshold (5x baseline): {FLAG_THRESHOLD*100:.2f}% fraud density")
print(f"Flagged communities: {len(flagged)}")
print(f"Fraud-linked accounts captured by flagged communities: "
      f"{flagged_fraud_nodes} / {total_fraud_nodes}")
print(f"Ring-detection recall: {ring_detection_recall*100:.2f}%")

# --- Step 7: Save results ---
with open("data/community_results.pkl", "wb") as f:
    pickle.dump({
        "communities": community_stats,
        "flagged": flagged,
        "ring_detection_recall": ring_detection_recall,
    }, f)

with open("data/results.json", "r") as f:
    results = json.load(f)

results["ring_detection_recall"] = ring_detection_recall
results["num_communities"] = len(communities)
results["num_flagged_communities"] = len(flagged)

with open("data/results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved community results to data/community_results.pkl")
print("Updated data/results.json with ring-detection recall")