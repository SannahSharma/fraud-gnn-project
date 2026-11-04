"""
export_dashboard_data.py

Purpose: Export slim, deploy-friendly copies of the pipeline outputs so the
Streamlit dashboard can run in the cloud WITHOUT PyTorch / PyTorch Geometric
and without the large pickles in data/ (which are gitignored).

Run once on your laptop, after the full pipeline, from the project root:
    python scripts/export_dashboard_data.py

Writes to dashboard_data/ (small enough to commit to GitHub):
    nodes.parquet               per account: id, GraphSAGE risk score, fraud label
    edges.parquet               per transaction link: accounts, amount, type, step, fraud flag
    rings.parquet               the most suspicious Louvain communities (triage table)
    communities.parquet         every candidate ring's size and fraud density
    feature_importance.parquet  XGBoost importances before/after removing the leak
    stats.json                  headline KPIs (graph size + model results)
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data")
OUT = Path("dashboard_data")
OUT.mkdir(exist_ok=True)
TOP_RINGS = 50

print("Loading pipeline outputs from data/ ...")
with open(DATA / "transaction_graph.pkl", "rb") as f:
    G = pickle.load(f)
with open(DATA / "graph_data.pkl", "rb") as f:
    graph_data = pickle.load(f)          # needs torch_geometric installed locally
with open(DATA / "node_scores.pkl", "rb") as f:
    node_score_data = pickle.load(f)
with open(DATA / "community_results.pkl", "rb") as f:
    community_data = pickle.load(f)
with open(DATA / "results.json", "r") as f:
    results = json.load(f)

node_to_idx = graph_data["node_to_idx"]
labels = graph_data["data"].y.numpy()
node_scores = np.asarray(node_score_data["node_scores"], dtype=float)

# --- Nodes: plain pandas table, no torch objects ---
node_ids = list(node_to_idx.keys())
idx = np.array([node_to_idx[n] for n in node_ids])
nodes = pd.DataFrame({
    "node": [str(n) for n in node_ids],
    "score": node_scores[idx].astype("float32"),
    "label": labels[idx].astype("int8"),
})
nodes.to_parquet(OUT / "nodes.parquet", index=False)
print(f"  nodes.parquet: {len(nodes):,} accounts")

# --- Edges: one row per transaction link, with its attributes ---
edges = pd.DataFrame(
    ((str(u), str(v), d.get("amount", np.nan), d.get("type", "UNKNOWN"),
      d.get("step", -1), d.get("isFraud", 0)) for u, v, d in G.edges(data=True)),
    columns=["src", "dst", "amount", "type", "step", "is_fraud"],
)
edges = edges.astype({"amount": "float32", "type": "category", "step": "int32", "is_fraud": "int8"})
edges.to_parquet(OUT / "edges.parquet", index=False)
print(f"  edges.parquet: {len(edges):,} links")

# --- Community summary: every candidate ring (size >= 3), for the ring scatter ---
overall_fraud_rate = float(labels.mean())
flag_threshold = overall_fraud_rate * 5          # same rule as detect_communities.py
comm = pd.DataFrame(
    [(c["size"], c["fraud_count"], c["fraud_density"])
     for c in community_data["communities"] if c["size"] >= 3],
    columns=["size", "fraud_count", "fraud_density"],
)
comm["flagged"] = comm["fraud_density"] >= flag_threshold
comm.to_parquet(OUT / "communities.parquet", index=False)
print(f"  communities.parquet: {len(comm):,} candidate rings")

# --- Feature importances: before vs after removing the PaySim balance leak ---
fi_rows = []
for label_, model_file, feat_file in [
    ("Naive (with balance leak)", "xgboost_tabular_model.pkl", "hybrid_features.pkl"),
    ("Hardened (leak removed)", "xgboost_tabular_hardened_model.pkl", "hardened_features.pkl"),
]:
    try:
        with open(DATA / model_file, "rb") as f:
            model = pickle.load(f)
        with open(DATA / feat_file, "rb") as f:
            names = list(pickle.load(f)["X_tab_train"].columns)
        for name, imp in zip(names, model.feature_importances_):
            fi_rows.append({"model": label_, "feature": name, "importance": float(imp)})
    except Exception as e:  # noqa: BLE001
        print(f"  (skipped feature importances for {label_}: {type(e).__name__})")
if fi_rows:
    pd.DataFrame(fi_rows).to_parquet(OUT / "feature_importance.parquet", index=False)
    print("  feature_importance.parquet")

# --- Rings: top communities by fraud density (size >= 3) ---
sizeable = [c for c in community_data["communities"] if c["size"] >= 3]
sizeable.sort(key=lambda c: c["fraud_density"], reverse=True)
score_of = dict(zip(nodes["node"], nodes["score"]))
rows = []
for rank, c in enumerate(sizeable[:TOP_RINGS], start=1):
    members = [str(n) for n in c["nodes"]]
    # Show the ring's highest-risk account, so clicking through is informative.
    lead = max(members, key=lambda n: score_of.get(n, 0.0))
    rows.append({
        "ring": rank,
        "size": int(c["size"]),
        "fraud_count": int(c["fraud_count"]),
        "fraud_density": float(c["fraud_density"]),
        "lead_account": lead,
    })
pd.DataFrame(rows).to_parquet(OUT / "rings.parquet", index=False)
print(f"  rings.parquet: top {len(rows)} rings")

# --- Stats: KPIs shown in the header ---
stats = dict(results)
stats["num_nodes"] = G.number_of_nodes()
stats["num_edges"] = G.number_of_edges()
stats["overall_fraud_rate"] = overall_fraud_rate
stats["flag_threshold"] = flag_threshold
stats["num_fraud_accounts"] = int(labels.sum())
(OUT / "stats.json").write_text(json.dumps(stats, indent=2))
print("  stats.json")

total_mb = sum(p.stat().st_size for p in OUT.iterdir()) / 1e6
print(f"\nDone. dashboard_data/ is {total_mb:.1f} MB "
      f"({'fine for GitHub' if total_mb < 90 else 'TOO BIG for GitHub (limit 100 MB per file)'}).")
