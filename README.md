# Graph-Based Fraud Detection & Transaction Intelligence System

**[Live demo](https://sannah-fraud-gnn.streamlit.app/)** · Python · PyTorch Geometric · GraphSAGE · XGBoost · NetworkX · FastAPI · Streamlit

A solo-built fraud detection system that represents transactions as a graph
(accounts as nodes, both customers and merchants; transactions as edges), trains a
GraphSAGE GNN to learn what a normal account neighborhood looks like versus a
suspicious one, combines that with an XGBoost tabular model into a hybrid
fraud score, and uses community detection to surface clusters of connected
suspicious accounts -- catching ring-based fraud that a flat, per-transaction
model cannot see.

Built end-to-end in 3 days as a scoped-down prototype: NetworkX + PyTorch
Geometric instead of a graph database, batch processing instead of Kafka
streaming, a single FastAPI endpoint instead of a full production API, and a
Streamlit dashboard instead of a custom investigator console.

## Dashboard

The [live dashboard](https://YOUR-APP-NAME.streamlit.app) has five tabs:

| Tab | What it shows |
|---|---|
| 🔎 Investigate | Search any account to see its 2-hop transaction neighborhood, colored by GraphSAGE risk score, next to the most suspicious fraud rings |
| 📊 Model results | PR-AUC for all six models, and feature importances showing the PaySim balance leak |
| 💸 Fraud patterns | Fraud rate by transaction type, fraud vs legitimate amounts, fraud over time, and GraphSAGE score separation |
| 🕸️ Ring detection | Every candidate Louvain community by size and fraud density, with the flagging threshold |
| 💡 Key findings | The three main takeaways and the project's limitations |

The deployed app reads precomputed results from `dashboard_data/` (about 17 MB),
so it runs on Streamlit's free tier without PyTorch. Scoring new transactions
runs through the FastAPI endpoint.

## Dataset

[PaySim](https://www.kaggle.com/datasets/ntnu-testimon/paysim1) -- synthetic
mobile money transactions (6,362,620 rows), 8,213 of which (0.1291%) are
labeled fraud. To keep the graph tractable on a laptop, the working dataset
is a sample of every fraud transaction plus 300,000 randomly sampled
legitimate transactions (308,213 rows total, 2.66% fraud).

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 1 | `scripts/inspect_data.py` | Loads and sanity-checks the raw dataset |
| 2 | `scripts/build_graph.py` | Samples the data and builds the transaction graph (NetworkX) |
| 3 | `scripts/prepare_graph_data.py` | Converts the graph into PyTorch Geometric format with per-account features/labels |
| 4 | `scripts/train_graphsage.py` | Trains a 2-layer GraphSAGE model to classify accounts as fraud-associated |
| 5 | `scripts/prepare_hybrid_features.py` / `prepare_hardened_features.py` | Builds tabular + graph-embedding feature sets for XGBoost |
| 6 | `scripts/train_xgboost.py` / `train_xgboost_hardened.py` / `train_score_hybrid.py` | Trains and compares tabular-only vs. hybrid XGBoost models |
| 7 | `scripts/detect_communities.py` | Runs Louvain community detection to find fraud rings |
| 8 | `scripts/export_dashboard_data.py` | Exports slim, deploy-ready results to `dashboard_data/` |
| 9 | `api.py` | FastAPI endpoint that scores a single transaction |
| 10 | `dashboard.py` | Streamlit dashboard (see above) |

## Results

| Model | Precision-Recall AUC | Notes |
|---|---|---|
| GraphSAGE alone (node classification) | 0.2521 | ~8.5x better than random guessing (baseline 0.0294) |
| XGBoost, tabular-only (naive, all balance columns) | 0.9962 | **Inflated** -- see "PaySim balance leak" below |
| XGBoost, hybrid + raw embeddings (naive tabular) | 0.9962 | No improvement -- tabular already at ceiling |
| XGBoost, tabular-only (hardened, leak removed) | 0.8832 | Fair baseline |
| XGBoost, hybrid + raw 64-dim embeddings (hardened) | 0.8770 | -0.7% vs. hardened tabular |
| XGBoost, hybrid + GraphSAGE risk-score columns (hardened) | 0.8817 | -0.2% vs. hardened tabular |
| **Ring-detection recall** (community detection) | **17.13%** | 2,806 of 16,382 fraud-linked accounts captured by 1,381 flagged high-density communities (out of 111,629 total) |

## Key findings (the honest part)

**PaySim balance leak.** The first tabular model scored a suspiciously
perfect 0.9962 PR-AUC. Feature importance analysis showed `newbalanceOrig`
and `oldbalanceOrg` accounted for 68% of the model's decisions -- a known
PaySim simulator artifact where a fraudulent transaction's origin balance is
frequently driven to exactly 0 as a side effect of how the simulator
generates fraud, not realistic fraudster behavior. Removing those two
columns ("hardened" features) dropped tabular-only to a more credible 0.8832
and made the tabular-vs-hybrid comparison meaningful.

**The hybrid classifier didn't beat tabular on PR-AUC, twice.** Both raw
GraphSAGE embeddings (-0.7%) and a compressed per-account risk score (-0.2%)
slightly underperformed tabular alone. Two likely reasons: PaySim's hardened
tabular features (amount, type, destination balance) are already strong
predictors on their own, leaving little headroom; and PaySim fraud is
typically a short 2-step chain (TRANSFER into a one-time mule account,
immediate CASH_OUT) rather than a multi-account ring, so an account-level
"was this account ever involved in fraud" label is a fairly blunt signal at
the individual-transaction level.

**Where the graph actually earns its keep: ring detection.** Community
detection via Louvain clustering surfaced 1,381 communities with fraud
density at least 5x the graph-wide average, capturing 17.13% of all
fraud-linked accounts -- something a flat tabular model has no mechanism to
do at all, since it only ever scores one transaction at a time and can never
represent "these accounts form a connected cluster." This is the project's
real differentiator, not the transaction-level PR-AUC.

## Project structure

```
fraud-gnn-project/
├── .streamlit/config.toml     dashboard theme
├── data/                      gitignored -- regenerate via the scripts below
├── dashboard_data/            slim exported results used by the deployed dashboard
├── scripts/                   the full pipeline, in run order (see table above)
├── api.py                     FastAPI scoring endpoint
├── dashboard.py               Streamlit dashboard
├── requirements.txt           dashboard-only dependencies (for Streamlit Cloud)
└── README.md
```

## How to run it

**Just the dashboard** (uses the committed `dashboard_data/`):

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1        # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
streamlit run dashboard.py
```

**The full pipeline:**

1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   pip install pandas networkx torch torch_geometric scikit-learn xgboost fastapi uvicorn streamlit plotly pyarrow
   ```
2. Download [PaySim](https://www.kaggle.com/datasets/ntnu-testimon/paysim1) from Kaggle and place the CSV at `data/paysim.csv`.
3. Run the pipeline scripts in the order listed in the table above, ending with `python scripts/export_dashboard_data.py`.
4. Start the API: `uvicorn api:app --reload` -- test at `http://127.0.0.1:8000/docs`.
5. Start the dashboard: `streamlit run dashboard.py`.

## Limitations & what a production version would add

This is a 3-day scoped prototype, not a production system. A real version
would need: a proper graph database (Neo4j/TigerGraph) instead of an
in-memory NetworkX snapshot, streaming ingestion (Kafka) instead of a static
sample, retraining on the full 6.3M-row dataset instead of a 300k sample,
adversarial-evasion testing (fraud rings adapt once they know they're being
detected), and a real investigator console instead of a single-page
dashboard. The tabular-vs-hybrid PR-AUC result is also specific to PaySim's
fraud mechanic (short mule chains) -- a dataset with genuine multi-hop rings
(shared devices/IPs across many accounts) would likely show a larger hybrid
lift.
