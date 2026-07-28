"""
api.py

Purpose: A single FastAPI endpoint that scores one transaction for fraud risk,
using the hybrid model: hardened tabular features + each account's
GraphSAGE-derived risk score.

Run with: uvicorn api:app --reload
Then open: http://127.0.0.1:8000/docs (FastAPI's auto-generated test page)
"""

import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# --- Step 1: Load everything ONCE, when the server starts ---
# Loading the model/data on every request would be far too slow -- we load
# once here, and every request afterward reuses what's already in memory.
print("Loading model and account risk scores...")

with open("data/xgboost_score_hybrid_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("data/node_scores.pkl", "rb") as f:
    node_score_data = pickle.load(f)

node_scores = node_score_data["node_scores"]
node_to_idx = node_score_data["node_to_idx"]
mean_risk_score = float(node_scores.mean())  # fallback for accounts we've never seen

# The exact column order the model was trained on -- must match exactly.
FEATURE_COLUMNS = [
    "step", "amount", "oldbalanceDest", "newbalanceDest",
    "type_CASH_IN", "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER",
    "orig_risk_score", "dest_risk_score",
]

print("Ready.")

# --- Step 2: Create the FastAPI app ---
app = FastAPI(title="Fraud Detection API")


# --- Step 3: Define what a valid request looks like ---
# Pydantic validates incoming JSON automatically -- if a field is missing or
# the wrong type, FastAPI rejects it with a clear error before our code runs.
class Transaction(BaseModel):
    step: int
    type: str            # one of: CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER
    amount: float
    nameOrig: str
    oldbalanceOrg: float
    newbalanceOrig: float
    nameDest: str
    oldbalanceDest: float
    newbalanceDest: float


def get_risk_score(account_id: str) -> float:
    """Look up an account's GraphSAGE risk score. Falls back to the
    graph-wide average if the account was never seen during training."""
    idx = node_to_idx.get(account_id)
    if idx is None:
        return mean_risk_score
    return float(node_scores[idx])


# --- Step 4: The scoring endpoint ---
@app.post("/score")
def score_transaction(txn: Transaction):
    """Score a single transaction and return its fraud probability."""

    orig_risk = get_risk_score(txn.nameOrig)
    dest_risk = get_risk_score(txn.nameDest)

    # Build a one-row feature vector matching the model's training format exactly.
    row = {col: 0.0 for col in FEATURE_COLUMNS}
    row["step"] = txn.step
    row["amount"] = txn.amount
    row["oldbalanceDest"] = txn.oldbalanceDest
    row["newbalanceDest"] = txn.newbalanceDest
    row[f"type_{txn.type}"] = 1.0
    row["orig_risk_score"] = orig_risk
    row["dest_risk_score"] = dest_risk

    X = pd.DataFrame([row])[FEATURE_COLUMNS]
    fraud_probability = float(model.predict_proba(X)[0, 1])

    return {
        "nameOrig": txn.nameOrig,
        "nameDest": txn.nameDest,
        "fraud_probability": round(fraud_probability, 4),
        "is_flagged": fraud_probability >= 0.5,
        "orig_risk_score": round(orig_risk, 4),
        "dest_risk_score": round(dest_risk, 4),
    }


@app.get("/")
def root():
    return {"message": "Fraud Detection API is running. POST to /score to score a transaction."}