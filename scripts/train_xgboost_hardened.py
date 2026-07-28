"""
train_xgboost_hardened.py

Purpose: Retrain tabular-only and hybrid XGBoost models on the HARDENED
feature set (which excludes the leaky oldbalanceOrg/newbalanceOrig columns),
to get a fair comparison of whether GraphSAGE embeddings add real value.

Results are merged into the existing data/results.json, alongside the
original (leaky) numbers -- both are worth keeping for the README, since the
leaky result is itself an interesting, explainable finding.
"""

import pickle
import json
from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score

# --- Step 1: Load the hardened feature sets ---
print("Loading hardened feature sets...")
with open("data/hardened_features.pkl", "rb") as f:
    saved = pickle.load(f)

X_tab_train, X_tab_test = saved["X_tab_train"], saved["X_tab_test"]
X_hyb_train, X_hyb_test = saved["X_hyb_train"], saved["X_hyb_test"]
y_train, y_test = saved["y_train"], saved["y_test"]

# --- Step 2: Handle class imbalance (same approach as before) ---
num_legit = (y_train == 0).sum()
num_fraud = (y_train == 1).sum()
scale_pos_weight = num_legit / num_fraud
print(f"scale_pos_weight: {scale_pos_weight:.2f}")


def train_and_evaluate(X_train, X_test, name):
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42,
    )
    model.fit(X_train, y_train)

    test_probs = model.predict_proba(X_test)[:, 1]
    pr_auc = average_precision_score(y_test, test_probs)
    print(f"{name} Precision-Recall AUC: {pr_auc:.4f}")
    return model, pr_auc


# --- Step 3: Train hardened tabular-only baseline ---
print("\nTraining hardened tabular-only XGBoost model...")
tabular_model, tabular_pr_auc = train_and_evaluate(X_tab_train, X_tab_test, "Hardened tabular-only")

# --- Step 4: Train hardened hybrid model ---
print("\nTraining hardened hybrid (tabular + GraphSAGE) XGBoost model...")
hybrid_model, hybrid_pr_auc = train_and_evaluate(X_hyb_train, X_hyb_test, "Hardened hybrid")

# --- Step 5: Compare ---
improvement = ((hybrid_pr_auc - tabular_pr_auc) / tabular_pr_auc) * 100
print("\n=== Hardened Results Summary ===")
print(f"Hardened tabular-only PR-AUC: {tabular_pr_auc:.4f}")
print(f"Hardened hybrid PR-AUC:       {hybrid_pr_auc:.4f}")
print(f"Improvement:                  {improvement:+.1f}%")

# --- Step 6: Save models ---
with open("data/xgboost_tabular_hardened_model.pkl", "wb") as f:
    pickle.dump(tabular_model, f)

with open("data/xgboost_hybrid_hardened_model.pkl", "wb") as f:
    pickle.dump(hybrid_model, f)

# --- Step 7: Merge into existing results.json (keep the original leaky numbers too) ---
with open("data/results.json", "r") as f:
    results = json.load(f)

results.update({
    "hardened_tabular_pr_auc": tabular_pr_auc,
    "hardened_hybrid_pr_auc": hybrid_pr_auc,
    "hardened_improvement_pct": improvement,
})

with open("data/results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nUpdated data/results.json with hardened results")