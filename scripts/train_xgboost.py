"""
train_xgboost.py

Purpose: Train two XGBoost fraud classifiers on the feature sets built in
prepare_hybrid_features.py:
  1. Tabular-only baseline -- what a standard fraud model looks like today.
  2. Hybrid model -- tabular features + GraphSAGE embeddings.

We evaluate both with Precision-Recall AUC and compare them directly. This
comparison is the main proof point of the whole project: does adding graph
awareness improve fraud detection over a strong tabular baseline?
"""

import pickle
import json
from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score

# --- Step 1: Load the feature sets we built earlier ---
print("Loading hybrid feature sets...")
with open("data/hybrid_features.pkl", "rb") as f:
    saved = pickle.load(f)

X_tab_train, X_tab_test = saved["X_tab_train"], saved["X_tab_test"]
X_hyb_train, X_hyb_test = saved["X_hyb_train"], saved["X_hyb_test"]
y_train, y_test = saved["y_train"], saved["y_test"]

# --- Step 2: Handle class imbalance ---
# scale_pos_weight tells XGBoost how much more to weight mistakes on the
# rare fraud class -- similar to the class_weights we used for GraphSAGE.
num_legit = (y_train == 0).sum()
num_fraud = (y_train == 1).sum()
scale_pos_weight = num_legit / num_fraud
print(f"scale_pos_weight: {scale_pos_weight:.2f}")


def train_and_evaluate(X_train, X_test, name):
    """Train an XGBoost classifier and return the model plus its test-set PR-AUC."""
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


# --- Step 3: Train the tabular-only baseline ---
print("\nTraining tabular-only XGBoost model...")
tabular_model, tabular_pr_auc = train_and_evaluate(X_tab_train, X_tab_test, "Tabular-only")

# --- Step 4: Train the hybrid model (tabular + graph embeddings) ---
print("\nTraining hybrid (tabular + GraphSAGE) XGBoost model...")
hybrid_model, hybrid_pr_auc = train_and_evaluate(X_hyb_train, X_hyb_test, "Hybrid")

# --- Step 5: Compare results ---
improvement = ((hybrid_pr_auc - tabular_pr_auc) / tabular_pr_auc) * 100
print("\n=== Results Summary ===")
print(f"Tabular-only PR-AUC: {tabular_pr_auc:.4f}")
print(f"Hybrid PR-AUC:       {hybrid_pr_auc:.4f}")
print(f"Improvement:         {improvement:+.1f}%")

# --- Step 6: Save models and results for the dashboard/README later ---
with open("data/xgboost_tabular_model.pkl", "wb") as f:
    pickle.dump(tabular_model, f)

with open("data/xgboost_hybrid_model.pkl", "wb") as f:
    pickle.dump(hybrid_model, f)

results = {
    "graphsage_pr_auc": 0.2521,   # from Day 1
    "tabular_pr_auc": tabular_pr_auc,
    "hybrid_pr_auc": hybrid_pr_auc,
    "improvement_pct": improvement,
}
with open("data/results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved both models and data/results.json")