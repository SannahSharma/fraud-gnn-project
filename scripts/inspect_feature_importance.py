"""
inspect_feature_importance.py

Purpose: Check which features the tabular-only XGBoost model relied on most.
This helps us diagnose why it scored a suspiciously high PR-AUC (0.9962) --
specifically, whether it's leaning heavily on PaySim's known balance-field
artifact rather than learning genuine fraud behavior.
"""

import pickle

with open("data/xgboost_tabular_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("data/hybrid_features.pkl", "rb") as f:
    saved = pickle.load(f)

feature_names = list(saved["X_tab_train"].columns)
importances = model.feature_importances_

# Pair up feature names with their importance scores and sort, most important first.
ranked = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

print("Feature importances (tabular-only model), most important first:\n")
for name, score in ranked:
    print(f"{name:20s} {score:.4f}")