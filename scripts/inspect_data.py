"""
inspect_data.py

Purpose: Load the raw PaySim CSV and print basic facts about it —
row count, column names, and how rare fraud actually is.

This is a one-time sanity check before we build anything else.
"""

import pandas as pd

# Load the CSV into a pandas DataFrame (an in-memory table).
# This may take 10-30 seconds since the file has ~6.3 million rows.
df = pd.read_csv("data/paysim.csv")

# How many rows (transactions) and columns do we have?
print("Shape (rows, columns):", df.shape)

# What are the column names?
print("\nColumns:", list(df.columns))

# Show the first 5 rows so we can visually confirm the data looks right.
print("\nFirst 5 rows:")
print(df.head())

# Count how many transactions are fraud (isFraud == 1) vs not (0).
fraud_counts = df["isFraud"].value_counts()
print("\nFraud counts:")
print(fraud_counts)

# Fraud as a percentage of all transactions — this is the "class imbalance"
# we'll need to handle carefully when evaluating our model later.
fraud_percentage = (fraud_counts[1] / len(df)) * 100
print(f"\nFraud is {fraud_percentage:.4f}% of all transactions.")