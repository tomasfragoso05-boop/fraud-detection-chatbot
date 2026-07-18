import pandas as pd
import numpy as np

# Load the cleaned dataset
df = pd.read_csv("transactions_limpo.csv")

print("Original shape:", df.shape)
print("Columns:", df.columns.tolist())

# Feature Engineering

# 1. Balance differences
df["balance_diff_orig"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
df["balance_diff_dest"] = df["newbalanceDest"] - df["oldbalanceDest"]

# 2. Check if origin account is emptied after transaction
df["is_empty_after_orig"] = (df["newbalanceOrig"] == 0).astype(int)

# 3. Check if amount exceeds origin balance (potential issue)
df["amount_exceeds_balance"] = (df["amount"] > df["oldbalanceOrg"]).astype(int)

# 4. Ratio of amount to origin balance (avoid division by zero)
df["amount_ratio_orig"] = df["amount"] / (df["oldbalanceOrg"] + 1)

# 5. Balance mismatches (should be close to amount in legitimate transactions)
df["orig_balance_mismatch"] = abs(df["balance_diff_orig"] - df["amount"])
df["dest_balance_mismatch"] = abs(df["balance_diff_dest"] - df["amount"])

# 6. Encode transaction type (TRANSFER=0, CASH_OUT=1)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["type_encoded"] = le.fit_transform(df["type"])

print("New shape after feature engineering:", df.shape)
print("New columns:", df.columns.tolist())

# Save the engineered dataset
df.to_csv("transactions_engineered.csv", index=False)
print("Engineered dataset saved as 'transactions_engineered.csv'")


