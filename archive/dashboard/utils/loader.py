"""
loader.py
---------
Central place for loading data and the trained model.

Two Streamlit caching decorators are used:
  @st.cache_data     — caches DataFrames and computed stats.
                       Re-runs if the function arguments change.
  @st.cache_resource — caches heavy objects that should be shared
                       across all users and pages (the ML model).

Both run only ONCE per server session, so every page that calls
these functions gets the same pre-loaded object instantly.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split

# ROOT points to the archive/ folder regardless of which page imports this
ROOT = Path(__file__).parent.parent.parent

FEATURES = [
    "amount", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
    "balance_diff_orig", "balance_diff_dest",
    "is_empty_after_orig", "amount_exceeds_balance",
    "amount_ratio_orig", "orig_balance_mismatch",
    "dest_balance_mismatch", "type_encoded",
]

FEATURE_LABELS = {
    "amount":                "Transaction Amount",
    "oldbalanceOrg":         "Origin Balance (Before)",
    "newbalanceOrig":        "Origin Balance (After)",
    "oldbalanceDest":        "Destination Balance (Before)",
    "newbalanceDest":        "Destination Balance (After)",
    "balance_diff_orig":     "Balance Drained from Origin",
    "balance_diff_dest":     "Balance Added to Destination",
    "is_empty_after_orig":   "Account Drained to Zero",
    "amount_exceeds_balance":"Amount Exceeds Origin Balance",
    "amount_ratio_orig":     "Fraction of Origin Balance Transferred",
    "orig_balance_mismatch": "Origin Balance Mismatch",
    "dest_balance_mismatch": "Destination Balance Mismatch",
    "type_encoded":          "Transaction Type (1=Transfer, 0=Cash Out)",
}

COLOR_FRAUD = "#e74c3c"
COLOR_LEGIT = "#3b82f6"


@st.cache_data(show_spinner="Loading dataset...")
def load_data() -> pd.DataFrame:
    """
    Loads the full engineered dataset (2.77M rows).
    Cached after the first call — every subsequent call is instant.
    """
    path = ROOT / "transactions_engineered.csv"
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def get_stats() -> dict:
    """
    Pre-computes all aggregate statistics shown across the dashboard.
    Runs once, reused everywhere.
    """
    df = load_data()
    fraud = df[df["isFraud"] == 1]
    legit = df[df["isFraud"] == 0]

    fraud_by_type = (
        df[df["isFraud"] == 1]["type"].value_counts().to_dict()
    )

    return {
        "total":              len(df),
        "n_fraud":            len(fraud),
        "n_legit":            len(legit),
        "fraud_rate":         len(fraud) / len(df),
        "total_fraud_amount": fraud["amount"].sum(),
        "avg_fraud_amount":   fraud["amount"].mean(),
        "max_fraud_amount":   fraud["amount"].max(),
        "fraud_by_type":      fraud_by_type,
        "avg_legit_amount":   legit["amount"].mean(),
    }


@st.cache_data(show_spinner=False)
def load_sample(n: int = 80_000) -> pd.DataFrame:
    """
    Returns a stratified sample of the dataset for fast visualisation.
    Keeps ALL fraud cases so none are lost in the charts.
    n = total rows to return (fraud + sampled non-fraud).
    """
    df    = load_data()
    fraud = df[df["isFraud"] == 1]
    legit = df[df["isFraud"] == 0].sample(n - len(fraud), random_state=42)
    return pd.concat([fraud, legit]).sample(frac=1, random_state=42).reset_index(drop=True)


@st.cache_resource(show_spinner="Training fraud detection model...")
def get_model() -> ExtraTreesClassifier:
    """
    Trains the Extra Trees model on the same sampled dataset used in
    model_comparison.py (all fraud + 10x undersampled non-fraud).

    @st.cache_resource means the trained model object is shared across
    all users and pages — it is only trained once per server session.
    """
    df       = load_data()
    fraud_df = df[df["isFraud"] == 1]
    legit_df = df[df["isFraud"] == 0]

    n_legit      = min(len(fraud_df) * 10, len(legit_df))
    legit_sample = legit_df.sample(n_legit, random_state=42)
    df_model     = (
        pd.concat([fraud_df, legit_sample])
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

    X = df_model[FEATURES]
    y = df_model["isFraud"]
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )

    model = ExtraTreesClassifier(
        n_estimators=200, max_depth=15, class_weight="balanced",
        random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model
