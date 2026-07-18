"""
3_Transaction_Scorer.py — Live Transaction Scorer
==================================================
This page lets you input the details of any transaction and the
trained Extra Trees model will predict whether it is fraud or not.

How it works:
  1. You fill in the raw transaction fields (amount, balances, type).
  2. The app automatically computes the engineered features
     (balance_diff_orig, is_empty_after_orig, etc.) — same logic as
     feature_engineering.py — so the model sees the correct inputs.
  3. The model outputs a fraud probability (0–100%).
  4. The app highlights which features are "suspicious" (match
     patterns seen in confirmed fraud cases).
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.loader import get_model, FEATURES, FEATURE_LABELS, COLOR_FRAUD, COLOR_LEGIT

st.set_page_config(page_title="Transaction Scorer", layout="wide")

st.title("Transaction Scorer")
st.caption("Enter a transaction's details and get an instant fraud risk assessment.")
st.divider()

model = get_model()

# ── Input form ────────────────────────────────────────────────────────────────
st.subheader("Transaction Details")
st.caption(
    "Fill in the raw transaction fields below. "
    "The engineered features are computed automatically."
)

with st.form("scorer_form"):
    col1, col2 = st.columns(2)

    with col1:
        tx_type    = st.selectbox("Transaction Type", ["TRANSFER", "CASH_OUT"])
        amount     = st.number_input("Transaction Amount ($)", min_value=0.0, value=50000.0, step=100.0)
        old_orig   = st.number_input("Origin Balance BEFORE ($)", min_value=0.0, value=50000.0, step=100.0)
        new_orig   = st.number_input("Origin Balance AFTER ($)", min_value=0.0, value=0.0, step=100.0)

    with col2:
        old_dest   = st.number_input("Destination Balance BEFORE ($)", min_value=0.0, value=0.0, step=100.0)
        new_dest   = st.number_input("Destination Balance AFTER ($)", min_value=0.0, value=0.0, step=100.0)

    submitted = st.form_submit_button("Analyse Transaction", type="primary", use_container_width=True)

# ── Feature engineering (mirrors feature_engineering.py exactly) ──────────────
if submitted:
    type_encoded         = 1 if tx_type == "TRANSFER" else 0
    balance_diff_orig    = old_orig - new_orig
    balance_diff_dest    = new_dest - old_dest
    is_empty_after_orig  = int(new_orig == 0)
    amount_exceeds_bal   = int(amount > old_orig)
    amount_ratio_orig    = amount / (old_orig + 1)
    orig_mismatch        = abs(balance_diff_orig - amount)
    dest_mismatch        = abs(balance_diff_dest - amount)

    input_row = pd.DataFrame([{
        "amount":                amount,
        "oldbalanceOrg":         old_orig,
        "newbalanceOrig":        new_orig,
        "oldbalanceDest":        old_dest,
        "newbalanceDest":        new_dest,
        "balance_diff_orig":     balance_diff_orig,
        "balance_diff_dest":     balance_diff_dest,
        "is_empty_after_orig":   is_empty_after_orig,
        "amount_exceeds_balance":amount_exceeds_bal,
        "amount_ratio_orig":     amount_ratio_orig,
        "orig_balance_mismatch": orig_mismatch,
        "dest_balance_mismatch": dest_mismatch,
        "type_encoded":          type_encoded,
    }])[FEATURES]

    fraud_prob = model.predict_proba(input_row)[0][1]
    is_fraud   = fraud_prob >= 0.5

    st.divider()

    # ── Result banner ─────────────────────────────────────────────────────────
    st.subheader("Result")
    if is_fraud:
        st.error(
            f"**FRAUD DETECTED**  |  Confidence: {fraud_prob:.1%}  |  "
            f"The model flags this transaction as highly suspicious."
        )
    else:
        st.success(
            f"**LEGITIMATE**  |  Fraud probability: {fraud_prob:.1%}  |  "
            f"The model does not flag this transaction."
        )

    # ── Probability gauge ─────────────────────────────────────────────────────
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=fraud_prob * 100,
        number={"suffix": "%", "font": {"size": 36}},
        title={"text": "Fraud Probability"},
        gauge={
            "axis":  {"range": [0, 100]},
            "bar":   {"color": COLOR_FRAUD if is_fraud else COLOR_LEGIT},
            "steps": [
                {"range": [0, 30],  "color": "rgba(59,130,246,0.2)"},
                {"range": [30, 60], "color": "rgba(234,179,8,0.2)"},
                {"range": [60, 100],"color": "rgba(231,76,60,0.2)"},
            ],
            "threshold": {
                "line":  {"color": "white", "width": 2},
                "thickness": 0.75,
                "value": 50,
            },
        },
    ))
    fig_gauge.update_layout(
        height=280,
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#f1f5f9",
        margin=dict(t=30, b=10),
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

    st.divider()

    # ── Engineered features table ─────────────────────────────────────────────
    st.subheader("Computed Features")
    st.caption(
        "These are the values the model actually sees. "
        "Flags highlight features that match known fraud patterns."
    )

    # Define thresholds that are considered suspicious
    suspicious_checks = {
        "is_empty_after_orig":    lambda v: v == 1,
        "amount_ratio_orig":      lambda v: v > 0.9,
        "orig_balance_mismatch":  lambda v: v > 1000,
        "dest_balance_mismatch":  lambda v: v > 1000,
        "amount_exceeds_balance": lambda v: v == 1,
    }

    feature_rows = []
    for feat in FEATURES:
        val   = input_row[feat].values[0]
        check = suspicious_checks.get(feat)
        flag  = check(val) if check else False
        feature_rows.append({
            "Feature":     FEATURE_LABELS[feat],
            "Value":       val,
            "Suspicious":  "Yes" if flag else "",
        })

    feat_df = pd.DataFrame(feature_rows)

    def highlight_suspicious(row):
        if row["Suspicious"] == "Yes":
            return ["background-color: rgba(231,76,60,0.25);"] * len(row)
        return [""] * len(row)

    st.dataframe(
        feat_df.style
        .apply(highlight_suspicious, axis=1)
        .format({"Value": "{:.4f}"}),
        use_container_width=True,
        hide_index=True,
    )

    # ── Feature contribution bar ───────────────────────────────────────────────
    st.divider()
    st.subheader("Feature Contribution to This Prediction")
    st.caption(
        "Approximate contribution: feature importance × normalised feature value. "
        "This shows which features pushed the model toward fraud."
    )

    importances = model.feature_importances_
    values_norm = (input_row.values[0] - input_row.values[0].min())
    values_norm = values_norm / (values_norm.max() + 1e-9)
    contributions = importances * values_norm

    contrib_df = pd.DataFrame({
        "Feature":      [FEATURE_LABELS[f] for f in FEATURES],
        "Contribution": contributions,
    }).sort_values("Contribution", ascending=True)

    colors_c = [COLOR_FRAUD if v > contrib_df["Contribution"].median() else COLOR_LEGIT
                for v in contrib_df["Contribution"]]

    fig_contrib = go.Figure(go.Bar(
        x=contrib_df["Contribution"],
        y=contrib_df["Feature"],
        orientation="h",
        marker_color=colors_c,
    ))
    fig_contrib.update_layout(
        height=450,
        xaxis=dict(title="Contribution Score", gridcolor="rgba(255,255,255,0.1)"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#f1f5f9",
        margin=dict(l=20, r=20, t=10, b=20),
    )
    st.plotly_chart(fig_contrib, use_container_width=True)

# ── Instructions when form not submitted ──────────────────────────────────────
else:
    st.info(
        "Fill in the transaction fields above and click **Analyse Transaction**.  \n\n"
        "**Tip:** Try a typical fraud pattern: set Amount = 50,000, "
        "Origin Before = 50,000, Origin After = 0, Type = TRANSFER — "
        "the account is drained to zero, which is the strongest fraud signal."
    )
