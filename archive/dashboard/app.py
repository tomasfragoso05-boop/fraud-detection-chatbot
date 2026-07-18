"""
app.py — Overview Page
======================
This is the entry point of the Streamlit dashboard.
Streamlit automatically creates a sidebar navigation for every file
placed inside the pages/ folder, so this file is Page 1 (Overview).

What this page shows:
  - 4 KPI metric cards (total transactions, fraud cases, fraud rate, fraud amount)
  - Pie chart: fraud vs legitimate split
  - Bar chart: fraud cases broken down by transaction type
  - Table: sample of detected fraud transactions
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import sys
from pathlib import Path

# Make sure utils/ is importable from any page
sys.path.insert(0, str(Path(__file__).parent))

from utils.loader import load_data, get_stats, load_sample, COLOR_FRAUD, COLOR_LEGIT

# ── Page config ───────────────────────────────────────────────────────────────
# This must be the FIRST Streamlit call in the script.
# layout="wide" uses the full browser width instead of the narrow default.
st.set_page_config(
    page_title="Tax Fraud Detection",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("Tax Fraud Detection Dashboard")
st.caption("Powered by Extra Trees Classifier  |  Dataset: transactions_engineered.csv")
st.divider()

# ── Load data (cached — runs only once) ───────────────────────────────────────
stats  = get_stats()
sample = load_sample()

# ── KPI Cards ─────────────────────────────────────────────────────────────────
# st.columns(n) splits the row into n equal-width columns.
c1, c2, c3, c4 = st.columns(4)

c1.metric(
    label="Total Transactions",
    value=f"{stats['total']:,}",
)
c2.metric(
    label="Fraud Cases Detected",
    value=f"{stats['n_fraud']:,}",
    delta=f"{stats['fraud_rate']:.4%} of all transactions",
    delta_color="inverse",
)
c3.metric(
    label="Total Fraud Amount",
    value=f"${stats['total_fraud_amount']:,.0f}",
)
c4.metric(
    label="Avg Fraud Transaction",
    value=f"${stats['avg_fraud_amount']:,.0f}",
    delta=f"vs ${stats['avg_legit_amount']:,.0f} legitimate avg",
    delta_color="inverse",
)

st.divider()

# ── Charts row ────────────────────────────────────────────────────────────────
col_left, col_right = st.columns(2)

# Pie chart — fraud vs legitimate proportion
with col_left:
    st.subheader("Transaction Split")
    pie_df = pd.DataFrame({
        "Category": ["Legitimate", "Fraud"],
        "Count":    [stats["n_legit"], stats["n_fraud"]],
    })
    fig_pie = px.pie(
        pie_df, names="Category", values="Count",
        color="Category",
        color_discrete_map={"Legitimate": COLOR_LEGIT, "Fraud": COLOR_FRAUD},
        hole=0.45,
    )
    fig_pie.update_traces(textposition="outside", textinfo="percent+label")
    fig_pie.update_layout(
        showlegend=False,
        margin=dict(t=20, b=20, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#f1f5f9",
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# Bar chart — fraud cases by transaction type
with col_right:
    st.subheader("Fraud Cases by Transaction Type")
    type_df = pd.DataFrame(
        stats["fraud_by_type"].items(), columns=["Type", "Fraud Cases"]
    ).sort_values("Fraud Cases", ascending=False)

    fig_bar = px.bar(
        type_df, x="Type", y="Fraud Cases",
        color_discrete_sequence=[COLOR_FRAUD],
        text="Fraud Cases",
    )
    fig_bar.update_traces(texttemplate="%{text:,}", textposition="outside")
    fig_bar.update_layout(
        xaxis_title="Transaction Type",
        yaxis_title="Number of Fraud Cases",
        margin=dict(t=20, b=20, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#f1f5f9",
        yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# ── Fraud amount distribution ─────────────────────────────────────────────────
st.subheader("Transaction Amount Distribution — Fraud vs Legitimate")
st.caption("Showing a sampled subset for performance. All fraud cases are included.")

fraud_sample  = sample[sample["isFraud"] == 1]
legit_sample  = sample[sample["isFraud"] == 0].sample(5000, random_state=42)
plot_df = pd.concat([fraud_sample, legit_sample])
plot_df["Label"] = plot_df["isFraud"].map({1: "Fraud", 0: "Legitimate"})

fig_hist = px.histogram(
    plot_df, x="amount", color="Label",
    nbins=80, barmode="overlay", opacity=0.7,
    color_discrete_map={"Fraud": COLOR_FRAUD, "Legitimate": COLOR_LEGIT},
    log_y=True,
    labels={"amount": "Transaction Amount ($)", "count": "Count (log scale)"},
)
fig_hist.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#f1f5f9",
    yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
    xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
    legend_title_text="",
)
st.plotly_chart(fig_hist, use_container_width=True)

st.divider()

# ── Fraud cases table ─────────────────────────────────────────────────────────
st.subheader("Sample Fraud Transactions")
st.caption("25 randomly selected confirmed fraud cases from the dataset.")

fraud_table = (
    sample[sample["isFraud"] == 1]
    [["type", "amount", "oldbalanceOrg", "newbalanceOrig",
      "oldbalanceDest", "newbalanceDest", "is_empty_after_orig", "amount_ratio_orig"]]
    .sample(min(25, len(sample[sample["isFraud"] == 1])), random_state=1)
    .rename(columns={
        "type":             "Type",
        "amount":           "Amount ($)",
        "oldbalanceOrg":    "Origin Balance Before",
        "newbalanceOrig":   "Origin Balance After",
        "oldbalanceDest":   "Dest Balance Before",
        "newbalanceDest":   "Dest Balance After",
        "is_empty_after_orig": "Account Drained",
        "amount_ratio_orig":   "% of Balance Taken",
    })
)

# Style the table: highlight rows where account was drained
st.dataframe(
    fraud_table.style.format({
        "Amount ($)":           "${:,.2f}",
        "Origin Balance Before":"${:,.2f}",
        "Origin Balance After": "${:,.2f}",
        "Dest Balance Before":  "${:,.2f}",
        "Dest Balance After":   "${:,.2f}",
        "% of Balance Taken":   "{:.2%}",
    }),
    use_container_width=True,
    hide_index=True,
)
