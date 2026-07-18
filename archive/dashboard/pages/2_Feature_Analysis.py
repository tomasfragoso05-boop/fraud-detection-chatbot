"""
2_Feature_Analysis.py — Feature Analysis Page
=============================================
What this page shows:
  - Feature importance ranking from the trained Extra Trees model
    (how much each feature contributed to splitting fraud from legitimate)
  - Interactive distribution chart: pick any feature and see how its
    values differ between fraud and legitimate transactions
  - Correlation heatmap: which features are most correlated with fraud
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.loader import get_model, load_sample, FEATURES, FEATURE_LABELS, COLOR_FRAUD, COLOR_LEGIT

st.set_page_config(page_title="Feature Analysis", layout="wide")

st.title("Feature Analysis")
st.caption("Understanding which signals drive the fraud detection model")
st.divider()

# ── Load model and data (both cached) ────────────────────────────────────────
model  = get_model()
sample = load_sample()

sample["Label"] = sample["isFraud"].map({1: "Fraud", 0: "Legitimate"})

# ── Feature Importance ────────────────────────────────────────────────────────
st.subheader("Feature Importance")
st.caption(
    "Mean Decrease in Impurity: how much each feature reduces uncertainty "
    "when the model makes a split. Higher = more important for detecting fraud."
)

feat_imp = (
    pd.Series(model.feature_importances_, index=FEATURES)
    .sort_values(ascending=True)
)
readable_labels = [FEATURE_LABELS.get(f, f) for f in feat_imp.index]
colors = [COLOR_FRAUD if v >= feat_imp.quantile(0.7) else COLOR_LEGIT for v in feat_imp.values]

fig_imp = go.Figure(go.Bar(
    x=feat_imp.values,
    y=readable_labels,
    orientation="h",
    marker_color=colors,
    text=[f"{v:.4f}" for v in feat_imp.values],
    textposition="outside",
))
fig_imp.update_layout(
    height=500,
    xaxis=dict(title="Importance Score", gridcolor="rgba(255,255,255,0.1)"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#f1f5f9",
    margin=dict(l=20, r=80, t=10, b=20),
)
st.plotly_chart(fig_imp, use_container_width=True)

# Insights below the chart
top3 = feat_imp.sort_values(ascending=False).head(3)
st.info(
    f"**Top 3 most important features:**  \n"
    f"1. `{top3.index[0]}` ({FEATURE_LABELS[top3.index[0]]}) — {top3.values[0]:.4f}  \n"
    f"2. `{top3.index[1]}` ({FEATURE_LABELS[top3.index[1]]}) — {top3.values[1]:.4f}  \n"
    f"3. `{top3.index[2]}` ({FEATURE_LABELS[top3.index[2]]}) — {top3.values[2]:.4f}"
)

st.divider()

# ── Distribution per feature ──────────────────────────────────────────────────
st.subheader("Feature Distributions — Fraud vs Legitimate")
st.caption(
    "Select a feature to see how its values are distributed "
    "differently between fraud and legitimate transactions."
)

selected = st.selectbox(
    "Choose a feature:",
    options=FEATURES,
    format_func=lambda f: f"{f}  ({FEATURE_LABELS[f]})",
    index=FEATURES.index("is_empty_after_orig"),
)

plot_df = sample[[selected, "Label"]].copy()

# Binary features: use bar chart; continuous: use violin/histogram
binary_features = {"is_empty_after_orig", "amount_exceeds_balance", "type_encoded"}

if selected in binary_features:
    count_df = (
        plot_df.groupby(["Label", selected])
        .size()
        .reset_index(name="Count")
    )
    count_df[selected] = count_df[selected].astype(str)
    fig_dist = px.bar(
        count_df, x=selected, y="Count", color="Label", barmode="group",
        color_discrete_map={"Fraud": COLOR_FRAUD, "Legitimate": COLOR_LEGIT},
        labels={selected: FEATURE_LABELS[selected]},
    )
else:
    fig_dist = px.violin(
        plot_df, x="Label", y=selected, color="Label", box=True,
        color_discrete_map={"Fraud": COLOR_FRAUD, "Legitimate": COLOR_LEGIT},
        labels={selected: FEATURE_LABELS[selected], "Label": ""},
        points=False,
    )

fig_dist.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#f1f5f9",
    yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
    legend_title_text="",
    height=420,
)
st.plotly_chart(fig_dist, use_container_width=True)

# Stats comparison table for selected feature
fraud_vals = plot_df[plot_df["Label"] == "Fraud"][selected]
legit_vals  = plot_df[plot_df["Label"] == "Legitimate"][selected]
stats_df = pd.DataFrame({
    "Statistic": ["Mean", "Median", "Std Dev", "Min", "Max"],
    "Fraud":      [fraud_vals.mean(), fraud_vals.median(), fraud_vals.std(), fraud_vals.min(), fraud_vals.max()],
    "Legitimate": [legit_vals.mean(),  legit_vals.median(),  legit_vals.std(),  legit_vals.min(),  legit_vals.max()],
})
st.dataframe(
    stats_df.style.format({"Fraud": "{:.4f}", "Legitimate": "{:.4f}"}),
    use_container_width=True,
    hide_index=True,
)

st.divider()

# ── Correlation heatmap ───────────────────────────────────────────────────────
st.subheader("Correlation with Fraud")
st.caption(
    "Pearson correlation between each feature and the isFraud label. "
    "Values close to +1 or -1 are strong predictors."
)

corr_cols = FEATURES + ["isFraud"]
corr = sample[corr_cols].corr()["isFraud"].drop("isFraud").sort_values(ascending=False)
corr_df = corr.reset_index().rename(columns={"index": "Feature", "isFraud": "Correlation"})
corr_df["Label"] = corr_df["Feature"].map(FEATURE_LABELS)

colors_corr = [COLOR_FRAUD if v > 0 else COLOR_LEGIT for v in corr_df["Correlation"]]

fig_corr = go.Figure(go.Bar(
    x=corr_df["Correlation"],
    y=corr_df["Label"],
    orientation="h",
    marker_color=colors_corr,
    text=[f"{v:.4f}" for v in corr_df["Correlation"]],
    textposition="outside",
))
fig_corr.update_layout(
    xaxis=dict(title="Correlation with isFraud", gridcolor="rgba(255,255,255,0.1)"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#f1f5f9",
    height=480,
    margin=dict(l=20, r=80, t=10, b=20),
)
st.plotly_chart(fig_corr, use_container_width=True)
st.caption("Red = positively correlated with fraud | Blue = negatively correlated.")
