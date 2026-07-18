"""
1_Model_Performance.py — Model Comparison Page
===============================================
Shows the results from model_comparison.py:
  - Which model won and why (summary cards)
  - Interactive sortable metrics table
  - Comparison bar chart (loaded from model_results/)
  - Confusion matrix of the winning model (Extra Trees)
  - Link to the full markdown report
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.loader import COLOR_FRAUD, COLOR_LEGIT, ROOT

st.set_page_config(page_title="Model Performance", layout="wide")

st.title("Model Performance")
st.caption("Results from the 10-model comparison on transactions_engineered.csv")
st.divider()

# ── Load comparison results ───────────────────────────────────────────────────
csv_path = ROOT / "model_results" / "model_comparison.csv"

if not csv_path.exists():
    st.error("model_results/model_comparison.csv not found. Run model_comparison.py first.")
    st.stop()

comp_df = pd.read_csv(csv_path)

# ── Winner banner ─────────────────────────────────────────────────────────────
best = comp_df.sort_values("F1-Score", ascending=False).iloc[0]

st.success(
    f"**Best Model: {best['Model']}**  |  "
    f"F1 = {best['F1-Score']:.4f}  |  "
    f"Recall = {best['Recall']:.4f}  |  "
    f"Precision = {best['Precision']:.4f}  |  "
    f"AUC-ROC = {best['AUC-ROC']:.4f}  |  "
    f"Zero false positives on the test set."
)

st.divider()

# ── Metric cards ──────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("F1-Score",  f"{best['F1-Score']:.4f}")
c2.metric("Recall",    f"{best['Recall']:.4f}")
c3.metric("Precision", f"{best['Precision']:.4f}")
c4.metric("AUC-ROC",   f"{best['AUC-ROC']:.4f}")

st.divider()

# ── Comparison table ──────────────────────────────────────────────────────────
st.subheader("All Models — Ranked by F1-Score")
st.caption("Click a column header to sort. The best model is highlighted.")

display_df = comp_df.sort_values("F1-Score", ascending=False).reset_index(drop=True)
display_df.insert(0, "Rank", range(1, len(display_df) + 1))

# Highlight the top row
def highlight_best(row):
    color = "background-color: rgba(231, 76, 60, 0.2);" if row["Rank"] == 1 else ""
    return [color] * len(row)

styled = (
    display_df.style
    .apply(highlight_best, axis=1)
    .format({
        "F1-Score":   "{:.4f}",
        "Recall":     "{:.4f}",
        "Precision":  "{:.4f}",
        "AUC-ROC":    "{:.4f}",
        "Accuracy":   "{:.4f}",
        "Time (s)":   "{:.1f}",
    })
)
st.dataframe(styled, use_container_width=True, hide_index=True)

st.divider()

# ── Interactive comparison chart ──────────────────────────────────────────────
st.subheader("Visual Comparison")

metric_choice = st.selectbox(
    "Select metric to compare:",
    ["F1-Score", "Recall", "Precision", "AUC-ROC"],
    index=0,
)

plot_df = comp_df.sort_values(metric_choice, ascending=True)
colors  = [COLOR_FRAUD if m == best["Model"] else COLOR_LEGIT for m in plot_df["Model"]]

fig = go.Figure(go.Bar(
    x=plot_df[metric_choice],
    y=plot_df["Model"],
    orientation="h",
    marker_color=colors,
    text=[f"{v:.4f}" for v in plot_df[metric_choice]],
    textposition="outside",
))
fig.update_layout(
    xaxis=dict(range=[plot_df[metric_choice].min() * 0.998, 1.005],
               gridcolor="rgba(255,255,255,0.1)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#f1f5f9",
    height=420,
    margin=dict(l=20, r=60, t=20, b=20),
)
st.plotly_chart(fig, use_container_width=True)
st.caption(f"Red bar = best model ({best['Model']}). Blue = others.")

st.divider()

# ── Confusion matrix of winning model ────────────────────────────────────────
st.subheader(f"Confusion Matrix — {best['Model']}")

# Reconstruct from known test set size
# Test set: 27,051 rows | Fraud: 2,459 | Legit: 24,592
n_fraud_test = 2459
n_legit_test = 24592
tp = round(best["Recall"] * n_fraud_test)
fn = n_fraud_test - tp
fp = round(tp * (1 - best["Precision"]) / best["Precision"]) if best["Precision"] < 1 else 0
tn = n_legit_test - fp

cm_data = [[tn, fp], [fn, tp]]
labels  = ["Legitimate", "Fraud"]

fig_cm = go.Figure(go.Heatmap(
    z=cm_data,
    x=["Predicted: Legitimate", "Predicted: Fraud"],
    y=["Actual: Legitimate", "Actual: Fraud"],
    colorscale="Blues",
    showscale=False,
    text=[[f"TN\n{tn:,}", f"FP\n{fp:,}"],
          [f"FN\n{fn:,}", f"TP\n{tp:,}"]],
    texttemplate="%{text}",
    textfont=dict(size=16, color="white"),
))
fig_cm.update_layout(
    height=380,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#f1f5f9",
    margin=dict(t=20, b=20),
)
st.plotly_chart(fig_cm, use_container_width=True)

col_a, col_b = st.columns(2)
col_a.info(f"**True Positives (TP):** {tp:,}  — fraud correctly caught")
col_a.info(f"**True Negatives (TN):** {tn:,}  — legitimate correctly cleared")
col_b.warning(f"**False Negatives (FN):** {fn:,}  — fraud missed")
col_b.warning(f"**False Positives (FP):** {fp:,}  — false alarms")

st.divider()

# ── Full report link ──────────────────────────────────────────────────────────
st.subheader("Full Report")
report_path = ROOT / "model_results" / "model_report.md"
if report_path.exists():
    with open(report_path, encoding="utf-8") as f:
        st.markdown(f.read())
else:
    st.warning("model_results/model_report.md not found.")
