"""
Fraud Detection — Model Comparison Script
==========================================
Trains and evaluates 10 ML models on transactions_engineered.csv.
Outputs:
  model_results/cm_<ModelName>.png        — confusion matrix plots
  model_results/model_comparison_chart.png — bar-chart comparison
  model_results/model_comparison.csv      — metrics table
  model_results/model_report.md           — full markdown report
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
import os

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, f1_score, recall_score,
    precision_score, roc_auc_score, accuracy_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, AdaBoostClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("WARNING: xgboost not installed — skipping XGBoost. Install with: pip install xgboost")

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("WARNING: lightgbm not installed — skipping LightGBM. Install with: pip install lightgbm")

os.makedirs("model_results", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 65)
print("  LOADING DATA")
print("=" * 65)
df = pd.read_csv("transactions_engineered.csv")
n_total  = len(df)
n_fraud  = int(df['isFraud'].sum())
fraud_pct = df['isFraud'].mean()
print(f"  Total rows  : {n_total:>12,}")
print(f"  Fraud cases : {n_fraud:>12,}  ({fraud_pct:.4%})")

# ─────────────────────────────────────────────────────────────────────────────
# 2. SAMPLING STRATEGY
#    Keep ALL fraud rows; undersample non-fraud to a 1:10 ratio.
#    This avoids information loss on the rare class while keeping training fast.
# ─────────────────────────────────────────────────────────────────────────────

FRAUD_TO_LEGIT_RATIO = 10

fraud_df     = df[df['isFraud'] == 1]
non_fraud_df = df[df['isFraud'] == 0]

n_non_fraud_sample = min(n_fraud * FRAUD_TO_LEGIT_RATIO, len(non_fraud_df))
non_fraud_sample   = non_fraud_df.sample(n_non_fraud_sample, random_state=42)
df_model = (
    pd.concat([fraud_df, non_fraud_sample])
    .sample(frac=1, random_state=42)
    .reset_index(drop=True)
)

print(f"\n  Model dataset : {len(df_model):,} rows")
print(f"  Fraud in set  : {df_model['isFraud'].sum():,} ({df_model['isFraud'].mean():.2%})")

# ─────────────────────────────────────────────────────────────────────────────
# 3. FEATURES & SPLIT
# ─────────────────────────────────────────────────────────────────────────────

FEATURES = [
    'amount', 'oldbalanceOrg', 'newbalanceOrig',
    'oldbalanceDest', 'newbalanceDest',
    'balance_diff_orig', 'balance_diff_dest',
    'is_empty_after_orig', 'amount_exceeds_balance',
    'amount_ratio_orig', 'orig_balance_mismatch',
    'dest_balance_mismatch', 'type_encoded',
]

X = df_model[FEATURES]
y = df_model['isFraud']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)
print(f"\n  Train : {len(X_train):,}  |  Test : {len(X_test):,}")

# Scaled variants (for linear / distance-based models)
scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# KNN subset — training full-set KNN is O(n²); cap at 20 k rows
KNN_LIMIT = 20_000
if len(X_train_s) > KNN_LIMIT:
    rng = np.random.RandomState(42)
    idx = rng.choice(len(X_train_s), KNN_LIMIT, replace=False)
    X_knn_train = X_train_s[idx]
    y_knn_train = y_train.iloc[idx]
else:
    X_knn_train = X_train_s
    y_knn_train = y_train

# ─────────────────────────────────────────────────────────────────────────────
# 4. MODEL DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

n_neg = int((y_train == 0).sum())
n_pos = int((y_train == 1).sum())
spw   = n_neg / n_pos   # scale_pos_weight for XGBoost

models = {}

if HAS_XGB:
    models["XGBoost"] = dict(
        model=XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            scale_pos_weight=spw, subsample=0.8, colsample_bytree=0.8,
            eval_metric='logloss', random_state=42, n_jobs=-1, verbosity=0,
        ),
        scaled=False, knn_mode=False,
    )

if HAS_LGBM:
    models["LightGBM"] = dict(
        model=LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            class_weight='balanced', subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, verbose=-1,
        ),
        scaled=False, knn_mode=False,
    )

models.update({
    "Random Forest": dict(
        model=RandomForestClassifier(
            n_estimators=200, max_depth=15, class_weight='balanced',
            random_state=42, n_jobs=-1,
        ),
        scaled=False, knn_mode=False,
    ),
    "Extra Trees": dict(
        model=ExtraTreesClassifier(
            n_estimators=200, max_depth=15, class_weight='balanced',
            random_state=42, n_jobs=-1,
        ),
        scaled=False, knn_mode=False,
    ),
    "Gradient Boosting": dict(
        model=GradientBoostingClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.1,
            subsample=0.8, random_state=42,
        ),
        scaled=False, knn_mode=False,
    ),
    "Decision Tree": dict(
        model=DecisionTreeClassifier(
            max_depth=12, class_weight='balanced', random_state=42,
        ),
        scaled=False, knn_mode=False,
    ),
    "AdaBoost": dict(
        model=AdaBoostClassifier(
            n_estimators=200, learning_rate=0.5, random_state=42,
        ),
        scaled=False, knn_mode=False,
    ),
    "Logistic Regression": dict(
        model=LogisticRegression(
            C=0.1, class_weight='balanced', max_iter=1000,
            solver='saga', random_state=42, n_jobs=-1,
        ),
        scaled=True, knn_mode=False,
    ),
    "MLP (Neural Network)": dict(
        model=MLPClassifier(
            hidden_layer_sizes=(128, 64, 32), activation='relu',
            max_iter=300, early_stopping=True, validation_fraction=0.1,
            random_state=42,
        ),
        scaled=True, knn_mode=False,
    ),
    "KNN": dict(
        model=KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        scaled=True, knn_mode=True,
    ),
})

# ─────────────────────────────────────────────────────────────────────────────
# 5. TRAINING & EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def save_confusion_matrix_plot(cm, name, f1, recall, filepath):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', ax=ax,
        xticklabels=['Legítimo', 'Fraude'],
        yticklabels=['Legítimo', 'Fraude'],
        annot_kws={'size': 14, 'weight': 'bold'},
    )
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(
        f'{name}\nF1 = {f1:.4f}  |  Recall = {recall:.4f}',
        fontsize=13, fontweight='bold', pad=12,
    )
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()


results = {}

print("\n" + "=" * 65)
print("  TRAINING & EVALUATION  (10 models)")
print("=" * 65)

for name, cfg in models.items():
    model    = cfg['model']
    scaled   = cfg['scaled']
    knn_mode = cfg['knn_mode']

    print(f"\n  [{name}]")
    t0 = time.time()

    if knn_mode:
        model.fit(X_knn_train, y_knn_train)
        y_pred = model.predict(X_test_s)
        proba  = model.predict_proba(X_test_s)[:, 1]
    elif scaled:
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        proba  = model.predict_proba(X_test_s)[:, 1] if hasattr(model, 'predict_proba') else None
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        proba  = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    elapsed = time.time() - t0

    cm      = confusion_matrix(y_test, y_pred)
    f1      = f1_score(y_test, y_pred, zero_division=0)
    recall  = recall_score(y_test, y_pred, zero_division=0)
    prec    = precision_score(y_test, y_pred, zero_division=0)
    acc     = accuracy_score(y_test, y_pred)
    auc     = roc_auc_score(y_test, proba) if proba is not None else None

    results[name] = {
        'f1': f1, 'recall': recall, 'precision': prec,
        'accuracy': acc, 'auc_roc': auc,
        'confusion_matrix': cm.tolist(),
        'time_s': elapsed,
    }

    auc_s = f"{auc:.4f}" if auc is not None else "N/A"
    print(f"    F1={f1:.4f}  Recall={recall:.4f}  Precision={prec:.4f}  AUC-ROC={auc_s}  ({elapsed:.1f}s)")

    safe = name.replace(" ", "_").replace("(", "").replace(")", "")
    save_confusion_matrix_plot(cm, name, f1, recall, f"model_results/cm_{safe}.png")

# ─────────────────────────────────────────────────────────────────────────────
# 6. SUMMARY TABLE & CHART
# ─────────────────────────────────────────────────────────────────────────────

rows = [
    {
        'Model': name,
        'F1-Score': round(r['f1'], 4),
        'Recall': round(r['recall'], 4),
        'Precision': round(r['precision'], 4),
        'AUC-ROC': round(r['auc_roc'], 4) if r['auc_roc'] is not None else None,
        'Accuracy': round(r['accuracy'], 4),
        'Time (s)': round(r['time_s'], 1),
    }
    for name, r in results.items()
]
comp_df = pd.DataFrame(rows).sort_values('F1-Score', ascending=False).reset_index(drop=True)
comp_df.to_csv("model_results/model_comparison.csv", index=False)

print("\n" + "=" * 65)
print("  RESULTS (sorted by F1-Score)")
print("=" * 65)
print(comp_df.to_string(index=False))

# Bar chart — F1, Recall, Precision side by side
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
metrics_to_plot = [('F1-Score', 'steelblue'), ('Recall', 'coral'), ('Precision', 'seagreen')]

for ax, (metric, color) in zip(axes, metrics_to_plot):
    plot_df = comp_df.sort_values(metric, ascending=True)
    bars = ax.barh(plot_df['Model'], plot_df[metric], color=color, edgecolor='white', height=0.6)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel(metric, fontsize=12)
    ax.set_title(f'{metric} by Model', fontsize=13, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    for bar, val in zip(bars, plot_df[metric]):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', fontsize=9)

plt.suptitle('Fraud Detection — Model Comparison', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("model_results/model_comparison_chart.png", dpi=150, bbox_inches='tight')
plt.close()
print("\n  Chart saved → model_results/model_comparison_chart.png")

# ─────────────────────────────────────────────────────────────────────────────
# 7. MARKDOWN REPORT GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

MODEL_COMMENTARY = {
    "XGBoost": (
        "XGBoost is a highly optimised gradient boosting framework. "
        "It handles class imbalance natively via `scale_pos_weight`, supports parallel tree construction, "
        "and excels at capturing the non-linear interactions common in financial fraud patterns "
        "(e.g., between `balance_diff_orig`, `is_empty_after_orig`, and `amount_ratio_orig`). "
        "It is the industry standard for tabular fraud detection competitions."
    ),
    "LightGBM": (
        "LightGBM uses leaf-wise (best-first) tree growth rather than level-wise, making it faster "
        "and often more accurate than XGBoost on large datasets. Its histogram-based splitting "
        "and native support for `class_weight='balanced'` make it excellent for imbalanced fraud data. "
        "Preferred when training speed on millions of rows is critical."
    ),
    "Random Forest": (
        "Random Forest is a robust bagging ensemble of decision trees. It resists overfitting through "
        "feature randomisation at each split and handles class imbalance with `class_weight='balanced'`. "
        "It provides reliable feature importance scores — valuable for regulatory explainability — "
        "and trains fully in parallel."
    ),
    "Extra Trees": (
        "Extremely Randomised Trees use random thresholds for splits (instead of the optimal threshold "
        "used by Random Forest), making them faster to train and reducing variance further. "
        "Often competitive with Random Forest on high-dimensional tabular data."
    ),
    "Gradient Boosting": (
        "Sklearn's Gradient Boosting builds trees sequentially, each correcting the residuals of the last. "
        "It is slower than XGBoost/LightGBM (no native parallelism) but highly accurate. "
        "Useful as a reference for the boosting family performance without external libraries."
    ),
    "Decision Tree": (
        "A single decision tree is the most interpretable model — every prediction can be traced back "
        "to a readable sequence of rules. However, it tends to overfit without depth constraints and "
        "is weaker than ensembles. Best used as an interpretable baseline or for rule extraction."
    ),
    "AdaBoost": (
        "AdaBoost reweights training samples iteratively, focusing on misclassified cases. "
        "It can work well but is sensitive to noisy data and outliers, which are common in financial datasets. "
        "Generally outperformed by gradient boosting variants on fraud tasks."
    ),
    "Logistic Regression": (
        "Logistic Regression is the canonical linear baseline. It trains fast, is fully interpretable "
        "(coefficients map directly to feature importance), and is legally defensible in regulated environments. "
        "However, it can only model linear decision boundaries, which limits its detection power "
        "for complex fraud patterns."
    ),
    "MLP (Neural Network)": (
        "A Multi-Layer Perceptron can learn complex non-linear patterns via hidden layers. "
        "With `early_stopping`, it avoids overfitting. However, it is less interpretable than tree models, "
        "requires careful scaling, and often underperforms gradient boosting on tabular data "
        "without extensive architecture tuning."
    ),
    "KNN": (
        "K-Nearest Neighbours classifies by majority vote among the k closest training samples. "
        "It is non-parametric and requires no training phase, but inference is O(n) per sample "
        "making it impractical for real-time scoring on large datasets. "
        "Note: trained on a 20k subsample here for computational feasibility."
    ),
}


def best_model_justification(best_name: str, best_row, second_row=None) -> list:
    lines = []
    model_family = (
        "gradient boosting" if best_name in {"XGBoost", "LightGBM", "Gradient Boosting"}
        else "tree ensemble" if best_name in {"Random Forest", "Extra Trees", "AdaBoost"}
        else "linear" if best_name == "Logistic Regression"
        else "neural"
    )

    if model_family == "gradient boosting":
        lines += [
            "**1. Gradient boosting dominates on tabular fraud data.**  "
            "Sequential boosting focuses training effort on the hardest-to-classify samples — "
            "exactly what is needed when fraud patterns are rare and subtle.",
            "",
            "**2. Native class-imbalance handling.**  "
            "`scale_pos_weight` (XGBoost) / `class_weight='balanced'` (LightGBM) directly "
            "amplify the penalty for missing a fraud case, improving Recall without manual oversampling.",
            "",
            "**3. Feature importance for explainability.**  "
            "Both XGBoost and LightGBM expose per-feature gain/split importance, enabling the "
            "dashboard chatbot to explain *why* a transaction was flagged — essential for tax authority audits.",
            "",
            "**4. Production-ready scalability.**  "
            "Both libraries support distributed training and can score millions of transactions "
            "per second, making them suitable for the full 2.7M-row dataset and real-time pipelines.",
        ]
    elif model_family == "tree ensemble":
        lines += [
            "**1. Ensemble diversity reduces variance.**  "
            f"{best_name} averages hundreds of independent trees, making it robust to noisy labels "
            "common in synthetic fraud datasets.",
            "",
            "**2. No probability calibration quirks.**  "
            "Unlike boosting, Random Forest / Extra Trees output well-calibrated probability estimates "
            "straight away, useful for threshold optimisation in the dashboard.",
            "",
            "**3. Feature importance for explainability.**  "
            "Mean Decrease in Impurity scores identify the most fraud-predictive features "
            "(e.g., `is_empty_after_orig`, `balance_diff_orig`), useful for investigator reports.",
        ]
    else:
        lines += [
            f"**1. {best_name} achieves the best F1/Recall trade-off on this dataset.**  "
            "This may indicate a predominantly linear decision boundary in the engineered feature space.",
            "",
            "**2. Interpretability advantage.**  "
            "Coefficient-level explanations are straightforward to surface in the dashboard.",
        ]

    if second_row is not None:
        lines += [
            "",
            f"> **Runner-up:** {second_row['Model']} "
            f"(F1 = {second_row['F1-Score']:.4f}, Recall = {second_row['Recall']:.4f})  "
            "Consider an ensemble of the top-2 models for production (soft-voting).",
        ]

    return lines


def generate_markdown(results: dict, comp_df: pd.DataFrame):
    best   = comp_df.iloc[0]
    second = comp_df.iloc[1] if len(comp_df) > 1 else None

    L = []  # lines accumulator

    L += [
        "# Fraud Detection — Model Comparison Report",
        "",
        "> **Dataset:** `transactions_engineered.csv` (PaySim Synthetic Financial)  ",
        f"> **Sampling strategy:** All {n_fraud:,} fraud rows + {FRAUD_TO_LEGIT_RATIO}× undersampled non-fraud  ",
        "> **Train / Test split:** 70 % / 30 % (stratified by `isFraud`)  ",
        "> **Metric priority:** Recall › F1-Score › AUC-ROC › Precision  ",
        "",
        "---",
        "",
        "## 1. Why These Metrics?",
        "",
        "In tax fraud detection a **False Negative** (missed fraud) causes financial harm; "
        "a **False Positive** (legitimate transaction flagged) wastes analyst time. "
        "Therefore:",
        "",
        "| Priority | Metric | Rationale |",
        "|:--------:|--------|-----------|",
        "| 1 | **Recall** | Maximise caught fraud — FN is the costliest error |",
        "| 2 | **F1-Score** | Harmonic mean; penalises extreme Precision/Recall trade-offs |",
        "| 3 | **AUC-ROC** | Threshold-independent discriminative power |",
        "| 4 | **Precision** | Controls false-alarm rate for human investigators |",
        "",
        "---",
        "",
        "## 2. Summary Table",
        "",
        "| Rank | Model | F1-Score | Recall | Precision | AUC-ROC | Accuracy | Time (s) |",
        "|:----:|-------|:--------:|:------:|:---------:|:-------:|:--------:|:--------:|",
    ]

    for rank, row in comp_df.iterrows():
        auc = f"{row['AUC-ROC']:.4f}" if pd.notna(row['AUC-ROC']) else "N/A"
        L.append(
            f"| {rank + 1} | **{row['Model']}** | {row['F1-Score']:.4f} | {row['Recall']:.4f} | "
            f"{row['Precision']:.4f} | {auc} | {row['Accuracy']:.4f} | {row['Time (s)']} |"
        )

    L += [
        "",
        "![Model Comparison Chart](model_comparison_chart.png)",
        "",
        "---",
        "",
        "## 3. Individual Model Results",
        "",
    ]

    for name, r in results.items():
        safe = name.replace(" ", "_").replace("(", "").replace(")", "")
        cm   = np.array(r['confusion_matrix'])
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
        auc_s = f"{r['auc_roc']:.4f}" if r['auc_roc'] is not None else "N/A"
        rank_row = comp_df[comp_df['Model'] == name]
        rank_val = int(rank_row.index[0]) + 1 if len(rank_row) else "?"

        L += [
            f"### {rank_val}. {name}",
            "",
            f"![{name} Confusion Matrix](cm_{safe}.png)",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| **F1-Score** | `{r['f1']:.4f}` |",
            f"| **Recall (Sensitivity)** | `{r['recall']:.4f}` |",
            f"| **Precision** | `{r['precision']:.4f}` |",
            f"| **AUC-ROC** | `{auc_s}` |",
            f"| Accuracy | `{r['accuracy']:.4f}` |",
            f"| Training time | `{r['time_s']:.1f} s` |",
            "",
            "**Confusion Matrix breakdown:**",
            "",
            "| | Predicted: Legítimo | Predicted: Fraude |",
            "|---|:---:|:---:|",
            f"| **Actual: Legítimo** | {tn:,} *(TN)* | {fp:,} *(FP)* |",
            f"| **Actual: Fraude** | {fn:,} *(FN)* | {tp:,} *(TP)* |",
            "",
        ]

        comment = MODEL_COMMENTARY.get(name, "")
        if comment:
            L += [f"**Model notes:** {comment}", ""]

        L += ["---", ""]

    # ── Best model section ──────────────────────────────────────────────────
    L += [
        "## 4. Conclusion — Best Model",
        "",
        f"### Winner: {best['Model']}",
        "",
        f"With **F1-Score = {best['F1-Score']:.4f}** and **Recall = {best['Recall']:.4f}**, "
        f"**{best['Model']}** ranks first when evaluated by the fraud-detection priority metric order.",
        "",
        "### Justification",
        "",
    ]
    L += best_model_justification(best['Model'], best, second)

    L += [
        "",
        "---",
        "",
        "## 5. Recommended Next Steps",
        "",
        f"1. **Hyperparameter tuning** — run Optuna / RandomizedSearchCV on **{best['Model']}** "
        "with the full dataset to push F1 and Recall further.",
        "2. **Threshold optimisation** — shift the classification threshold below 0.5 to increase Recall "
        "at the cost of Precision; use a precision-recall curve to find the operational sweet spot.",
        "3. **SHAP explanations** — integrate SHAP values into the dashboard and chatbot so investigators "
        "can see which features drove each fraud flag (e.g., `is_empty_after_orig=1` + `amount_ratio_orig` high).",
        "4. **Calibration** — apply Isotonic Regression or Platt scaling so the model's probability output "
        "reflects true fraud likelihood, enabling risk-scoring instead of binary labels.",
        "5. **Retrain on full data** — the comparison used an undersampled subset for speed; "
        "retrain the winning model on the full 2.7 M rows for production deployment.",
        "",
        "---",
        "",
        "*Report auto-generated by `model_comparison.py`*",
    ]

    report_path = "model_results/model_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"\n  Markdown report saved → {report_path}")


generate_markdown(results, comp_df)

print("\n" + "=" * 65)
print("  ALL DONE  — check the model_results/ folder")
print("=" * 65)
