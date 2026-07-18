# Tax Fraud Detection System — Complete Project Explanation

---

## What This Project Is

A complete end-to-end tax fraud detection system built in three layers:

1. **Machine learning pipeline** — data is loaded, features are engineered, and 10 models are trained and compared to find the best fraud detector
2. **Visual dashboard** — a web application that presents the results interactively, lets you explore the data, and score any transaction live without writing any code
3. **AI chatbot** — a conversational interface powered by a real language model where you describe a transaction in plain language and it runs the actual fraud detection model to tell you whether it is fraud

The entire project is written in Python. No JavaScript, no HTML, no separate backend — just Python scripts that do everything from data processing to the web interface.

---

## The Dataset

**File:** `transactions_engineered.csv` — 2,770,393 rows

Each row represents one financial transaction. The raw fields are:

| Column | What it means |
|--------|--------------|
| `type` | Transaction type: TRANSFER, CASH_OUT, PAYMENT, DEBIT, or CASH_IN |
| `amount` | The transaction amount in dollars |
| `oldbalanceOrg` | Origin account balance before the transaction |
| `newbalanceOrig` | Origin account balance after the transaction |
| `oldbalanceDest` | Destination account balance before the transaction |
| `newbalanceDest` | Destination account balance after the transaction |
| `isFraud` | Ground truth label — 1 means fraud, 0 means legitimate |

**The core challenge:** Of 2,770,393 transactions, only **8,197 are fraud — that is 0.2959%**. This extreme imbalance means a model that blindly predicts "legitimate" for every transaction would be 99.7% accurate while catching zero fraud. This is why accuracy is a useless metric here and why we need a careful approach to both training and evaluation.

---

## Step 1 — Feature Engineering (`feature_engineering.py`)

The raw balance fields alone do not reveal fraud clearly. A $50,000 transfer could be completely normal or completely fraudulent depending on context. Feature engineering computes derived signals that capture the *patterns* fraud leaves behind — patterns that are far more informative than the raw numbers.

**The 8 engineered features added on top of the 5 raw numeric fields:**

| Feature | Formula | What it captures |
|---------|---------|-----------------|
| `balance_diff_orig` | `oldbalanceOrg - newbalanceOrig` | Exactly how much the origin account lost |
| `balance_diff_dest` | `newbalanceDest - oldbalanceDest` | Exactly how much the destination gained |
| `is_empty_after_orig` | `1 if newbalanceOrig == 0 else 0` | Did the origin account reach exactly zero? This is the single strongest fraud signal in the entire dataset |
| `amount_exceeds_balance` | `1 if amount > oldbalanceOrg else 0` | Did the transaction claim to move more than the account actually had? |
| `amount_ratio_orig` | `amount / (oldbalanceOrg + 1)` | What fraction of the origin balance was moved? Fraudsters typically drain close to 100% |
| `orig_balance_mismatch` | `abs(balance_diff_orig - amount)` | Does the stated amount match the actual balance change? Mismatches indicate manipulation |
| `dest_balance_mismatch` | `abs(balance_diff_dest - amount)` | Does the destination gain match the stated amount? |
| `type_encoded` | `1 if TRANSFER else 0` | Only TRANSFER and CASH_OUT transactions are ever fraudulent in this dataset — PAYMENT, DEBIT, CASH_IN have zero fraud cases |

**Why this matters:** After engineering these features, the Extra Trees model achieves near-perfect fraud detection. Without them, using only the raw balance numbers, the model would perform significantly worse because the signal is buried in the context, not the raw values.

---

## Step 2 — Model Comparison (`model_comparison.py`)

We trained and evaluated **10 different machine learning models** on the same dataset and compared them head-to-head.

### Why these metrics and not accuracy?

| Metric | What it measures | Why it matters here |
|--------|-----------------|---------------------|
| **Recall** | Of all real fraud cases, what % did the model catch? | Missing fraud (false negative) has real consequences — a fraudster escapes. This is the most important metric |
| **Precision** | Of all cases flagged as fraud, what % were actually fraud? | False alarms (false positives) waste investigator time and cause friction for legitimate users |
| **F1-Score** | Harmonic mean of Recall and Precision | Penalises models that sacrifice one for the other — a model with 100% recall that flags everything as fraud has 0% precision and a terrible F1 |
| **AUC-ROC** | Area under the ROC curve | How well the model separates the two classes across all possible thresholds |

### Sampling strategy

Training directly on the raw 2.77M rows with 0.3% fraud would bias every model toward predicting "legitimate" because it sees so few fraud examples. Instead:

- Keep **all 8,197 fraud rows**
- Randomly sample **10× non-fraud rows** (81,970 rows)
- Total training pool: **~90,167 rows** — a balanced enough dataset without discarding any fraud

### The 10 models compared

Logistic Regression, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, XGBoost, LightGBM, K-Nearest Neighbours, Naive Bayes, and Support Vector Machine.

### Winner: Extra Trees Classifier

| Metric | Score |
|--------|-------|
| F1-Score | 0.9973 |
| Recall | 0.9947 |
| Precision | 1.0000 |
| AUC-ROC | 0.9974 |

**Zero false positives on the test set.** Every transaction it flagged as fraud was actually fraud. It missed only 13 out of 2,459 fraud cases in the test set (recall of 99.47%).

**Why Extra Trees over Random Forest?** Both are ensemble methods that build many decision trees. Extra Trees (Extremely Randomised Trees) introduces additional randomness in how it chooses split thresholds — this reduces variance and tends to generalise better on highly engineered feature sets like this one, where the features themselves are already very informative.

Results are saved to:
- `model_results/model_comparison.csv` — metrics table for all 10 models
- `model_results/model_report.md` — full written analysis with confusion matrices and conclusions

**Note on hyperparameter optimisation:** We considered running Optuna to tune the Extra Trees hyperparameters but concluded it was not worth the effort. The feature engineering was so effective that any reasonable configuration of the model already achieved near-perfect results. The baseline Extra Trees with default-adjacent settings already had zero false positives — there was nothing meaningful to improve.

---

## Step 3 — The Dashboard

### Why Streamlit?

Streamlit is a Python library that turns a regular `.py` script into a fully interactive web application that runs in the browser. You write pure Python — no HTML, no CSS, no JavaScript — and Streamlit renders buttons, charts, tables, and forms automatically.

```bash
python -m streamlit run dashboard/app.py
```

This starts a local web server at `http://localhost:8501` and opens the app in your browser. Every interaction (button click, dropdown change, form submission) re-runs the Python script and updates the page.

**Why not other technologies?**

| Option | Why not |
|--------|---------|
| Docker | A containerisation/deployment tool, not a UI framework — you would still need a UI library running inside the container |
| Flask / FastAPI | Requires writing HTML templates, CSS, and JavaScript separately from the Python logic |
| Dash (Plotly) | More complex setup and a steeper learning curve for the same result |
| Power BI / Tableau | Cannot integrate a live ML model, cannot build a chatbot, proprietary tools |
| **Streamlit** | Pure Python, built-in chat support, caching built in, ideal for data science projects |

**Important:** The terminal must stay running while you use the dashboard. The website only exists as a Python process on your machine — it is not hosted anywhere. Closing the terminal kills the server. To start it again, run the command above from the `archive` folder.

---

## Folder Structure

```
archive/
  dashboard/
    .streamlit/
      config.toml        ← Visual theme (colours, dark background)
      secrets.toml       ← API keys — loaded automatically, never share this file
    utils/
      __init__.py        ← Makes utils/ a Python package so pages can import from it
      loader.py          ← Central hub: loads data and trains the model, shared by all pages
    app.py               ← Page 1: Overview
    pages/
      1_Model_Performance.py   ← Page 2: Model comparison results
      2_Feature_Analysis.py    ← Page 3: Feature importance and distributions
      3_Transaction_Scorer.py  ← Page 4: Score any transaction live
      4_Chatbot.py             ← Page 5: AI chatbot with real fraud detection
  model_results/
    model_comparison.csv       ← Metrics table from model_comparison.py
    model_report.md            ← Full written analysis
  model_comparison.py          ← Script that trains and compares 10 models
  feature_engineering.py       ← Script that built transactions_engineered.csv
  transactions_engineered.csv  ← The full engineered dataset (2.77M rows)
  PROJECT_EXPLANATION.md       ← This file
```

Streamlit reads the `pages/` folder automatically and creates a sidebar navigation entry for each file. The numbers at the start of the filenames control the display order.

---

## File-by-File Explanation

---

### `.streamlit/config.toml` — Visual Theme

```toml
[theme]
primaryColor             = "#e74c3c"   ← Red — used for fraud highlights, buttons, alerts
backgroundColor          = "#0f172a"   ← Very dark navy — main page background
secondaryBackgroundColor = "#1e293b"   ← Slightly lighter navy — sidebar and card backgrounds
textColor                = "#f1f5f9"   ← Off-white — all text
```

This file is read by Streamlit on startup and applies the colour scheme across every page of the app automatically. Red was chosen as the primary colour because it is the universal signal for danger and alerts — appropriate for a fraud detection system. The dark navy background reduces eye strain for extended analytical use.

---

### `.streamlit/secrets.toml` — API Key Storage

```toml
GROQ_API_KEY = "your_key_here"
```

Streamlit loads this file automatically every time the app starts. This is why you only had to paste your Groq API key once — it is now read from disk on every startup. The chatbot page reads it with:

```python
api_key = st.secrets.get("GROQ_API_KEY", "")
```

If the key is not in secrets, it falls back to an environment variable, and if that is also missing, it shows a text input in the sidebar. This three-level fallback means the app works in any environment.

**This file must never be shared or committed to git.** It contains your private API key.

---

### `utils/loader.py` — The Foundation of the Dashboard

Every page imports from here. Its entire purpose is to **load the data and train the model exactly once, then share the results with every page instantly**.

**The problem without caching:**

Streamlit re-runs the entire script every time you click anything. Without caching that would mean:
- Re-reading 2.77 million rows from disk on every click — roughly 10 seconds each time
- Re-training the Extra Trees model on every click — roughly 2 seconds each time
- The app would be completely unusable

**The solution — two Streamlit caching decorators:**

`@st.cache_data` caches DataFrames and computed values:
```python
@st.cache_data
def load_data():
    return pd.read_csv("transactions_engineered.csv")
```
The first call reads the file and stores the DataFrame in memory. Every subsequent call — from any page, from any user — returns the already-loaded copy instantly. The CSV file is never read again until the server restarts.

`@st.cache_resource` caches heavy objects that should be shared across all users:
```python
@st.cache_resource
def get_model():
    # ... sampling, splitting, training ...
    model = ExtraTreesClassifier(n_estimators=200, max_depth=15, class_weight="balanced")
    model.fit(X_train, y_train)
    return model
```
Same idea but for objects like the trained model. It trains once (~2 seconds) and every page that calls `get_model()` gets the same already-trained model object from memory.

**What loader.py provides to the rest of the app:**

| Function | What it returns |
|----------|----------------|
| `load_data()` | The full 2,770,393 row DataFrame |
| `get_stats()` | A dictionary: total transactions, fraud count, fraud rate, total fraud amount, average amounts, fraud breakdown by type |
| `load_sample(n=80_000)` | A stratified 80,000-row sample that keeps ALL 8,197 fraud cases and randomly samples the rest from legitimate transactions — used for charts so we don't try to plot 2.77M points |
| `get_model()` | The trained ExtraTreesClassifier, ready to call `.predict_proba()` |

It also defines two constants used for consistent chart colours across every page:
- `COLOR_FRAUD = "#e74c3c"` — red
- `COLOR_LEGIT = "#3b82f6"` — blue

---

### `app.py` — Overview Page

The home page. Answers the question: **"What is happening in this dataset at a glance?"**

**1. Four KPI metric cards**
```
Total Transactions    Fraud Cases Detected    Total Fraud Amount    Avg Fraud Transaction
2,770,393             8,197 (0.2959%)         $X,XXX,XXX            $X,XXX
```
Built with `st.metric()`. The delta values below each number show comparisons — fraud rate %, or the difference between average fraud and average legitimate transaction sizes.

**2. Pie chart — Fraud vs Legitimate**
A donut chart showing that 99.7% of transactions are legitimate and 0.3% are fraud. Built with Plotly Express — hovering over a slice shows the exact count. The extreme imbalance is immediately visible here, which is what motivates the careful sampling strategy used in model training.

**3. Bar chart — Fraud by Transaction Type**
Shows that fraud cases only appear in TRANSFER and CASH_OUT transactions. PAYMENT, DEBIT, and CASH_IN have zero fraud cases in the entire dataset. This is one of the most actionable findings — any fraud investigation system can immediately ignore half the transaction types.

**4. Amount distribution histogram**
Fraud vs legitimate transaction amounts overlaid on the same chart with a log scale on the Y-axis. Log scale is necessary because legitimate transactions vastly outnumber fraud at every amount level — without it, the fraud distribution would be invisible.

**5. Sample fraud transactions table**
25 randomly selected confirmed fraud cases displayed as a styled table with all balance fields and the computed `% of Balance Taken` column. In almost every case this is close to 100% — fraudsters drain the account completely.

---

### `pages/1_Model_Performance.py` — Model Comparison Results

Answers: **"Which model performed best and by how much?"**

This page does **not** use the trained model — it reads from `model_results/model_comparison.csv` which was generated when `model_comparison.py` was run. This is intentional: the comparison results are static facts that don't need to be recomputed.

**What it shows:**

**Winner banner** — a green success box at the top announcing Extra Trees as the best model with its exact F1, Recall, Precision, and AUC-ROC. The first thing anyone sees when opening this page is the conclusion.

**Four metric cards** — F1, Recall, Precision, AUC-ROC of the winning model displayed prominently.

**Sortable metrics table** — all 10 models ranked by F1-Score. The winning model row is highlighted in red. Clicking any column header re-sorts the table by that metric — so you can instantly see which model had the best recall, or which was fastest to train.

**Interactive comparison bar chart** — a horizontal bar chart with a dropdown to switch between F1, Recall, Precision, and AUC-ROC. The winning model's bar is always red, others are blue. Hovering shows exact values.

**Confusion matrix heatmap** — TN/FP/FN/TP for the winning model, reconstructed from the known test set size (27,051 rows, 2,459 fraud). Below the matrix, four info boxes explain what each cell means in plain language: "fraud correctly caught", "false alarms", "fraud missed", "legitimate correctly cleared".

**Full model report** — the complete `model_results/model_report.md` rendered inline at the bottom of the page. This is the full written analysis with conclusions about the FP/FN trade-off.

---

### `pages/2_Feature_Analysis.py` — Feature Analysis

Answers: **"Which signals in the data actually reveal fraud, and how strongly?"**

**Feature importance bar chart**

The Extra Trees model exposes `model.feature_importances_` — an array of 13 values (one per feature) representing how much each feature reduced uncertainty (impurity) across all split decisions in all 200 trees. Higher value = more important for separating fraud from legitimate.

Features above the 70th percentile of importance are shown in red, others in blue. This makes the most important features immediately obvious.

**Feature distribution — fraud vs legitimate**

A dropdown lets you select any of the 13 features. For the selected feature:
- **Binary features** (`is_empty_after_orig`, `amount_exceeds_balance`, `type_encoded`): a grouped bar chart showing how many fraud vs legitimate transactions have each value (0 or 1)
- **Continuous features** (`amount`, `amount_ratio_orig`, etc.): a violin plot showing the full distribution shape, median, and spread for fraud and legitimate side by side

Below the chart, a statistics table shows mean, median, standard deviation, min, and max for both classes — making it easy to quantify exactly how different fraud looks from legitimate for any given feature.

**Correlation with fraud bar chart**

Pearson correlation between each feature and the `isFraud` column. Features with high positive correlation (shown in red) are strongly associated with fraud. Features with negative correlation (blue) are more common in legitimate transactions.

This is different from feature importance. Feature importance measures how useful a feature is to the model's tree-splitting decisions. Correlation measures the raw linear relationship between the feature's values and the fraud label — a feature can be highly correlated but provide little additional value if another feature already captures the same information.

---

### `pages/3_Transaction_Scorer.py` — Live Transaction Scorer

Answers: **"Is this specific transaction fraud?"**

This is the direct interface to the model. You fill in six raw transaction fields and the app runs the full prediction pipeline.

**Step 1 — Input form**

A form with two columns:
- Left: Transaction Type (TRANSFER or CASH_OUT), Amount, Origin Balance Before, Origin Balance After
- Right: Destination Balance Before, Destination Balance After
- Submit button: "Analyse Transaction"

**Step 2 — Feature engineering (automatic, invisible to user)**

The moment you submit, the app computes all 13 features using the exact same arithmetic as `feature_engineering.py`:

```python
balance_diff_orig  = old_orig - new_orig
balance_diff_dest  = new_dest - old_dest
is_empty           = int(new_orig == 0)
exceeds_bal        = int(amount > old_orig)
amount_ratio       = amount / (old_orig + 1)
orig_mismatch      = abs(balance_diff_orig - amount)
dest_mismatch      = abs(balance_diff_dest - amount)
type_encoded       = 1 if tx_type == "TRANSFER" else 0
```

**Step 3 — Prediction**

```python
fraud_prob = model.predict_proba(input_row)[0][1]
is_fraud   = fraud_prob >= 0.5
```

**Step 4 — Results display**

- **Result banner** — red "FRAUD DETECTED" or green "LEGITIMATE" with exact confidence percentage
- **Probability gauge** — a speedometer chart from 0% to 100% with colour zones (blue = safe, yellow = uncertain, red = fraud) and a threshold line at 50%
- **Computed features table** — all 13 engineered feature values. Rows highlighted in red if they match known fraud patterns: `is_empty_after_orig = 1`, `amount_ratio_orig > 0.9`, mismatches above $1,000, amount exceeds balance
- **Feature contribution chart** — approximate measure of which features pushed the model toward its decision for this specific transaction, computed as feature importance × normalised feature value

---

### `pages/4_Chatbot.py` — AI Chatbot with Real Fraud Detection

Answers: **"I want to describe a transaction in plain language — is it fraud?"**

This page combines a large language model with the actual Extra Trees fraud detector using a mechanism called **tool calling** (also called function calling).

---

#### Technologies

**Groq**

Groq is a company that runs AI language models on their own custom hardware called LPUs (Language Processing Units). They offer a completely free tier with no meaningful quota limits for this use case. The models run extremely fast — responses typically arrive in under a second.

We switched to Groq after trying two other providers:

| Provider tried | What happened | Why we moved on |
|----------------|--------------|-----------------|
| Anthropic (Claude API) | Requires a paid credit balance | No free tier |
| Google Gemini (new `google-genai` SDK) | Free tier models not available via API | Returned 404 |
| Google Gemini (old `google-generativeai` SDK) | Tool calls routed through v1beta endpoint | Returned 404 for gemini-1.5-flash |
| **Groq + Llama 3.3 70B** | Works reliably, completely free | Current solution |

**Llama 3.3 70B**

Llama is Meta's open-source language model. The "70B" refers to 70 billion parameters — this is a large, capable model. It is hosted on Groq's infrastructure and supports tool calling. Groq's free tier gives access to it with generous rate limits.

**Tool calling / function calling**

This is the mechanism that makes the chatbot a real fraud detector rather than just a chatbot that talks about fraud. A tool is a Python function that the language model is allowed to call when it decides it needs to. The model receives a description of the function and its parameters. When the user describes a transaction, the model decides to call `score_transaction()` with the extracted numbers rather than guessing the result itself.

---

#### How it works step by step

**1. System prompt**

Before any user message is processed, the model receives a system prompt that establishes its role and constraints:

```
You are a fraud detection analyst with access to a trained Extra Trees ML model.

Dataset: 2,770,393 transactions | 8,197 fraud cases (0.2959%)
Model: Extra Trees | F1=0.9973 | Recall=0.9947 | Precision=1.0000

Key fraud signals:
1. Account drained to zero after transaction (strongest signal)
2. >90% of balance transferred
3. Balance mismatch — change doesn't match the transaction amount
4. Only TRANSFER and CASH_OUT are ever fraudulent

Rules:
- Call score_transaction whenever the user provides transaction details.
- Start your answer with FRAUD or LEGITIMATE in bold.
- ONLY answer questions related to fraud detection, financial fraud, this dataset,
  or this model. If the user asks about anything else, reply with:
  "I can only help with fraud detection and financial transaction analysis."
```

The last rule restricts the chatbot to fraud-related topics only. If you ask it about anything unrelated — cooking, sports, coding help — it refuses and redirects you back to fraud topics.

**2. Tool definition**

The `score_transaction` tool is defined in OpenAI-compatible JSON schema format and passed to the model:

```python
SCORE_TOOL = {
    "type": "function",
    "function": {
        "name": "score_transaction",
        "description": "Run the trained Extra Trees fraud detection model...",
        "parameters": {
            "type": "object",
            "properties": {
                "transaction_type": {"type": "string", "enum": ["TRANSFER", "CASH_OUT"]},
                "amount":           {"type": "number"},
                "old_balance_orig": {"type": "number"},
                "new_balance_orig": {"type": "number"},
                "old_balance_dest": {"type": "number"},
                "new_balance_dest": {"type": "number"},
            },
            "required": [...]
        }
    }
}
```

**3. The full conversation flow for a fraud query**

```
User types: "TRANSFER of 75000, origin had 75000 before and 0 after, dest had 0 before and 0 after"
        ↓
Llama 3.3 reads the message and decides to call score_transaction()
        ↓
score_transaction(transaction_type="TRANSFER", amount=75000,
                  old_balance_orig=75000, new_balance_orig=0,
                  old_balance_dest=0, new_balance_dest=0)
        ↓
Feature engineering runs: is_empty=1, amount_ratio=1.0, no mismatches
        ↓
model.predict_proba(row) → [[0.0004, 0.9996]]
        ↓
Returns: {verdict: "FRAUD", fraud_probability: 99.96%, signals: ["Account drained to zero", "100% of balance transferred"]}
        ↓
UI shows a result card with the fraud details
        ↓
Result is sent back to Llama 3.3
        ↓
Llama writes: "**FRAUD** — The model flags this transaction with 99.96% confidence..."
        ↓
User sees both the result card and the written explanation
```

**4. What happens when fields are missing**

If you write "analyse a transfer of 50,000" without providing balance information, Llama will ask you for the missing fields before calling the tool — this is defined in the system prompt and the tool's required fields list.

**5. Why it still works regardless of how you phrase things**

Llama 3.3 is a 70-billion parameter language model trained on the entire internet. Parsing natural language into structured numbers — whether you write "75 thousand", "75,000", "75k", or "seventy-five thousand dollars" — is exactly what it is built for. The Extra Trees model never sees your text at all. By the time it runs, Llama has already extracted six clean numbers and called the function with them.

**6. Conversation memory**

Every message is stored in `st.session_state.groq_messages` as a list of dictionaries. The full conversation history is sent to Groq on every turn:

```python
messages = [{"role": "system", "content": SYSTEM_PROMPT}] + st.session_state.groq_messages
```

This is why the chatbot remembers what was said earlier in the conversation.

**7. API key persistence**

The Groq API key is stored in `.streamlit/secrets.toml` and read automatically on every startup. You set it once, and the chatbot works every time you open the app without entering it again.

---

## How the Model Connects to the Dashboard

The model is not saved to disk as a file and loaded. Instead, **it is trained from scratch every time the Streamlit server starts**, and then held in memory for the entire session.

### Why train on startup instead of saving the model?

Saving the model to disk (with pickle or joblib) and loading it would be faster on startup, but introduces a risk: if the model file was trained with a different version of scikit-learn or on different data, it might behave unexpectedly or fail to load. Training from scratch takes about 2 seconds and guarantees the model always matches the current data and code.

### The startup sequence

When you run `python -m streamlit run dashboard/app.py`, the first page that imports `get_model()` triggers this (once):

```python
@st.cache_resource
def get_model():
    df       = load_data()                            # read 2.77M rows
    fraud_df = df[df["isFraud"] == 1]                 # all 8,197 fraud rows
    legit_df = df[df["isFraud"] == 0]
    legit_sample = legit_df.sample(n_legit, random_state=42)   # 10× undersampling
    df_model = pd.concat([fraud_df, legit_sample])             # ~90,167 rows

    X = df_model[FEATURES]     # the 13 engineered features, in the defined order
    y = df_model["isFraud"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.30, stratify=y)

    model = ExtraTreesClassifier(
        n_estimators=200, max_depth=15, class_weight="balanced",
        random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model
```

`@st.cache_resource` stores the returned model object in the server's memory. Every subsequent call to `get_model()` from any page returns the same already-trained object — no retraining, no disk read, just a memory reference.

### The critical constraint: feature engineering must match exactly

The model was trained on 13 specific features in a specific order. When scoring a new transaction, those exact same 13 features must be computed the exact same way and passed in the exact same column order — otherwise the model receives numbers that mean something completely different from what it learned during training.

Both `3_Transaction_Scorer.py` and `4_Chatbot.py` replicate the feature engineering locally, and both end with:

```python
row = pd.DataFrame([{ ...all 13 features... }])[FEATURES]
```

The `[FEATURES]` at the end enforces the correct column order by selecting columns in the order defined in `loader.py`. Without this, pandas might place columns in insertion order which could differ, and the model would silently produce wrong predictions.

### Getting a prediction

```python
fraud_prob = model.predict_proba(row)[0][1]
is_fraud   = fraud_prob >= 0.5
```

`predict_proba` returns a 2-column array where column 0 is the probability of being legitimate and column 1 is the probability of being fraud. `[0]` selects the first (and only) row. `[1]` selects the fraud probability. A value of 0.5 or above is classified as fraud.

### The full prediction chain

```
User provides 6 raw values: type, amount, origin before, origin after, dest before, dest after
        ↓
8 lines of arithmetic produce 8 new features
        ↓
Combined with the 5 original numeric fields = 13 total features
        ↓
pd.DataFrame with columns in the exact order FEATURES defines
        ↓
model.predict_proba(row) → e.g. [[0.0004, 0.9996]]
        ↓
fraud_prob = 0.9996 → is_fraud = True → "FRAUD DETECTED — 99.96% confidence"
```

### Which pages use the model and how

| Page | Model usage |
|------|------------|
| `app.py` | Does not use the model. Reads the raw dataset for statistics and charts only |
| `1_Model_Performance.py` | Does not use the model. Reads the pre-saved `model_comparison.csv` |
| `2_Feature_Analysis.py` | Reads `model.feature_importances_` for the importance chart. Does not call predict |
| `3_Transaction_Scorer.py` | Feature engineering on form inputs → `model.predict_proba()` → gauge + signals table |
| `4_Chatbot.py` | Llama calls `score_transaction()` → feature engineering → `model.predict_proba()` → result sent back to Llama |

---

## How All the Pieces Connect

```
transactions_engineered.csv
         │
         ▼
   utils/loader.py                  ← reads and caches everything once
    ├── load_data()                  → Overview, Feature Analysis
    ├── get_stats()                  → Overview (KPI cards), Chatbot (system prompt numbers)
    ├── load_sample()                → Overview (charts/table), Feature Analysis (distributions)
    └── get_model()                  → Feature Analysis, Transaction Scorer, Chatbot
         │
         ├──▶ app.py                         Overview — KPIs, charts, fraud sample table
         ├──▶ 1_Model_Performance.py         Reads model_results/model_comparison.csv
         ├──▶ 2_Feature_Analysis.py          model.feature_importances_, distributions, correlations
         ├──▶ 3_Transaction_Scorer.py        Form → feature engineering → predict_proba → gauge
         └──▶ 4_Chatbot.py                   Groq API → Llama 3.3 → tool call → predict_proba
                    │
                    └── .streamlit/secrets.toml  ← GROQ_API_KEY loaded automatically
```

The Transaction Scorer and the Chatbot run the **same model** through the **same feature engineering logic**. The only difference is the input method: the scorer uses a web form, while the chatbot uses a language model to extract the transaction details from natural language before running the same pipeline.

---

## Summary Table

| File | Role | Key technology |
|------|------|----------------|
| `feature_engineering.py` | Builds engineered dataset from raw transactions | Pandas arithmetic |
| `model_comparison.py` | Trains 10 models, selects the best, saves results | scikit-learn, XGBoost, LightGBM |
| `model_results/model_comparison.csv` | Metrics for all 10 models | Output of model_comparison.py |
| `model_results/model_report.md` | Full written analysis and conclusions | Output of model_comparison.py |
| `config.toml` | Visual theme | Streamlit theming |
| `secrets.toml` | Groq API key — loaded automatically on startup | Streamlit secrets |
| `loader.py` | Data + model hub, caches everything | `@st.cache_data` / `@st.cache_resource` |
| `app.py` | Overview home page | KPI cards, Plotly charts |
| `1_Model_Performance.py` | Model comparison results | Sortable table, confusion matrix heatmap |
| `2_Feature_Analysis.py` | Feature insights | Importance bar chart, violin plots, correlation chart |
| `3_Transaction_Scorer.py` | Live transaction prediction via form | Feature engineering → `predict_proba` → gauge chart |
| `4_Chatbot.py` | AI fraud detection via natural language | Groq + Llama 3.3 70B + tool calling → Extra Trees |

---

*Tax Fraud Detection Project | Model: Extra Trees (F1=0.9973, Recall=0.9947, Precision=1.0000) | Dataset: transactions_engineered.csv | Chatbot: Groq / Llama 3.3 70B*
