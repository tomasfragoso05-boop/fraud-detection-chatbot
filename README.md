# Fraud Detection Chatbot

An end-to-end financial fraud detection system built entirely in Python — combining a machine learning pipeline, an interactive dashboard, and an AI chatbot that understands plain language.

## What it does

The system analyses financial transactions and predicts whether they are fraudulent. You can interact with it in two ways: fill in a form with transaction details, or just describe the transaction in natural language and let the AI figure out the rest.

## The dataset

2,770,393 synthetic financial transactions (PaySim dataset), of which only 8,197 are fraud — just 0.3% of all records. This extreme class imbalance is the core challenge: a model that blindly predicts "legitimate" for every transaction would be 99.7% accurate while catching zero fraud.

## Machine learning pipeline

Ten models were trained and evaluated head-to-head:

Logistic Regression, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, XGBoost, LightGBM, KNN, Naive Bayes, and SVM.

Rather than relying on raw balance fields, eight additional features were engineered to capture the patterns fraud leaves behind — including whether the origin account was drained to exactly zero (the single strongest fraud signal in the dataset), what fraction of the balance was transferred, and whether the stated transaction amount matches the actual balance change.

**Winner: Extra Trees Classifier**

| Metric | Score |
|--------|-------|
| F1-Score | 0.9973 |
| Recall | 0.9947 |
| Precision | 1.0000 |
| AUC-ROC | 0.9974 |

Zero false positives on the test set. Every transaction it flagged as fraud was actually fraud.

## Dashboard

A five-page Streamlit web app:

- **Overview** — KPI cards, fraud vs legitimate split, fraud by transaction type, amount distributions
- **Model Performance** — comparison table for all 10 models, confusion matrix, full analysis report
- **Feature Analysis** — feature importance rankings, per-feature distributions (fraud vs legitimate), correlation with fraud label
- **Transaction Scorer** — enter six transaction fields, get an instant prediction with confidence score and a breakdown of which signals triggered the alert
- **AI Chatbot** — describe a transaction in plain English; the chatbot extracts the numbers, runs them through the Extra Trees model, and explains the result

## AI Chatbot

The chatbot uses Groq's API (Llama 3.3 70B) with tool calling. When you describe a transaction, the language model extracts the relevant numbers and calls the actual fraud detection model — it does not guess or hallucinate a result. The Extra Trees classifier makes the prediction; the language model only handles the natural language parsing and the explanation.

The chatbot is also restricted to fraud-related topics only and will not answer unrelated questions.

## Tech stack

- **ML:** scikit-learn, XGBoost, LightGBM
- **Dashboard:** Streamlit, Plotly
- **AI:** Groq API, Llama 3.3 70B
- **Data:** Pandas, NumPy

## Running the app

```bash
cd archive
python -m streamlit run dashboard/app.py
```

Add your Groq API key to `dashboard/.streamlit/secrets.toml` to enable the chatbot:

```toml
GROQ_API_KEY = "your_key_here"
```

The model trains automatically on startup (~2 seconds) and is cached in memory for the entire session.
