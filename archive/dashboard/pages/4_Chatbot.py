"""
4_Chatbot.py — Fraud Detection Chatbot (Groq / Llama 3.3)
"""

import streamlit as st
from groq import Groq
import pandas as pd
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.loader import get_model, get_stats, FEATURES, COLOR_FRAUD, COLOR_LEGIT

st.set_page_config(page_title="Fraud Detection Chatbot", layout="wide")

st.title("Fraud Detection Chatbot")
st.caption("Describe any transaction and the model will evaluate it for fraud.")
st.divider()

# ── API key ────────────────────────────────────────────────────────────────────
api_key = st.secrets.get("GROQ_API_KEY", "") or os.environ.get("GROQ_API_KEY", "")
with st.sidebar:
    st.subheader("Groq API Key")
    st.caption("Free key at console.groq.com")
    if not api_key:
        api_key = st.text_input("API Key", type="password", placeholder="gsk_...")
    else:
        st.success("API key loaded from secrets.")

    st.divider()
    st.subheader("How to use")
    st.markdown(
        "Describe a transaction:  \n"
        "*'Analyse: TRANSFER of 75000, origin had 75000 before and 0 after, "
        "destination had 0 before and 0 after'*  \n\n"
        "Or ask a question:  \n"
        "*'Why is account draining a fraud signal?'*"
    )
    if st.button("Clear conversation", type="secondary"):
        st.session_state.display_messages = []
        st.session_state.groq_messages = []
        st.rerun()

if not api_key:
    st.warning(
        "A Groq API key is required.  \n"
        "Get one free at **console.groq.com** → API Keys → Create API Key."
    )
    st.stop()

# ── Load fraud model and stats ─────────────────────────────────────────────────
fraud_model = get_model()
stats       = get_stats()

# ── Feature engineering + scoring ─────────────────────────────────────────────
def run_score_transaction(args: dict) -> dict:
    tx_type  = args["transaction_type"]
    amount   = float(args["amount"])
    old_orig = float(args["old_balance_orig"])
    new_orig = float(args["new_balance_orig"])
    old_dest = float(args["old_balance_dest"])
    new_dest = float(args["new_balance_dest"])

    type_encoded        = 1 if tx_type == "TRANSFER" else 0
    balance_diff_orig   = old_orig - new_orig
    balance_diff_dest   = new_dest - old_dest
    is_empty            = int(new_orig == 0)
    exceeds_bal         = int(amount > old_orig)
    amount_ratio        = amount / (old_orig + 1)
    orig_mismatch       = abs(balance_diff_orig - amount)
    dest_mismatch       = abs(balance_diff_dest - amount)

    row = pd.DataFrame([{
        "amount": amount, "oldbalanceOrg": old_orig, "newbalanceOrig": new_orig,
        "oldbalanceDest": old_dest, "newbalanceDest": new_dest,
        "balance_diff_orig": balance_diff_orig, "balance_diff_dest": balance_diff_dest,
        "is_empty_after_orig": is_empty, "amount_exceeds_balance": exceeds_bal,
        "amount_ratio_orig": amount_ratio, "orig_balance_mismatch": orig_mismatch,
        "dest_balance_mismatch": dest_mismatch, "type_encoded": type_encoded,
    }])[FEATURES]

    prob     = float(fraud_model.predict_proba(row)[0][1])
    is_fraud = prob >= 0.5

    signals = []
    if is_empty:
        signals.append("Origin account drained to exactly zero")
    if amount_ratio > 0.9:
        signals.append(f"{amount_ratio:.1%} of the origin balance was transferred")
    if orig_mismatch > 1000:
        signals.append(f"Origin balance mismatch of ${orig_mismatch:,.2f}")
    if dest_mismatch > 1000:
        signals.append(f"Destination balance mismatch of ${dest_mismatch:,.2f}")
    if exceeds_bal:
        signals.append("Amount exceeds origin account balance")
    if not signals and is_fraud:
        signals.append("Pattern of features matches fraud in training data")

    return {
        "verdict": "FRAUD" if is_fraud else "LEGITIMATE",
        "fraud_probability": round(prob * 100, 2),
        "is_fraud": is_fraud,
        "suspicious_signals": signals,
        "features": {
            "is_empty_after_orig": is_empty,
            "amount_ratio_orig": round(amount_ratio, 4),
            "orig_balance_mismatch": round(orig_mismatch, 2),
            "dest_balance_mismatch": round(dest_mismatch, 2),
        },
        "input": {
            "type": tx_type, "amount": amount,
            "origin_before": old_orig, "origin_after": new_orig,
            "dest_before": old_dest, "dest_after": new_dest,
        },
    }

# ── Tool definition (OpenAI-compatible format) ─────────────────────────────────
SCORE_TOOL = {
    "type": "function",
    "function": {
        "name": "score_transaction",
        "description": (
            "Run the trained Extra Trees fraud detection model on a transaction. "
            "Call this whenever the user describes a transaction to evaluate. "
            "Ask for missing fields before calling."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "transaction_type": {
                    "type": "string",
                    "enum": ["TRANSFER", "CASH_OUT"],
                    "description": "TRANSFER or CASH_OUT",
                },
                "amount":           {"type": "number", "description": "Amount in dollars"},
                "old_balance_orig": {"type": "number", "description": "Origin balance before"},
                "new_balance_orig": {"type": "number", "description": "Origin balance after"},
                "old_balance_dest": {"type": "number", "description": "Dest balance before"},
                "new_balance_dest": {"type": "number", "description": "Dest balance after"},
            },
            "required": [
                "transaction_type", "amount",
                "old_balance_orig", "new_balance_orig",
                "old_balance_dest", "new_balance_dest",
            ],
        },
    },
}

SYSTEM_PROMPT = f"""You are a fraud detection analyst with access to a trained Extra Trees ML model.

Dataset: {stats['total']:,} transactions | {stats['n_fraud']:,} fraud cases ({stats['fraud_rate']:.4%})
Model: Extra Trees | F1=0.9973 | Recall=0.9947 | Precision=1.0000

Key fraud signals:
1. Account drained to zero after transaction (strongest signal)
2. >90% of balance transferred
3. Balance mismatch — change doesn't match the transaction amount
4. Only TRANSFER and CASH_OUT are ever fraudulent

Rules:
- Call score_transaction whenever the user provides transaction details.
- Start your answer with **FRAUD** or **LEGITIMATE** in bold.
- Explain which signals triggered the result and why they indicate fraud.
- If fields are missing ask for them before calling the tool.
- Be concise and data-driven.
- ONLY answer questions related to fraud detection, financial fraud, this dataset, or this model. If the user asks about anything else, reply with: "I can only help with fraud detection and financial transaction analysis."
"""

# ── Session state ──────────────────────────────────────────────────────────────
if "display_messages" not in st.session_state:
    st.session_state.display_messages = []
if "groq_messages" not in st.session_state:
    st.session_state.groq_messages = []

# ── Render result card ─────────────────────────────────────────────────────────
def render_tool_result(result: dict):
    inp = result["input"]
    if result["is_fraud"]:
        st.error(f"**FRAUD DETECTED** — {result['fraud_probability']:.1f}% confidence")
    else:
        st.success(f"**LEGITIMATE** — {result['fraud_probability']:.1f}% fraud probability")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Transaction:**")
        st.markdown(
            f"- Type: `{inp['type']}`  \n"
            f"- Amount: `${inp['amount']:,.2f}`  \n"
            f"- Origin before: `${inp['origin_before']:,.2f}`  \n"
            f"- Origin after: `${inp['origin_after']:,.2f}`  \n"
            f"- Dest before: `${inp['dest_before']:,.2f}`  \n"
            f"- Dest after: `${inp['dest_after']:,.2f}`"
        )
    with col2:
        st.markdown("**Key signals:**")
        f = result["features"]
        st.markdown(
            f"- Account drained: `{bool(f['is_empty_after_orig'])}`  \n"
            f"- Fraction taken: `{f['amount_ratio_orig']:.2%}`  \n"
            f"- Origin mismatch: `${f['orig_balance_mismatch']:,.2f}`  \n"
            f"- Dest mismatch: `${f['dest_balance_mismatch']:,.2f}`"
        )
    if result["suspicious_signals"]:
        st.markdown("**Suspicious signals:**")
        for s in result["suspicious_signals"]:
            st.markdown(f"- {s}")

# ── Render chat history ────────────────────────────────────────────────────────
for msg in st.session_state.display_messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    elif msg["role"] == "assistant":
        with st.chat_message("assistant"):
            st.markdown(msg["content"])
    elif msg["role"] == "tool_result":
        with st.chat_message("assistant"):
            render_tool_result(msg["result"])

if not st.session_state.display_messages:
    st.info(
        "**Test fraud:** *'Analyse: TRANSFER of 75000, origin had 75000 before and 0 after, destination had 0 before and 0 after'*  \n\n"
        "**Test legitimate:** *'CASH_OUT of 200, origin had 1500 before and 1300 after, destination had 8000 before and 8200 after'*"
    )

# ── Chat input ─────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Describe a transaction or ask a question..."):
    st.session_state.display_messages.append({"role": "user", "content": prompt})
    st.session_state.groq_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    success = False
    with st.spinner("Analysing..."):
        try:
            client   = Groq(api_key=api_key)
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "system", "content": SYSTEM_PROMPT}]
                         + st.session_state.groq_messages,
                tools=[SCORE_TOOL],
                tool_choice="auto",
            )

            msg = response.choices[0].message

            # Check if the model wants to call score_transaction
            if msg.tool_calls:
                tool_call  = msg.tool_calls[0]
                args       = json.loads(tool_call.function.arguments)
                tool_result = run_score_transaction(args)

                # Show result card in UI
                st.session_state.display_messages.append({
                    "role": "tool_result", "result": tool_result
                })

                # Add assistant tool-call message + tool result to Groq history
                st.session_state.groq_messages.append(msg)
                st.session_state.groq_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(tool_result),
                })

                # Get follow-up text from model
                follow_up = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "system", "content": SYSTEM_PROMPT}]
                             + st.session_state.groq_messages,
                )
                reply = follow_up.choices[0].message.content
                st.session_state.groq_messages.append({"role": "assistant", "content": reply})
            else:
                reply = msg.content
                st.session_state.groq_messages.append({"role": "assistant", "content": reply})

            st.session_state.display_messages.append({"role": "assistant", "content": reply})
            success = True

        except Exception as e:
            import traceback
            st.session_state.display_messages.append({
                "role": "assistant",
                "content": f"Error: `{e}`\n\n```\n{traceback.format_exc()}\n```",
            })
            success = True

    if success:
        st.rerun()
