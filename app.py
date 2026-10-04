"""Financial Fraud Detection System - Streamlit app."""

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from fraud_logic import (
    FLAG_COLUMNS,
    FRAUD_SCORE_THRESHOLD,
    HIGH_AMOUNT_THRESHOLD,
    MAX_FAILED_ATTEMPTS,
    NEW_ACCOUNT_MONTHS,
    REQUIRED_COLUMNS,
    evaluate_dataframe,
    evaluate_transaction,
    rule_trigger_counts,
    score_distribution,
    summary_stats,
)

SAMPLE_CSV = Path(__file__).parent / "sample_transactions.csv"

st.set_page_config(page_title="Financial Fraud Detection", page_icon="🛡️", layout="wide")

# All checked transactions (single and batch) live in one DataFrame in session state
if "records" not in st.session_state:
    st.session_state.records = pd.DataFrame()
if "added_files" not in st.session_state:
    st.session_state.added_files = set()


def pretty(df: pd.DataFrame) -> pd.DataFrame:
    """Friendlier column names for display."""
    return df.rename(columns=lambda c: c.replace("_", " ").title())


def result_table(df: pd.DataFrame) -> None:
    """Show a results DataFrame with a score bar and without the raw flag columns."""
    shown = df.drop(columns=FLAG_COLUMNS, errors="ignore")
    st.dataframe(
        pretty(shown),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Fraud Score": st.column_config.ProgressColumn(
                "Fraud Score", min_value=0, max_value=len(FLAG_COLUMNS), format="%d"
            ),
            "Transaction Amount": st.column_config.NumberColumn("Transaction Amount", format="₹%.0f"),
        },
    )


def add_to_records(df: pd.DataFrame, source: str) -> None:
    df = df.copy()
    df.insert(0, "source", source)
    df.insert(0, "checked_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    st.session_state.records = pd.concat([st.session_state.records, df], ignore_index=True)


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Detection rules")
    st.markdown(
        f"""
Each rule that is triggered adds 1 to the fraud score.

- Amount greater than ₹{HIGH_AMOUNT_THRESHOLD:,}
- Account age less than {NEW_ACCOUNT_MONTHS} months
- Failed login attempts of {MAX_FAILED_ATTEMPTS} or more
- Transaction from an untrusted location

A score of {FRAUD_SCORE_THRESHOLD} or more is flagged as Potential Fraud Detected.
"""
    )
    st.divider()
    st.subheader("Quick test cases")
    samples = {
        "Rahul (score 4)": ("Rahul", 75000.0, 3, 4, "No"),
        "Priya (score 0)": ("Priya", 12000.0, 18, 0, "Yes"),
        "Amit (score 1)": ("Amit", 60000.0, 24, 1, "Yes"),
        "Neha (score 2)": ("Neha", 30000.0, 2, 3, "Yes"),
        "Arjun (score 3)": ("Arjun", 55000.0, 12, 5, "No"),
    }
    choice = st.selectbox("Load a sample into the single check form", ["None"] + list(samples))
    if st.button("Load sample", use_container_width=True) and choice != "None":
        name, amt, age, fails, loc = samples[choice]
        st.session_state.name = name
        st.session_state.amount = amt
        st.session_state.age = age
        st.session_state.fails = fails
        st.session_state.loc = loc
        st.rerun()

# ---------------- Main ----------------
st.title("🛡️ Financial Fraud Detection System")
st.caption("A simple rule-based system that checks transactions for suspicious signs.")

tab_single, tab_batch, tab_dash = st.tabs(["Single check", "Batch check (CSV)", "Dashboard"])

# ================= Single check =================
with tab_single:
    with st.form("fraud_form"):
        name = st.text_input("Customer name", key="name", placeholder="e.g. Rahul")
        col1, col2 = st.columns(2)
        with col1:
            amount = st.number_input(
                "Transaction amount (₹)", min_value=0.0, step=500.0, key="amount", format="%.2f"
            )
            fails = st.number_input("Failed login attempts", min_value=0, step=1, key="fails")
        with col2:
            age = st.number_input("Account age (months)", min_value=0, step=1, key="age")
            loc = st.radio(
                "Is the transaction from a trusted location?",
                ["Yes", "No"],
                horizontal=True,
                key="loc",
            )
        submitted = st.form_submit_button("Check transaction", use_container_width=True)

    if submitted:
        if not name.strip():
            st.error("Please enter the customer name.")
        else:
            result = evaluate_transaction(name.strip(), amount, int(age), int(fails), loc)

            st.subheader("Result")
            m1, m2 = st.columns(2)
            m1.metric("Customer", result.customer_name)
            m2.metric("Fraud score", f"{result.fraud_score} / {len(FLAG_COLUMNS)}")
            st.progress(result.fraud_score / len(FLAG_COLUMNS))

            if result.warnings:
                for w in result.warnings:
                    st.warning(w)
            else:
                st.info("No suspicious conditions found.")

            if result.is_fraud:
                st.error(f"Status: {result.status}")
            else:
                st.success(f"Status: {result.status}")

            # Record it using the same pandas logic used for batches
            one_row = pd.DataFrame(
                [
                    {
                        "customer_name": name.strip(),
                        "transaction_amount": amount,
                        "account_age_months": int(age),
                        "failed_attempts": int(fails),
                        "trusted_location": loc,
                    }
                ]
            )
            scored, _ = evaluate_dataframe(one_row)
            add_to_records(scored, "Single check")
            st.caption("Added to the Dashboard tab.")

# ================= Batch check =================
with tab_batch:
    st.write(
        "Upload a CSV file to check many transactions at once. "
        "Required columns: " + ", ".join(REQUIRED_COLUMNS) + "."
    )
    if SAMPLE_CSV.exists():
        st.download_button(
            "Download sample CSV",
            SAMPLE_CSV.read_bytes(),
            file_name="sample_transactions.csv",
            mime="text/csv",
        )

    uploaded = st.file_uploader("Upload transactions CSV", type=["csv"])

    if uploaded is not None:
        try:
            raw = pd.read_csv(uploaded)
            batch, dropped = evaluate_dataframe(raw)
        except ValueError as err:
            st.error(str(err))
            batch = None
        except Exception:
            st.error("Could not read this file. Please upload a valid CSV.")
            batch = None

        if batch is not None:
            if dropped:
                st.warning(f"{dropped} row(s) were skipped because of missing or non-numeric values.")
            if batch.empty:
                st.info("No valid rows to analyse.")
            else:
                stats = summary_stats(batch)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Transactions", stats["total"])
                c2.metric("Flagged as fraud", stats["flagged"])
                c3.metric("Flag rate", f"{stats['flag_rate']:.1f}%")
                c4.metric("Average score", f"{stats['avg_score']:.2f}")

                # Filter the results table with a pandas boolean mask
                view = st.radio(
                    "Show",
                    ["All", "Potential fraud only", "Safe only"],
                    horizontal=True,
                    key="batch_view",
                )
                shown = batch
                if view == "Potential fraud only":
                    shown = batch[batch["fraud_score"] >= FRAUD_SCORE_THRESHOLD]
                elif view == "Safe only":
                    shown = batch[batch["fraud_score"] < FRAUD_SCORE_THRESHOLD]

                result_table(shown.sort_values("fraud_score", ascending=False))

                b1, b2 = st.columns(2)
                b1.download_button(
                    "Download results as CSV",
                    batch.drop(columns=FLAG_COLUMNS).to_csv(index=False).encode("utf-8"),
                    file_name="fraud_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
                already = uploaded.name in st.session_state.added_files
                if b2.button(
                    "Added to dashboard" if already else "Add to dashboard",
                    disabled=already,
                    use_container_width=True,
                ):
                    add_to_records(batch, f"Batch: {uploaded.name}")
                    st.session_state.added_files.add(uploaded.name)
                    st.rerun()

# ================= Dashboard =================
with tab_dash:
    records = st.session_state.records
    if records.empty:
        st.info("Nothing here yet. Check a transaction or add a batch to see the dashboard.")
    else:
        stats = summary_stats(records)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Transactions checked", stats["total"])
        c2.metric("Flagged as fraud", stats["flagged"])
        c3.metric("Flag rate", f"{stats['flag_rate']:.1f}%")
        c4.metric("Amount at risk", f"₹{stats['flagged_amount']:,.0f}")

        left, right = st.columns(2)
        with left:
            st.subheader("Rules triggered")
            st.bar_chart(rule_trigger_counts(records))
        with right:
            st.subheader("Fraud score distribution")
            st.bar_chart(score_distribution(records))

        st.subheader("Summary by source")
        by_source = (
            records.assign(flagged=records["fraud_score"] >= FRAUD_SCORE_THRESHOLD)
            .groupby("source")
            .agg(
                transactions=("customer_name", "count"),
                flagged=("flagged", "sum"),
                avg_score=("fraud_score", "mean"),
                total_amount=("transaction_amount", "sum"),
            )
            .round({"avg_score": 2})
            .reset_index()
        )
        st.dataframe(pretty(by_source), use_container_width=True, hide_index=True)

        st.subheader("Highest value flagged transactions")
        top_flagged = records[records["fraud_score"] >= FRAUD_SCORE_THRESHOLD].nlargest(
            5, "transaction_amount"
        )
        if top_flagged.empty:
            st.write("No flagged transactions yet.")
        else:
            result_table(top_flagged[["customer_name", "transaction_amount", "fraud_score", "warnings"]])

        st.subheader("All checked transactions")
        result_table(records)

        d1, d2 = st.columns(2)
        d1.download_button(
            "Download all as CSV",
            records.drop(columns=FLAG_COLUMNS).to_csv(index=False).encode("utf-8"),
            file_name="fraud_check_history.csv",
            mime="text/csv",
            use_container_width=True,
        )
        if d2.button("Clear dashboard", use_container_width=True):
            st.session_state.records = pd.DataFrame()
            st.session_state.added_files = set()
            st.rerun()
