"""Rule-based financial fraud detection logic (no Streamlit dependency).

Two entry points:
- evaluate_transaction(): scores a single transaction (used for the single check form)
- evaluate_dataframe():   scores many transactions at once with vectorised pandas code
"""

from dataclasses import dataclass, field
from typing import List, Tuple

import pandas as pd

HIGH_AMOUNT_THRESHOLD = 50000   # rupees
NEW_ACCOUNT_MONTHS = 6          # months
MAX_FAILED_ATTEMPTS = 3         # attempts
FRAUD_SCORE_THRESHOLD = 2       # score at which a transaction is flagged

REQUIRED_COLUMNS = [
    "customer_name",
    "transaction_amount",
    "account_age_months",
    "failed_attempts",
    "trusted_location",
]

# One boolean column per rule, and the warning text shown for each
RULE_LABELS = {
    "high_amount": "High transaction amount.",
    "new_account": "New account detected.",
    "failed_logins": "Multiple failed login attempts.",
    "untrusted_location": "Transaction from an untrusted location.",
}
FLAG_COLUMNS = list(RULE_LABELS.keys())

STATUS_FRAUD = "Potential Fraud Detected"
STATUS_SAFE = "Transaction is Safe"


# ---------------------------------------------------------------- single transaction
@dataclass
class FraudResult:
    customer_name: str
    fraud_score: int
    warnings: List[str] = field(default_factory=list)

    @property
    def is_fraud(self) -> bool:
        return self.fraud_score >= FRAUD_SCORE_THRESHOLD

    @property
    def status(self) -> str:
        return STATUS_FRAUD if self.is_fraud else STATUS_SAFE


def evaluate_transaction(
    customer_name: str,
    transaction_amount: float,
    account_age_months: int,
    failed_attempts: int,
    trusted_location: str,
) -> FraudResult:
    """Apply the four fraud rules to one transaction."""
    score = 0
    warnings: List[str] = []

    if transaction_amount > HIGH_AMOUNT_THRESHOLD:
        score += 1
        warnings.append(RULE_LABELS["high_amount"])

    if account_age_months < NEW_ACCOUNT_MONTHS:
        score += 1
        warnings.append(RULE_LABELS["new_account"])

    if failed_attempts >= MAX_FAILED_ATTEMPTS:
        score += 1
        warnings.append(RULE_LABELS["failed_logins"])

    if str(trusted_location).strip().lower() == "no":
        score += 1
        warnings.append(RULE_LABELS["untrusted_location"])

    return FraudResult(customer_name=customer_name, fraud_score=score, warnings=warnings)


# ---------------------------------------------------------------- many transactions
def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case column names and replace spaces with underscores."""
    return df.rename(columns=lambda c: str(c).strip().lower().replace(" ", "_"))


def evaluate_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Score every row of a DataFrame using vectorised pandas operations.

    Returns (result_df, dropped_rows). Rows with a missing name or a non-numeric
    amount, age or attempts value are dropped and counted in dropped_rows.
    """
    df = normalise_columns(df)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("Missing required column(s): " + ", ".join(missing))

    out = df[REQUIRED_COLUMNS].copy()

    # Coerce numeric columns; anything unreadable becomes NaN
    for col in ["transaction_amount", "account_age_months", "failed_attempts"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    before = len(out)
    out = out.dropna(subset=["customer_name"] + ["transaction_amount", "account_age_months", "failed_attempts"])
    out = out.reset_index(drop=True)
    dropped = before - len(out)

    # One boolean column per rule
    out["high_amount"] = out["transaction_amount"] > HIGH_AMOUNT_THRESHOLD
    out["new_account"] = out["account_age_months"] < NEW_ACCOUNT_MONTHS
    out["failed_logins"] = out["failed_attempts"] >= MAX_FAILED_ATTEMPTS
    out["untrusted_location"] = (
        out["trusted_location"].astype(str).str.strip().str.lower().eq("no")
    )

    # Score = number of rules triggered in each row
    out["fraud_score"] = out[FLAG_COLUMNS].sum(axis=1).astype(int)
    out["status"] = (out["fraud_score"] >= FRAUD_SCORE_THRESHOLD).map(
        {True: STATUS_FRAUD, False: STATUS_SAFE}
    )

    # Readable list of triggered warnings per row
    if out.empty:
        out["warnings"] = pd.Series(dtype="object")
    else:
        out["warnings"] = out[FLAG_COLUMNS].apply(
            lambda row: "; ".join(RULE_LABELS[c] for c in FLAG_COLUMNS if row[c]),
            axis=1,
        )

    return out, dropped


def rule_trigger_counts(result: pd.DataFrame) -> pd.Series:
    """How many transactions triggered each rule."""
    counts = result[FLAG_COLUMNS].sum().astype(int)
    counts.index = [RULE_LABELS[c].rstrip(".") for c in counts.index]
    return counts


def score_distribution(result: pd.DataFrame) -> pd.Series:
    """Number of transactions at each fraud score from 0 to 4."""
    return (
        result["fraud_score"]
        .value_counts()
        .reindex(range(len(FLAG_COLUMNS) + 1), fill_value=0)
        .sort_index()
    )


def summary_stats(result: pd.DataFrame) -> dict:
    """Headline numbers for a set of evaluated transactions."""
    total = len(result)
    flagged = int((result["fraud_score"] >= FRAUD_SCORE_THRESHOLD).sum())
    return {
        "total": total,
        "flagged": flagged,
        "flag_rate": (flagged / total * 100) if total else 0.0,
        "avg_score": float(result["fraud_score"].mean()) if total else 0.0,
        "total_amount": float(result["transaction_amount"].sum()) if total else 0.0,
        "flagged_amount": float(
            result.loc[result["fraud_score"] >= FRAUD_SCORE_THRESHOLD, "transaction_amount"].sum()
        ) if total else 0.0,
    }
