"""Tests based on the five test cases in the project report. Run: python -m unittest -v"""

import unittest

import pandas as pd

from fraud_logic import (
    evaluate_dataframe,
    evaluate_transaction,
    rule_trigger_counts,
    score_distribution,
    summary_stats,
)

CASES = [
    ("Rahul", 75000, 3, 4, "No", 4, "Potential Fraud Detected"),
    ("Priya", 12000, 18, 0, "Yes", 0, "Transaction is Safe"),
    ("Amit", 60000, 24, 1, "Yes", 1, "Transaction is Safe"),
    ("Neha", 30000, 2, 3, "Yes", 2, "Potential Fraud Detected"),
    ("Arjun", 55000, 12, 5, "No", 3, "Potential Fraud Detected"),
]
COLS = ["customer_name", "transaction_amount", "account_age_months", "failed_attempts", "trusted_location"]


class TestSingle(unittest.TestCase):
    def test_report_cases(self):
        for name, amt, age, fails, loc, score, status in CASES:
            r = evaluate_transaction(name, amt, age, fails, loc)
            self.assertEqual(r.fraud_score, score, name)
            self.assertEqual(r.status, status, name)

    def test_no_warnings_when_safe(self):
        self.assertEqual(evaluate_transaction("Priya", 12000, 18, 0, "Yes").warnings, [])

    def test_boundaries(self):
        self.assertEqual(evaluate_transaction("a", 50000, 6, 2, "Yes").fraud_score, 0)
        self.assertEqual(evaluate_transaction("a", 50001, 5, 3, "Yes").fraud_score, 3)

    def test_location_case_insensitive(self):
        for v in ("NO", "no", "No", " no "):
            self.assertEqual(evaluate_transaction("a", 0, 10, 0, v).fraud_score, 1)


class TestDataFrame(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame([c[:5] for c in CASES], columns=COLS)

    def test_report_cases(self):
        result, dropped = evaluate_dataframe(self.df)
        self.assertEqual(dropped, 0)
        self.assertEqual(result["fraud_score"].tolist(), [c[5] for c in CASES])
        self.assertEqual(result["status"].tolist(), [c[6] for c in CASES])

    def test_matches_single_function(self):
        result, _ = evaluate_dataframe(self.df)
        for _, row in result.iterrows():
            single = evaluate_transaction(
                row["customer_name"], row["transaction_amount"], row["account_age_months"],
                row["failed_attempts"], row["trusted_location"],
            )
            self.assertEqual(row["fraud_score"], single.fraud_score)
            self.assertEqual(row["warnings"], "; ".join(single.warnings))

    def test_column_names_are_normalised(self):
        df = self.df.rename(columns={"customer_name": "Customer Name", "trusted_location": " TRUSTED_LOCATION "})
        result, _ = evaluate_dataframe(df)
        self.assertEqual(len(result), 5)

    def test_missing_column_raises(self):
        with self.assertRaises(ValueError):
            evaluate_dataframe(self.df.drop(columns=["failed_attempts"]))

    def test_bad_rows_are_dropped(self):
        df = pd.concat(
            [self.df, pd.DataFrame([["Bad", "abc", 5, 1, "Yes"], [None, 100, 5, 1, "Yes"]], columns=COLS)],
            ignore_index=True,
        )
        result, dropped = evaluate_dataframe(df)
        self.assertEqual(dropped, 2)
        self.assertEqual(len(result), 5)

    def test_empty_dataframe(self):
        result, dropped = evaluate_dataframe(pd.DataFrame(columns=COLS))
        self.assertTrue(result.empty)
        self.assertEqual(summary_stats(result)["total"], 0)

    def test_summary_helpers(self):
        result, _ = evaluate_dataframe(self.df)
        stats = summary_stats(result)
        self.assertEqual(stats["total"], 5)
        self.assertEqual(stats["flagged"], 3)
        self.assertAlmostEqual(stats["flag_rate"], 60.0)
        self.assertAlmostEqual(stats["avg_score"], 2.0)
        self.assertEqual(score_distribution(result).tolist(), [1, 1, 1, 1, 1])
        counts = rule_trigger_counts(result)
        self.assertEqual(int(counts["High transaction amount"]), 3)

    def test_sample_csv_loads(self):
        result, dropped = evaluate_dataframe(pd.read_csv("sample_transactions.csv"))
        self.assertEqual(dropped, 0)
        self.assertEqual(len(result), 12)


if __name__ == "__main__":
    unittest.main()
