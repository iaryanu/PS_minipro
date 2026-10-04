Financial Fraud Detection System (Streamlit + pandas)

A rule-based fraud detection app. Each triggered rule adds 1 to the fraud score,
and a score of 2 or more is flagged as "Potential Fraud Detected".

Rules
- Transaction amount greater than 50,000 rupees
- Account age less than 6 months
- Failed login attempts of 3 or more
- Transaction from an untrusted location

Features
- Single check: enter one transaction and see the score, warnings and status
- Batch check: upload a CSV and score every row at once (vectorised pandas)
- Dashboard: summary metrics, rules-triggered chart, score distribution,
  groupby summary by source, top flagged transactions, CSV download

Where pandas is used (fraud_logic.py and app.py)
- Boolean column per rule using vectorised comparisons
- fraud_score = sum of the rule columns across each row
- str.strip().str.lower() for the location column
- pd.to_numeric(errors="coerce") plus dropna for bad rows
- value_counts().reindex() for the score distribution
- groupby().agg() for the summary by source
- nlargest() for the top flagged transactions
- boolean masks to filter flagged and safe rows
- pd.concat to build the session history
- to_csv for downloads

requirements.txt
- streamlit: the web app framework
- pandas: data handling (numpy is installed automatically as a dependency)

CSV format for batch check (see sample_transactions.csv)
Columns: customer_name, transaction_amount, account_age_months,
failed_attempts, trusted_location (Yes or No)

Project files
- app.py                  Streamlit user interface
- fraud_logic.py          Fraud scoring logic (single and pandas batch)
- test_fraud_logic.py     Unit tests including the five report test cases
- sample_transactions.csv Sample file for the batch tab
- requirements.txt        Python dependencies
- .streamlit/config.toml  Theme settings

How to run
1. Install Python 3.9 or newer.
2. Open a terminal in this folder.
3. (Optional) Create a virtual environment:
     python -m venv venv
     venv\Scripts\activate        (Windows)
     source venv/bin/activate     (Mac/Linux)
4. Install dependencies:
     pip install -r requirements.txt
5. Start the app:
     streamlit run app.py
6. The app opens at http://localhost:8501

Run the tests
     python -m unittest -v

Deploy on Streamlit Community Cloud
Push these files to a GitHub repo (app.py and requirements.txt at the repo root),
then create an app at share.streamlit.io and set the main file to app.py.
