"""Run after starting server: python client_excel.py

Default preview does NOT alter server state. To commit actual accepted transactions,
set COMMIT_REAL_TRANSACTIONS = True; never commit a demo batch.
"""
from pathlib import Path
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
INPUT_FILE = ROOT / "Current_1K_From_FundTransfer_History.xlsx"
OUTPUT_FILE = ROOT / "Current_1K_RIVER_API_Predictions.xlsx"
API_URL = "http://127.0.0.1:8002/predict/batch"
COMMIT_REAL_TRANSACTIONS = False

current = pd.read_excel(
    INPUT_FILE,
    dtype={"EVENT_ID": str, "USER_ID": str, "RECEIVER_ID": str, "TO_ACCOUNT_NO": str},
)

# Replace pandas NaN/NaT with JSON null and datetimes with strings.
records = []
for _, series in current.iterrows():
    data = {}
    for key, value in series.items():
        if isinstance(value, pd.Timestamp):
            data[key] = value.isoformat()
        elif isinstance(value, (list, dict)):
            data[key] = value
        elif pd.isna(value):
            data[key] = None
        elif hasattr(value, "item"):
            data[key] = value.item()
        else:
            data[key] = value
    records.append(data)

if not 1 <= len(records) <= 1000:
    raise ValueError("The endpoint accepts 1–1000 transactions")

response = requests.post(
    API_URL,
    params={"commit": str(COMMIT_REAL_TRANSACTIONS).lower()},
    json=records,
    timeout=300,
)
if not response.ok:
    print("API response:", response.text[:5000])
response.raise_for_status()

result = pd.DataFrame(response.json())
if len(result) != len(current):
    raise RuntimeError("Returned prediction count differs from Excel row count")

# API returns results in original input row order despite chronological scoring.
for col in ["RIVER_NORMAL_SCORE", "RIVER_FRAUD_SCORE", "RIVER_PREDICTED_FRAUD", "RIVER_RISK_LEVEL", "user_txn_count_before"]:
    current[col] = result[col].to_numpy()

current.to_excel(OUTPUT_FILE, index=False)
print("Saved:", OUTPUT_FILE.resolve())
print(current["RIVER_PREDICTED_FRAUD"].value_counts(dropna=False))
