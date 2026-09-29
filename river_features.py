"""Normalization and behavioral features matched to River(2).ipynb cells 9,11,12,24,26.

DO NOT change feature names/order without retraining the River model.
"""
from collections import Counter, defaultdict
import math
import re

import numpy as np
import pandas as pd


def clean_text(value):
    if pd.isna(value):
        return "__MISSING__"
    value = str(value).strip().lower()
    if value == "":
        return "__MISSING__"
    return value


def clean_id(value):
    if pd.isna(value):
        return "__MISSING__"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if np.isfinite(value) and value.is_integer():
            return str(int(value))
        return str(value).strip().lower()
    value = str(value).strip().lower()
    if re.fullmatch(r"\d+\.0", value):
        value = value[:-2]
    return value if value else "__MISSING__"


def parse_mixed_datetime_one(value):
    """Single-event variant of the notebook's parse_mixed_datetime."""
    if value is None or pd.isna(value) or str(value).strip() == "":
        raise ValueError("EVENT_TIME cannot be blank")
    if isinstance(value, (int, float, np.integer, np.floating)):
        if not np.isfinite(value):
            raise ValueError("EVENT_TIME must be a valid datetime")
        timestamp = pd.Timestamp("1899-12-30") + pd.to_timedelta(value, unit="D")
    else:
        # ISO yyyy-mm-dd must NOT be parsed day-first: pandas can silently
        # turn 2026-08-12 into 2026-12-08 when dayfirst=True.
        text = str(value).strip()
        if re.match(r"^\d{4}-\d{2}-\d{2}(?:$|[T ])", text):
            timestamp = pd.to_datetime(text, format="ISO8601", errors="coerce")
        else:
            timestamp = pd.to_datetime(value, dayfirst=True, errors="coerce")
    if pd.isna(timestamp):
        raise ValueError("EVENT_TIME must be a valid datetime or Excel date serial")
    timestamp = pd.Timestamp(timestamp)
    if timestamp.tzinfo is not None:
        raise ValueError("Use a timezone-naive EVENT_TIME, consistent with the notebook")
    return timestamp


def new_user_state():
    return {
        "count": 0, "amount_sum": 0.0, "amount_sq_sum": 0.0,
        "max_amount": 0.0, "last_time": None,
        "receivers": Counter(), "accounts": Counter(),
        "purposes": Counter(), "transfer_types": Counter(),
    }


def as_defaultdict(state_mapping):
    """Restore default-factory behavior without pickling notebook __main__ functions."""
    return defaultdict(new_user_state, state_mapping)


def normalize_event(event):
    """Equivalent to the notebook's current_stream mapping for one transaction."""
    amount = float(event.AMOUNT_LCY)
    if not math.isfinite(amount) or amount < 0:
        raise ValueError("AMOUNT_LCY must be a finite non-negative number")
    user = clean_id(event.USER_ID)
    if user == "__MISSING__":
        raise ValueError("USER_ID cannot be blank")
    return {
        "USER_ID": user,
        "DT": parse_mixed_datetime_one(event.EVENT_TIME),
        "AMOUNT": amount,
        "RECEIVER_ID": clean_id(event.RECEIVER_ID),
        "RECEIVER_NM": clean_text(event.RECEIVER_NM),
        "TO_ACCOUNT_NO": clean_id(event.TO_ACCOUNT_NO),
        "PURPOSE": clean_text(event.PURPOSE),
        "TRANSFER_TYPE": clean_text(event.TRANSFER_TYPE),
    }


def build_features(row, states):
    # Exact feature formulas and names from River(2).ipynb cell 24.
    user = row["USER_ID"]
    state = states[user]
    amount = float(row["AMOUNT"])
    dt = row["DT"]
    receiver_key = row["RECEIVER_ID"] if row["RECEIVER_ID"] != "__MISSING__" else row["RECEIVER_NM"]
    account = row["TO_ACCOUNT_NO"]
    purpose = row["PURPOSE"]
    transfer_type = row["TRANSFER_TYPE"]
    count = state["count"]
    if count > 0:
        avg_amount = state["amount_sum"] / count
        variance = max(state["amount_sq_sum"] / count - avg_amount ** 2, 0)
        std_amount = math.sqrt(variance)
    else:
        avg_amount = 0.0
        std_amount = 0.0
    amount_ratio = amount / avg_amount if avg_amount > 0 else 1.0
    if state["last_time"] is not None and pd.notna(dt):
        hours_since_last = max((dt - state["last_time"]).total_seconds() / 3600, 0.0)
    else:
        hours_since_last = 0.0
    if pd.notna(dt):
        hour, day_of_week = int(dt.hour), int(dt.dayofweek)
    else:
        hour, day_of_week = 0, 0
    is_weekend = int(day_of_week in [5, 6])
    is_night = int(hour in [0, 1, 2, 3, 4, 5])
    receiver_count = state["receivers"][receiver_key]
    account_count = state["accounts"][account]
    purpose_count = state["purposes"][purpose]
    transfer_count = state["transfer_types"][transfer_type]
    return {
        "amount": amount,
        "hour_of_day": hour,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "is_night": is_night,
        "user_txn_count_before": count,
        "user_avg_amount_before": avg_amount,
        "user_std_amount_before": std_amount,
        "user_max_amount_before": state["max_amount"],
        "amount_to_user_avg": amount_ratio,
        "hours_since_last_txn": hours_since_last,
        "receiver_count_before": receiver_count,
        "receiver_seen_before": int(receiver_count > 0),
        "new_receiver": int(receiver_count == 0),
        "account_count_before": account_count,
        "account_seen_before": int(account_count > 0),
        "new_account": int(account_count == 0),
        "purpose_count_before": purpose_count,
        "purpose_seen_before": int(purpose_count > 0),
        "transfer_type_count_before": transfer_count,
        "transfer_type_seen_before": int(transfer_count > 0),
        "purpose": purpose,
        "transfer_type": transfer_type,
    }


def update_user_state(row, states):
    # Equivalent to River(2).ipynb cell 26; does NOT call model.learn_one.
    state = states[row["USER_ID"]]
    amount = float(row["AMOUNT"])
    receiver_key = row["RECEIVER_ID"] if row["RECEIVER_ID"] != "__MISSING__" else row["RECEIVER_NM"]
    state["count"] += 1
    state["amount_sum"] += amount
    state["amount_sq_sum"] += amount ** 2
    state["max_amount"] = max(state["max_amount"], amount)
    if pd.notna(row["DT"]):
        state["last_time"] = row["DT"]
    state["receivers"][receiver_key] += 1
    state["accounts"][row["TO_ACCOUNT_NO"]] += 1
    state["purposes"][row["PURPOSE"]] += 1
    state["transfer_types"][row["TRANSFER_TYPE"]] += 1
