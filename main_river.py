"""Run: python -m uvicorn main_river:app --reload --port 8002

Local demonstration API. Do not expose /feedback on a public network without
user authentication, TLS, rate limiting and an audited label-verification flow.
"""
from __future__ import annotations

import copy
import math
import os
import pickle
import threading
from pathlib import Path
from typing import Any, Annotated

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from river_features import as_defaultdict, build_features, normalize_event, update_user_state, clean_id

ROOT = Path(__file__).resolve().parent
BUNDLE_PATH = ROOT / "river_api_bundle.pkl"
RUNTIME_PATH = ROOT / "river_api_runtime.pkl"

if not BUNDLE_PATH.is_file():
    raise RuntimeError(
        f"Missing {BUNDLE_PATH}. Run notebook_export.py cell after historical River "
        "training, then put river_api_bundle.pkl beside main_river.py."
    )

# Load ONLY a pickle you created and trust. Match notebook's River version.
with BUNDLE_PATH.open("rb") as f:
    bundle = pickle.load(f)

if not {"model", "user_states", "trained_through"} <= set(bundle):
    raise RuntimeError("River bundle lacks model / user_states / trained_through")

trained_through = pd.Timestamp(bundle["trained_through"])
lock = threading.RLock()

# This file preserves committed predictions and verified feedback across restarts.
# Delete it with the server stopped to return to the original exported bundle.
if RUNTIME_PATH.is_file():
    with RUNTIME_PATH.open("rb") as f:
        runtime = pickle.load(f)
    model = runtime["model"]
    user_states = as_defaultdict(runtime["user_states"])
    seen_events = runtime["seen_events"]
    pending_features = runtime["pending_features"]
    learned_events = runtime["learned_events"]
else:
    model = bundle["model"]
    user_states = as_defaultdict(bundle["user_states"])
    seen_events = set()
    pending_features = {}
    learned_events = set()

app = FastAPI(title="River Fraud Prediction API (No ECOD / XGBoost)", version="1.0")


class Transaction(BaseModel):
    model_config = ConfigDict(extra="ignore")

    EVENT_ID: str | int = Field(description="Unique transaction ID")
    USER_ID: str | int | float
    EVENT_TIME: str | int | float
    AMOUNT_LCY: float = Field(ge=0, allow_inf_nan=False)
    RECEIVER_ID: str | int | float | None = None
    RECEIVER_NM: str | None = None
    TO_ACCOUNT_NO: str | int | float | None = None
    PURPOSE: str | None = None
    TRANSFER_TYPE: str | None = None


class VerifiedFeedback(BaseModel):
    USER_ID: str | int | float
    EVENT_ID: str | int
    FRAUD_LABEL: int = Field(ge=0, le=1, description="Verified: 0=legitimate, 1=fraud")


def event_key(user, event_id):
    event_id = str(event_id).strip()
    if not event_id:
        raise HTTPException(status_code=422, detail="EVENT_ID cannot be blank")
    return (user, event_id)


def save_runtime():
    # Atomic replacement prevents a half-written runtime file.
    tmp = RUNTIME_PATH.with_suffix(".tmp")
    with tmp.open("wb") as f:
        pickle.dump({
            "model": model,
            "user_states": dict(user_states),
            "seen_events": seen_events,
            "pending_features": pending_features,
            "learned_events": learned_events,
        }, f)
    os.replace(tmp, RUNTIME_PATH)


def score_transactions(events: list[Transaction], commit: bool):
    global user_states, seen_events, pending_features
    if not events:
        raise HTTPException(status_code=422, detail="Provide at least one transaction")
    if len(events) > 1000:
        raise HTTPException(status_code=422, detail="Maximum 1000 transactions per batch")

    with lock:
        # Work on copies so the whole batch is accepted or rejected together.
        temp_states = as_defaultdict(copy.deepcopy(dict(user_states)))
        temp_seen = seen_events.copy()
        temp_pending = pending_features.copy()
        normalized = []
        input_keys = set()

        for idx, event in enumerate(events):
            try:
                row = normalize_event(event)
            except (ValueError, OverflowError, TypeError) as exc:
                raise HTTPException(status_code=422, detail=f"Row {idx + 1}: {exc}") from exc
            key = event_key(row["USER_ID"], event.EVENT_ID)
            if key in input_keys:
                raise HTTPException(status_code=409, detail=f"Duplicate EVENT_ID in batch at row {idx + 1}")
            if commit and key in temp_seen:
                raise HTTPException(status_code=409, detail=f"Already committed: {key}")
            if row["DT"] < trained_through:
                raise HTTPException(
                    status_code=422,
                    detail=f"Row {idx + 1} is earlier than historical training cutoff "
                           f"{trained_through.isoformat()}; export a correctly time-scoped model."
                )
            input_keys.add(key)
            normalized.append((idx, event, row, key))

        # Notebook also sorts current transactions by (DT, original ROW_ID).
        normalized.sort(key=lambda item: (item[2]["DT"], item[0]))
        results = [None] * len(events)
        for idx, event, row, key in normalized:
            last_time = temp_states[row["USER_ID"]]["last_time"]
            if last_time is not None and row["DT"] < last_time:
                raise HTTPException(
                    status_code=422,
                    detail=f"Row {idx + 1}: transaction is older than this user's latest "
                           "committed transaction. Process in chronological order."
                )
            x = build_features(row, temp_states)
            proba = model.predict_proba_one(x) or {}
            fraud_score = float(proba.get(True, 0.0))
            normal_score = float(proba.get(False, 0.0))
            if not (math.isfinite(fraud_score) and math.isfinite(normal_score)):
                raise HTTPException(status_code=500, detail="Model returned invalid probability")
            prediction = model.predict_one(x)
            prediction = False if prediction is None else bool(prediction)
            risk = "HIGH" if fraud_score >= 0.75 else "MEDIUM" if fraud_score >= 0.50 else "LOW"
            results[idx] = {
                "event_id": str(event.EVENT_ID),
                "user_id": row["USER_ID"],
                "event_time": row["DT"].isoformat(),
                "amount": row["AMOUNT"],
                "user_txn_count_before": x["user_txn_count_before"],
                "RIVER_NORMAL_SCORE": normal_score,
                "RIVER_FRAUD_SCORE": fraud_score,
                "RIVER_PREDICTED_FRAUD": int(prediction),
                "RIVER_RISK_LEVEL": risk,
                "behavior_committed": commit,
                "message": "Model prediction; fraud is not confirmed without review.",
            }
            # Exactly as notebook cell 42: behavior updates; model does not learn
            # until a separately verified FRAUD_LABEL is supplied via /feedback.
            update_user_state(row, temp_states)
            if commit:
                temp_seen.add(key)
                temp_pending[key] = x

        if commit:
            user_states = temp_states
            seen_events = temp_seen
            pending_features = temp_pending
            save_runtime()
        return results


@app.get("/health")
def health():
    with lock:
        return {
            "status": "ready", "model": "River (no ECOD or XGBoost)",
            "historical_transactions": bundle.get("historical_count"),
            "historical_users": len(bundle["user_states"]),
            "trained_through": trained_through.isoformat(),
            "committed_events": len(seen_events),
            "awaiting_verified_labels": len(pending_features),
        }


@app.post("/predict")
def predict(event: Transaction, commit: Annotated[bool, Query(description="True only for a real accepted transaction; demo defaults to preview")] = False):
    return score_transactions([event], commit)[0]


@app.post("/predict/batch")
def predict_batch(events: list[Transaction], commit: Annotated[bool, Query(description="Persist a real accepted batch; default preview avoids duplicate history")] = False):
    return score_transactions(events, commit)


@app.post("/feedback")
def feedback(item: VerifiedFeedback):
    """Only after a human/investigation confirms the true fraud label."""
    global model
    key = event_key(clean_id(item.USER_ID), item.EVENT_ID)
    with lock:
        if key in learned_events:
            raise HTTPException(status_code=409, detail="This event has already been learned")
        if key not in pending_features:
            raise HTTPException(
                status_code=404,
                detail="No committed prediction for that event. Predict with ?commit=true first."
            )
        model.learn_one(pending_features[key], bool(item.FRAUD_LABEL))
        pending_features.pop(key)
        learned_events.add(key)
        save_runtime()
        return {
            "status": "updated", "event_id": key[1],
            "user_id": key[0], "verified_label": item.FRAUD_LABEL,
            "message": "River classifier learned from verified feedback. Behavior not double-counted."
        }
