# River-only Fraud API — VS Code

API for uploaded `River(2).ipynb`, not ECOD or XGBoost. Based on River 0.25.0, supervised logistic regression + online RandomOverSampler, with chronological user behavior.

## 1. Export BEFORE scoring current 1K

In `River(2).ipynb`, run cells through the **historical training loop** (cell index 35 when counting all notebook cells from zero). Immediately after its successful completion, paste/run `notebook_export.py` AS A NOTEBOOK CELL. It needs existing notebook variables `model`, `user_states`, `history_stream`, `pd`. This creates `river_api_bundle.pkl` in notebook's working directory. Copy it into this folder, beside `main_river.py`.

If you previously ran the `for ... current_stream.iterrows()` prediction cell (notebook cell 42), it already mutated `user_states`. Restart the notebook and run through cell 35 again; do NOT export post-current state or 1K predictions will be counted twice. If you decide to export post-current state for *later* unseen events, you must not rescore the same current 1K.

The notebook treats missing historical FRAUD_LABEL as 0. Review your data: unreviewed does not mean legitimate. Use only real verified fraud labels when interpreting the classifier's fraud scores as predictive evidence.

## 2. Folder

```
river_api_vscode/
  main_river.py
  river_features.py
  notebook_export.py (reference cell; not imported by the API)
  client_excel.py
  example_request.json
  requirements.txt
  river_api_bundle.pkl  <-- export from your notebook; not included
```

## 3. Install / run (VS Code terminal)

```powershell
python -m pip install -r requirements.txt
python -m uvicorn main_river:app --reload --port 8002
```

Use the SAME compatible Python/River version as notebook for loading the pickle. If `river` is not installed in the selected VS Code interpreter, select the right interpreter/kernel and install requirements in it.

Open http://127.0.0.1:8002/docs. Test GET `/health`, then POST `/predict` with `example_request.json`. Swagger lists HTTP 422 as a *possible* validation response; success is HTTP 200.

## 4. Live behavior, preview and feedback

* Default POST `/predict` and `/predict/batch` is **preview (`commit=false`)**: nothing persists. Chronological scoring *within* a batch still updates a temporary state as in the notebook.
* To commit a **real accepted transaction** (not demo), use `/predict?commit=true` or `/predict/batch?commit=true`. Event ID + user ID must be unique; duplicate submissions return 409. Committed behavior/model state survives restarts in `river_api_runtime.pkl`.
* To make the classifier learn **only a verified label**, POST `/feedback` for a previously committed event, e.g. `{"USER_ID":"zubair","EVENT_ID":"LIVE001","FRAUD_LABEL":1}`. This calls `model.learn_one`, and does NOT count the transaction twice in behavior history. Do NOT feed an unverified model prediction back as a label.
* A current event earlier than `trained_through` in the bundle is rejected because the fitted model has already seen future historical data. Chronological order is also enforced relative to committed user activity.
* To discard local testing commitments, stop the server, delete `river_api_runtime.pkl`, then restart. This returns to the exported model bundle. Back up real activity before doing so.

Do not expose this demonstration server publicly; /feedback is unauthenticated. Production requires auth, TLS, access controls, logging, durable storage, versioned models and careful temporal evaluation.

## 5. Full current Excel

Place `Current_1K_From_FundTransfer_History.xlsx` in this folder. First terminal runs server; a second terminal runs:

```powershell
python client_excel.py
```

Output: `Current_1K_RIVER_API_Predictions.xlsx`. This defaults to preview, so it is safe to test repeatedly without accumulating demo events. Set `COMMIT_REAL_TRANSACTIONS = True` only for real transactions that should permanently alter user behavior state.

## 6. Notebook's exact output columns

- `RIVER_NORMAL_SCORE`
- `RIVER_FRAUD_SCORE`
- `RIVER_PREDICTED_FRAUD`
- `RIVER_RISK_LEVEL` (`HIGH` >= 0.75, `MEDIUM` >= 0.50, else `LOW`)

The score is the River model's class-1 estimate, not a guarantee of calibrated fraud probability or confirmed fraud. If historical `FRAUD_LABEL` is simulated, treat results as a demonstration only.
