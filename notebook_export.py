"""PASTE THE CELL BELOW into River(2).ipynb IMMEDIATELY AFTER CELL 35.

Cell 35 trains River on history. Run this BEFORE cell 42 predicts current 1K,
since cell 42 mutates user_states. If cell 42 has already run, restart
and rerun through the historical training loop before exporting.
"""

import copy
import pickle
from pathlib import Path

# Execute this in the training notebook (not as a standalone script).
river_api_bundle = {
    "model": copy.deepcopy(model),
    # Convert defaultdict to plain dict so unpickling needs no notebook function.
    "user_states": copy.deepcopy(dict(user_states)),
    "trained_through": pd.Timestamp(history_stream["DT"].max()).isoformat(),
    "historical_count": len(history_stream),
    "model_type": "River online supervised fraud classifier",
    "risk_high": 0.75,
    "risk_medium": 0.50,
}

MODEL_PATH = Path.cwd() / "river_api_bundle.pkl"
with MODEL_PATH.open("wb") as f:
    pickle.dump(river_api_bundle, f)

print("River API bundle created:", MODEL_PATH.resolve())
print("File exists:", MODEL_PATH.exists())
