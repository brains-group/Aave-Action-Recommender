import os
import json

seed = 42

DATA_PATH = os.environ.get("AAVE_SURVIVAL_DATA", "./data/")
CACHE_DIR = "./cache/"
if os.environ.get("AAVE_SURVIVAL_DATA"):
    import hashlib
    from pathlib import Path
    source = Path(DATA_PATH).resolve()
    manifest = source / "manifest.json"
    if not manifest.is_file():
        raise ValueError("AAVE_SURVIVAL_DATA requires a pipeline manifest.json")
    identity = hashlib.sha256(str(source).encode()+manifest.read_bytes()).hexdigest()[:16]
    CACHE_DIR = str(Path(CACHE_DIR) / "survival" / identity)

os.makedirs(CACHE_DIR, exist_ok=True)
MODEL_CACHE_DIR = os.path.join(CACHE_DIR, "models")
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
DATA_CACHE_DIR = os.path.join(CACHE_DIR, "data")
os.makedirs(DATA_CACHE_DIR, exist_ok=True)
RESULTS_CACHE_DIR = os.path.join(CACHE_DIR, "results")
os.makedirs(RESULTS_CACHE_DIR, exist_ok=True)
SIMULATION_RESULTS_CACHE_DIR = os.path.join(CACHE_DIR, "simulation_results")
os.makedirs(SIMULATION_RESULTS_CACHE_DIR, exist_ok=True)

EVENTS = ["Deposit", "Withdraw", "Repay", "Borrow", "Liquidated"]

DEFAULT_TIME_DELTA_SECONDS = 600

# Dust liquidation prevention thresholds
# Based on analysis: dust liquidations occur when positions are extremely small
MIN_RECOMMENDATION_DEBT_USD = 1.0  # Minimum debt to avoid dust liquidation
MIN_RECOMMENDATION_COLLATERAL_USD = 10.0  # Minimum collateral to avoid dust liquidation
MIN_RECOMMENDATION_AMOUNT = 50.0  # Minimum recommended amount (USD equivalent) to prevent creating tiny positions

RECOMMENDATIONS_FILE = os.path.join(CACHE_DIR, "recommendations.pkl")

PROFILES_DIR = "./profiles/"

LABEL_TIME, LABEL_EVENT = "timeDiff", "status"

DEFAULT_LOOKAHEAD_DAYS = 7
DEFAULT_LOOKAHEAD_SECONDS = 86400 * DEFAULT_LOOKAHEAD_DAYS
