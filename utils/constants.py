import os
import json

DATA_PATH = "./data/"
CACHE_DIR = "./cache/"
os.makedirs(CACHE_DIR, exist_ok=True)
MODEL_CACHE_DIR = os.path.join(CACHE_DIR, "models")
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
DATA_CACHE_DIR = os.path.join(CACHE_DIR, "data")
os.makedirs(DATA_CACHE_DIR, exist_ok=True)
RESULTS_CACHE_DIR = os.path.join(CACHE_DIR, "results")
os.makedirs(RESULTS_CACHE_DIR, exist_ok=True)

EVENTS = ["Deposit", "Withdraw", "Repay", "Borrow", "Liquidated"]

RECOMMENDATIONS_FILE = os.path.join(CACHE_DIR, "recommendations.pkl")

if os.path.exists("./Aave-Simulator/config.json"):
    with open("./Aave-Simulator/config.json", 'r') as f:
        PROFILES_DIR = json.load(f)["sample_user_profile_path"]
PROFILE_CACHE_DIR = os.path.join(CACHE_DIR, "profile_backups")
os.makedirs(PROFILE_CACHE_DIR, exist_ok=True)
