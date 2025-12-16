import os

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

PROFILES_DIR = "./Aave-Simulator/results/profile-generation/"
PROFILE_CACHE_DIR = os.path.join(CACHE_DIR, "profile_backups")
os.makedirs(PROFILE_CACHE_DIR, exist_ok=True)
