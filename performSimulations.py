import pickle as pkl
from utils.constants import *
import pandas as pd
import json
import shutil
import importlib  
WalletInferencer = importlib.import_module("Aave-simulator/profile_gen/wallet_inference").WalletInferencer

# Return any previously leftover profiles
for filename in os.listdir(PROFILE_CACHE_DIR):
    shutil.move(os.path.join(PROFILE_CACHE_DIR, filename), os.path.join(PROFILES_DIR, filename))

with open(RECOMMENDATIONS_FILE, "rb") as f:
    recommendations = pkl.load(f)

wallet_inferencer = WalletInferencer()

for recommendation in recommendations.values():
    user = recommendation["user"]
    user_filename = user + ".json"
    user_profile_file = os.path.join(PROFILES_DIR, user_filename)
    if not os.path.exists(user_profile_file):
        print(f"Did not find profile for {user}. Skipping...")
        continue
    shutil.copyfile(user_profile_file, os.path.join(PROFILE_CACHE_DIR, user_filename))
    with open(user_profile_file, "r") as f:
        user_profile = json.load(f)

    user_transactions = user_profile["transactions"]
    user_transactions = [user_transaction for user_transaction in user_transactions if user_transaction["timestamp"] < recommendation['timestamp']]
    user_transactions.append({
        "action": recommendation["Index Event"],
        "symbol": wallet_inferencer.reserve_mapping.get(recommendation['reserve'], 'USDC'),
        "amount": recommendation["amount"],
        "timestamp": recommendation["timestamp"]
    })


