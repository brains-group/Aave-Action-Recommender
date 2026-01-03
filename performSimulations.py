import pickle as pkl
from utils.constants import *
import pandas as pd
import json
import shutil
import os
import sys
import importlib

# TODO: Check how many of the recommendations/liquidation risks line up with what the data had

# Make the bundled Aave-Simulator directory importable (it's next to this file)
this_dir = os.path.dirname(os.path.realpath(__file__))
aave_sim_path = os.path.join(this_dir, "Aave-Simulator")
if aave_sim_path not in sys.path:
    sys.path.insert(0, aave_sim_path)

# Import the wallet inferencer and the simulator `main` module from the
# Aave-Simulator package directory. The previous code tried to use
# importlib.import_module with a path containing a slash/hyphen, which is
# not a valid module name.
from profile_gen.user_profile_generator import UserProfileGenerator
from profile_gen.wallet_inference import WalletInferencer
from tools.run_single_simulation import run_simulation

# # Return any previously leftover profiles
# for filename in os.listdir(PROFILE_CACHE_DIR):
#     shutil.move(
#         os.path.join(PROFILE_CACHE_DIR, filename), os.path.expanduser(os.path.join(PROFILES_DIR, filename))
#     )

with open(RECOMMENDATIONS_FILE, "rb") as f:
    recommendations = pkl.load(f)

user_profile_generator = UserProfileGenerator(None, WalletInferencer())

for recommendation, liquidation_info in recommendations.values():
    is_at_risk = liquidation_info["is_at_risk"]
    is_at_immediate_risk = liquidation_info["is_at_immediate_risk"]
    most_recent_predictions = liquidation_info["most_recent_predictions"]
    trend_slopes = liquidation_info["trend_slopes"]
    user = recommendation["user"]
    user_filename = "user_" + user + ".json"
    user_profile_file = os.path.expanduser(os.path.join(PROFILES_DIR, user_filename))
    if not os.path.exists(user_profile_file):
        print(f"Did not find profile for {user} ({user_profile_file}). Skipping...")
        continue
    with open(user_profile_file, "r") as f:
        user_profile = json.load(f)

    user_transactions = user_profile["transactions"]
    new_user_transactions = [
        user_transaction
        for user_transaction in user_transactions
        if user_transaction["timestamp"] < recommendation["timestamp"]
    ]
    user_profile["transactions"] = new_user_transactions
    results_without_recommendation = run_simulation(
        user_profile, lookahead_seconds=10000
    )

    user_profile["transactions"].append(
        user_profile_generator._row_to_transaction(recommendation)
    )
    results_with_recommendation = run_simulation(user_profile, lookahead_seconds=10000)
