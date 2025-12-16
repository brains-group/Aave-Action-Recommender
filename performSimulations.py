import pickle as pkl
from utils.constants import *
import pandas as pd
import json
import shutil
import os
import sys
import importlib

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
import main as simulator

# Return any previously leftover profiles
for filename in os.listdir(PROFILE_CACHE_DIR):
    shutil.move(
        os.path.join(PROFILE_CACHE_DIR, filename), os.path.expanduser(os.path.join(PROFILES_DIR, filename))
    )

with open(RECOMMENDATIONS_FILE, "rb") as f:
    recommendations = pkl.load(f)

user_profile_generator = UserProfileGenerator(None, WalletInferencer())

for recommendation in recommendations.values():
    user = recommendation["user"]
    user_filename = "user_" + user + ".json"
    user_profile_file = os.path.expanduser(os.path.join(PROFILES_DIR, user_filename))
    if not os.path.exists(user_profile_file):
        print(f"Did not find profile for {user} ({user_profile_file}). Skipping...")
        continue
    user_profile_cache = os.path.join(PROFILE_CACHE_DIR, user_filename)
    with open(user_profile_file, "r") as f:
        user_profile = json.load(f)
    shutil.move(user_profile_file, user_profile_cache)

    user_transactions = user_profile["transactions"]
    user_transactions = [
        user_transaction
        for user_transaction in user_transactions
        if user_transaction["timestamp"] < recommendation["timestamp"]
    ]
    user_transactions.append(user_profile_generator._row_to_transaction(recommendation))
    user_profile["transactions"] = user_transactions

    with open(user_profile_file, "w") as f:
        json.dump(user_profile, f, indent=2)

    simulator.main()

    # Return original profile
    shutil.move(user_profile_cache, user_profile_file)
