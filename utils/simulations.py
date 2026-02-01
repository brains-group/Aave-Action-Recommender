import bisect
import json
from pathlib import Path
import pickle as pkl
import os
import sys

import numpy as np

from performSimulations import DEFAULT_LOOKAHEAD_SECONDS
from utils.logger import logger
from utils.constants import (
    DEFAULT_TIME_DELTA_SECONDS,
    MIN_RECOMMENDATION_DEBT_USD,
    SIMULATION_RESULTS_CACHE_DIR,
    PROFILES_DIR,
)

# Make the bundled Aave-Simulator directory importable (it's next to this file)
this_dir = os.path.dirname(os.path.realpath(__file__))
aave_sim_path = os.path.join(this_dir, "..", "Aave-Simulator")
if aave_sim_path not in sys.path:
    sys.path.insert(0, aave_sim_path)

from tools.run_single_simulation import run_simulation
from simulator.utils import get_price_history


def get_simulation_outcome(recommendation, suffix, **passed_args):
    """Load cached simulation results or compute and cache them."""
    key = f"{recommendation['user']}_{int(recommendation.get('timestamp', 0))}_{suffix}"
    results_cache_file = Path(SIMULATION_RESULTS_CACHE_DIR) / f"{key}.pkl"
    # logger.debug("Checkpoint 7.6")

    # Try to load from cache
    if results_cache_file.exists():
        try:
            with open(results_cache_file, "rb") as f:
                return pkl.load(f)
        except Exception as e:
            logger.debug(f"Failed to load cache {results_cache_file}: {e}")
    # logger.debug("Checkpoint 7.7")

    # Cache miss - compute results
    results = run_simulation(**passed_args)
    # logger.debug("Checkpoint 7.8")

    # Save to cache (best effort)
    try:
        Path(SIMULATION_RESULTS_CACHE_DIR).mkdir(parents=True, exist_ok=True)
        with open(results_cache_file, "wb") as f:
            pkl.dump(results, f)
    except Exception as e:
        logger.debug(f"Failed to save cache {results_cache_file}: {e}")
    # logger.debug("Checkpoint 7.9")

    return results


def updateAmountOrUSD(recommendation, amount=None, amountUSD=None):
    if amount is None and amountUSD is None:
        return
    price = recommendation["priceInUSD"]
    recommendation["amount"] = amount if amount is not None else amountUSD / price
    recommendation["amountUSD"] = amountUSD if amountUSD is not None else amount * price

    # Use numpy.log1p when available; fall back to math.log1p if `np` is shadowed.
    try:
        log1p_fn = np.log1p
    except Exception:
        import math

        log1p_fn = math.log1p
    recommendation["logAmountUSD"] = float(log1p_fn(float(recommendation["amountUSD"])))
    recommendation["logAmount"] = float(log1p_fn(float(recommendation["amount"])))


def would_create_dust_position(recommendation, results_without_recommendation):
    total_debt_usd = results_without_recommendation["final_state"]["total_debt_usd"]
    amount_usd = recommendation["amountUSD"]
    estimated_remaining_debt = max(0, total_debt_usd - amount_usd)
    return recommendation["Index Event"] == "repay" and (
        estimated_remaining_debt > 0
        and estimated_remaining_debt < MIN_RECOMMENDATION_DEBT_USD
    )


def update_recommendation_if_necessary(recommendation, results_without_recommendation):
    if recommendation["Index Event"] != "repay":
        return recommendation

    # Get symbol from recommendation - try 'symbol' first, then 'reserve' as fallback
    symbol = recommendation.get("symbol") or recommendation.get("reserve")
    if not symbol:
        logger.warning(
            f"Recommendation missing 'symbol' and 'reserve' fields for repay action. "
            f"Available keys: {list(recommendation.keys())}. Skipping update."
        )
        return recommendation

    # walletSymbolAmount = wallet_balances.get(symbol, 0)
    # if walletSymbolAmount < recommendation['amount']:
    #     updateAmountOrUSD(recommendation, amount = walletSymbolAmount)
    total_debt_usd = results_without_recommendation["final_state"]["total_debt_usd"]
    amount_usd = recommendation["amountUSD"]
    estimated_remaining_debt = max(0, total_debt_usd - amount_usd)
    # if not (
    #     estimated_remaining_debt > 0
    #     and estimated_remaining_debt < MIN_RECOMMENDATION_DEBT_USD
    # ):
    #     return recommendation
    # elif (
    #     recommendation["Index Event"] != "repay"
    # ):  # Comment out these if and return statements for potential performance increase for deposit recommendations
    #     return None
    if recommendation["Index Event"] != "repay" and (
        estimated_remaining_debt > 0
        and estimated_remaining_debt < MIN_RECOMMENDATION_DEBT_USD
    ):  # Comment out these if and return statements for potential performance increase for deposit recommendations
        return None

    wallet_balances = results_without_recommendation["final_state"]["wallet_balances"]
    maxWalletSymbol = max(wallet_balances, key=wallet_balances.get)
    maxWalletValue = wallet_balances.get(maxWalletSymbol, 0)

    recommendation["symbol"] = recommendation["reserve"] = maxWalletSymbol
    updateAmountOrUSD(recommendation, amount=maxWalletValue)

    # updateAmountOrUSD(recommendation, amountUSD = total_debt_usd*1.01)

    # if walletSymbolAmount < recommendation['amount']:
    #     updateAmountOrUSD(recommendation, amount = walletSymbolAmount)

    return recommendation


def get_user_profile(transaction):
    user = transaction.get("user")

    # Search for profile in both non_liquidated_profiles and liquidated_profiles subdirectories
    # PROFILES_DIR should point to a directory containing both subdirectories
    profiles_base = Path(PROFILES_DIR).expanduser()

    # Primary search paths: check the standard structure first
    # 1. non_liquidated_profiles/profiles/
    # 2. liquidated_profiles/profiles/
    search_paths = [
        profiles_base / "non_liquidated_profiles" / "profiles" / f"user_{user}.json",
        profiles_base / "liquidated_profiles" / "profiles" / f"user_{user}.json",
    ]

    # Fallback: try directly in PROFILES_DIR (backward compatibility)
    search_paths.append(profiles_base / f"user_{user}.json")

    user_profile_file = None
    for search_path in search_paths:
        if search_path.exists():
            user_profile_file = search_path
            logger.debug(f"Found profile for user {user} at: {user_profile_file}")
            break

    if user_profile_file is None:
        # Only show first few paths in warning to avoid cluttering logs
        paths_displayed = "\n".join([f"  - {p}" for p in search_paths[:3]])
        logger.warning(
            f"Did not find profile for user {user} in any of these locations:\n{paths_displayed}\n"
            f"  ... (checked {len(search_paths)} total locations)\n"
            f"Skipping..."
        )
        return None

    try:
        with user_profile_file.open("r") as f:
            user_profile = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in profile file {user_profile_file}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error reading profile file {user_profile_file}: {e}")
        return None

    if not isinstance(user_profile, dict):
        logger.error(
            f"Invalid profile format for user {user}: expected dict, got {type(user_profile)}"
        )
        return None

    if "transactions" not in user_profile:
        logger.warning(f"Profile for user {user} missing 'transactions' field")
        user_profile["transactions"] = []

    return user_profile


def get_limited_user_profile(recommendation, return_extras=False):
    user_profile = get_user_profile(recommendation)
    if user_profile is None:
        logger.warning("User profile not found.")
        return None
    user = user_profile.get("user_address")

    recommendation_timestamp = recommendation.get("timestamp")
    if recommendation_timestamp is None:
        logger.warning(f"Recommendation missing 'timestamp' field for user {user}")
        return None
    cutoff_timestamp = recommendation_timestamp - DEFAULT_TIME_DELTA_SECONDS

    original_transactions = user_profile.get("transactions", [])
    if not original_transactions:
        logger.warning(f"No transactions found in profile for user {user}")

    # Filter transactions in a single pass: historical (<= timestamp) vs future (> timestamp)
    # Note: We use <= for historical to include transactions at exactly the recommendation time
    historical_transactions = []
    future_transactions = []
    for tx in original_transactions:
        if not isinstance(tx, dict):
            continue
        tx_timestamp = tx.get("timestamp", 0)
        if tx_timestamp <= cutoff_timestamp:
            historical_transactions.append(tx)
        else:
            # Don't bother with this if not being returned
            if return_extras:
                future_transactions.append(tx)

    # For "without recommendation" simulation: use only historical transactions
    # (No need to copy - we'll create a deepcopy later for "with" profile)
    user_profile["transactions"] = historical_transactions

    if return_extras:
        if future_transactions:
            future_transactions.sort(key=lambda x: x.get("timestamp", 0))

        # Calculate lookahead: simulate forward from recommendation time to check for liquidation
        # Use the last future transaction timestamp if available, otherwise default to 7 days ahead
        if future_transactions:
            last_future_tx_timestamp = future_transactions[-1].get("timestamp")
            if last_future_tx_timestamp:
                lookahead_seconds = (
                    max(1, int(last_future_tx_timestamp - cutoff_timestamp)) * 2
                )
            else:
                lookahead_seconds = DEFAULT_LOOKAHEAD_SECONDS
        else:
            # No future transactions, simulate 7 days ahead to see if liquidation occurs
            lookahead_seconds = DEFAULT_LOOKAHEAD_SECONDS

        return (
            user_profile,
            recommendation_timestamp,
            cutoff_timestamp,
            user,
            future_transactions,
            lookahead_seconds,
        )
    return user_profile


def get_price_history_value(symbol, timestamp):
    global _price_timestamps_cache
    symbol_price_history = get_price_history()[symbol]
    if symbol not in _price_timestamps_cache:
        _price_timestamps_cache[symbol] = sorted(symbol_price_history.keys())
    sorted_timestamps = _price_timestamps_cache[symbol]
    closest_timestamp = sorted_timestamps[
        bisect.bisect_left(sorted_timestamps, timestamp, hi=len(sorted_timestamps) - 1)
    ]
    return symbol_price_history[closest_timestamp]
