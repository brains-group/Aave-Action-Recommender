from pathlib import Path
import pickle as pkl
import os
import sys

import numpy as np

from utils.logger import logger
from utils.constants import MIN_RECOMMENDATION_DEBT_USD, SIMULATION_RESULTS_CACHE_DIR

# Make the bundled Aave-Simulator directory importable (it's next to this file)
this_dir = os.path.dirname(os.path.realpath(__file__))
aave_sim_path = os.path.join(this_dir, "..", "Aave-Simulator")
if aave_sim_path not in sys.path:
    sys.path.insert(0, aave_sim_path)

from tools.run_single_simulation import run_simulation


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
    if (
        recommendation["Index Event"] != "repay" and (estimated_remaining_debt > 0
        and estimated_remaining_debt < MIN_RECOMMENDATION_DEBT_USD)
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
