import bisect
from pathlib import Path
import pickle as pkl
import os
import sys

import numpy as np

from utils.data import get_user_profile
from utils.logger import logger
from utils.constants import (
    DEFAULT_TIME_DELTA_SECONDS,
    MIN_RECOMMENDATION_DEBT_USD,
    SIMULATION_RESULTS_CACHE_DIR,
    DEFAULT_LOOKAHEAD_SECONDS,
)

# Make the bundled Aave-Simulator directory importable (it's next to this file)
this_dir = os.path.dirname(os.path.realpath(__file__))
aave_sim_path = os.path.join(this_dir, "..", "Aave-Simulator")
if aave_sim_path not in sys.path:
    sys.path.insert(0, aave_sim_path)

from tools.run_single_simulation import run_simulation
from simulator.utils import get_price_history


from functools import lru_cache
@lru_cache(maxsize=1)
def simulation_code_identity():
    import hashlib
    root = Path(aave_sim_path).resolve()
    digest = hashlib.sha256()
    for directory in ['simulator', 'tools', 'analysis', 'market']:
        for source in sorted((root / directory).rglob('*.py')):
            digest.update(str(source.relative_to(root)).encode()); digest.update(source.read_bytes())
    for source in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(source.name.encode()); digest.update(source.read_bytes())
    price = root / 'data/reserves/price_history.json'
    with price.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''): digest.update(chunk)
    return 'code_' + digest.hexdigest()[:20]


def get_simulation_outcome(recommendation, suffix, **passed_args):
    """Load cached simulation results or compute and cache them."""
    key = f"{recommendation['user']}_{int(recommendation.get('timestamp', 0))}_{suffix}"
    # New logic/configuration must not consume paper-era simulation caches.
    import hashlib
    canonical = pkl.dumps({k:v for k,v in passed_args.items() if k != "output_file"}, protocol=4)
    fingerprint = hashlib.sha256(canonical).hexdigest()[:16]
    results_cache_file = Path(SIMULATION_RESULTS_CACHE_DIR) / simulation_code_identity() / fingerprint / f"{key}.pkl"
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
        results_cache_file.parent.mkdir(parents=True, exist_ok=True)
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
    """Bound repayment to same-asset funds/debt in the pre-projection checkpoint.

    No conversion, wallet sweep, future-state funding, or automatic dust upsizing.
    Inferred initial wallets remain an explicit retrospective assumption.
    """
    if str(recommendation["Index Event"]).lower() != "repay":
        return recommendation
    state = results_without_recommendation.get("checkpoint_state")
    if state is None or state["timestamp"] > recommendation["timestamp"]:
        logger.warning("Repayment requires an available pre-recommendation checkpoint state")
        return None
    symbol = recommendation.get("symbol") or recommendation.get("reserve")
    amount = min(float(recommendation["amount"]),
                 state["wallet_balances"].get(symbol, 0),
                 state["debt_balances"].get(symbol, 0))
    if not np.isfinite(amount) or amount <= 0:
        return None
    recommendation = recommendation.copy()
    updateAmountOrUSD(recommendation, amount=amount)
    return recommendation


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

    from utils.indexed_checkpoint import attach_checkpoint
    attach_checkpoint(user_profile)

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

_price_timestamps_cache = {}

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
