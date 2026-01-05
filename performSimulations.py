import pickle as pkl
from utils.constants import *
import pandas as pd
import json
import shutil
import os
import sys
import importlib
import copy
import time
import numpy as np
import logging
import threading
import concurrent.futures
import threading

outputFile = "simulationOutput3.log"
# Module logger
logger = logging.getLogger(__name__)
# File handler: capture all log levels to file
_file_handler = logging.FileHandler(outputFile)
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(_file_handler)
# # Console handler: show INFO+ by default
# _console_handler = logging.StreamHandler()
# _console_handler.setLevel(logging.INFO)
# _console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
# logger.addHandler(_console_handler)
# Ensure logger forwards all levels to handlers (file will receive DEBUG+)
logger.setLevel(logging.DEBUG)


def set_log_file(path: str, file_level: str | int = "DEBUG"):
    """Change the log file path and level. File will receive all levels at or above file_level."""
    # remove old file handler
    global _file_handler
    try:
        logger.removeHandler(_file_handler)
    except Exception:
        pass
    _file_handler = logging.FileHandler(path)
    if isinstance(file_level, str):
        lvl = getattr(logging, file_level.upper(), logging.DEBUG)
    else:
        lvl = int(file_level)
    _file_handler.setLevel(lvl)
    _file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    logger.addHandler(_file_handler)


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

PROFILES_DIR = "./profiles/"

with open(RECOMMENDATIONS_FILE, "rb") as f:
    recommendations = pkl.load(f)

# # Results cache: load previously computed simulation results (if any)
# results_cache_file = os.path.join(
#     SIMULATION_RESULTS_CACHE_DIR, "simulation_results.pkl"
# )
# try:
#     with open(results_cache_file, "rb") as f:
#         results_cache = pkl.load(f)
# except Exception:
#     results_cache = {}


def get_new_stats_dict():
    """Returns a new dictionary for tracking statistics."""
    return {
        "processed": 0,
        "skipped_cached": 0,
        "liquidated_without": 0,
        "liquidated_with": 0,
        "improved": 0,
        "worsened": 0,
        "no_change": 0,
        "time_deltas": [],
        # New stats for prediction validation
        "at_risk": 0,
        "not_at_risk": 0,
        "not_at_risk_but_liquidated": 0,
        "at_immediate_risk": 0,
        "immediate_risk_followed_by_liquidation": 0,
        "immediate_risk_not_followed_by_liquidation": 0,
        "immediate_risk_no_future": 0,
        "at_risk_eventual_liquidation_count": 0,
        "at_risk_time_to_liquidation": [],
        "at_risk_no_future_liquidation_count": 0,
        "at_risk_no_future": 0,
        "total_predictions_checked_mrp": 0,
        "prediction_matches_next_action_mrp": 0,
        "total_predictions_checked_ts": 0,
        "prediction_matches_next_action_ts": 0,
    }


stats = {
    "overall": get_new_stats_dict(),
    "by_index_action": {},
    "by_outcome_action": {},
    "by_action_pair": {},
}

stats_lock = threading.Lock()
processed_count_lock = threading.Lock()
processed_count = 0


# def save_results_cache():
#     try:
#         with open(results_cache_file, "wb") as f:
#             pkl.dump(results_cache, f)
#     except Exception as e:
#         logger.warning(f"Failed to save results cache: {e}")


def load_results_cache(recommendation, suffix, get_results):
    key = f"{recommendation['user']}_{int(recommendation.get('timestamp', 0))}_{suffix}"
    results_cache_file = os.path.join(SIMULATION_RESULTS_CACHE_DIR, f"{key}.pkl")
    try:
        with open(results_cache_file, "rb") as f:
            return pkl.load(f)
    except Exception:
        results = get_results()
        with open(results_cache_file, "wb") as f:
            pkl.dump(results, f)
        return get_results()


def _print_stats_summary(s: dict, title: str):
    processed = s.get("processed", 0)
    skipped = s.get("skipped_cached", 0)
    liq_without = s.get("liquidated_without", 0)
    liq_with = s.get("liquidated_with", 0)
    improved = s.get("improved", 0)
    worsened = s.get("worsened", 0)
    no_change = s.get("no_change", 0)
    deltas = s.get("time_deltas", []) or []

    def pct(n):
        return f"{(n / processed * 100):.1f}%" if processed else "N/A"

    logger.info(f"\n{title}")
    logger.info(f"Processed: {processed}")
    logger.info(f"Skipped (cached runs): {skipped} ({pct(skipped)})")
    logger.info(
        f"Liquidated without recommendation: {liq_without} ({pct(liq_without)})"
    )
    logger.info(
        f"Liquidated with recommendation:    {liq_with} ({pct(liq_with)})"
    )
    logger.info(
        f"Improved (avoided liquidation):     {improved} ({pct(improved)})"
    )
    logger.info(
        f"Worsened (introduced liquidation):  {worsened} ({pct(worsened)})"
    )
    logger.info(
        f"No change:                          {no_change} ({pct(no_change)})"
    )

    if deltas:
        try:
            avg = sum(deltas) / len(deltas)
            sd = (sum((x - avg) ** 2 for x in deltas) / len(deltas)) ** 0.5
            sorted_d = sorted(deltas)
            mid = len(sorted_d) // 2
            median = (
                sorted_d[mid]
                if len(sorted_d) % 2 == 1
                else (sorted_d[mid - 1] + sorted_d[mid]) / 2
            )
            logger.info(
                "\nTime-to-liquidation deltas (with - without) in seconds:"
            )
            logger.info(
                f"  Count: {len(deltas)}  Avg: {avg:.1f}s  Median: {median:.1f}s  Std: {sd:.1f}s"
            )
            logger.info(f"  Min: {min(deltas):.1f}s  Max: {max(deltas):.1f}s")
        except Exception as e:
            logger.warning(f"  Could not compute time-delta stats: {e}")
    else:
        logger.info("\nNo time-to-liquidation delta data available.")

    logger.info("\n=== Liquidation Risk Prediction vs. Actual Future ===")
    at_risk = s.get("at_risk", 0)
    not_at_risk = s.get("not_at_risk", 0)
    total_risk_assessed = at_risk + not_at_risk
    if total_risk_assessed == 0:
        logger.info("No risk assessment data available.")
        return

    logger.info(
        f"Total recommendations assessed for risk: {total_risk_assessed}"
    )
    if total_risk_assessed > 0:
        logger.info(f"At Risk: {at_risk} ({at_risk/total_risk_assessed:.1%})")
        logger.info(
            f"Not At Risk: {not_at_risk} ({not_at_risk/total_risk_assessed:.1%})"
        )
    not_at_risk_but_liquidated = s.get("not_at_risk_but_liquidated", 0)
    if not_at_risk > 0:
        logger.info(
            f"  - Users not at risk but liquidated anyway: {not_at_risk_but_liquidated} ({not_at_risk_but_liquidated/not_at_risk:.1%})"
        )

    logger.info("\n--- Analysis of 'At Risk' Users ---")
    if at_risk > 0:
        # Immediate Risk
        at_immediate_risk = s.get("at_immediate_risk", 0)
        logger.info(
            f"Immediate Risk: {at_immediate_risk} / {at_risk} ({at_immediate_risk/at_risk:.1%})"
        )
        if at_immediate_risk > 0:
            imm_followed_by_liq = s.get(
                "immediate_risk_followed_by_liquidation", 0
            )
            imm_not_followed_by_liq = s.get(
                "immediate_risk_not_followed_by_liquidation", 0
            )
            total_imm_with_future = (
                imm_followed_by_liq + imm_not_followed_by_liq
            )
            if total_imm_with_future > 0:
                logger.info(
                    f"  - Next transaction was a liquidation: {imm_followed_by_liq} ({imm_followed_by_liq/total_imm_with_future:.1%})"
                )
            else:
                logger.info(
                    "  - No users at immediate risk had future transactions to check."
                )

        # Eventual Liquidation
        eventual_liq_count = s.get("at_risk_eventual_liquidation_count", 0)
        no_future_liq_count = s.get("at_risk_no_future_liquidation_count", 0)
        total_at_risk_with_future = eventual_liq_count + no_future_liq_count
        if total_at_risk_with_future > 0:
            logger.info(
                f"\nEventual Liquidation (for all 'at risk' users with a future history):"
            )
            logger.info(
                f"  - Were eventually liquidated: {eventual_liq_count} ({eventual_liq_count/total_at_risk_with_future:.1%})"
            )
            logger.info(
                f"  - Were NOT eventually liquidated: {no_future_liq_count} ({no_future_liq_count/total_at_risk_with_future:.1%})"
            )

        # Time to liquidation
        times_to_liq = s.get("at_risk_time_to_liquidation", [])
        if times_to_liq:
            avg_ttl_hours = np.mean(times_to_liq) / 3600
            avg_ttl_days = avg_ttl_hours / 24
            logger.info(
                f"  - Average time to liquidation: {avg_ttl_days:.2f} days ({avg_ttl_hours:.2f} hours)"
            )
            median_ttl_days = np.median(times_to_liq) / 3600 / 24
            min_ttl_days = np.min(times_to_liq) / 3600 / 24
            max_ttl_days = np.max(times_to_liq) / 3600 / 24
            logger.info(
                f"    - Median: {median_ttl_days:.2f} days, Min: {min_ttl_days:.2f} days, Max: {max_ttl_days:.2f} days"
            )
    else:
        logger.info("No users were flagged as 'at risk'.")

    logger.info("\n--- Next Action Prediction Accuracy ---")
    total_mrp = s.get("total_predictions_checked_mrp", 0)
    matches_mrp = s.get("prediction_matches_next_action_mrp", 0)
    if total_mrp > 0:
        logger.info(
            f"Based on `most_recent_predictions` (lowest time-to-event):"
        )
        logger.info(
            f"  - Correctly predicted next action: {matches_mrp} / {total_mrp} ({matches_mrp/total_mrp:.1%})"
        )

    total_ts = s.get("total_predictions_checked_ts", 0)
    matches_ts = s.get("prediction_matches_next_action_ts", 0)
    if total_ts > 0:
        logger.info(f"Based on `trend_slopes` (most negative slope):")
        logger.info(
            f"  - Correctly predicted next action: {matches_ts} / {total_ts} ({matches_ts/total_ts:.1%})"
        )


user_profile_generator = UserProfileGenerator(None, WalletInferencer())

def process_recommendation(item):
    global processed_count
    recommendation, liquidation_info = item

    is_at_risk = liquidation_info["is_at_risk"]
    is_at_immediate_risk = liquidation_info["is_at_immediate_risk"]
    most_recent_predictions = liquidation_info["most_recent_predictions"]
    trend_slopes = liquidation_info["trend_slopes"]
    user = recommendation["user"]
    user_filename = "user_" + user + ".json"
    user_profile_file = os.path.expanduser(os.path.join(PROFILES_DIR, user_filename))
    if not os.path.exists(user_profile_file):
        logger.warning(
            f"Did not find profile for {user} ({user_profile_file}). Skipping..."
        )
        return
    with open(user_profile_file, "r") as f:
        user_profile = json.load(f)

    cutoff_timestamp = recommendation["timestamp"] - DEFAULT_TIME_DELTA_SECONDS

    # --- Start of new logic for prediction validation ---
    original_transactions = user_profile.get("transactions", [])
    future_transactions = sorted(
        [tx for tx in original_transactions if tx["timestamp"] > cutoff_timestamp],
        key=lambda x: x.get("timestamp", 0),
    )

    user_transactions = user_profile["transactions"]
    new_user_transactions = [
        user_transaction
        for user_transaction in user_transactions
        if user_transaction["timestamp"] <= cutoff_timestamp
    ]
    user_profile["transactions"] = new_user_transactions

    last_transaction_before_recommendation = user_profile["transactions"][-1]
    last_action_before_recommendation = last_transaction_before_recommendation["action"]
    outcome_transaction = (
        future_transactions[0]
        if len(future_transactions)
        else last_transaction_before_recommendation
    )
    outcome_action = outcome_transaction["action"]

    # Get stat buckets to update
    with stats_lock:
        overall_stats = stats["overall"]
        if last_action_before_recommendation not in stats["by_index_action"]:
            stats["by_index_action"][
                last_action_before_recommendation
            ] = get_new_stats_dict()
        action_stats = stats["by_index_action"][last_action_before_recommendation]
        if outcome_action not in stats["by_outcome_action"]:
            stats["by_outcome_action"][outcome_action] = get_new_stats_dict()
        outcome_action_stats = stats["by_outcome_action"][outcome_action]
        action_pair = (last_action_before_recommendation, outcome_action)
        if action_pair not in stats["by_action_pair"]:
            stats["by_action_pair"][action_pair] = get_new_stats_dict()
        action_pair_stats = stats["by_action_pair"][action_pair]
        stat_buckets = [
            overall_stats,
            action_stats,
            outcome_action_stats,
            action_pair_stats,
        ]

        if not is_at_risk:
            for bucket in stat_buckets:
                bucket["not_at_risk"] += 1
            if future_transactions:
                if any(tx["action"].lower() == "liquidated" for tx in future_transactions):
                    for bucket in stat_buckets:
                        bucket["not_at_risk_but_liquidated"] += 1
        else:  # is_at_risk is True
            for bucket in stat_buckets:
                bucket["at_risk"] += 1

            # Immediate risk check
            if is_at_immediate_risk:
                for bucket in stat_buckets:
                    bucket["at_immediate_risk"] += 1
                if future_transactions:
                    next_action = future_transactions[0]["action"]
                    if next_action.lower() == "liquidated":
                        for bucket in stat_buckets:
                            bucket["immediate_risk_followed_by_liquidation"] += 1
                    else:
                        for bucket in stat_buckets:
                            bucket["immediate_risk_not_followed_by_liquidation"] += 1
                else:
                    for bucket in stat_buckets:
                        bucket["immediate_risk_no_future"] += 1

            # Eventual liquidation check for all 'at_risk' cases
            if future_transactions:
                future_liquidations = [
                    tx for tx in future_transactions if tx["action"].lower() == "liquidated"
                ]
                if future_liquidations:
                    first_liquidation = future_liquidations[0]
                    time_to_liquidation = first_liquidation["timestamp"] - cutoff_timestamp
                    for bucket in stat_buckets:
                        bucket["at_risk_time_to_liquidation"].append(time_to_liquidation)
                        bucket["at_risk_eventual_liquidation_count"] += 1
                else:
                    for bucket in stat_buckets:
                        bucket["at_risk_no_future_liquidation_count"] += 1
            else:
                for bucket in stat_buckets:
                    bucket["at_risk_no_future"] += 1

            # `most_recent_predictions` check
            if most_recent_predictions and future_transactions:
                for bucket in stat_buckets:
                    bucket["total_predictions_checked_mrp"] += 1
                valid_predictions = {
                    k: v for k, v in most_recent_predictions.items() if v is not None
                }
                if valid_predictions:
                    predicted_next_event = min(valid_predictions, key=valid_predictions.get)
                    next_actual_action = future_transactions[0]["action"]
                    if predicted_next_event.lower() == next_actual_action.lower():
                        for bucket in stat_buckets:
                            bucket["prediction_matches_next_action_mrp"] += 1

            # `trend_slopes` check
            if trend_slopes and future_transactions:
                for bucket in stat_buckets:
                    bucket["total_predictions_checked_ts"] += 1
                valid_slopes = {k: v for k, v in trend_slopes.items() if v is not None}
                if valid_slopes:
                    predicted_next_event_by_slope = min(valid_slopes, key=valid_slopes.get)
                    next_actual_action = future_transactions[0]["action"]
                    if predicted_next_event_by_slope.lower() == next_actual_action.lower():
                        for bucket in stat_buckets:
                            bucket["prediction_matches_next_action_ts"] += 1
    # --- End of new logic ---

    # # Unique cache key per user + recommendation timestamp
    # key = f"{user}_{int(recommendation.get('timestamp', 0))}"

    # cached = results_cache.get(key, {})

    lookahead_seconds = (
        outcome_transaction["timestamp"]
        - last_transaction_before_recommendation["timestamp"]
    ) * 2

    # Run (or load) results without the recommendation
    # if "without" in cached:
    #     results_without_recommendation = cached["without"]
    #     for bucket in stat_buckets:
    #         bucket["skipped_cached"] += 1
    # else:
    #     results_without_recommendation = run_simulation(
    #         user_profile, lookahead_seconds=lookahead_seconds, output_file=outputFile
    #     )
    #     results_cache.setdefault(key, {})["without"] = results_without_recommendation
    #     save_results_cache()
    results_without_recommendation = load_results_cache(
        recommendation,
        "without",
        lambda: run_simulation(
            user_profile, lookahead_seconds=lookahead_seconds, output_file=outputFile
        ),
    )

    # Prepare a copy of the profile that includes the recommendation transaction
    user_profile_with = copy.deepcopy(user_profile)
    user_profile_with["transactions"].append(
        user_profile_generator._row_to_transaction(recommendation)
    )

    # # Run (or load) results with the recommendation
    # if "with" in cached:
    #     results_with_recommendation = cached["with"]
    # else:
    #     results_with_recommendation = run_simulation(
    #         user_profile_with,
    #         lookahead_seconds=lookahead_seconds,
    #         output_file=outputFile,
    #     )
    #     results_cache.setdefault(key, {})["with"] = results_with_recommendation
    #     results_cache[key]["recommendation"] = recommendation
    #     # summary (quick stats) stored alongside detailed results
    #     results_cache[key]["summary"] = {
    #         "user": user,
    #         "timestamp": int(recommendation.get("timestamp", 0)),
    #         "liquidated_without": bool(
    #             results_without_recommendation.get("liquidation_stats", {}).get(
    #                 "liquidated"
    #             )
    #         ),
    #         "liquidated_with": bool(
    #             results_with_recommendation.get("liquidation_stats", {}).get(
    #                 "liquidated"
    #             )
    #         ),
    #         "time_to_liquidation_without": results_without_recommendation.get(
    #             "liquidation_stats", {}
    #         ).get("time_to_liquidation"),
    #         "time_to_liquidation_with": results_with_recommendation.get(
    #             "liquidation_stats", {}
    #         ).get("time_to_liquidation"),
    #     }
    #     save_results_cache()
    results_with_recommendation = load_results_cache(
        recommendation,
        "with",
        lambda: run_simulation(
            user_profile_with,
            lookahead_seconds=lookahead_seconds,
            output_file=outputFile,
        ),
    )

    # Update runtime stats
    with stats_lock:
        for bucket in stat_buckets:
            bucket["processed"] += 1
        lw = bool(
            results_without_recommendation.get("liquidation_stats", {}).get("liquidated")
        )
        lw_with = bool(
            results_with_recommendation.get("liquidation_stats", {}).get("liquidated")
        )
        if lw:
            for bucket in stat_buckets:
                bucket["liquidated_without"] += 1
        if lw_with:
            for bucket in stat_buckets:
                bucket["liquidated_with"] += 1

        if lw and not lw_with:
            for bucket in stat_buckets:
                bucket["improved"] += 1
        elif (not lw) and lw_with:
            for bucket in stat_buckets:
                bucket["worsened"] += 1
        else:
            for bucket in stat_buckets:
                bucket["no_change"] += 1

        t_without = results_without_recommendation.get("liquidation_stats", {}).get(
            "time_to_liquidation"
        )
        t_with = results_with_recommendation.get("liquidation_stats", {}).get(
            "time_to_liquidation"
        )
        if (t_without is not None) and (t_with is not None):
            try:
                delta = float(t_with) - float(t_without)
                for bucket in stat_buckets:
                    bucket["time_deltas"].append(delta)
            except Exception:
                pass

    with processed_count_lock:
        processed_count += 1
        current_count = processed_count
        logger.info(f"Processed {current_count} recommendations...")

    if current_count % 100 == 0:
        with stats_lock:
            _print_stats_summary(stats["overall"], "=== Overall Simulation Summary ===")

            for action, action_stats in sorted(stats["by_index_action"].items()):
                _print_stats_summary(
                    action_stats, f"=== Simulation Summary for index_action = {action} ==="
                )
            for action, action_stats in sorted(stats["by_outcome_action"].items()):
                _print_stats_summary(
                    action_stats,
                    f"=== Simulation Summary for outcome_action = {action} ===",
                )
            for action_pair, action_pair_stats in sorted(stats["by_action_pair"].items()):
                _print_stats_summary(
                    action_pair_stats,
                    f"=== Simulation Summary for action_pair = {action_pair} ===",
                )


with concurrent.futures.ThreadPoolExecutor() as executor:
    list(executor.map(process_recommendation, recommendations.values()))
