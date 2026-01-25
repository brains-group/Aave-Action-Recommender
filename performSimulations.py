import pickle as pkl
from utils.constants import *
import pandas as pd
import json
import os
import sys
import copy
import time
import numpy as np
import logging
import argparse
from multiprocessing import Pool, cpu_count
from pathlib import Path
from datetime import datetime

outputFile = "output7.log"
# Module logger
logger = logging.getLogger(__name__)
# File handler: capture all log levels to file
_file_handler = logging.FileHandler(outputFile)
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(_file_handler)
# Ensure logger forwards all levels to handlers (file will receive DEBUG+)
logger.setLevel(logging.DEBUG)
logger.propagate = False


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

PROFILES_DIR = "./profiles/"

# Constants
DEFAULT_LOOKAHEAD_DAYS = 7
DEFAULT_LOOKAHEAD_SECONDS = 86400 * DEFAULT_LOOKAHEAD_DAYS

# Ensure cache directories exist at startup (performance optimization)
Path(SIMULATION_RESULTS_CACHE_DIR).mkdir(parents=True, exist_ok=True)

# RECOMMENDATIONS_FILE = "./backup/recommendations.pkl"
# Load recommendations with error handling
try:
    if not os.path.exists(RECOMMENDATIONS_FILE):
        logger.error(f"Recommendations file not found: {RECOMMENDATIONS_FILE}")
        raise FileNotFoundError(
            f"Recommendations file not found: {RECOMMENDATIONS_FILE}"
        )

    logger.info(f"Loading recommendations from: {RECOMMENDATIONS_FILE}")
    with open(RECOMMENDATIONS_FILE, "rb") as f:
        recommendations = pkl.load(f)

    if not isinstance(recommendations, dict):
        logger.error(
            f"Invalid recommendations format: expected dict, got {type(recommendations)}"
        )
        raise ValueError(
            f"Invalid recommendations format: expected dict, got {type(recommendations)}"
        )

    logger.info(f"Loaded {len(recommendations)} recommendations")

except FileNotFoundError:
    logger.error(
        "Recommendations file not found. Please run actionAgentTraining.py first."
    )
    raise
except Exception as e:
    logger.error(f"Error loading recommendations file: {e}")
    raise


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
        "no_change_with_liquidation": 0,
        "no_change_without_liquidation": 0,
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
        "not_at_risk_time_to_liquidation": [],
        "liquidation_prediction_correlation": [],
        "at_risk_no_future_liquidation_count": 0,
        "at_risk_no_future": 0,
        "total_predictions_checked_mrp": 0,
        "prediction_matches_next_action_mrp": 0,
        "total_predictions_checked_ts": 0,
        "prediction_matches_next_action_ts": 0,
        # Liquidation reason tracking
        "liquidation_reasons_without": [],  # List of liquidation reasons for "without recommendation"
        "liquidation_reasons_with": [],  # List of liquidation reasons for "with recommendation"
        "dust_liquidations_without": 0,
        "dust_liquidations_with": 0,
        "hf_based_liquidations_without": 0,
        "hf_based_liquidations_with": 0,
        "threshold_based_liquidations_without": 0,
        "threshold_based_liquidations_with": 0,
        # Strategy comparison tracking
        "strategy_comparisons": {
            "without": {
                "event_driven": {"detected": 0, "checks": 0, "time": []},
                "adaptive_granularity": {"detected": 0, "checks": 0, "time": []},
                "binary_search": {"detected": 0, "checks": 0, "time": []},
                "hybrid": {"detected": 0, "checks": 0, "time": []},
                "model_based": {"detected": 0, "checks": 0, "time": []},
                "interest_milestones": {"detected": 0, "checks": 0, "time": []},
            },
            "with": {
                "event_driven": {"detected": 0, "checks": 0, "time": []},
                "adaptive_granularity": {"detected": 0, "checks": 0, "time": []},
                "binary_search": {"detected": 0, "checks": 0, "time": []},
                "hybrid": {"detected": 0, "checks": 0, "time": []},
                "model_based": {"detected": 0, "checks": 0, "time": []},
                "interest_milestones": {"detected": 0, "checks": 0, "time": []},
            },
            "consensus_agreement_without": [],
            "consensus_agreement_with": [],
            "best_strategy_counts": {"without": {}, "with": {}},
            # Per-simulation breakdown: track which strategies detected liquidation for each user
            "per_simulation_without": [],  # List of dicts: [{"user": "...", "strategies_detected": [...], "times": {...}}, ...]
            "per_simulation_with": [],  # Same structure for "with recommendation"
        },
    }


def merge_stats_updates(base_stats, updates):
    """
    Merge stats updates into base stats dictionary.

    Args:
        base_stats: Base statistics dictionary to update
        updates: Dictionary with stats updates in the same structure
    """

    def merge_dict(base, update):
        """Recursively merge update dict into base dict."""
        for key, value in update.items():
            if key not in base:
                # Key doesn't exist, copy the value (handles both dicts and other types)
                if isinstance(value, dict):
                    base[key] = copy.deepcopy(value)
                elif isinstance(value, list):
                    base[key] = copy.deepcopy(value)
                else:
                    base[key] = value
            elif isinstance(value, dict):
                # Key exists and value is a dict, recursively merge
                if not isinstance(base[key], dict):
                    base[key] = copy.deepcopy(value)
                else:
                    merge_dict(base[key], value)
            elif isinstance(value, list):
                # Key exists and value is a list, extend the existing list
                if isinstance(base[key], list):
                    base[key].extend(value)
                else:
                    base[key] = copy.deepcopy(value)
            else:
                # Key exists and value is a scalar, add to existing value
                base[key] += value

    merge_dict(base_stats, updates)


stats = {
    "overall": get_new_stats_dict(),
    "by_index_action": {},
    "by_outcome_action": {},
    "by_action_pair": {},
}
stats_no_dust = {
    "overall": get_new_stats_dict(),
    "by_index_action": {},
    "by_outcome_action": {},
    "by_action_pair": {},
}


def load_results_cache(recommendation, suffix, **passed_args):
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


def convert_to_json_serializable(obj):
    """
    Recursively convert numpy types and other non-JSON-serializable types to Python native types.

    Handles:
    - numpy integers/floats/bools
    - numpy arrays
    - pandas NA values
    - nested dicts and lists
    - tuple keys (converts to strings like "action1->action2")
    """
    if isinstance(obj, dict):
        # Convert dict, handling tuple keys by converting them to strings
        result = {}
        for k, v in obj.items():
            # Convert tuple keys to strings (e.g., ("Deposit", "Liquidated") -> "Deposit->Liquidated")
            if isinstance(k, tuple):
                key_str = "->".join(str(item) for item in k)
            elif isinstance(k, (np.integer, np.int64, np.int32)):
                key_str = int(k)
            elif isinstance(k, (np.floating, np.float64, np.float32)):
                key_str = float(k)
            else:
                key_str = k
            result[key_str] = convert_to_json_serializable(v)
        return result
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    else:
        return obj


def save_statistics_to_json(stats_dict: dict, suffix: str = ""):
    """
    Save statistics dictionary to JSON file.

    Args:
        stats_dict: Statistics dictionary to save
        suffix: Optional suffix for filename (e.g., "INTERRUPTED", "ERROR")

    Returns:
        Path to saved file, or None if failed
    """
    try:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        stats_output_dir = Path(CACHE_DIR) / "statistics"
        stats_output_dir.mkdir(parents=True, exist_ok=True)

        filename = (
            f"simulation_statistics_{suffix}_{timestamp_str}.json"
            if suffix
            else f"simulation_statistics_{timestamp_str}.json"
        )
        stats_json_path = stats_output_dir / filename

        stats_serializable = convert_to_json_serializable(stats_dict)

        with open(stats_json_path, "w") as f:
            json.dump(stats_serializable, f, indent=2)

        logger.info(f"✓ Statistics saved to: {stats_json_path}")
        return stats_json_path
    except Exception as e:
        logger.error(f"Error saving statistics to JSON: {e}", exc_info=True)
        return None


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
    logger.info(f"Liquidated with recommendation:    {liq_with} ({pct(liq_with)})")
    logger.info(f"Improved (avoided liquidation):     {improved} ({pct(improved)})")
    logger.info(f"Worsened (introduced liquidation):  {worsened} ({pct(worsened)})")
    logger.info(f"No change:                          {no_change} ({pct(no_change)})")
    if no_change > 0:
        no_change_with_liq = s.get("no_change_with_liquidation", 0)
        no_change_without_liq = s.get("no_change_without_liquidation", 0)
        logger.info(
            f"  - Both liquidated:                {no_change_with_liq} ({pct(no_change_with_liq)})"
        )
        logger.info(
            f"  - Neither liquidated:             {no_change_without_liq} ({pct(no_change_without_liq)})"
        )

    # Display liquidation type breakdown
    if liq_without > 0 or liq_with > 0:
        logger.info("\n=== Liquidation Type Breakdown ===")

        dust_without = s.get("dust_liquidations_without", 0)
        hf_without = s.get("hf_based_liquidations_without", 0)
        threshold_without = s.get("threshold_based_liquidations_without", 0)
        reasons_without = s.get("liquidation_reasons_without", [])

        if liq_without > 0:
            logger.info(f"Liquidations WITHOUT recommendation ({liq_without} total):")
            if dust_without > 0:
                logger.info(
                    f"  - Dust liquidations: {dust_without} ({dust_without/liq_without:.1%})"
                )
            if hf_without > 0:
                logger.info(
                    f"  - HF-based liquidations: {hf_without} ({hf_without/liq_without:.1%})"
                )
            if threshold_without > 0:
                logger.info(
                    f"  - Threshold-based liquidations: {threshold_without} ({threshold_without/liq_without:.1%})"
                )
            # Show sample reasons (first 3 unique)
            if reasons_without:
                unique_reasons = list(set(reasons_without))[:3]
                logger.info(f"  - Sample reasons: {unique_reasons}")

        dust_with = s.get("dust_liquidations_with", 0)
        hf_with = s.get("hf_based_liquidations_with", 0)
        threshold_with = s.get("threshold_based_liquidations_with", 0)
        reasons_with = s.get("liquidation_reasons_with", [])

        if liq_with > 0:
            logger.info(f"Liquidations WITH recommendation ({liq_with} total):")
            if dust_with > 0:
                logger.info(
                    f"  - Dust liquidations: {dust_with} ({dust_with/liq_with:.1%})"
                )
            if hf_with > 0:
                logger.info(
                    f"  - HF-based liquidations: {hf_with} ({hf_with/liq_with:.1%})"
                )
            if threshold_with > 0:
                logger.info(
                    f"  - Threshold-based liquidations: {threshold_with} ({threshold_with/liq_with:.1%})"
                )
            # Show sample reasons (first 3 unique)
            if reasons_with:
                unique_reasons = list(set(reasons_with))[:3]
                logger.info(f"  - Sample reasons: {unique_reasons}")

    # Display strategy comparison summary
    strategy_comparisons = s.get("strategy_comparisons", {})
    if strategy_comparisons:
        logger.info("\n=== Liquidation Detection Strategy Comparison ===")

        # Consensus agreement rates
        consensus_without = strategy_comparisons.get("consensus_agreement_without", [])
        consensus_with = strategy_comparisons.get("consensus_agreement_with", [])
        if consensus_without:
            avg_agreement_without = (
                np.mean(consensus_without) if consensus_without else 0.0
            )
            logger.info(
                f"Average consensus agreement (WITHOUT recommendation): {avg_agreement_without:.1%}"
            )
        if consensus_with:
            avg_agreement_with = np.mean(consensus_with) if consensus_with else 0.0
            logger.info(
                f"Average consensus agreement (WITH recommendation): {avg_agreement_with:.1%}"
            )

        # Best strategy counts
        best_strategy_counts_without = strategy_comparisons.get(
            "best_strategy_counts", {}
        ).get("without", {})
        best_strategy_counts_with = strategy_comparisons.get(
            "best_strategy_counts", {}
        ).get("with", {})

        if best_strategy_counts_without:
            logger.info("\nMost frequently best strategy (WITHOUT recommendation):")
            # Sort once, reuse for display
            sorted_strategies_without = sorted(
                best_strategy_counts_without.items(), key=lambda x: x[1], reverse=True
            )
            for strategy, count in sorted_strategies_without[:3]:
                logger.info(f"  - {strategy}: {count} times ({count/processed:.1%})")

        if best_strategy_counts_with:
            logger.info("\nMost frequently best strategy (WITH recommendation):")
            # Sort once, reuse for display
            sorted_strategies_with = sorted(
                best_strategy_counts_with.items(), key=lambda x: x[1], reverse=True
            )
            for strategy, count in sorted_strategies_with[:3]:
                logger.info(f"  - {strategy}: {count} times ({count/processed:.1%})")

        # Per-strategy detection rates and efficiency (optimized: sort once, reuse)
        strategy_stats_without = strategy_comparisons.get("without", {})
        strategy_stats_with = strategy_comparisons.get("with", {})

        if strategy_stats_without and processed > 0:
            logger.info("\nStrategy Performance (WITHOUT recommendation):")
            # Sort once, iterate multiple times if needed
            sorted_strategies_without = sorted(strategy_stats_without.items())
            for strategy_name, stats in sorted_strategies_without:
                detected = stats.get("detected", 0)
                checks = stats.get("checks", 0)
                avg_checks = checks / processed if processed > 0 else 0
                detection_rate = detected / processed if processed > 0 else 0.0

                # Calculate average liquidation time for this strategy
                times = stats.get("time", [])
                time_info = ""
                if times:
                    avg_time = np.mean(times)
                    avg_time_days = avg_time / (24 * 3600)
                    time_info = f", avg time: {avg_time_days:.2f} days"
                    if len(times) > 1:
                        std_time = np.std(times)
                        std_time_days = std_time / (24 * 3600)
                        time_info += f" (±{std_time_days:.2f} days)"

                logger.info(
                    f"  - {strategy_name}: {detected}/{processed} detected ({detection_rate:.1%}), "
                    f"avg {avg_checks:.0f} checks/simulation{time_info}"
                )

        if strategy_stats_with and processed > 0:
            logger.info("\nStrategy Performance (WITH recommendation):")
            # Sort once, iterate multiple times if needed
            sorted_strategies_with = sorted(strategy_stats_with.items())
            for strategy_name, stats in sorted_strategies_with:
                detected = stats.get("detected", 0)
                checks = stats.get("checks", 0)
                avg_checks = checks / processed if processed > 0 else 0
                detection_rate = detected / processed if processed > 0 else 0.0

                # Calculate average liquidation time for this strategy
                times = stats.get("time", [])
                time_info = ""
                if times:
                    avg_time = np.mean(times)
                    avg_time_days = avg_time / (24 * 3600)
                    time_info = f", avg time: {avg_time_days:.2f} days"
                    if len(times) > 1:
                        std_time = np.std(times)
                        std_time_days = std_time / (24 * 3600)
                        time_info += f" (±{std_time_days:.2f} days)"

                logger.info(
                    f"  - {strategy_name}: {detected}/{processed} detected ({detection_rate:.1%}), "
                    f"avg {avg_checks:.0f} checks/simulation{time_info}"
                )

        # Strategy agreement analysis
        if strategy_stats_without and processed > 0:
            logger.info("\nStrategy Agreement Analysis (WITHOUT recommendation):")
            # Count how many times each combination of strategies detected liquidation
            # This would require storing per-simulation data, but we can show detection overlap
            total_detections = sum(
                stats.get("detected", 0) for stats in strategy_stats_without.values()
            )
            if total_detections > 0:
                logger.info(
                    f"  Total liquidation detections across all strategies: {total_detections}"
                )
                logger.info(
                    f"  Average detections per liquidated case: {total_detections / max(liq_without, 1):.1f} strategies"
                )

                # Show which strategies detected most/least
                detection_counts = [
                    (name, stats.get("detected", 0))
                    for name, stats in strategy_stats_without.items()
                ]
                detection_counts.sort(key=lambda x: x[1], reverse=True)
                logger.info("  Detection frequency:")
                for name, count in detection_counts:
                    pct = count / max(liq_without, 1) if liq_without > 0 else 0.0
                    logger.info(
                        f"    - {name}: {count} detections ({pct:.1%} of liquidations)"
                    )

        if strategy_stats_with and processed > 0:
            logger.info("\nStrategy Agreement Analysis (WITH recommendation):")
            total_detections = sum(
                stats.get("detected", 0) for stats in strategy_stats_with.values()
            )
            if total_detections > 0:
                logger.info(
                    f"  Total liquidation detections across all strategies: {total_detections}"
                )
                logger.info(
                    f"  Average detections per liquidated case: {total_detections / max(liq_with, 1):.1f} strategies"
                )

                # Show which strategies detected most/least
                detection_counts = [
                    (name, stats.get("detected", 0))
                    for name, stats in strategy_stats_with.items()
                ]
                detection_counts.sort(key=lambda x: x[1], reverse=True)
                logger.info("  Detection frequency:")
                for name, count in detection_counts:
                    pct = count / max(liq_with, 1) if liq_with > 0 else 0.0
                    logger.info(
                        f"    - {name}: {count} detections ({pct:.1%} of liquidations)"
                    )

        # Per-simulation breakdown (optimized: extract once, reuse)
        per_sim_without = strategy_comparisons.get("per_simulation_without", [])
        per_sim_with = strategy_comparisons.get("per_simulation_with", [])

        if per_sim_without or per_sim_with:
            logger.info("\n=== Per-Simulation Strategy Breakdown ===")

            # Pre-compute disagreement samples (cache results)
            disagreement_samples_without = []
            if per_sim_without:
                for sim in per_sim_without:
                    detected = sim.get("strategies_detected", [])
                    detected_count = len(detected)
                    if detected_count > 0 and detected_count < 6:
                        disagreement_samples_without.append(sim)

            logger.info(
                "\nSample Cases with Strategy Disagreement (WITHOUT recommendation):"
            )

            if disagreement_samples_without:
                # Show first 10 cases with disagreement
                for i, sim in enumerate(disagreement_samples_without[:10]):
                    user_id = sim.get("user", "unknown")
                    detected = sim.get("strategies_detected", [])
                    not_detected = sim.get("strategies_not_detected", [])
                    times = sim.get("times", {})
                    consensus_agreement = sim.get("consensus_agreement", 0.0)

                    # Cache sorted results (used multiple times)
                    detected_sorted = sorted(detected) if detected else []
                    not_detected_sorted = sorted(not_detected) if not_detected else []

                    logger.info(f"\n  User {user_id}:")
                    logger.info(
                        f"    Strategies DETECTED liquidation: {', '.join(detected_sorted)} ({len(detected)}/6)"
                    )
                    if detected and times:
                        # Sort times dict items once
                        sorted_times = sorted(times.items())
                        time_strs = [
                            f"{name}: {t/(24*3600):.2f} days"
                            for name, t in sorted_times
                        ]
                        logger.info(f"    Liquidation times: {', '.join(time_strs)}")
                    if not_detected:
                        logger.info(
                            f"    Strategies NOT detected: {', '.join(not_detected_sorted)} ({len(not_detected)}/6)"
                        )
                    logger.info(f"    Consensus agreement: {consensus_agreement:.1%}")

                    if i < len(disagreement_samples_without) - 1:
                        pass  # Continue to next

                if len(disagreement_samples_without) > 10:
                    logger.info(
                        f"\n  ... and {len(disagreement_samples_without) - 10} more cases with disagreement"
                    )
            else:
                logger.info(
                    "  No cases with strategy disagreement found (all strategies agreed)"
                )

            # Pre-compute disagreement samples for "with" (cache results)
            disagreement_samples_with = []
            if per_sim_with:
                for sim in per_sim_with:
                    detected = sim.get("strategies_detected", [])
                    detected_count = len(detected)
                    if detected_count > 0 and detected_count < 6:
                        disagreement_samples_with.append(sim)

            logger.info(
                "\nSample Cases with Strategy Disagreement (WITH recommendation):"
            )

            if disagreement_samples_with:
                # Show first 10 cases with disagreement
                for i, sim in enumerate(disagreement_samples_with[:10]):
                    user_id = sim.get("user", "unknown")
                    detected = sim.get("strategies_detected", [])
                    not_detected = sim.get("strategies_not_detected", [])
                    times = sim.get("times", {})
                    consensus_agreement = sim.get("consensus_agreement", 0.0)

                    # Cache sorted results (used multiple times)
                    detected_sorted = sorted(detected) if detected else []
                    not_detected_sorted = sorted(not_detected) if not_detected else []

                    logger.info(f"\n  User {user_id}:")
                    logger.info(
                        f"    Strategies DETECTED liquidation: {', '.join(detected_sorted)} ({len(detected)}/6)"
                    )
                    if detected and times:
                        # Sort times dict items once
                        sorted_times = sorted(times.items())
                        time_strs = [
                            f"{name}: {t/(24*3600):.2f} days"
                            for name, t in sorted_times
                        ]
                        logger.info(f"    Liquidation times: {', '.join(time_strs)}")
                    if not_detected:
                        logger.info(
                            f"    Strategies NOT detected: {', '.join(not_detected_sorted)} ({len(not_detected)}/6)"
                        )
                    logger.info(f"    Consensus agreement: {consensus_agreement:.1%}")

                if len(disagreement_samples_with) > 10:
                    logger.info(
                        f"\n  ... and {len(disagreement_samples_with) - 10} more cases with disagreement"
                    )
            else:
                logger.info(
                    "  No cases with strategy disagreement found (all strategies agreed)"
                )

            # Summary statistics on strategy agreement (optimized: compute once, reuse disagreement lists)
            logger.info("\nStrategy Agreement Statistics:")
            if per_sim_without:
                # Use pre-computed disagreement count
                partial_agreement_without = len(disagreement_samples_without)
                full_agreement_without = (
                    len(per_sim_without) - partial_agreement_without
                )
                logger.info(f"  WITHOUT recommendation:")
                logger.info(
                    f"    Full agreement (all 6 strategies same): {full_agreement_without}/{len(per_sim_without)} ({full_agreement_without/len(per_sim_without):.1%})"
                )
                logger.info(
                    f"    Partial agreement (some disagreement): {partial_agreement_without}/{len(per_sim_without)} ({partial_agreement_without/len(per_sim_without):.1%})"
                )

            if per_sim_with:
                # Use pre-computed disagreement count
                partial_agreement_with = len(disagreement_samples_with)
                full_agreement_with = len(per_sim_with) - partial_agreement_with
                logger.info(f"  WITH recommendation:")
                logger.info(
                    f"    Full agreement (all 6 strategies same): {full_agreement_with}/{len(per_sim_with)} ({full_agreement_with/len(per_sim_with):.1%})"
                )
                logger.info(
                    f"    Partial agreement (some disagreement): {partial_agreement_with}/{len(per_sim_with)} ({partial_agreement_with/len(per_sim_with):.1%})"
                )

            # Pre-compute liquidated cases (optimized: cache results)
            liquidated_without = [
                sim for sim in per_sim_without if sim.get("strategies_detected")
            ]
            liquidated_with = [
                sim for sim in per_sim_with if sim.get("strategies_detected")
            ]

            # Show sample of all liquidations (not just disagreements) with strategy breakdown
            logger.info(
                "\nSample Liquidations with Full Strategy Breakdown (WITHOUT recommendation):"
            )
            if liquidated_without:
                # Show first 5 liquidations (can include both agreed and disagreed cases)
                for i, sim in enumerate(liquidated_without[:5]):
                    user_id = sim.get("user", "unknown")
                    detected = sim.get("strategies_detected", [])
                    times = sim.get("times", {})
                    checks = sim.get("checks", {})

                    # Cache sorted results
                    detected_sorted = sorted(detected) if detected else []

                    logger.info(f"\n  User {user_id}:")
                    logger.info(
                        f"    Detected by: {', '.join(detected_sorted)} ({len(detected)}/6 strategies)"
                    )
                    if times:
                        logger.info(f"    Liquidation times by strategy:")
                        # Iterate over sorted detected strategies (matches displayed order)
                        for strategy_name in detected_sorted:
                            if strategy_name in times:
                                time_days = times[strategy_name] / (24 * 3600)
                                check_count = checks.get(strategy_name, "N/A")
                                logger.info(
                                    f"      - {strategy_name}: {time_days:.2f} days ({check_count} checks)"
                                )
                    if len(detected) < 6:
                        not_detected = sim.get("strategies_not_detected", [])
                        logger.info(
                            f"    NOT detected by: {', '.join(sorted(not_detected))} ({len(not_detected)}/6 strategies)"
                        )

                if len(liquidated_without) > 5:
                    logger.info(
                        f"\n  ... and {len(liquidated_without) - 5} more liquidations"
                    )
            else:
                logger.info("  No liquidations detected (WITHOUT recommendation)")

            # Same for WITH recommendation (using pre-computed list)
            logger.info(
                "\nSample Liquidations with Full Strategy Breakdown (WITH recommendation):"
            )
            if liquidated_with:
                for i, sim in enumerate(liquidated_with[:5]):
                    user_id = sim.get("user", "unknown")
                    detected = sim.get("strategies_detected", [])
                    times = sim.get("times", {})
                    checks = sim.get("checks", {})

                    # Cache sorted results
                    detected_sorted = sorted(detected) if detected else []

                    logger.info(f"\n  User {user_id}:")
                    logger.info(
                        f"    Detected by: {', '.join(detected_sorted)} ({len(detected)}/6 strategies)"
                    )
                    if times:
                        logger.info(f"    Liquidation times by strategy:")
                        # Iterate over sorted detected strategies (matches displayed order)
                        for strategy_name in detected_sorted:
                            if strategy_name in times:
                                time_days = times[strategy_name] / (24 * 3600)
                                check_count = checks.get(strategy_name, "N/A")
                                logger.info(
                                    f"      - {strategy_name}: {time_days:.2f} days ({check_count} checks)"
                                )
                    if len(detected) < 6:
                        not_detected = sim.get("strategies_not_detected", [])
                        logger.info(
                            f"    NOT detected by: {', '.join(sorted(not_detected))} ({len(not_detected)}/6 strategies)"
                        )

                if len(liquidated_with) > 5:
                    logger.info(
                        f"\n  ... and {len(liquidated_with) - 5} more liquidations"
                    )
            else:
                logger.info("  No liquidations detected (WITH recommendation)")

    if deltas:
        try:
            # Use numpy for efficient statistics if available
            deltas_array = np.array(deltas, dtype=float)
            avg = float(np.mean(deltas_array))
            sd = float(np.std(deltas_array))
            median = float(np.median(deltas_array))
            logger.info("\nTime-to-liquidation deltas (with - without) in seconds:")
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

    logger.info(f"Total recommendations assessed for risk: {total_risk_assessed}")
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
            imm_followed_by_liq = s.get("immediate_risk_followed_by_liquidation", 0)
            imm_not_followed_by_liq = s.get(
                "immediate_risk_not_followed_by_liquidation", 0
            )
            total_imm_with_future = imm_followed_by_liq + imm_not_followed_by_liq
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
        logger.info(f"Based on `most_recent_predictions` (lowest time-to-event):")
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


def process_recommendation_wrapper(args_tuple):
    """
    Wrapper function for multiprocessing.Pool.

    This function is required because multiprocessing.Pool.map() needs a function
    that takes a single argument. We pack all arguments into a tuple and unpack
    them here before calling process_recommendation().

    Args:
        args_tuple: Tuple of (item, output_file, profiles_dir)

    Returns:
        Result from process_recommendation() (dict)

    Note:
        This function must be at module level (not nested) to be picklable
        for multiprocessing.
    """
    item, output_file, profiles_dir = args_tuple
    # Update global outputFile and PROFILES_DIR for this process
    global outputFile, PROFILES_DIR
    outputFile = output_file
    PROFILES_DIR = profiles_dir
    return process_recommendation(item)


def normalize_recommendation(item):
    """
    Normalize recommendation format to handle current format.

    Current format: (pandas.Series, dict) - tuple of (recommended_action_series, liquidation_info_dict)

    Returns: (recommendation_dict, liquidation_info_dict)
    """
    try:
        # Expected format: (pandas.Series, dict)
        if isinstance(item, tuple) and len(item) == 2:
            recommendation, liquidation_info = item

            # Convert pandas.Series to dict if needed
            if isinstance(recommendation, pd.Series):
                recommendation_dict = recommendation.to_dict()
            elif isinstance(recommendation, dict):
                recommendation_dict = recommendation
            else:
                raise ValueError(
                    f"Unexpected recommendation type: {type(recommendation)}, expected pandas.Series or dict"
                )

            # Ensure 'symbol' field exists (use 'reserve' as fallback if needed)
            # This is needed because transactions in the simulator use 'symbol' but
            # the training data may have 'reserve' instead
            if "symbol" not in recommendation_dict and "reserve" in recommendation_dict:
                recommendation_dict["symbol"] = recommendation_dict["reserve"]
            elif (
                "symbol" not in recommendation_dict
                and "reserve" not in recommendation_dict
            ):
                logger.warning(
                    f"Recommendation missing both 'symbol' and 'reserve' fields. "
                    f"Available keys: {list(recommendation_dict.keys())}"
                )

            # Validate liquidation_info is a dict
            if isinstance(liquidation_info, dict):
                liquidation_info_dict = liquidation_info
            else:
                logger.warning(
                    f"Liquidation info is not a dict (type: {type(liquidation_info)}), using defaults"
                )
                liquidation_info_dict = {
                    "is_at_risk": False,
                    "is_at_immediate_risk": False,
                    "most_recent_predictions": None,
                    "trend_slopes": None,
                }

            return recommendation_dict, liquidation_info_dict

        # If item is just a dict (shouldn't happen but handle gracefully)
        if isinstance(item, dict):
            logger.warning(
                "Recommendation is a single dict, extracting liquidation_info from dict itself"
            )
            liquidation_info = {
                "is_at_risk": item.get("is_at_risk", False),
                "is_at_immediate_risk": item.get("is_at_immediate_risk", False),
                "most_recent_predictions": item.get("most_recent_predictions"),
                "trend_slopes": item.get("trend_slopes"),
            }
            # Remove liquidation_info fields from recommendation dict
            recommendation_dict = {
                k: v
                for k, v in item.items()
                if k
                not in [
                    "is_at_risk",
                    "is_at_immediate_risk",
                    "most_recent_predictions",
                    "trend_slopes",
                ]
            }
            return recommendation_dict, liquidation_info

        raise ValueError(
            f"Unknown recommendation format: expected tuple of (Series/dict, dict), got {type(item)}"
        )

    except Exception as e:
        logger.error(
            f"Error normalizing recommendation format: {e}, item type: {type(item)}"
        )
        raise


def process_recommendation(item):
    """
    Process a single recommendation and return stats updates.

    Returns:
        dict: Dictionary with keys:
            - 'success': bool
            - 'stats_updates': dict with stats to merge
            - 'error': str (if failed)
    """
    # Initialize user_profile_generator for this process (needed for multiprocessing)
    local_user_profile_generator = UserProfileGenerator(None, WalletInferencer())

    try:
        # Normalize the recommendation format
        recommendation, liquidation_info = normalize_recommendation(item)

        # Validate required fields
        if not isinstance(recommendation, dict):
            logger.error(
                f"Invalid recommendation format after normalization: {type(recommendation)}"
            )
            return {
                "success": False,
                "error": f"Invalid recommendation format: {type(recommendation)}",
                "stats_updates": {},
            }

        user = recommendation.get("user")
        if not user:
            logger.warning(
                f"Recommendation missing 'user' field: {recommendation.keys() if hasattr(recommendation, 'keys') else 'N/A'}"
            )
            return {
                "success": False,
                "error": "Missing user field",
                "stats_updates": {},
            }

        timestamp = recommendation.get("timestamp")
        if timestamp is None:
            logger.warning(f"Recommendation missing 'timestamp' field for user {user}")
            return {
                "success": False,
                "error": "Missing timestamp field",
                "stats_updates": {},
            }

        # Extract liquidation info with defaults
        is_at_risk = liquidation_info.get("is_at_risk", False)
        is_at_immediate_risk = liquidation_info.get("is_at_immediate_risk", False)
        most_recent_predictions = liquidation_info.get("most_recent_predictions")
        trend_slopes = liquidation_info.get("trend_slopes")
        # Search for profile in both non_liquidated_profiles and liquidated_profiles subdirectories
        # PROFILES_DIR should point to a directory containing both subdirectories
        profiles_base = Path(PROFILES_DIR).expanduser()

        # Primary search paths: check the standard structure first
        # 1. non_liquidated_profiles/profiles/
        # 2. liquidated_profiles/profiles/
        search_paths = [
            profiles_base
            / "non_liquidated_profiles"
            / "profiles"
            / f"user_{user}.json",
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
            return {
                "success": False,
                "error": f"Profile file not found for user {user}",
                "stats_updates": {},
            }
        # logger.debug("Checkpoint 1")

        try:
            with user_profile_file.open("r") as f:
                user_profile = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in profile file {user_profile_file}: {e}")
            return {
                "success": False,
                "error": f"Invalid JSON: {e}",
                "stats_updates": {},
            }
        except Exception as e:
            logger.error(f"Error reading profile file {user_profile_file}: {e}")
            return {
                "success": False,
                "error": f"Error reading profile: {e}",
                "stats_updates": {},
            }
        # logger.debug("Checkpoint 2")

        if not isinstance(user_profile, dict):
            logger.error(
                f"Invalid profile format for user {user}: expected dict, got {type(user_profile)}"
            )
            return {
                "success": False,
                "error": f"Invalid profile format: {type(user_profile)}",
                "stats_updates": {},
            }
        # logger.debug("Checkpoint 3")

        if "transactions" not in user_profile:
            logger.warning(f"Profile for user {user} missing 'transactions' field")
            user_profile["transactions"] = []
        # logger.debug("Checkpoint 4")

        # Use the recommendation timestamp as the cutoff point
        # Transactions <= this timestamp are historical, > this timestamp are future
        recommendation_timestamp = float(timestamp)
        cutoff_timestamp = recommendation_timestamp - DEFAULT_TIME_DELTA_SECONDS

        # --- Start of new logic for prediction validation ---
        original_transactions = user_profile.get("transactions", [])

        if not original_transactions:
            logger.warning(f"No transactions found in profile for user {user}")
            return {
                "success": False,
                "error": "No transactions in profile",
                "stats_updates": {},
            }
        # logger.debug("Checkpoint 5")

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
                future_transactions.append(tx)
        # logger.debug("Checkpoint 6")

        # Sort future transactions by timestamp (needed for first liquidation lookup)
        if future_transactions:
            future_transactions.sort(key=lambda x: x.get("timestamp", 0))

        # For "without recommendation" simulation: use only historical transactions
        # (No need to copy - we'll create a deepcopy later for "with" profile)
        user_profile["transactions"] = historical_transactions

        if not user_profile["transactions"]:
            logger.warning(
                f"No historical transactions found before cutoff timestamp {cutoff_timestamp} for user {user}"
            )
            # Still proceed - maybe the user has no history yet
        logger.debug("Checkpoint 7")

        last_transaction_before_recommendation = (
            user_profile["transactions"][-1]
            if user_profile["transactions"]
            else {"timestamp": cutoff_timestamp, "action": "Unknown"}
        )
        last_action_before_recommendation = last_transaction_before_recommendation.get(
            "action", "Unknown"
        )
        outcome_transaction = (
            future_transactions[0]
            if len(future_transactions) > 0
            else last_transaction_before_recommendation
        )
        outcome_action = outcome_transaction.get("action", "Unknown")

        # Initialize stat buckets
        overall_stats = get_new_stats_dict()
        action_stats = get_new_stats_dict()
        outcome_action_stats = get_new_stats_dict()
        action_pair_stats = get_new_stats_dict()

        stat_buckets = [
            overall_stats,
            action_stats,
            outcome_action_stats,
            action_pair_stats,
        ]

        # Build stats updates dictionary
        action_pair = (last_action_before_recommendation, outcome_action)
        stats_updates = {
            "overall": overall_stats,
            "by_index_action": {last_action_before_recommendation: action_stats},
            "by_outcome_action": {outcome_action: outcome_action_stats},
            "by_action_pair": {action_pair: action_pair_stats},
        }

        # Pre-compute liquidation check (single pass, only need first)
        has_future_liquidations = False
        first_liquidation = None
        actual_time_to_liquidation = None
        if future_transactions:
            for tx in future_transactions:
                if tx.get("action", "").lower() == "liquidated":
                    has_future_liquidations = True
                    first_liquidation = tx
                    actual_time_to_liquidation = first_liquidation["timestamp"] - cutoff_timestamp
                    break  # Only need first liquidation

        # Update stats - consolidated logic
        if not is_at_risk:
            for bucket in stat_buckets:
                bucket["not_at_risk"] += 1
            if has_future_liquidations:
                for bucket in stat_buckets:
                    bucket["not_at_risk_but_liquidated"] += 1
                    if actual_time_to_liquidation is not None:
                        bucket.setdefault("not_at_risk_time_to_liquidation", []).append(actual_time_to_liquidation)
        else:  # is_at_risk is True
            for bucket in stat_buckets:
                bucket["at_risk"] += 1

            # Immediate risk check
            if is_at_immediate_risk:
                for bucket in stat_buckets:
                    bucket["at_immediate_risk"] += 1
                if future_transactions:
                    next_action = future_transactions[0].get("action", "").lower()
                    if next_action == "liquidated":
                        for bucket in stat_buckets:
                            bucket["immediate_risk_followed_by_liquidation"] += 1
                    else:
                        for bucket in stat_buckets:
                            bucket["immediate_risk_not_followed_by_liquidation"] += 1
                else:
                    for bucket in stat_buckets:
                        bucket["immediate_risk_no_future"] += 1

            # Eventual liquidation check
            if future_transactions:
                if first_liquidation:
                    time_to_liquidation = (
                        first_liquidation["timestamp"] - cutoff_timestamp
                    )
                    for bucket in stat_buckets:
                        bucket["at_risk_time_to_liquidation"].append(
                            time_to_liquidation
                        )
                        bucket["at_risk_eventual_liquidation_count"] += 1
                else:
                    for bucket in stat_buckets:
                        bucket["at_risk_no_future_liquidation_count"] += 1
            else:
                for bucket in stat_buckets:
                    bucket["at_risk_no_future"] += 1

            # Capture correlation data (regardless of risk classification)
            if has_future_liquidations and actual_time_to_liquidation is not None and most_recent_predictions:
                predicted_time = most_recent_predictions.get("Liquidated")
                if predicted_time is not None:
                    for bucket in stat_buckets:
                        bucket.setdefault("liquidation_prediction_correlation", []).append({
                            "predicted": float(predicted_time),
                            "actual": float(actual_time_to_liquidation)
                        })

            # Cache next action for prediction checks (compute once, use for both)
            next_actual_action_lower = None
            if future_transactions:
                next_actual_action_lower = (
                    future_transactions[0].get("action", "").lower()
                )

            # `most_recent_predictions` check
            if (
                most_recent_predictions
                and future_transactions
                and next_actual_action_lower
            ):
                for bucket in stat_buckets:
                    bucket["total_predictions_checked_mrp"] += 1
                valid_predictions = {
                    k: v for k, v in most_recent_predictions.items() if v is not None
                }
                if valid_predictions:
                    predicted_next_event = min(
                        valid_predictions, key=valid_predictions.get
                    )
                    if predicted_next_event.lower() == next_actual_action_lower:
                        for bucket in stat_buckets:
                            bucket["prediction_matches_next_action_mrp"] += 1

            # `trend_slopes` check
            if trend_slopes and future_transactions and next_actual_action_lower:
                for bucket in stat_buckets:
                    bucket["total_predictions_checked_ts"] += 1
                valid_slopes = {k: v for k, v in trend_slopes.items() if v is not None}
                if valid_slopes:
                    predicted_next_event_by_slope = min(
                        valid_slopes, key=valid_slopes.get
                    )
                    if (
                        predicted_next_event_by_slope.lower()
                        == next_actual_action_lower
                    ):
                        for bucket in stat_buckets:
                            bucket["prediction_matches_next_action_ts"] += 1
        # --- End of new logic ---

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
        # logger.debug("Checkpoint 7.5")

        # Run (or load) results without the recommendation
        # This simulates: "What happens if user continues without taking the recommendation?"
        # Note: run_simulation() uses sophisticated liquidation detection consistent with Aave-Simulator:
        # - Dynamic margin thresholds (scales with time gap, base 1.10)
        # - Enhanced HF calculations (oracle delay, volatility adjustments)
        # - Dust liquidation detection
        # - Effective liquidation threshold checks
        # Uses INVESTIGATION_PARAMS defaults (oracle_delay=300s, margin_threshold=1.10, volatility_discount=0.08)
        try:
            results_without_recommendation = load_results_cache(
                recommendation,
                "without",
                profile=user_profile,
                lookahead_seconds=lookahead_seconds,
                output_file=outputFile
            )
        except Exception as e:
            import traceback

            logger.error(
                f"Error running simulation without recommendation for user {user}: {e}"
            )
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            return {
                "success": False,
                "error": f"Simulation without recommendation failed: {e}",
                "stats_updates": {},
            }
        # logger.debug("Checkpoint 8")

        recommendation = update_recommendation_if_necessary(
            recommendation, results_without_recommendation
        )
        # logger.debug("Checkpoint 9")

        if recommendation is None:
            logger.info(
                f"Skipping Non-Repay Recommendation in Dust Scenario for user {user}."
            )
            return {
                "success": False,
                "stats_updates": stats_updates,
                "user": user,
                "skipped": True,
            }

        # logger.debug("Checkpoint 10")
        # Prepare a copy of the profile that includes the recommendation transaction
        # This simulates: "What happens if user takes the recommendation?"
        try:
            user_profile_with = copy.deepcopy(user_profile)
            recommended_transaction = local_user_profile_generator._row_to_transaction(
                recommendation
            )
            if recommended_transaction is None:
                logger.warning(
                    f"Failed to convert recommendation to transaction for user {user}"
                )
                return {
                    "success": False,
                    "error": "Failed to convert recommendation to transaction",
                    "stats_updates": {},
                }

            # Ensure the recommended transaction has the correct timestamp
            # It should be at the recommendation time, or slightly after the last historical transaction
            recommended_transaction["timestamp"] = recommendation_timestamp

            # Insert the recommendation transaction in the correct chronological position
            # Since transactions are sorted by timestamp in run_simulation, we just need to append
            # and ensure it's at the right time
            user_profile_with["transactions"].append(recommended_transaction)

            # Note: run_simulation will sort by timestamp, so the recommendation will be executed
            # after historical transactions and before any future transactions we might add
        except Exception as e:
            logger.error(
                f"Error preparing profile with recommendation for user {user}: {e}"
            )
            return {
                "success": False,
                "error": f"Error preparing profile with recommendation: {e}",
                "stats_updates": {},
            }
        # logger.debug("Checkpoint 11")

        # Run (or load) results with the recommendation
        # Uses same sophisticated liquidation detection as above (consistent with Aave-Simulator)
        lookahead_seconds_for_recommendation = max(
            1, lookahead_seconds - (recommendation_timestamp - cutoff_timestamp) * 2
        )
        try:
            results_with_recommendation = load_results_cache(
                recommendation,
                "with",
                profile=user_profile_with,
                lookahead_seconds=lookahead_seconds_for_recommendation,
                output_file=outputFile
            )
        except Exception as e:
            import traceback

            logger.error(
                f"Error running simulation with recommendation for user {user}: {e}"
            )
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            return {
                "success": False,
                "error": f"Simulation with recommendation failed: {e}",
                "stats_updates": {},
            }
        logger.debug("Checkpoint 12")

        if results_with_recommendation['liquidation_stats']["liquidated"] and (results_with_recommendation['final_state']["total_debt_usd"] == 0 or results_with_recommendation['liquidation_stats']["liquidation_reason"] == "dust_liquidation (total_debt_usd (0.000000) < $1.00)"):
            logger.info(
                f"Liquidated despite 0 debt."
            )
            return {
                "success": False,
                "error": f"Liquidated despite 0 debt.",
                "stats_updates": {},
            }
        
        if (results_without_recommendation['liquidation_stats']['liquidated']) and (results_without_recommendation['liquidation_stats']['time_to_liquidation'] < DEFAULT_TIME_DELTA_SECONDS) and (results_with_recommendation['liquidation_stats']["liquidated"]):
            logger.info(
                f"Skipping because liquidation too fast for user {user}. {results_without_recommendation['liquidation_stats']['time_to_liquidation'] < DEFAULT_TIME_DELTA_SECONDS} {results_without_recommendation['liquidation_stats']['liquidated']} {results_with_recommendation['liquidation_stats']["liquidated"]}"
            )
            return {
                "success": False,
                "error": f"Skipping because liquidation too fast for user {user}.",
                "stats_updates": stats_updates,
                "user": user,
                "skipped": True,
            }

        # Update runtime stats with simulation results
        liquidation_stats_without = results_without_recommendation.get(
            "liquidation_stats", {}
        )
        liquidation_stats_with = results_with_recommendation.get(
            "liquidation_stats", {}
        )

        # Extract strategy comparison data if available
        strategy_comparison_without = liquidation_stats_without.get(
            "strategy_comparison", {}
        )
        strategy_comparison_with = liquidation_stats_with.get("strategy_comparison", {})

        lw = bool(liquidation_stats_without.get("liquidated", False))
        lw_with = bool(liquidation_stats_with.get("liquidated", False))

        # Track strategy comparison results (optimized: extract once, use for all buckets)
        if strategy_comparison_without and strategy_comparison_without.get(
            "results_by_strategy"
        ):
            consensus_without = strategy_comparison_without.get("consensus", {})
            results_by_strategy = strategy_comparison_without.get(
                "results_by_strategy", {}
            )
            best_strategy_without = strategy_comparison_without.get("best_strategy")
            consensus_agreement_rate = consensus_without.get("agreement_rate", 0.0)

            # Extract per-simulation data ONCE (same for all buckets)
            per_sim_data_without = {
                "user": user,
                "strategies_detected": [],  # List of strategy names that detected liquidation
                "strategies_not_detected": [],  # List of strategy names that did NOT detect
                "times": {},  # Dict: {strategy_name: time_to_liquidation_seconds}
                "checks": {},  # Dict: {strategy_name: checks_performed}
                "consensus_agreement": consensus_agreement_rate,
                "consensus_liquidated": consensus_without.get("liquidated", False),
                "average_time": consensus_without.get("average_liquidation_time"),
            }

            # Process strategy results ONCE (data is same for all buckets)
            strategy_updates = (
                {}
            )  # {strategy_name: {"detected": bool, "time": float|None, "checks": int}}
            for strategy_name, strategy_result in results_by_strategy.items():
                liquidated = strategy_result.get("liquidated", False)
                time_to_liquidation = strategy_result.get("time_to_liquidation")
                checks_performed = strategy_result.get("checks_performed", 0)

                strategy_updates[strategy_name] = {
                    "detected": liquidated,
                    "time": time_to_liquidation,
                    "checks": checks_performed,
                }

                # Populate per-simulation data
                if liquidated:
                    per_sim_data_without["strategies_detected"].append(strategy_name)
                    if time_to_liquidation is not None:
                        per_sim_data_without["times"][
                            strategy_name
                        ] = time_to_liquidation
                else:
                    per_sim_data_without["strategies_not_detected"].append(
                        strategy_name
                    )

                per_sim_data_without["checks"][strategy_name] = checks_performed

            # Apply updates to all buckets (reuse extracted data)
            for bucket in stat_buckets:
                bucket["strategy_comparisons"]["consensus_agreement_without"].append(
                    consensus_agreement_rate
                )

                if best_strategy_without:
                    bucket["strategy_comparisons"]["best_strategy_counts"]["without"][
                        best_strategy_without
                    ] = (
                        bucket["strategy_comparisons"]["best_strategy_counts"][
                            "without"
                        ].get(best_strategy_without, 0)
                        + 1
                    )

                # Apply pre-computed strategy updates
                for strategy_name, updates in strategy_updates.items():
                    if updates["detected"]:
                        bucket["strategy_comparisons"]["without"][strategy_name][
                            "detected"
                        ] += 1
                        if updates["time"] is not None:
                            bucket["strategy_comparisons"]["without"][strategy_name][
                                "time"
                            ].append(updates["time"])
                    bucket["strategy_comparisons"]["without"][strategy_name][
                        "checks"
                    ] += updates["checks"]

            # Store per-simulation data ONCE (only in first bucket to avoid duplication)
            if (
                per_sim_data_without["strategies_detected"]
                or per_sim_data_without["strategies_not_detected"]
            ):
                stat_buckets[0]["strategy_comparisons"][
                    "per_simulation_without"
                ].append(per_sim_data_without)

        if strategy_comparison_with and strategy_comparison_with.get(
            "results_by_strategy"
        ):
            consensus_with = strategy_comparison_with.get("consensus", {})
            results_by_strategy = strategy_comparison_with.get(
                "results_by_strategy", {}
            )
            best_strategy_with = strategy_comparison_with.get("best_strategy")
            consensus_agreement_rate = consensus_with.get("agreement_rate", 0.0)

            # Extract per-simulation data ONCE (same for all buckets)
            per_sim_data_with = {
                "user": user,
                "strategies_detected": [],  # List of strategy names that detected liquidation
                "strategies_not_detected": [],  # List of strategy names that did NOT detect
                "times": {},  # Dict: {strategy_name: time_to_liquidation_seconds}
                "checks": {},  # Dict: {strategy_name: checks_performed}
                "consensus_agreement": consensus_agreement_rate,
                "consensus_liquidated": consensus_with.get("liquidated", False),
                "average_time": consensus_with.get("average_liquidation_time"),
            }

            # Process strategy results ONCE (data is same for all buckets)
            strategy_updates = (
                {}
            )  # {strategy_name: {"detected": bool, "time": float|None, "checks": int}}
            for strategy_name, strategy_result in results_by_strategy.items():
                liquidated = strategy_result.get("liquidated", False)
                time_to_liquidation = strategy_result.get("time_to_liquidation")
                checks_performed = strategy_result.get("checks_performed", 0)

                strategy_updates[strategy_name] = {
                    "detected": liquidated,
                    "time": time_to_liquidation,
                    "checks": checks_performed,
                }

                # Populate per-simulation data
                if liquidated:
                    per_sim_data_with["strategies_detected"].append(strategy_name)
                    if time_to_liquidation is not None:
                        per_sim_data_with["times"][strategy_name] = time_to_liquidation
                else:
                    per_sim_data_with["strategies_not_detected"].append(strategy_name)

                per_sim_data_with["checks"][strategy_name] = checks_performed

            # Apply updates to all buckets (reuse extracted data)
            for bucket in stat_buckets:
                bucket["strategy_comparisons"]["consensus_agreement_with"].append(
                    consensus_agreement_rate
                )

                if best_strategy_with:
                    bucket["strategy_comparisons"]["best_strategy_counts"]["with"][
                        best_strategy_with
                    ] = (
                        bucket["strategy_comparisons"]["best_strategy_counts"][
                            "with"
                        ].get(best_strategy_with, 0)
                        + 1
                    )

                # Apply pre-computed strategy updates
                for strategy_name, updates in strategy_updates.items():
                    if updates["detected"]:
                        bucket["strategy_comparisons"]["with"][strategy_name][
                            "detected"
                        ] += 1
                        if updates["time"] is not None:
                            bucket["strategy_comparisons"]["with"][strategy_name][
                                "time"
                            ].append(updates["time"])
                    bucket["strategy_comparisons"]["with"][strategy_name][
                        "checks"
                    ] += updates["checks"]

            # Store per-simulation data ONCE (only in first bucket to avoid duplication)
            if (
                per_sim_data_with["strategies_detected"]
                or per_sim_data_with["strategies_not_detected"]
            ):
                stat_buckets[0]["strategy_comparisons"]["per_simulation_with"].append(
                    per_sim_data_with
                )

        # Extract liquidation reasons and categorize (optimized: compute .lower() once per reason)
        reason_without = liquidation_stats_without.get("liquidation_reason")
        reason_with = liquidation_stats_with.get("liquidation_reason")

        # Pre-compute categorization flags (compute once, use for all buckets)
        is_dust_without = False
        is_hf_without = False
        is_threshold_without = False
        if lw and reason_without:
            reason_without_lower = reason_without.lower()
            is_dust_without = "dust" in reason_without_lower
            is_hf_without = (
                "hf" in reason_without_lower
                or "health factor" in reason_without_lower
                or "margin" in reason_without_lower
            )
            is_threshold_without = (
                "threshold" in reason_without_lower
                or "effective lt" in reason_without_lower
                or "effective liquidation" in reason_without_lower
            )

        is_dust_with = False
        is_hf_with = False
        is_threshold_with = False
        if lw_with and reason_with:
            reason_with_lower = reason_with.lower()
            is_dust_with = "dust" in reason_with_lower
            is_hf_with = (
                "hf" in reason_with_lower
                or "health factor" in reason_with_lower
                or "margin" in reason_with_lower
            )
            is_threshold_with = (
                "threshold" in reason_with_lower
                or "effective lt" in reason_with_lower
                or "effective liquidation" in reason_with_lower
            )

        if lw:
            for bucket in stat_buckets:
                bucket["liquidated_without"] += 1
                if reason_without:
                    bucket["liquidation_reasons_without"].append(reason_without)
                    if is_dust_without:
                        bucket["dust_liquidations_without"] += 1
                    if is_hf_without:
                        bucket["hf_based_liquidations_without"] += 1
                    if is_threshold_without:
                        bucket["threshold_based_liquidations_without"] += 1

        if lw_with:
            for bucket in stat_buckets:
                bucket["liquidated_with"] += 1
                if reason_with:
                    bucket["liquidation_reasons_with"].append(reason_with)
                    if is_dust_with:
                        bucket["dust_liquidations_with"] += 1
                    if is_hf_with:
                        bucket["hf_based_liquidations_with"] += 1
                    if is_threshold_with:
                        bucket["threshold_based_liquidations_with"] += 1

        if lw and not lw_with:
            for bucket in stat_buckets:
                bucket["improved"] += 1
        elif (not lw) and lw_with:
            for bucket in stat_buckets:
                bucket["worsened"] += 1
        else:
            for bucket in stat_buckets:
                bucket["no_change"] += 1
                if lw:
                    bucket["no_change_with_liquidation"] += 1
                else:
                    bucket["no_change_without_liquidation"] += 1

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

        # Mark as processed (only increment once at the end)
        for bucket in stat_buckets:
            bucket["processed"] += 1

        stats_updates_no_dust = {}
        if not (is_dust_without or is_dust_with):
            stats_updates_no_dust = copy.deepcopy(stats_updates)

        return {
            "success": True,
            "stats_updates": stats_updates,
            "stats_updates_no_dust": stats_updates_no_dust,
            "user": user,
        }

    except Exception as e:
        logger.error(f"Error processing recommendation: {e}", exc_info=True)
        return {"success": False, "error": str(e), "stats_updates": {}}


# Main execution with error handling
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run simulations for recommendations with multiprocessing support"
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=None,
        help=f"Number of worker processes (default: min(available_cores, total_recommendations)). Available cores: {cpu_count()}",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Custom log file path (default: output7.log)",
    )
    args = parser.parse_args()

    # Set log file if specified
    if args.log_file:
        set_log_file(args.log_file)
        outputFile = args.log_file

    try:
        total_recommendations = len(recommendations)
        logger.info(
            f"Starting processing of {total_recommendations} recommendations..."
        )

        if total_recommendations == 0:
            logger.warning("No recommendations to process!")
        else:
            # Determine number of workers
            if args.workers is not None:
                num_workers = min(args.workers, total_recommendations, cpu_count())
                logger.info(
                    f"Using {num_workers} worker processes (requested: {args.workers}, available cores: {cpu_count()})"
                )
            else:
                num_workers = min(cpu_count(), total_recommendations)
                logger.info(
                    f"Using {num_workers} worker processes (auto-detected, out of {cpu_count()} available CPU cores)"
                )

            # Prepare arguments for multiprocessing
            args_list = [
                (item, outputFile, PROFILES_DIR) for item in recommendations.values()
            ]

            processed_count = 0
            start_time = time.time()

            # Process in parallel with progress tracking
            if num_workers > 1:
                with Pool(processes=num_workers) as pool:
                    # Use imap for progress tracking
                    results_iter = pool.imap(process_recommendation_wrapper, args_list)
                    for result in results_iter:
                        processed_count += 1

                        # Merge stats updates from this result
                        if result and result.get("success", False):
                            stats_updates = result.get("stats_updates", {})
                            if stats_updates:
                                merge_stats_updates(stats, stats_updates)

                            stats_updates_no_dust = result.get(
                                "stats_updates_no_dust", {}
                            )
                            if stats_updates_no_dust:
                                merge_stats_updates(
                                    stats_no_dust, stats_updates_no_dust
                                )

                        # Progress logging (optimized: compute elapsed once, reduce logging frequency)
                        if processed_count % 100 == 0:
                            elapsed = time.time() - start_time
                            rate = processed_count / elapsed if elapsed > 0 else 0
                            remaining = total_recommendations - processed_count
                            eta_seconds = remaining / rate if rate > 0 else 0

                            # Format ETA message efficiently
                            if eta_seconds >= 3600:
                                eta_str = f"{eta_seconds/3600:.1f} hours"
                            elif eta_seconds >= 60:
                                eta_str = f"{eta_seconds/60:.1f} minutes"
                            else:
                                eta_str = f"{eta_seconds:.0f} seconds"

                            logger.info(
                                f"Processed {processed_count}/{total_recommendations} recommendations... "
                                f"({rate:.1f} recs/sec, ETA: {eta_str})"
                            )

                            # Print overall stats summary (detailed breakdowns only every 500 items)
                            _print_stats_summary(
                                stats["overall"], "=== Overall Simulation Summary ==="
                            )

                            # Detailed breakdowns less frequently (reduces I/O overhead)
                            if processed_count % 500 == 0:
                                for action, action_stats in sorted(
                                    stats["by_index_action"].items()
                                ):
                                    _print_stats_summary(
                                        action_stats,
                                        f"=== Simulation Summary for index_action = {action} ===",
                                    )
                                for action, action_stats in sorted(
                                    stats["by_outcome_action"].items()
                                ):
                                    _print_stats_summary(
                                        action_stats,
                                        f"=== Simulation Summary for outcome_action = {action} ===",
                                    )
                                for action_pair, action_pair_stats in sorted(
                                    stats["by_action_pair"].items()
                                ):
                                    _print_stats_summary(
                                        action_pair_stats,
                                        f"=== Simulation Summary for action_pair = {action_pair} ===",
                                    )
            else:
                # Sequential processing (fallback for single worker)
                logger.info("Using sequential processing (1 worker)")
                for args_tuple in args_list:
                    result = process_recommendation_wrapper(args_tuple)
                    processed_count += 1

                    # Merge stats updates
                    if result and result.get("success", False):
                        stats_updates = result.get("stats_updates", {})
                        if stats_updates:
                            merge_stats_updates(stats, stats_updates)

                        stats_updates_no_dust = result.get("stats_updates_no_dust", {})
                        if stats_updates_no_dust:
                            merge_stats_updates(stats_no_dust, stats_updates_no_dust)

                    if processed_count % 100 == 0:
                        elapsed = time.time() - start_time
                        rate = processed_count / elapsed if elapsed > 0 else 0
                        logger.info(
                            f"Processed {processed_count}/{total_recommendations} recommendations... "
                            f"({rate:.1f} recs/sec)"
                        )
                        _print_stats_summary(
                            stats["overall"], "=== Overall Simulation Summary ==="
                        )

            elapsed_time = time.time() - start_time
            hours = int(elapsed_time // 3600)
            minutes = int((elapsed_time % 3600) // 60)
            seconds = int(elapsed_time % 60)
            if hours > 0:
                time_str = f"{hours}h {minutes}m {seconds}s"
            elif minutes > 0:
                time_str = f"{minutes}m {seconds}s"
            else:
                time_str = f"{seconds}s"

            logger.info(
                f"Processing completed in {time_str} ({elapsed_time:.2f} seconds)"
            )
            logger.info(f"Processed {processed_count} recommendations")
            if processed_count > 0:
                avg_time_per_rec = elapsed_time / processed_count
                logger.info(
                    f"Average time per recommendation: {avg_time_per_rec:.2f} seconds"
                )

            # Print final summary
            logger.info("\n" + "=" * 80)
            logger.info("FINAL SUMMARY")
            logger.info("=" * 80)
            _print_stats_summary(stats["overall"], "=== Overall Simulation Summary ===")

            # Save statistics to JSON file
            save_statistics_to_json(stats)
            save_statistics_to_json(stats_no_dust, suffix="no_dust")

    except KeyboardInterrupt:
        logger.warning("Processing interrupted by user")
        # Try to save statistics even if interrupted
        try:
            save_statistics_to_json(stats, suffix="INTERRUPTED")
            save_statistics_to_json(stats_no_dust, suffix="no_dust_INTERRUPTED")
        except Exception as save_error:
            logger.error(f"Failed to save statistics after interruption: {save_error}")
    except Exception as e:
        logger.error(f"Fatal error during processing: {e}", exc_info=True)
        # Try to save statistics even on fatal error
        try:
            save_statistics_to_json(stats, suffix="ERROR")
            save_statistics_to_json(stats_no_dust, suffix="no_dust_ERROR")
        except Exception:
            pass
        raise
