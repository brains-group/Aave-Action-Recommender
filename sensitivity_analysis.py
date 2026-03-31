#!/usr/bin/env python3
"""
This script performs a sensitivity analysis on the coefficients of the
`calculate_trend_slope` function in `actionAgentTraining.py`.

It iterates through different combinations of coefficients for a sample of
the training data and reports how many times the liquidation risk prediction
changes compared to the baseline coefficients.

This helps justify the choice of the "magic numbers" in the trend calculation.
"""

import argparse
import pandas as pd
import numpy as np
from itertools import product, groupby
from functools import partial
from tqdm import tqdm
import os

from actionAgentTraining import (
    get_train_set,
    determine_liquidation_risk,
    init_worker,
)
from utils.logger import logger, set_log_file
import utils.data
import utils.model_training
from multiprocessing import Pool, cpu_count, Manager

set_log_file("output_sensitivity_analysis.log")


def process_sample_row_binary_search(row, grouped_by_dist, original_coeffs):
    """
    Worker function to find the minimum distance for a prediction change
    using a binary search approach on distance groups.
    """
    # Get baseline prediction
    try:
        base_at_risk, _, _, _ = determine_liquidation_risk(row, *original_coeffs)
    except Exception as e:
        logger.warning(
            f"Could not process row {row['user']} at {row['timestamp']} for baseline. Error: {e}"
        )
        return None

    # Per user request, first check the most extreme difference.
    # If no change is detected, we assume no smaller change will cause one and stop.
    # This is an optimization based on an assumption of monotonicity.
    if grouped_by_dist:
        extreme_group = grouped_by_dist[-1]
        change_found_at_extreme = False
        for coeffs in extreme_group["combinations"]:
            try:
                new_at_risk, _, _, _ = determine_liquidation_risk(row, *coeffs)
                if new_at_risk != base_at_risk:
                    change_found_at_extreme = True
                    break
            except Exception:
                continue  # Ignore errors in specific coefficient runs

        if not change_found_at_extreme:
            return {
                "user": row["user"],
                "timestamp": row["timestamp"],
                "base_prediction": base_at_risk,
                "change_found": False,
                "min_distance": None,
            }

    # If a change was found at the extreme, proceed with binary search to find the minimum distance.
    low = 0
    high = len(grouped_by_dist) - 1
    min_change_distance = float("inf")

    while low <= high:
        mid_idx = (low + high) // 2
        group = grouped_by_dist[mid_idx]
        coeffs_to_test = group["combinations"]
        distance_of_group = group["distance"]

        change_found_in_group = False
        for coeffs in coeffs_to_test:
            try:
                new_at_risk, _, _, _ = determine_liquidation_risk(row, *coeffs)
                if new_at_risk != base_at_risk:
                    change_found_in_group = True
                    break  # Stop checking this group, as requested
            except Exception as e:
                logger.debug(f"Could not process row with coeffs {coeffs}. Error: {e}")
                continue

        if change_found_in_group:
            # Change found, this distance is a candidate for the minimum.
            # Try to find an even smaller distance.
            min_change_distance = min(min_change_distance, distance_of_group)
            high = mid_idx - 1
        else:
            # No change found, need to look at larger distances.
            low = mid_idx + 1

    final_distance = min_change_distance if min_change_distance != float("inf") else None

    return {
        "user": row["user"],
        "timestamp": row["timestamp"],
        "base_prediction": base_at_risk,
        "change_found": final_distance is not None,
        "min_distance": final_distance,
    }


def run_sensitivity_analysis(
    sample_size=100, num_workers=None, cache_file="./cache/sensitivity_results.pkl", load_cache=False
):
    """
    Runs a sensitivity analysis on the trend slope coefficients.
    """
    logger.info("Starting sensitivity analysis for trend slope coefficients.")

    # 1. Define coefficient ranges, calculate distances, and group them.
    original_coeffs = (0.8, 0.6, 0.3)

    # Define wider, incremental coefficient ranges
    acceleration_coeffs = np.round(np.linspace(0.6, 1.0, 21), 2)
    volatility_coeffs = np.round(np.linspace(0.4, 0.8, 21), 2)
    momentum_coeffs = np.round(np.linspace(0.1, 0.5, 21), 2)

    all_combinations = list(
        product(acceleration_coeffs, volatility_coeffs, momentum_coeffs)
    )

    original_coeffs_arr = np.array(original_coeffs)

    # Calculate distance for each combination and round for grouping
    combs_with_dist = [
        (coeffs, round(np.linalg.norm(np.array(coeffs) - original_coeffs_arr), 4))
        for coeffs in all_combinations
    ]

    # Sort by distance to prepare for grouping
    combs_with_dist.sort(key=lambda x: x[1])

    # Group combinations by their rounded distance, excluding the zero-distance group
    grouped_by_dist = [
        {"distance": distance, "combinations": [item[0] for item in group]}
        for distance, group in groupby(combs_with_dist, key=lambda x: x[1])
        if distance > 0
    ]

    max_distance_tested = grouped_by_dist[-1]["distance"]

    if load_cache and os.path.exists(cache_file):
        logger.info(f"Loading cached results from {cache_file}")
        df_results = pd.read_pickle(cache_file)
    else:
        logger.info(
            f"Created {len(grouped_by_dist)} groups of coefficients based on distance to test."
        )

        # 2. Get a sample from the training set
        train_set = get_train_set()
        if sample_size >= len(train_set):
            sample_set = train_set
            logger.info(f"Using full training set of {len(train_set)} samples.")
        else:
            sample_set = train_set.sample(n=sample_size, random_state=42)
            logger.info(f"Using a sample of {sample_size} from the training set.")

        # 3. Run analysis
        if num_workers is None:
            num_workers = cpu_count()
        num_workers = min(num_workers, len(sample_set))

        if num_workers > 1:
            logger.info(f"Using {num_workers} worker processes.")
            manager = Manager()
            # Share caches
            shared_event_df_cache = manager.dict()
            shared_event_df_cache.update(utils.data.EVENT_DF_CACHE)
            shared_preprocess_cache = manager.dict()
            shared_preprocess_cache.update(utils.model_training.PREPROCESS_CACHE)
            shared_models_cache = manager.dict()
            shared_models_cache.update(utils.model_training.MODELS_CACHE)

            with Pool(
                processes=num_workers,
                initializer=init_worker,
                initargs=(
                    shared_event_df_cache,
                    shared_preprocess_cache,
                    shared_models_cache,
                ),
            ) as pool:
                worker_func = partial(
                    process_sample_row_binary_search,
                    grouped_by_dist=grouped_by_dist,
                    original_coeffs=original_coeffs,
                )

                results = list(
                    tqdm(
                        pool.imap(worker_func, [row for _, row in sample_set.iterrows()]),
                        total=len(sample_set),
                        desc="Analyzing samples",
                    )
                )
                results = [r for r in results if r is not None]

        else:  # Sequential
            logger.info("Using sequential processing.")
            results = []
            for _, row in tqdm(
                sample_set.iterrows(), total=len(sample_set), desc="Analyzing samples"
            ):
                result = process_sample_row_binary_search(
                    row, grouped_by_dist, original_coeffs
                )
                if result:
                    results.append(result)

        if not results:
            logger.warning("No results were generated. Exiting.")
            return

        df_results = pd.DataFrame(results)
        df_results.to_pickle(cache_file)
        logger.info(f"Saved results to {cache_file}")

    # 4. Summarize results
    if df_results.empty:
        logger.warning("No results are available for summarization. Exiting.")
        return

    total_samples = len(df_results)
    samples_with_changes = len(df_results[df_results["change_found"] == True])

    # Calculate Stability Score
    # For samples with no change, effective distance is max_distance_tested
    effective_distances = df_results.apply(
        lambda row: row["min_distance"] if row["change_found"] else max_distance_tested,
        axis=1,
    )

    # Stability Score: Mean of (effective_distance / max_distance_tested)
    # Ranges from 0 to 1, where 1 means completely robust/stable across all tests.
    stability_score = (effective_distances / max_distance_tested).mean()
    finickiness_score = 1.0 - stability_score

    logger.info("\n--- Sensitivity Analysis Summary ---")
    logger.info(f"Total samples analyzed: {total_samples}")
    if total_samples > 0:
        logger.info(
            f"Samples where risk prediction changed: {samples_with_changes} ({samples_with_changes/total_samples:.2%})"
        )

        at_risk_base = len(df_results[df_results["base_prediction"] == True])
        logger.info(
            f"Samples predicted 'at risk' with original coefficients: {at_risk_base} ({at_risk_base/total_samples:.2%})"
        )

        logger.info(f"Stability Score:   {stability_score:.4f} (1.0 is perfectly stable)")
        logger.info(
            f"Finickiness Score: {finickiness_score:.4f} (0.0 means not finicky at all)"
        )

        if samples_with_changes > 0:
            changed_df = df_results[df_results["change_found"] == True]

            base_risk_in_changed = len(changed_df[changed_df["base_prediction"] == True])
            if samples_with_changes > 0:
                logger.info(
                    f"Of the samples that changed, {base_risk_in_changed} ({base_risk_in_changed/samples_with_changes:.2%}) were originally 'at risk'."
                )

            # Analyze the magnitude of coefficient changes
            all_distances = changed_df["min_distance"].dropna().tolist()
            if all_distances:
                logger.info("\n--- Analysis of Coefficient Change Magnitude ---")
                logger.info(
                    "Statistics on the minimum Euclidean distance of coefficient vectors needed to cause a prediction change:"
                )
                logger.info(f"  Total samples with a change: {len(all_distances)}")
                logger.info(f"  Average min distance for a change: {np.mean(all_distances):.4f}")
                logger.info(f"  Median min distance for a change:  {np.median(all_distances):.4f}")
                logger.info(f"  Min distance for a change:         {np.min(all_distances):.4f}")
                logger.info(f"  Max distance for a change:         {np.max(all_distances):.4f}")
                logger.info(f"  Std Dev of distance:               {np.std(all_distances):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run sensitivity analysis on trend slope coefficients."
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Number of samples to use from the training set.",
    )
    parser.add_argument(
        "--num-workers",
        "-w",
        type=int,
        default=None,
        help=f"Number of worker processes (default: all available cores).",
    )
    parser.add_argument(
        "--cache-file",
        type=str,
        default="./cache/sensitivity_results.pkl",
        help="File to save or load cached results.",
    )
    parser.add_argument(
        "--load-cache",
        action="store_true",
        help="Load results from the cache file instead of re-running the predictions.",
    )
    args = parser.parse_args()

    run_sensitivity_analysis(
        sample_size=args.sample_size,
        num_workers=args.num_workers,
        cache_file=args.cache_file,
        load_cache=args.load_cache,
    )
