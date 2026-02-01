import pandas as pd
import numpy as np
import os
import pickle as pkl
import glob
from utils.data import get_date_ranges, get_event_df, get_train_set
from utils.model_training import preprocess, get_model_for_pair_and_date
from utils.constants import *
from utils.logger import logger, set_log_file
from utils.simulations import (
    get_limited_user_profile,
    get_price_history_value,
    get_simulation_outcome,
    would_create_dust_position,
)

set_log_file("output_recommendationGeneration.log")

np.random.seed(seed)


def get_expected_time_to_event(model, X_test, baseline_meta, max_prediction_days=365):
    """
    Calculates the 'Return Period' (inverse probability scaled by time).
    This serves as a robust 'Time to Event' proxy for rare events.
    Lower value = Higher Risk (Event expected sooner).
    """
    times = baseline_meta["times"]
    cum_hazards = baseline_meta["cum_hazards"]
    max_t = baseline_meta["max_time"]
    final_rate = baseline_meta["final_rate"]
    log_shift = baseline_meta["log_shift"]

    # 1. Define a standard window for probability estimation (e.g., 7 days)
    # This acts as our "instantaneous" risk measurement window.
    window_seconds = 7 * 24 * 3600

    # 2. Get Baseline Hazard for this window
    if window_seconds <= max_t:
        idx = np.searchsorted(times, window_seconds) - 1
        idx = max(0, min(idx, len(cum_hazards) - 1))
        h0 = cum_hazards[idx]
    else:
        excess = window_seconds - max_t
        h0 = cum_hazards[-1] + (excess * final_rate)

    # 3. Calculate Risk Probability for each user
    log_margin = model.predict(X_test, output_margin=True)
    relative_risk = np.exp(np.clip(log_margin - log_shift, -20, 20))

    # Prob = 1 - exp(-H0 * RR)
    # We clip the exponent to avoid overflow/underflow
    exponent = -h0 * relative_risk
    exponent = np.clip(exponent, -50, 0)
    probability = 1.0 - np.exp(exponent)

    # 4. Calculate Return Period: Window / Probability
    # If prob is 1.0, time is Window.
    # If prob is tiny, time is huge.
    # Add epsilon to prob to prevent division by zero.
    probability = np.maximum(probability, 1e-12)

    return window_seconds / probability



def get_user_history(user_id: str, up_to_timestamp: int) -> pd.DataFrame:
    all_events = []
    for index_event in EVENTS:
        for outcome_event in EVENTS:
            # event_path = os.path.join(DATA_PATH, index_event, outcome_event, "data.csv")
            # use cached loader
            event_df = get_event_df(index_event, outcome_event)
            if not (event_df is None):
                user_events = event_df[
                    (event_df["user"] == user_id)
                    & (event_df["timestamp"] <= up_to_timestamp)
                ]
                if len(user_events):
                    all_events.append(user_events)

                liquidated_events = user_events[
                    (user_events["Index Event"] == "liquidated")
                    & (user_events["Outcome Event"] == "liquidated")
                ]
                if liquidated_events.shape[0] > 0:
                    logger.warning(
                        f"User {user_id} had liquidated->liquidated events in {index_event}->{outcome_event} before {up_to_timestamp}"
                    )
                    logger.debug("\n" + str(liquidated_events))
    if all_events:
        user_history_df = pd.concat(all_events).sort_values(by="timestamp")
    else:
        user_history_df = pd.DataFrame()
    if len(user_history_df) > 1000:
        user_history_df = user_history_df.tail(10000).reset_index(drop=True)
    return user_history_df


def calc_predictions(index_event_value, group, results, model_date, user_history):
    # use title-case when requesting model (keeps previous behavior)
    index_event_title = str(index_event_value).title()

    # For each possible outcome, preprocess the whole group once and predict
    total_outcomes = len(EVENTS)
    for outcome_idx, outcome_event in enumerate(EVENTS, start=1):
        logger.debug(
            "%s -> outcome %s/%s: %s",
            index_event_title,
            outcome_idx,
            total_outcomes,
            outcome_event,
        )
        # Skip invalid liquidated->liquidated pairs (preserve previous behavior)
        if index_event_title == outcome_event and index_event_title == "Liquidated":
            for ts in group["timestamp"]:
                results[int(ts)][outcome_event] = None
            continue

        model_pack = get_model_for_pair_and_date(
            index_event_title,
            outcome_event,
            model_date=model_date,
            verbose=True,
        )
        if model_pack is None:
            for ts in group["timestamp"]:
                results[int(ts)][outcome_event] = None
            continue
        model, baseline_meta = model_pack

        # Prepare a copy of the group's rows with the requested Outcome Event
        test_df = group.copy()
        test_df["Outcome Event"] = outcome_event.lower()

        # Preprocess the entire group's test features at once
        _, _, test_features, test_features_index = preprocess(
            test_df=test_df, model_date=model_date
        )

        if test_features is None or test_features.num_row() == 0:
            for ts in group["timestamp"]:
                results[int(ts)][outcome_event] = None
            continue

        # Predict in batch and map predictions back to timestamps
        try:
            time_preds = get_expected_time_to_event(model, test_features, baseline_meta)
        except Exception:
            # If prediction fails for any reason, mark as None
            logger.warning(
                f"Warning: prediction failed for {index_event_title}->{outcome_event} at {model_date}"
            )
            logger.exception("Exception details:")
            for ts in group["timestamp"]:
                results[int(ts)][outcome_event] = None
            continue

        # Align by index: test_features_index corresponds to rows in user_history
        for idx_i, time_pred in zip(test_features_index, time_preds):
            ts = int(user_history.loc[idx_i, "timestamp"])
            results[ts][outcome_event] = float(time_pred)


def get_transaction_history_predictions(row: pd.Series) -> pd.DataFrame:
    results_cache_file = os.path.join(
        RESULTS_CACHE_DIR,
        f"{row['user']}_{row['timestamp']}_{row['amount']}.pkl",
    )
    # If exact cache exists for this amount, return it immediately
    if os.path.exists(results_cache_file):
        with open(results_cache_file, "rb") as f:
            return pkl.load(f)

    # Otherwise, check for any cache for the same user+timestamp (different amount).
    # If found, we'll reuse that cache and only recompute the predictions for
    # the most-recent transaction (the row passed in), then save a new cache
    # file for the current amount.
    pattern = os.path.join(RESULTS_CACHE_DIR, f"{row['user']}_{row['timestamp']}_*.pkl")
    matches = glob.glob(pattern)
    cached_results = None
    if matches:
        for m in matches:
            try:
                with open(m, "rb") as f:
                    cached_results = pkl.load(f)
                logger.debug("Loaded alternative cache %s for user/timestamp", m)
            except Exception:
                continue
            else:

                break

    # Build cache-aware, batched prediction: group history rows by Index Event
    # If we loaded an alternative cache file above, start from that and only
    # recompute the predictions for the most recent transaction (the row).
    train_dates, test_dates = get_date_ranges()
    dates = train_dates.union(test_dates)

    user_history = get_user_history(
        user_id=row["user"], up_to_timestamp=row["timestamp"] - 1
    )
    if user_history.empty:
        user_history = row.to_frame().T
    else:
        user_history = pd.concat([user_history, row.to_frame().T]).reset_index(
            drop=True
        )

    model_date = dates[dates <= pd.to_datetime(row["timestamp"], unit="s")].max()

    if cached_results is not None:
        results = cached_results
        calc_predictions(
            row["Index Event"],
            user_history.iloc[[-1]],
            results,
            model_date,
            user_history,
        )
    else:
        # initialize results structure for each timestamp (preserve existing cached keys)
        results = {}
        for ts in user_history["timestamp"]:
            results.setdefault(int(ts), {})

        # Group by original Index Event to preprocess/predict in batches
        grouped = list(user_history.groupby("Index Event"))
        total_groups = len(grouped)
        for group_idx, (index_event_value, group) in enumerate(grouped, start=1):
            logger.debug(
                "Processing IndexEvent group %s/%s: %s",
                group_idx,
                total_groups,
                index_event_value,
            )
            calc_predictions(
                index_event_value, group, results, model_date, user_history
            )

    with open(
        results_cache_file,
        "wb",
    ) as f:
        pkl.dump(results, f)
    return results


def calculate_trend_slope(data):
    """
    Calculates the linear regression slope of a dataset to determine the trend.

    Args:
        data (dict): A dictionary where keys are timestamps (int/float)
                     and values are numbers.

    Returns:
        float: The slope of the trend line.
               > 0 means increasing, < 0 means decreasing.
               Returns 0.0 if not enough data.
    """
    """
    Compute a robust, dimensionless trend-risk score for the provided
    time-series `data` (mapping timestamps -> values). The returned value
    is larger for stronger upward trends relative to recent volatility and
    recent acceleration; negative values indicate downward tendency.

    This keeps the same input contract and returns a single float so the
    rest of the code can compare scores as before.
    """

    # Sort data by timestamp and normalize times to start at 0 to avoid
    # precision issues with large epoch timestamps.
    sorted_items = [
        item
        for item in sorted(data.items())
        if item[1] is not None and np.isfinite(item[1])
    ]
    if len(sorted_items) < 2:
        return 0.0
    start_time = sorted_items[0][0]
    # debug: list of (timestamp, value) pairs
    # logger = logging.getLogger(__name__)
    # logger.debug(f"sorted_items: {sorted_items}")
    xs = np.array([float(x - start_time) for x, _ in sorted_items], dtype=float)
    ys = np.array([float(y) for _, y in sorted_items], dtype=float)

    n = len(xs)
    eps = 1e-8

    # Full-series least squares slope (y per unit time)
    mean_x = xs.mean()
    mean_y = ys.mean()
    logger.debug(f"mean_x: {mean_x}, mean_y: {mean_y}")
    num = ((xs - mean_x) * (ys - mean_y)).sum()
    den = ((xs - mean_x) ** 2).sum()
    slope = float(num / den) if den != 0 else 0.0

    # Predicted values and residuals to estimate volatility
    y_pred = slope * (xs - mean_x) + mean_y
    residuals = ys - y_pred
    y_std = float(ys.std(ddof=0))
    res_std = float(residuals.std(ddof=0))

    # Make slope dimensionless by scaling with time span and y variability.
    time_span = float(xs[-1] - xs[0]) if xs[-1] - xs[0] > 0 else 1.0
    slope_z = slope * time_span / (y_std + eps)

    # Recent slope (last half of points, at least 2) to capture short-term
    # acceleration relative to the past slope.
    k = max(2, n // 2)
    recent_xs = xs[-k:]
    recent_ys = ys[-k:]
    mean_rx = recent_xs.mean()
    mean_ry = recent_ys.mean()
    num_r = ((recent_xs - mean_rx) * (recent_ys - mean_ry)).sum()
    den_r = ((recent_xs - mean_rx) ** 2).sum()
    recent_slope = float(num_r / den_r) if den_r != 0 else 0.0
    recent_time_span = (
        float(recent_xs[-1] - recent_xs[0]) if recent_xs[-1] - recent_xs[0] > 0 else 1.0
    )
    recent_slope_z = recent_slope * recent_time_span / (y_std + eps)

    # Past slope (first segment) when we have enough points; otherwise reuse
    # the full-series normalized slope.
    if n >= 4 and (n - k) >= 2:
        past_xs = xs[: n - k]
        past_ys = ys[: n - k]
        mean_px = past_xs.mean()
        mean_py = past_ys.mean()
        num_p = ((past_xs - mean_px) * (past_ys - mean_py)).sum()
        den_p = ((past_xs - mean_px) ** 2).sum()
        past_slope = float(num_p / den_p) if den_p != 0 else 0.0
        past_time_span = (
            float(past_xs[-1] - past_xs[0]) if past_xs[-1] - past_xs[0] > 0 else 1.0
        )
        past_slope_z = past_slope * past_time_span / (y_std + eps)
    else:
        past_slope_z = slope_z

    # Acceleration (dimensionless)
    acceleration_z = recent_slope_z - past_slope_z

    # Normalize volatility relative to the series amplitude
    volatility_norm = res_std / (y_std + eps)

    # Combine into a single score: base normalized slope, boosted by
    # recent acceleration, penalized by volatility. We also slightly
    # amplify when the last value is above the mean (momentum).
    score = slope_z + 0.8 * acceleration_z - 0.6 * volatility_norm

    last_rel = (ys[-1] - mean_y) / (abs(mean_y) + eps)
    score = score * (1.0 + 0.3 * last_rel)

    return float(score)


def determine_liquidation_risk(row: pd.Series):
    predict_transaction_history = {
        key: value
        for key, value in get_transaction_history_predictions(row).items()
        if value
    }

    is_at_risk = False
    is_at_immediate_risk = False

    most_recent_predictions = predict_transaction_history[
        max(predict_transaction_history.keys())
    ]
    if (most_recent_predictions["Liquidated"] is not None) and (
        most_recent_predictions["Liquidated"] <= min(most_recent_predictions.values())
    ):
        is_at_risk = True
        is_at_immediate_risk = True
        trend_slopes = None
        logger.info(
            "Liquidation Risk Immediate: "
            + str(most_recent_predictions["Liquidated"])
            + " <= "
            + str(list(most_recent_predictions.values()))
        )
    else:
        trend_slopes = {
            outcome_event: calculate_trend_slope(
                {
                    timestamp: preds[outcome_event]
                    for timestamp, preds in predict_transaction_history.items()
                    if preds and outcome_event in preds
                }
            )
            for outcome_event in predict_transaction_history[
                sorted(predict_transaction_history.keys(), reverse=True)[0]
            ].keys()
        }
        if trend_slopes["Liquidated"] < 0 and trend_slopes["Liquidated"] <= min(
            trend_slopes.values()
        ):
            is_at_risk = True
            logger.info(
                "Liquidation Risk Gradual: "
                + str(trend_slopes["Liquidated"])
                + " <= "
                + str(list(trend_slopes.values()))
            )

    if not is_at_risk:
        logger.info("No Liquidation Risk.")
    return is_at_risk, is_at_immediate_risk, most_recent_predictions, trend_slopes
    # return is_at_risk, trend_slopes


def generate_next_transaction(
    prev_row: pd.Series,
    action_type: str,
    amount=10.0,
    time_delta_seconds=DEFAULT_TIME_DELTA_SECONDS,
    reserve=None,
):
    """
    Generates a new transaction row 'soon after' the previous row.
    Updates time features and cumulative user stats.

    Args:
        prev_row (pd.Series): The 'Index Event' row to build upon.
        action_type (str): The new action, e.g., 'Deposit' or 'Repay'.
        amount (float): The amount for the new transaction.
        time_delta_seconds (int): How many seconds after the previous row this occurs.

    Returns:
        pd.Series: The new, complete transaction row.
    """
    # 1. Initialize new row
    new_row = prev_row.copy()
    action = action_type.lower()

    # 2. Update Identifiers
    new_row["Index Event"] = action
    new_row["type"] = action
    new_row["Outcome Event"] = None  # As requested

    # 3. Update Time
    # Increment timestamp
    prev_ts = new_row["timestamp"]
    new_row["timestamp"] = prev_ts + time_delta_seconds

    # Update time intervals
    new_row["timeDiff"] = time_delta_seconds
    new_row["userSecondsSincePreviousTransaction"] = time_delta_seconds
    new_row["userSecondsSinceFirstTransaction"] += time_delta_seconds

    # Update 'timeOfDay' (Assuming data is in hours 0-24)
    hours_added = time_delta_seconds / 3600.0
    new_time_of_day = (new_row["timeOfDay"] + hours_added) % 24
    new_row["timeOfDay"] = new_time_of_day

    # Recalculate Cyclical Time Features (to maintain model consistency)
    # sin/cos TimeOfDay (Period: 24h)
    new_row["sinTimeOfDay"] = np.sin(2 * np.pi * new_time_of_day / 24)
    new_row["cosTimeOfDay"] = np.cos(2 * np.pi * new_time_of_day / 24)

    # Check if we crossed a day boundary (simple approximation)
    if new_time_of_day < (prev_row["timeOfDay"] + hours_added):
        new_row["dayOfWeek"] = (new_row["dayOfWeek"] % 7) + 1
        new_row["dayOfYear"] += 1
        # Update day-based cyclicals
        new_row["sinDayOfWeek"] = np.sin(2 * np.pi * new_row["dayOfWeek"] / 7)
        new_row["cosDayOfWeek"] = np.cos(2 * np.pi * new_row["dayOfWeek"] / 7)
        new_row["sinDayOfYear"] = np.sin(2 * np.pi * new_row["dayOfYear"] / 365)
        new_row["cosDayOfYear"] = np.cos(2 * np.pi * new_row["dayOfYear"] / 365)

    # 4. Update Amount
    new_row["amount"] = amount
    # Use numpy.log1p when available; fall back to math.log1p if `np` is shadowed.
    try:
        log1p_fn = np.log1p
    except Exception:
        import math

        log1p_fn = math.log1p
    new_row["logAmount"] = float(log1p_fn(float(amount)))

    # Calculate USD Amount (assuming price is static for the short interval)
    # We can't get the actual conversion rate for the future time because that would be cheating
    if reserve:
        new_row["reserve"] = reserve
        new_row["priceInUSD"] = get_price_history_value(reserve, prev_ts)
    price = new_row["priceInUSD"]
    amount_usd = amount * price
    new_row["amountUSD"] = amount_usd
    try:
        log1p_fn = np.log1p
    except Exception:
        import math

        log1p_fn = math.log1p
    new_row["logAmountUSD"] = float(log1p_fn(float(amount_usd)))

    # 5. Update Cumulative User Stats
    # These specific columns track the user's history
    if action == "deposit":
        new_row["userDepositCount"] += 1
        new_row["userDepositSum"] += amount
        new_row["userDepositSumUSD"] += amount_usd

        # Recalculate Averages
        if new_row["userDepositCount"] > 0:
            new_row["userDepositAvgAmount"] = (
                new_row["userDepositSum"] / new_row["userDepositCount"]
            )
            new_row["userDepositAvgAmountUSD"] = (
                new_row["userDepositSumUSD"] / new_row["userDepositCount"]
            )

    elif action == "repay":
        new_row["userRepayCount"] += 1
        new_row["userRepaySum"] += amount
        new_row["userRepaySumUSD"] += amount_usd

        if new_row["userRepayCount"] > 0:
            new_row["userRepayAvgAmount"] = (
                new_row["userRepaySum"] / new_row["userRepayCount"]
            )
            new_row["userRepayAvgAmountUSD"] = (
                new_row["userRepaySumUSD"] / new_row["userRepayCount"]
            )

    # Note: 'Borrow' stats generally don't change on a 'Repay' action
    # (BorrowSum usually tracks total borrowed volume, not current debt balance).

    return new_row


def optimize_recommendation(row: pd.Series, recommended_action: str):
    min_recommendation = MIN_RECOMMENDATION_AMOUNT
    max_recommendation = 100000

    sample_action = generate_next_transaction(row, recommended_action)
    return_values = get_limited_user_profile(sample_action, return_extras=True)
    reserve = None
    if return_values is not None:
        (
            user_profile,
            _,
            _,
            _,
            _,
            lookahead_seconds,
        ) = return_values
        simulation_results = get_simulation_outcome(
            sample_action,
            "without",
            profile=user_profile,
            lookahead_seconds=lookahead_seconds,
            output_file=logger.handlers[0].baseFilename,
        )
        value_pairs = [
            (reserve, amount * get_price_history_value(reserve, row["timestamp"]))
            for reserve, amount in simulation_results["final_state"][
                "wallet_balances"
            ].items()
        ]
        reserve = max(value_pairs, key=value_pairs[1])[0]
        max_recommendation = simulation_results["final_state"]["wallet_balances"][
            reserve
        ]
        min_recommendation = min(
            max(MIN_RECOMMENDATION_AMOUNT, max_recommendation / (2 * 6)),
            max_recommendation,
        )

    new_action = generate_next_transaction(
        row,
        recommended_action,
        amount=min_recommendation,
        reserve=reserve,
    )

    while (
        new_action["amount"] < max_recommendation
        and determine_liquidation_risk(new_action)[0]
    ):
        new_action = generate_next_transaction(
            row,
            recommended_action,
            amount=min(max_recommendation, new_action["amount"] * 2),
            reserve=reserve,
        )
        logger.info("Increased amount to: %s", new_action["amount"])

    # Final check: if recommendation would create dust, return None or mark as invalid
    if return_values is not None and would_create_dust_position(
        new_action, simulation_results
    ):
        new_action = generate_next_transaction(
            row,
            recommended_action,
            amount=max_recommendation,
            reserve=reserve,
        )
    return new_action


def recommend_action(row: pd.Series):
    """Analyze predicted transaction history to determine
    whether the user is currently at risk of liquidation and
    provide a simple recommended action. Returns a dictionary
    with keys: liquidation_risk, is_at_risk, risk_trend,
    recommended_action, reason, details.

    Now includes dust position filtering to prevent recommendations
    that would create tiny positions likely to be liquidated."""

    is_at_risk, is_at_immediate_risk, most_recent_predictions, trend_slopes = (
        determine_liquidation_risk(row)
    )

    recommended_action = (
        "Repay"
        if (
            most_recent_predictions["Deposit"] is None
            or most_recent_predictions["Repay"] >= most_recent_predictions["Deposit"]
        )
        and is_at_risk
        else "Deposit"
    )

    # Optimize recommendation with dust filtering
    optimized_action = optimize_recommendation(row, recommended_action)

    return optimized_action, {
        "is_at_risk": is_at_risk,
        "is_at_immediate_risk": is_at_immediate_risk,
        "most_recent_predictions": most_recent_predictions,
        "trend_slopes": trend_slopes,
    }


def run_training_pipeline():
    recommendation_cache_file = RECOMMENDATIONS_FILE
    if os.path.exists(recommendation_cache_file):
        with open(recommendation_cache_file, "rb") as f:
            recommendations = pkl.load(f)
    else:
        recommendations = {}
    train_set = get_train_set()
    # with logging_redirect_tqdm():
    total_rows = train_set.shape[0]
    for iter_count, row in train_set.iterrows():
        logger.debug("Processing row %s/%s", iter_count + 1, total_rows)
        rowID = str(row)
        if rowID not in recommendations:
            recommendations[rowID] = recommend_action(row)
            if iter_count % 1 == 0:
                with open(recommendation_cache_file, "wb") as f:
                    pkl.dump(recommendations, f)
        logger.info("Recommended %s\nfor %s", str(recommendations[rowID]), rowID)
    # with open(recommendation_cache_file, "wb") as f:
    #     pkl.dump(recommendations, f)


if __name__ == "__main__":
    run_training_pipeline()
