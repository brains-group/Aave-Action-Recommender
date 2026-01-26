import pandas as pd
import numpy as np
import os
import pickle as pkl
import glob
import pyreadr
from utils.logger import logger
from utils.data import get_event_df
from utils.model_training import preprocess, get_model_for_pair_and_date

from utils.constants import *

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


DATE_RANGES_CACHE = None

def get_date_ranges():
    global DATE_RANGES_CACHE
    if os.path.exists(os.path.join(CACHE_DIR, "date_ranges.pkl")):
        with open(os.path.join(CACHE_DIR, "date_ranges.pkl"), "rb") as f:
            DATE_RANGES_CACHE = pkl.load(f)
    if DATE_RANGES_CACHE is not None:
        return DATE_RANGES_CACHE
    transactions_df = pyreadr.read_r("./data/transactions.rds")[None]
    min_date = transactions_df["timestamp"].min() * 1e9
    logger.debug(f"min_date: {min_date}")
    max_date = transactions_df["timestamp"].max() * 1e9
    logger.debug(f"max_date: {max_date}")
    train_start_date = min_date + 0.4 * (max_date - min_date)
    logger.debug(f"train_start_date: {train_start_date}")
    test_start_date = min_date + 0.8 * (max_date - min_date)
    logger.debug(f"test_start_date: {test_start_date}")
    train_dates = pd.date_range(start=train_start_date, end=test_start_date, freq="2W")
    test_dates = pd.date_range(start=test_start_date, end=max_date, freq="2W")
    with open(os.path.join(CACHE_DIR, "date_ranges.pkl"), "wb") as f:
        pkl.dump((train_dates, test_dates), f)
    DATE_RANGES_CACHE = (train_dates, test_dates)
    return train_dates, test_dates


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
            time_preds = get_expected_time_to_event(
                model, test_features, baseline_meta
            )
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


def get_transaction_history_predictions(
    row: pd.Series
) -> pd.DataFrame:
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
    row_df = row.to_frame().T
    user_history = pd.concat([user_history, row_df]).reset_index(drop=True)

    model_date = dates[dates <= pd.to_datetime(row["timestamp"], unit="s")].max()

    if cached_results is not None:
        results = cached_results
        calc_predictions(row["Index Event"], row_df, results, model_date, user_history)
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
            calc_predictions(index_event_value, group, results, model_date, user_history)

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


def _would_create_dust_position(
    row: pd.Series, recommended_action: str, amount: float
) -> bool:
    """
    Estimate if a recommendation would create a dust position.

    Based on analysis findings:
    - Dust liquidations: total_debt_usd < $1.00 or total_collateral_usd < $10.00
    - We estimate the final position after the recommendation

    Args:
        row: Current user state row
        recommended_action: Action type (Deposit, Repay, etc.)
        amount: Recommended amount

    Returns:
        True if recommendation would likely create dust position, False otherwise
    """
    from utils.constants import (
        MIN_RECOMMENDATION_DEBT_USD,
        MIN_RECOMMENDATION_COLLATERAL_USD,
        MIN_RECOMMENDATION_AMOUNT,
    )

    action = recommended_action.lower()
    amount_usd = (
        amount * row.get("priceInUSD", 1.0)
        if pd.notna(row.get("priceInUSD"))
        else amount
    )

    # Check minimum amount threshold
    if amount_usd < MIN_RECOMMENDATION_AMOUNT:
        return True

    # For Repay actions: estimate if remaining debt would be dust
    if action == "repay":
        # Estimate current debt from row data
        # Try to get debt from available fields, or use a conservative estimate
        estimated_debt_usd = row.get("totalDebtUSD", 0.0)
        if pd.isna(estimated_debt_usd) or estimated_debt_usd == 0:
            # Fallback: use a conservative estimate based on user stats
            estimated_debt_usd = row.get("userBorrowSumUSD", 0.0)

        estimated_remaining_debt = max(0, estimated_debt_usd - amount_usd)
        if (
            estimated_remaining_debt > 0
            and estimated_remaining_debt < MIN_RECOMMENDATION_DEBT_USD
        ):
            return True

    # For Deposit actions: estimate if position would result in dust collateral
    elif action == "deposit":
        # For deposits, we're adding collateral, so unlikely to create dust debt
        # But if the user has very little debt and we're depositing, check if
        # the resulting position might be problematic
        estimated_debt_usd = row.get("totalDebtUSD", 0.0)
        if pd.isna(estimated_debt_usd):
            estimated_debt_usd = row.get("userBorrowSumUSD", 0.0)

        # If debt is very small (< $1) and we're just depositing without repaying,
        # this might create a dust-like position
        if estimated_debt_usd > 0 and estimated_debt_usd < MIN_RECOMMENDATION_DEBT_USD:
            # Depositing while having dust debt could be problematic
            return True

    # For Withdraw actions: check if withdrawal would leave insufficient collateral
    elif action == "withdraw":
        estimated_collateral_usd = row.get("totalCollateralUSD", 0.0)
        if pd.isna(estimated_collateral_usd):
            estimated_collateral_usd = row.get("userDepositSumUSD", 0.0)

        estimated_remaining_collateral = max(0, estimated_collateral_usd - amount_usd)
        if (
            estimated_remaining_collateral > 0
            and estimated_remaining_collateral < MIN_RECOMMENDATION_COLLATERAL_USD
        ):
            return True

    return False


def optimize_recommendation(row: pd.Series, recommended_action: str):
    from utils.constants import MIN_RECOMMENDATION_AMOUNT

    # Start with minimum amount instead of fixed 10
    price = max(row.get("priceInUSD", 1.0) if pd.notna(row.get("priceInUSD")) else 1.0, 0.0001)
    initial_amount = max(
        MIN_RECOMMENDATION_AMOUNT / price, 10.0
    )  # At least MIN_RECOMMENDATION_AMOUNT USD or 10 tokens

    new_action = generate_next_transaction(
        row,
        recommended_action,
        amount=initial_amount,
    )

    # If risk remains, iteratively increase the amount and log single-action predictions
    max_iterations = 15  # Prevent infinite loops
    iteration = 0
    while (
        determine_liquidation_risk(new_action)[0]
        and new_action["amount"] < 100000
        and iteration < max_iterations
    ):
        # Check if current recommendation would create dust position
        if _would_create_dust_position(row, recommended_action, new_action["amount"]):
            # Skip dust-creating recommendations - increase amount before checking risk again
            new_action = generate_next_transaction(
                row,
                recommended_action,
                amount=new_action["amount"] * 2,
            )
            logger.info(
                "Skipped dust-creating amount, increased to: %s", new_action["amount"]
            )
            iteration += 1
            continue

        # If not dust, increase amount to reduce liquidation risk
        new_action = generate_next_transaction(
            row,
            recommended_action,
            amount=new_action["amount"] * 2,
        )
        logger.info("Increased amount to: %s", new_action["amount"])
        iteration += 1

    # Final check: if recommendation would create dust, return None or mark as invalid
    if _would_create_dust_position(row, recommended_action, new_action["amount"]):
        logger.warning(
            "Recommendation would create dust position (amount=%.2f, action=%s). "
            "Consider not recommending this action.",
            new_action["amount"],
            recommended_action,
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

    # Final check: if optimized action would still create dust, log warning
    # The optimize_recommendation function already handles dust checking internally,
    # but we do a final verification here
    if optimized_action is not None:
        amount_usd = (
            optimized_action.get("amountUSD", 0.0)
            if hasattr(optimized_action, "get")
            else (
                optimized_action["amount"] * optimized_action.get("priceInUSD", 1.0)
                if pd.notna(optimized_action.get("priceInUSD"))
                else optimized_action["amount"]
            )
        )
        if _would_create_dust_position(
            row, recommended_action, optimized_action["amount"]
        ):
            logger.warning(
                "Final recommendation for user %s would create dust position. "
                "Action: %s, Amount: %.2f (%.2f USD). Recommendation may be filtered out.",
                row.get("user", "unknown"),
                recommended_action,
                optimized_action["amount"],
                amount_usd,
            )

    return optimized_action, {
        "is_at_risk": is_at_risk,
        "is_at_immediate_risk": is_at_immediate_risk,
        "most_recent_predictions": most_recent_predictions,
        "trend_slopes": trend_slopes,
    }


def get_train_set():
    TRAIN_SET_CACHE_PATH = os.path.join(CACHE_DIR, "train_set.csv")
    if os.path.exists(TRAIN_SET_CACHE_PATH):
        train_set = pd.read_csv(TRAIN_SET_CACHE_PATH)
    else:
        train_set = pd.DataFrame()
        train_ranges, test_ranges = get_date_ranges()
        min_train_date = train_ranges[0].timestamp()
        max_train_date = test_ranges[0].timestamp()
        event_pairs = [
            (ie, oe) for ie in EVENTS for oe in EVENTS if not (ie == oe == "Liquidated")
        ]
        for index_event, outcome_event in event_pairs:
            # log progress for building the train set
            logger.info(
                "Building train_set: processing %s->%s", index_event, outcome_event
            )
            df = get_event_df(index_event, outcome_event)
            if df is None:
                continue
            # logger = logging.getLogger(__name__)
            logger.info(f"Processing {index_event}->{outcome_event}")
            logger.debug(f"Data has {len(df)} rows")
            logger.debug(f"Min timestamp: {df['timestamp'].min()}")
            logger.debug(f"Max timestamp: {df['timestamp'].max()}")
            subsetInRange = df[
                (df["timestamp"] >= min_train_date) & (df["timestamp"] < max_train_date)
            ]
            logger.debug(
                f"Loaded {len(subsetInRange)} rows for {index_event}->{outcome_event}"
            )
            train_set = pd.concat(
                [
                    train_set,
                    subsetInRange.sample(n=300, random_state=seed),
                ],
                ignore_index=True,
            )
        for index_event in EVENTS:
            if index_event == "Liquidated":
                continue
            outcome_event = "Liquidated"
            logger.info(
                "Building train_set: processing %s->%s", index_event, outcome_event
            )
            df = get_event_df(index_event, outcome_event)
            if df is None:
                continue
            logger.info(f"Processing {index_event}->{outcome_event}")
            logger.debug(f"Data has {len(df)} rows")
            logger.debug(f"Min timestamp: {df['timestamp'].min()}")
            logger.debug(f"Max timestamp: {df['timestamp'].max()}")
            subsetInRange = df[
                (df["timestamp"] >= min_train_date) & (df["timestamp"] < max_train_date)
            ]
            logger.debug(
                f"Loaded {len(subsetInRange)} rows for {index_event}->{outcome_event}"
            )
            train_set = pd.concat(
                [
                    train_set,
                    subsetInRange.sort_values(by="timeDiff").head(300),
                ],
                ignore_index=True,
            )
        with open(TRAIN_SET_CACHE_PATH, "w") as f:
            train_set.to_csv(f, index=False)

    return train_set


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
