# %%
# !export CUDA_VISIBLE_DEVICES=0

# Install required packages
# pip install -q pandas xgboost scikit-learn numpy pyreadr

# Import libraries
import pandas as pd
import numpy as np
import os
import shutil
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
import pickle as pkl
from itertools import chain
import pyreadr
import json

# %%
DATA_PATH = "./data/"
CACHE_DIR = "./cache/"
os.makedirs(CACHE_DIR, exist_ok=True)
MODEL_CACHE_DIR = os.path.join(CACHE_DIR, "models")
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
DATA_CACHE_DIR = os.path.join(CACHE_DIR, "data")
os.makedirs(DATA_CACHE_DIR, exist_ok=True)
RESULTS_CACHE_DIR = os.path.join(CACHE_DIR, "results")
os.makedirs(RESULTS_CACHE_DIR, exist_ok=True)

seed = 42
np.random.seed(seed)

EVENTS = ["Deposit", "Withdraw", "Repay", "Borrow", "Liquidated"]


# %%
def preprocess(
    train_df_with_labels: Optional[pd.DataFrame] = None,
    test_features_df: Optional[pd.DataFrame] = None,
    model_date: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:

    # Create unique prefix for saving/loading preprocessing objects
    unique_prefix = (
        (
            present_dataframe := (
                train_df_with_labels
                if train_df_with_labels is not None
                else test_features_df
            )
        )["Index Event"].iloc[0]
        + "_"
        + present_dataframe["Outcome Event"].iloc[0]
        + (f"_{model_date}_" if model_date is not None else "_")
    )
    # Define paths for saving/loading preprocessing objects
    scaler_path = os.path.join(DATA_CACHE_DIR, unique_prefix + "scaler.pkl")
    train_cols = os.path.join(DATA_CACHE_DIR, unique_prefix + "train_cols.pkl")
    top_categories_dict_path = os.path.join(
        DATA_CACHE_DIR, unique_prefix + "top_categories_dict.pkl"
    )
    categorical_cols_path = os.path.join(
        DATA_CACHE_DIR, unique_prefix + "categorical_cols.pkl"
    )
    numerical_cols_path = os.path.join(
        DATA_CACHE_DIR, unique_prefix + "numerical_cols.pkl"
    )
    cols_to_keep_path = os.path.join(DATA_CACHE_DIR, unique_prefix + "cols_to_keep.pkl")

    target_columns = ["timeDiff", "status"]
    cols_to_drop = [
        "id",
        "user",
        "pool",
        "Index Event",
        "Outcome Event",
        "type",
        "timestamp",
    ]

    if train_df_with_labels is not None:
        if model_date is not None:
            # model_date may be a pandas.Timestamp while dataframe timestamps are numeric.
            # Convert model_date to numeric epoch seconds for a safe comparison.
            if isinstance(model_date, pd.Timestamp):
                model_date_value = model_date.timestamp()
            else:
                try:
                    model_date_value = float(model_date)
                except Exception:
                    model_date_value = model_date

            train_df_with_labels = train_df_with_labels[
                (train_df_with_labels["timestamp"] + train_df_with_labels["timeDiff"])
                <= model_date_value
            ]

        # Separate features and targets (and drop unneeded columns from features)
        train_targets = train_df_with_labels[target_columns]
        train_features = train_df_with_labels.drop(
            columns=target_columns + cols_to_drop, errors="ignore"
        )

        # Make uncommon categories "Other" and one-hot encode categorical features
        categorical_cols = train_features.select_dtypes(
            include=["object", "category"]
        ).columns
        top_categories_dict = {}
        for col in categorical_cols:
            top_categories_dict[col] = (
                train_features[col].value_counts().nlargest(10).index
            )
            train_features[col] = train_features[col].where(
                train_features[col].isin(top_categories_dict[col]), "Other"
            )
        train_features_encoded = pd.get_dummies(
            train_features, columns=categorical_cols, dummy_na=True, drop_first=True
        )

        # Standardize numerical features
        numerical_cols = train_features_encoded.select_dtypes(include=np.number).columns
        scaler = StandardScaler()
        train_features_scaled = scaler.fit_transform(
            train_features_encoded[numerical_cols]
        )
        train_features_final = pd.DataFrame(
            train_features_scaled,
            index=train_features_encoded.index,
            columns=numerical_cols,
        ).fillna(0)

        # Remove zero-variance columns
        cols_to_keep = train_features_final.columns[train_features_final.var() != 0]
        train_features_final = train_features_final[cols_to_keep]

        # Save preprocessing objects
        with open(scaler_path, "wb") as f:
            pkl.dump(scaler, f)
        with open(train_cols, "wb") as f:
            pkl.dump(train_features_encoded.columns, f)
        with open(top_categories_dict_path, "wb") as f:
            pkl.dump(top_categories_dict, f)
        with open(categorical_cols_path, "wb") as f:
            pkl.dump(categorical_cols, f)
        with open(numerical_cols_path, "wb") as f:
            pkl.dump(numerical_cols, f)
        with open(cols_to_keep_path, "wb") as f:
            pkl.dump(cols_to_keep, f)
    else:
        train_features_final = None
        train_targets = None

    # Process test features if provided
    if test_features_df is not None:
        test_features = test_features_df.drop(columns=cols_to_drop, errors="ignore")
        with open(categorical_cols_path, "rb") as f:
            categorical_cols = pkl.load(f)
        with open(top_categories_dict_path, "rb") as f:
            top_categories_dict = pkl.load(f)
            print(top_categories_dict)  # Debug print to verify loaded categories
        for col in categorical_cols:
            top_categories = top_categories_dict[col]
            test_features[col] = test_features[col].where(
                test_features[col].isin(top_categories), "Other"
            )
        test_features_encoded = pd.get_dummies(
            test_features, columns=categorical_cols, dummy_na=True, drop_first=True
        )
        with open(train_cols, "rb") as f:
            train_cols = pkl.load(f)
        test_features_aligned = test_features_encoded.reindex(
            columns=train_cols, fill_value=0
        )
        with open(scaler_path, "rb") as f:
            scaler = pkl.load(f)
        with open(numerical_cols_path, "rb") as f:
            numerical_cols = pkl.load(f)
        test_features_scaled = scaler.transform(test_features_aligned[numerical_cols])
        test_features_final = pd.DataFrame(
            test_features_scaled,
            index=test_features_aligned.index,
            columns=numerical_cols,
        ).fillna(0)
        with open(cols_to_keep_path, "rb") as f:
            cols_to_keep = pkl.load(f)
        test_processed_features = test_features_final[cols_to_keep]
    else:
        test_processed_features = None
    return train_features_final, train_targets, test_processed_features


# %%
def get_model_for_pair_and_date(
    index_event: str,
    outcome_event: str,
    model_date: int | None = None,
    verbose: bool = False,
):
    # normalize model_date for filename
    model_date_str = str(model_date) if model_date is not None else "latest"
    model_filename = f"xgboost_cox_{index_event}_{outcome_event}_{model_date_str}.ubj"
    model_path = os.path.join(MODEL_CACHE_DIR, model_filename)

    # Create model with Cox objective
    model = XGBRegressor(
        objective="survival:cox",
        eval_metric="cox-nloglik",
        tree_method="hist",
        predictor="gpu_predictor",
        device="cuda",
        seed=42,
        verbosity=0,
        max_bin=64,
        learning_rate=0.04,
        max_depth=5,
        subsample=0.85,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=1.0,
        reg_alpha=0.1,
    )

    # If model file exists, try to load into the estimator and return the estimator
    if os.path.exists(model_path):
        if verbose:
            print(f"Loading existing model from {model_path}")
        try:
            model.load_model(model_path)
            if verbose:
                print(f"model loaded from {model_path}")
            return model
        except Exception as e:
            print(
                f"Warning: failed to load model from {model_path}: {e}. Will retrain."
            )

    dataset_path = os.path.join(index_event, outcome_event)

    # --- Load and Preprocess ---
    if verbose:
        print(f"Loading data from {os.path.join(DATA_PATH, dataset_path, 'data.csv')}")
    train_df = pd.read_csv(os.path.join(DATA_PATH, dataset_path, "data.csv"))

    X_train, y_train, _ = preprocess(train_df, model_date=model_date)

    # --- Train Model ---
    # Prepare target variables for Cox regression
    y_train_duration = y_train["timeDiff"].values
    y_train_event = y_train["status"].values

    # Fit model: XGBoost Cox expects labels to be the event indicators
    # and the sample_weight to be the durations
    if verbose:
        print("Training model...")
    try:
        model.fit(X_train, y_train_event, sample_weight=y_train_duration)
    except Exception as e:
        print(f"ERROR: Model training failed for {dataset_path}: {e}")
        raise

    # Save model: try estimator's save_model, fall back to Booster.save_model
    try:
        # XGBRegressor implements save_model; call it and confirm file created
        model.save_model(model_path)
        if verbose:
            print(f"Model saved to {model_path}")
    except Exception:
        try:
            booster = model.get_booster()
            booster.save_model(model_path)
            if verbose:
                print(f"Model booster saved to {model_path}")
        except Exception as e:
            print(f"Warning: Failed to save model to {model_path}: {e}")

    return model


# %%
def train_models_for_all_event_pairs(
    model_date: int | None = None, verbose: bool = False
):
    # Define all 16 event pairs
    index_events = EVENTS
    outcome_events = index_events
    event_pairs = [
        event_pair
        for sub_event_pairs in [
            [(index_event, outcome_event) for outcome_event in outcome_events]
            for index_event in index_events
        ]
        for event_pair in sub_event_pairs
    ]

    for index_event, outcome_event in event_pairs:
        if index_event == outcome_event and index_event == "Liquidated":
            continue
        if verbose:
            print(f"\n{'='*50}")
            print(f"Training for: {index_event} -> {outcome_event}")
            print(f"{'='*50}")

        get_model_for_pair_and_date(
            index_event, outcome_event, model_date=model_date, verbose=verbose
        )

    if verbose:
        print("\n\nAll prediction files have been generated.")


# %%
def get_date_ranges():
    if os.path.exists(os.path.join(CACHE_DIR, "date_ranges.pkl")):
        with open(os.path.join(CACHE_DIR, "date_ranges.pkl"), "rb") as f:
            return pkl.load(f)
    transactions_df = pyreadr.read_r("./data/transactions.rds")[None]
    min_date = transactions_df["timestamp"].min() * 1e9
    print(min_date)
    max_date = transactions_df["timestamp"].max() * 1e9
    print(max_date)
    train_start_date = min_date + 0.4 * (max_date - min_date)
    print(train_start_date)
    test_start_date = min_date + 0.8 * (max_date - min_date)
    print(test_start_date)
    train_dates = pd.date_range(start=train_start_date, end=test_start_date, freq="2W")
    test_dates = pd.date_range(start=test_start_date, end=max_date, freq="2W")
    with open(os.path.join(CACHE_DIR, "date_ranges.pkl"), "wb") as f:
        pkl.dump((train_dates, test_dates), f)
    return train_dates, test_dates


# %%
# for date in chain(*get_date_ranges()):
#     train_models_for_all_event_pairs(model_date=date.timestamp(), verbose=True)


# %%
def get_user_history(user_id: str, up_to_timestamp: int) -> pd.DataFrame:
    all_events = []
    for index_event in EVENTS:
        for outcome_event in EVENTS:
            event_path = os.path.join(DATA_PATH, index_event, outcome_event, "data.csv")
            if os.path.exists(event_path):
                event_df = pd.read_csv(event_path)
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
                    print(
                        f"User {user_id} had liquidated->liquidated events in {index_event}->{outcome_event} before {up_to_timestamp}"
                    )
                    print(liquidated_events)
    if all_events:
        user_history_df = pd.concat(all_events).sort_values(by="timestamp")
    else:
        user_history_df = pd.DataFrame()
    return user_history_df


def get_transaction_history_predictions(row: pd.Series) -> pd.DataFrame:
    results_cache_file = os.path.join(
        RESULTS_CACHE_DIR, f"{row['user']}_{row['timestamp']}_{row['amount']}.pkl"
    )
    if os.path.exists(results_cache_file):
        with open(results_cache_file, "rb") as f:
            return pkl.load(f)

    results = {}
    train_dates, test_dates = get_date_ranges()
    dates = train_dates.union(test_dates)

    user_history = get_user_history(
        user_id=row["user"], up_to_timestamp=row["timestamp"] - 1
    )
    user_history = pd.concat([user_history, row.to_frame().T]).reset_index(drop=True)

    model_date = dates[dates <= pd.to_datetime(row["timestamp"], unit="s")].max()
    for _, history_row in user_history.iterrows():
        history_timestamp = history_row["timestamp"]
        results[history_timestamp] = {}
        history_row = history_row.copy()

        index_event = history_row["Index Event"].title()
        for outcome_event in EVENTS:
            if index_event == outcome_event and index_event == "Liquidated":
                results[history_timestamp][outcome_event] = None
                continue
            model = get_model_for_pair_and_date(
                index_event, outcome_event, model_date=model_date, verbose=True
            )

            if model is None:
                results[history_timestamp][outcome_event] = None
                continue

            history_row["Outcome Event"] = outcome_event.lower()
            _, _, test_features = preprocess(
                test_features_df=history_row.to_frame().T,
                model_date=model_date,
            )

            prediction = model.predict(test_features)
            results[history_timestamp][outcome_event] = prediction[0]

    with open(
        results_cache_file,
        "wb",
    ) as f:
        pkl.dump(results, f)
    return results


# %%
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
    print(sorted_items)
    xs = np.array([float(x - start_time) for x, _ in sorted_items], dtype=float)
    ys = np.array([float(y) for _, y in sorted_items], dtype=float)

    n = len(xs)
    eps = 1e-8

    # Full-series least squares slope (y per unit time)
    mean_x = xs.mean()
    mean_y = ys.mean()
    print(mean_x, mean_y)
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

    most_recent_predictions = predict_transaction_history[
        max(predict_transaction_history.keys())
    ]
    if most_recent_predictions["Liquidated"] >= max(most_recent_predictions.values()):
        is_at_risk = True
        print(
            "Liquidation Risk Immediate: "
            + str(most_recent_predictions["Liquidated"])
            + " >= "
            + str(most_recent_predictions.values())
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
        if trend_slopes["Liquidated"] > 0 and trend_slopes["Liquidated"] >= max(
            trend_slopes.values()
        ):
            is_at_risk = True
            print(
                "Liquidation Risk Gradual: "
                + str(trend_slopes["Liquidated"])
                + " >= "
                + str(trend_slopes.values())
            )

    if not is_at_risk:
        print("No Liquidation Risk.")
    return is_at_risk, most_recent_predictions, trend_slopes


def generate_next_transaction(
    prev_row: pd.Series, action_type: str, amount=10.0, time_delta_seconds=600
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
    new_row["logAmount"] = np.log1p(amount)

    # Calculate USD Amount (assuming price is static for the short interval)
    price = new_row["priceInUSD"]
    amount_usd = amount * price
    new_row["amountUSD"] = amount_usd
    new_row["logAmountUSD"] = np.log1p(amount_usd)

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
    new_action = generate_next_transaction(
        row,
        recommended_action,
        amount=10,
    )
    while determine_liquidation_risk(new_action)[0]:
        new_action = generate_next_transaction(
            row,
            recommended_action,
            amount=new_action["amount"] * 2,
        )
        print("Increased amount to:", new_action["amount"])
    return new_action


def recommend_action(row: pd.Series):
    """Analyze predicted transaction history to determine
    whether the user is currently at risk of liquidation and
    provide a simple recommended action. Returns a dictionary
    with keys: liquidation_risk, is_at_risk, risk_trend,
    recommended_action, reason, details."""

    is_at_risk, most_recent_predictions, _ = determine_liquidation_risk(row)

    recommended_action = (
        "Repay"
        if most_recent_predictions["Repay"] >= most_recent_predictions["Deposit"]
        and is_at_risk
        else "Deposit"
    )

    return optimize_recommendation(row, recommended_action)


# %%
def get_train_set():
    train_set_dir = os.path.join(CACHE_DIR, "train_set.csv")
    if os.path.exists(train_set_dir):
        train_set = pd.read_csv(train_set_dir)
    else:
        train_set = pd.DataFrame()
        train_ranges, test_ranges = get_date_ranges()
        min_train_date = train_ranges[0].timestamp()
        max_train_date = test_ranges[0].timestamp()
        for index_event in EVENTS:
            if index_event == "Liquidated":
                continue
            for outcome_event in EVENTS:
                with open(
                    os.path.join(
                        DATA_PATH,
                        index_event,
                        outcome_event,
                        "data.csv",
                    ),
                    "r",
                ) as f:
                    df = pd.read_csv(f)
                print(f"Processing {index_event}->{outcome_event}")
                print(f"Data has {len(df)} rows")
                print(f"Min timestamp: {df['timestamp'].min()}")
                print(f"Max timestamp: {df['timestamp'].max()}")
                subsetInRange = df[
                    (df["timestamp"] >= min_train_date)
                    & (df["timestamp"] < max_train_date)
                ]
                print(
                    f"Loaded {len(subsetInRange)} rows for {index_event}->{outcome_event}"
                )
                train_set = pd.concat(
                    [
                        train_set,
                        df[
                            (df["timestamp"] >= min_train_date)
                            & (df["timestamp"] < max_train_date)
                        ].sample(n=100, random_state=seed),
                    ],
                    ignore_index=True,
                )
        with open(train_set_dir, "w") as f:
            train_set.to_csv(f, index=False)
    return train_set


def run_training_pipeline():
    recommendation_cache_file = os.path.join(CACHE_DIR, "recommendations.pkl")
    if os.path.exists(recommendation_cache_file):
        with open(recommendation_cache_file, "rb") as f:
            recommendations = pkl.load(f)
    else:
        recommendations = {}
    train_set = get_train_set()
    for i, row in train_set.iterrows():
        rowID = str(row)
        if rowID not in recommendations:
            recommendations[rowID] = recommend_action(row)
            if i % 10 == 0:
                with open(recommendation_cache_file, "wb") as f:
                    pkl.dump(recommendations, f)
        print("Recommended " + str(recommendations[rowID]) + "\nfor " + rowID)
    with open(recommendation_cache_file, "wb") as f:
        pkl.dump(recommendations, f)


run_training_pipeline()
