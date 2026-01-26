import json
import pandas as pd
from typing import Optional, Dict, Any
import os
import pickle as pkl
import numpy as np
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold

from utils.data import get_event_df
from utils.logger import logger
from utils.constants import (
    LABEL_TIME,
    LABEL_EVENT,
    DATA_CACHE_DIR,
    MODEL_CACHE_DIR,
    DATA_PATH,
    seed,
    EVENTS,
)

PREPROCESS_CACHE: Dict[str, Any] = {}


def preprocess(
    train_df: Optional[pd.DataFrame] = None,
    test_df: Optional[pd.DataFrame] = None,
    model_date: Optional[int] = None,
):
    # Create unique prefix for saving/loading preprocessing objects
    present_dataframe = train_df if train_df is not None else test_df
    unique_prefix = (
        present_dataframe["Index Event"].iloc[0]
        + "_"
        + present_dataframe["Outcome Event"].iloc[0]
        + (f"_{model_date}_" if model_date else "_")
    )
    PREPROCESS_CACHE_PATH = os.path.join(DATA_CACHE_DIR, unique_prefix)

    target_columns = [LABEL_TIME, LABEL_EVENT]
    cols_to_drop = target_columns + [
        "id",
        "user",
        "pool",
        "Index Event",
        "Outcome Event",
        "type",
        "timestamp",
    ]

    if train_df is not None:
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

            train_df = train_df[
                (train_df["timestamp"] + train_df["timeDiff"]) <= model_date_value
            ]

        train_targets = train_df[target_columns]
        train_features = train_df.drop(columns=cols_to_drop, errors="ignore")

        # 1. Custom "Top 10" Categorical Logic
        categorical_cols = train_features.select_dtypes(
            include=["object", "category"]
        ).columns
        top10_dict = {}
        for c in categorical_cols:
            top10 = train_features[c].value_counts(dropna=True).nlargest(10).index
            train_features[c] = (
                train_features[c]
                .where(train_features[c].isin(top10), "Other")
                .fillna("Other")
            )
            top10_dict[c] = top10

        # 2. Setup ColumnTransformer (Scaling + Encoding)
        numerical_cols = train_features.select_dtypes(include=np.number).columns

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numerical_cols),
                # handle_unknown='ignore' ensures test data with new categories doesn't crash
                (
                    "cat",
                    OneHotEncoder(
                        sparse_output=False, handle_unknown="ignore", dtype=np.float32
                    ),
                    categorical_cols,
                ),
            ],
            verbose_feature_names_out=False,
        )

        # Apply transformations (Returns numpy array, we wrap back to DataFrame)
        train_features_transformed = pd.DataFrame(
            preprocessor.fit_transform(train_features),
            columns=preprocessor.get_feature_names_out(),
            index=train_features.index,
        )

        # 3. Variance Threshold (Feature Selection)
        selector = VarianceThreshold(threshold=0)
        train_features_final = pd.DataFrame(
            selector.fit_transform(train_features_transformed),
            columns=selector.get_feature_names_out(),
            index=train_features.index,
        ).astype(np.float32)

        target_train_duration = train_targets["timeDiff"].astype(float).values
        target_train_event = train_targets["status"].astype(int).values
        target_signed = np.where(
            target_train_event == 1, target_train_duration, -target_train_duration
        )
        train_features_final = xgb.DMatrix(train_features_final, label=target_signed)

        # 4. Save entire pipeline objects to cache
        PREPROCESS_CACHE[unique_prefix] = {
            "categorical_cols": categorical_cols,
            "top10_dict": top10_dict,
            "preprocessor": preprocessor,  # Saves the fitted Scaler and Encoder
            "selector": selector,  # Saves which columns were kept
        }
        with open(PREPROCESS_CACHE_PATH, "wb") as f:
            pkl.dump(PREPROCESS_CACHE[unique_prefix], f)
    else:
        train_features_final = train_targets = target_signed = None

    if test_df is not None:
        if unique_prefix not in PREPROCESS_CACHE:
            with open(PREPROCESS_CACHE_PATH, "rb") as f:
                PREPROCESS_CACHE[unique_prefix] = pkl.load(f)

        cache_data = PREPROCESS_CACHE[unique_prefix]
        test_features = test_df.drop(columns=cols_to_drop, errors="ignore")

        # 1. Apply "Top 10" Logic (using cached lists)
        top10_dict = cache_data["top10_dict"]
        categorical_cols = cache_data["categorical_cols"]
        for c in categorical_cols:
            test_features[c] = (
                test_features[c]
                .where(
                    test_features[c].isin(top10_dict[c]),
                    "Other",
                )
                .fillna("Other")
            )

        # 2. Apply Preprocessor (Scaling + Encoding)
        # transform() automatically handles alignment, zero-filling, and scaling using train stats
        preprocessor = cache_data["preprocessor"]
        test_features_transformed = pd.DataFrame(
            preprocessor.transform(test_features),
            columns=preprocessor.get_feature_names_out(),
            index=test_features.index,
        )

        # 3. Apply Selector (Drop zero variance cols from train)
        selector = cache_data["selector"]
        test_features_final = pd.DataFrame(
            selector.transform(test_features_transformed),
            columns=selector.get_feature_names_out(),
            index=test_features.index,
        ).astype(np.float32)

        test_features_final_index = test_features_final.index
        test_features_final = xgb.DMatrix(test_features_final)
    else:
        test_features_final = test_features_final_index = None

    return (
        train_features_final,
        train_targets,
        test_features_final,
        test_features_final_index,
    )


def compute_baseline_hazard(model, X_train, y_train):
    """
    Computes the Breslow estimator using Vectorized operations.
    Replaces the O(N^2) loop with O(N) groupby/cumsum for massive speedup.
    """
    # 1. Get Log-Hazards and Shift
    log_partial_hazard = model.predict(X_train, output_margin=True)
    log_shift = np.median(log_partial_hazard)

    # Center and Clip
    log_partial_hazard_centered = np.clip(log_partial_hazard - log_shift, -10, 10)
    partial_hazard = np.exp(log_partial_hazard_centered)

    # 2. VECTORIZED BRESLOW ESTIMATOR
    # Instead of iterating, we group by time.
    # This collapses 500k rows into unique time buckets instantly.
    df_temp = pd.DataFrame(
        {
            "time": y_train["timeDiff"].values,
            "event": y_train["status"].values,
            "risk": partial_hazard,
        }
    )

    # Group by unique durations
    # sum('risk') = hazard of all people leaving at this time t
    # sum('event') = actual deaths at this time t
    grouped = df_temp.groupby("time").agg({"event": "sum", "risk": "sum"}).sort_index()

    unique_durations = grouped.index.values
    events_at_t = grouped["event"].values
    risk_leaving_at_t = grouped["risk"].values

    # STABILITY FIX: Use Reverse Cumulative Sum for Risk Set
    # risk_at_t[i] = Sum of all risk leaving from time i to end
    # This avoids "Total - Cumulative" subtraction errors.
    risk_at_t = np.cumsum(risk_leaving_at_t[::-1])[::-1]

    # Handle near-zero risk to avoid division errors
    risk_at_t = np.maximum(risk_at_t, 1e-9)

    # Calculate Baseline Increment (d_i / Sum_Risk_i)
    hazard_increments = events_at_t / risk_at_t
    cum_baseline_hazard = np.cumsum(hazard_increments)

    # 3. Memory Optimization (Downsampling)
    MAX_POINTS = 2000
    if len(unique_durations) > MAX_POINTS:
        indices = np.linspace(0, len(unique_durations) - 1, MAX_POINTS).astype(int)
        unique_durations = unique_durations[indices]
        cum_baseline_hazard = cum_baseline_hazard[indices]

    # 4. Extrapolation Logic
    if len(unique_durations) > 5:
        tail_idx = int(len(unique_durations) * 0.8)
        time_delta = unique_durations[-1] - unique_durations[tail_idx]
        hazard_delta = cum_baseline_hazard[-1] - cum_baseline_hazard[tail_idx]
        final_rate = hazard_delta / (time_delta + 1e-9)
    else:
        final_rate = 0.0

    logger.debug(f"Baseline max hazard: {cum_baseline_hazard[-1]}")
    logger.debug(f"Total events: {np.sum(events_at_t)}")

    return {
        "times": unique_durations,
        "cum_hazards": cum_baseline_hazard,
        "final_rate": final_rate,
        "max_time": unique_durations[-1],
        "log_shift": log_shift,
    }


# Module-level cache for loaded/trained models to reuse across calls
MODELS_CACHE: dict = {}


def get_model_for_pair_and_date(
    index_event: str,
    outcome_event: str,
    model_date: int | None = None,
    verbose: bool = False,
):
    # Try module-level cache first
    model_key = (index_event, outcome_event, str(model_date))
    if model_key in MODELS_CACHE:
        return MODELS_CACHE[model_key]
    # normalize model_date for filename
    model_date_str = (
        str(model_date).replace(" ", "_") if model_date is not None else "latest"
    )
    model_filename = f"xgboost_cox_{index_event}_{outcome_event}_{model_date_str}.pkl"
    model_path = os.path.join(MODEL_CACHE_DIR, model_filename)
    baseline_path = model_path.replace(".pkl", "_baseline.pkl")

    # If model file exists, try to load into the estimator and return the estimator
    needToTrainAndSaveModel = True
    if os.path.exists(model_path):
        if verbose:
            logger.info(f"Loading existing model from {model_path}")
        with open(model_path, "rb") as f:
            model = pkl.load(f)
        if os.path.exists(baseline_path):
            with open(baseline_path, "rb") as f:
                baseline_data = pkl.load(f)
            MODELS_CACHE[model_key] = (model, baseline_data)
            return model, baseline_data
        else:
            needToTrainAndSaveModel = False

    dataset_path = os.path.join(index_event, outcome_event)

    # --- Load and Preprocess ---
    if verbose:
        logger.info(
            f"Loading data from {os.path.join(DATA_PATH, dataset_path, 'data.csv')}"
        )
    train_df = get_event_df(index_event, outcome_event)
    if train_df is None:
        logger.warning(f"No training data found for {dataset_path}")
        MODELS_CACHE[model_key] = None
        return None

    X_train, y_train, _, _ = preprocess(train_df, model_date=model_date)

    if needToTrainAndSaveModel:
        # Fit model: XGBoost Cox expects labels to be the event indicators
        # and the sample_weight to be the durations
        if verbose:
            logger.info("Training model...")
        params = {
            "objective": "survival:cox",
            "eval_metric": "cox-nloglik",
            "device": "cuda",
            "tree_method": "hist",
            "device": "cuda",
            "seed": seed,
            "verbosity": 1,
            "max_bin": 64,
            "learning_rate": 0.04,
            "max_depth": 5,
            "subsample": 0.85,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "reg_lambda": 1.0,
            "reg_alpha": 0.1,
        }
        model = xgb.train(
            params,
            X_train,
            num_boost_round=1000,
            evals=[(X_train, "train")],
            verbose_eval=100,
        )

        with open(model_path, "wb") as f:
            pkl.dump(model, f)

    baseline_data = compute_baseline_hazard(model, X_train, y_train)
    with open(baseline_path, "wb") as f:
        pkl.dump(baseline_data, f)

    # cache model (even if training produced a fitted estimator)
    MODELS_CACHE[model_key] = (model, baseline_data)
    return model, baseline_data

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

    total_pairs = len(event_pairs)
    for pair_idx, (index_event, outcome_event) in enumerate(event_pairs, start=1):
        if index_event == outcome_event and index_event == "Liquidated":
            continue
        if verbose:
            logger.info("\n" + "=" * 50)
            logger.info(
                f"Training event pair {pair_idx}/{total_pairs}: {index_event} -> {outcome_event}"
            )
            logger.info("" + "=" * 50)

        get_model_for_pair_and_date(
            index_event, outcome_event, model_date=model_date, verbose=verbose
        )

    if verbose:
        logger.info("\n\nAll prediction files have been generated.")
