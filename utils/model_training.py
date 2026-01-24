from constants import LABEL_TIME, LABEL_EVENT, DATA_CACHE_DIR
import pandas as pd
from typing import Optional, Dict, Any
import os
import pickle as pkl
import numpy as np

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
        train_features_final = train_targets = None

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
    else:
        test_features_final = None

    return train_features_final, train_targets, test_features_final
