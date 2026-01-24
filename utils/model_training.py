from constants import LABEL_TIME, LABEL_EVENT, DATA_CACHE_DIR
import pandas as pd
from typing import Optional
import os
import pickle as pkl
import numpy as np

PREPROCESS_CACHE: dict = {}


def preprocess(
    train_df: Optional[pd.DataFrame] = None,
    test_df: Optional[pd.DataFrame] = None,
    model_date: Optional[int] = None,
):
    # Create unique prefix for saving/loading preprocessing objects
    unique_prefix = (
        (present_dataframe := (train_df or test_df))["Index Event"].iloc[0]
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

        train_features_encoded = pd.get_dummies(train_features, drop_first=False)
        train_features_encoded_columns = train_features_encoded.columns

        numerical_cols = train_features.select_dtypes(include=np.number).columns
        scaler = StandardScaler()
        train_features_scaled = pd.DataFrame(
            scaler.fit_transform(train_features[numerical_cols]),
            columns=numerical_cols,
            index=train_features.index,
        )

        train_features_final = pd.concat(
            [
                train_features_scaled,
                train_features_encoded.drop(columns=numerical_cols, errors="ignore"),
            ],
            axis=1,
        )

        cols_to_keep = train_features_final.columns[train_features_final.var() > 0]
        train_features_final = train_features_final[cols_to_keep].astype(np.float32)

        PREPROCESS_CACHE[unique_prefix] = {
            "categorical_cols": categorical_cols,
            "top10_dict": top10_dict,
            "train_features_encoded_columns": train_features_encoded_columns,
            "numerical_cols": numerical_cols,
            "scaler": scaler,
            "cols_to_keep": cols_to_keep,
        }
        with open(PREPROCESS_CACHE_PATH, "wb") as f:
            pkl.dump(PREPROCESS_CACHE[unique_prefix], f)
    else:
        train_features_final = train_targets = None

    if test_df is not None:
        if unique_prefix not in PREPROCESS_CACHE:
            with open(PREPROCESS_CACHE_PATH, "rb") as f:
                PREPROCESS_CACHE[unique_prefix] = pkl.load(f)

        test_features = test_df.drop(columns=cols_to_drop, errors="ignore")

        top10_dict = PREPROCESS_CACHE[unique_prefix]["top10_dict"]
        for c in PREPROCESS_CACHE[unique_prefix]["categorical_cols"]:
            test_features[c] = (
                test_features[c]
                .where(
                    test_features[c].isin(top10_dict[c]),
                    "Other",
                )
                .fillna("Other")
            )

        test_features_encoded = pd.get_dummies(test_features, drop_first=False)
        test_features_encoded = test_features_encoded.reindex(
            columns=PREPROCESS_CACHE[unique_prefix]["train_features_encoded_columns"],
            fill_value=0,
        )

        numerical_cols = PREPROCESS_CACHE[unique_prefix]["numerical_cols"]
        test_features_scaled = pd.DataFrame(
            PREPROCESS_CACHE[unique_prefix]["scaler"].transform(
                test_features[numerical_cols]
            ),
            columns=numerical_cols,
            index=test_features.index,
        )

        test_features_final = pd.concat(
            [
                test_features_scaled,
                test_features_encoded.drop(columns=numerical_cols, errors="ignore"),
            ],
            axis=1,
        )

        test_features_final = test_features_final.reindex(
            columns=PREPROCESS_CACHE[unique_prefix]["cols_to_keep"], fill_value=0
        ).astype(np.float32)
    else:
        test_features_final = None

    return train_features_final, train_targets, test_features_final
