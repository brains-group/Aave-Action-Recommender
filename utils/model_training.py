from constants import LABEL_TIME, LABEL_EVENT
import pandas as pd
from typing import Optional

PREPROCESS_CACHE: dict = {}


def preprocess(
    train_df: Optional[pd.DataFrame] = None,
    test_df: Optional[pd.DataFrame] = None,
    model_date: Optional[int] = None,
):
    target_columns = [LABEL_TIME, LABEL_EVENT]
    drop_cols = target_columns + [
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
                (train_df["timestamp"] + train_df["timeDiff"])
                <= model_date_value
            ]

        train_targets = train_df[target_columns]
        # y_time = train_df[LABEL_TIME].astype(float).values
        # y_event = train_df[LABEL_EVENT].astype(int).values

    X_train = train_df.drop(columns=drop_cols, errors="ignore")
    X_test = test_df.drop(columns=drop_cols, errors="ignore")

    # 類別與數值欄位區分
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns
    X_train, X_test = fit_top10_and_map(X_train, X_test, cat_cols)
    dtr = pd.get_dummies(X_train, drop_first=False)
    dte = pd.get_dummies(X_test, drop_first=False)
    dte = dte.reindex(columns=dtr.columns, fill_value=0)

    num_cols = X_train.select_dtypes(include=np.number).columns
    scaler = StandardScaler()
    Xtr_num = pd.DataFrame(
        scaler.fit_transform(X_train[num_cols]), columns=num_cols, index=X_train.index
    )
    Xte_num = pd.DataFrame(
        scaler.transform(X_test[num_cols]), columns=num_cols, index=X_test.index
    )

    Xtr = pd.concat([Xtr_num, dtr.drop(columns=num_cols, errors="ignore")], axis=1)
    Xte = pd.concat([Xte_num, dte.drop(columns=num_cols, errors="ignore")], axis=1)

    nz_cols = Xtr.columns[Xtr.var() > 0]
    Xtr = Xtr[nz_cols].astype(np.float32)
    Xte = Xte.reindex(columns=nz_cols, fill_value=0).astype(np.float32)

    return Xtr, Xte, y_time, y_event
