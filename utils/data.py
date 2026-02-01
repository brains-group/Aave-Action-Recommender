import pandas as pd
from typing import Optional
import os
import pickle as pkl

import pyreadr
from utils.logger import logger
from utils.constants import *

# In-memory cache for event CSV dataframes to avoid repeated disk I/O
EVENT_DF_CACHE: dict = {}


def get_event_df(index_event: str, outcome_event: str) -> Optional[pd.DataFrame]:
    """Return cached DataFrame for an event pair, loading it once if needed.

    Returns None if the CSV does not exist.
    """
    global EVENT_DF_CACHE
    key = (index_event, outcome_event)
    if key in EVENT_DF_CACHE:
        return EVENT_DF_CACHE[key]
    event_path = os.path.join(DATA_PATH, index_event, outcome_event, "data.csv")
    if not os.path.exists(event_path):
        EVENT_DF_CACHE[key] = None
        return None
    try:
        df = pd.read_csv(event_path)
    except Exception as e:
        logger.warning(f"Warning: failed to read {event_path}: {e}")
        EVENT_DF_CACHE[key] = None
        return None
    EVENT_DF_CACHE[key] = df
    return df

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
