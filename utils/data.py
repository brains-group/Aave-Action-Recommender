import bisect
import time
from pathlib import Path
import random
import pandas as pd
from typing import Optional
import os
import pickle as pkl

import pyreadr
from utils.constants import PROFILES_DIR
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

    # Unique token for this process's loading attempt
    loading_token = f"LOADING_{os.getpid()}_{random.randint(0, 1000000)}"

    while True:
        if key in EVENT_DF_CACHE:
            val = EVENT_DF_CACHE[key]
            # Check if it is a loading token
            if isinstance(val, str) and val.startswith("LOADING_"):
                time.sleep(1)
                continue
            return val

        # Try to claim loading responsibility
        current_val = EVENT_DF_CACHE.setdefault(key, loading_token)

        if isinstance(current_val, str) and current_val == loading_token:
            try:
                logger.info(f"Loading event dataframe for ({index_event}, {outcome_event})...")
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
                logger.info(f"Finished loading event dataframe for ({index_event}, {outcome_event})...")
                return df
            except Exception:
                # If loading fails, clear the token so others can retry
                if key in EVENT_DF_CACHE and EVENT_DF_CACHE[key] == loading_token:
                    del EVENT_DF_CACHE[key]
                raise

        # If we didn't get the lock, wait and retry
        if isinstance(current_val, str) and current_val.startswith("LOADING_"):
            time.sleep(1)
            continue

        return current_val


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


def get_user_profile(transaction=None, user=None):
    if user is None:
        user = transaction.get("user")

    # Search for profile in both non_liquidated_profiles and liquidated_profiles subdirectories
    # PROFILES_DIR should point to a directory containing both subdirectories
    profiles_base = Path(PROFILES_DIR).expanduser()

    # Primary search paths: check the standard structure first
    # 1. non_liquidated_profiles/profiles/
    # 2. liquidated_profiles/profiles/
    search_paths = [
        profiles_base / "non_liquidated_profiles" / "profiles" / f"user_{user}.json",
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
        return None

    try:
        with user_profile_file.open("r") as f:
            user_profile = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in profile file {user_profile_file}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error reading profile file {user_profile_file}: {e}")
        return None

    if not isinstance(user_profile, dict):
        logger.error(
            f"Invalid profile format for user {user}: expected dict, got {type(user_profile)}"
        )
        return None

    if "transactions" not in user_profile:
        logger.warning(f"Profile for user {user} missing 'transactions' field")
        user_profile["transactions"] = []

    return user_profile


def get_train_set():
    TRAIN_SET_CACHE_PATH = os.path.join(CACHE_DIR, "train_set.csv")
    if os.path.exists(TRAIN_SET_CACHE_PATH):
        return pd.read_csv(TRAIN_SET_CACHE_PATH)
    train_ranges, test_ranges = get_date_ranges()
    min_train_date = train_ranges[0].timestamp()
    max_train_date = test_ranges[0].timestamp()
    all_users = pd.read_csv(os.path.join(DATA_PATH, "users.csv"))["id"]
    old_users = pd.read_csv(os.path.join(DATA_PATH, "users_V2.csv"))["id"]
    filtered_users = list(set(all_users) - set(old_users))
    random.shuffle(filtered_users)
    event_pairs = [
        (ie, oe) for ie in EVENTS for oe in EVENTS if not (ie == oe == "Liquidated")
    ]
    num_per_event_pair = 500
    event_pair_counts = {event_pair: 0 for event_pair in event_pairs}
    collected_rows = []
    for user in filtered_users:
        user_profile = get_user_profile(user=user)
        if not user_profile:
            continue
        user_transactions = user_profile["transactions"]
        if not user_transactions:
            continue

        user_transactions.sort(key=lambda x: x.get("timestamp", 0))
        timestamps = [tx.get("timestamp", 0) for tx in user_transactions]
        start_index = bisect.bisect_left(timestamps, min_train_date)
        end_index = bisect.bisect_right(timestamps, max_train_date)
        if end_index <= start_index:
            continue
        filtered_transactions = user_transactions[start_index:end_index]

        user_actions = [transaction["action"] for transaction in filtered_transactions]
        if "Liquidated" in user_actions:
            user_actions = user_actions[: user_actions.index("Liquidated") + 1]
        if len(user_actions) < 10:
            continue

        user_action_pair_index_dict = {event_pair: [] for event_pair in event_pairs}
        for index in range(
            max(int(len(filtered_transactions) / 5) - start_index, 0),
            len(user_actions) - 1,
        ):
            user_action_pair_index_dict[
                (user_actions[index], user_actions[index + 1])
            ].append(index)

        for event_pair in event_pairs:
            if event_pair_counts[event_pair] < num_per_event_pair:
                pair_indexes = user_action_pair_index_dict[event_pair]
                if len(pair_indexes):
                    chosen_transaction = filtered_transactions[
                        random.choice(pair_indexes)
                    ]
                    event_df = get_event_df(event_pair[0], event_pair[1])
                    timestamp = chosen_transaction["timestamp"]
                    splice = event_df[
                        (event_df["user"] == user)
                        & (event_df["timestamp"] == timestamp)
                    ]
                    if splice.empty:
                        logger.warning(
                            f"Searched for test transaction ({user}, {timestamp}), but not found."
                        )
                        continue
                    row = splice.iloc[0]
                    collected_rows.append(row)
                    logger.info(
                        f"Added row of type {event_pair} to collected_rows and reached {len(collected_rows)} rows."
                    )
                    break

        if len(collected_rows) >= len(event_pairs) * num_per_event_pair:
            break
    train_set = pd.concat(collected_rows, axis=1).T
    with open(TRAIN_SET_CACHE_PATH, "w") as f:
        train_set.to_csv(f, index=False)
    return train_set


# def get_train_set():
#     TRAIN_SET_CACHE_PATH = os.path.join(CACHE_DIR, "train_set.csv")
#     if os.path.exists(TRAIN_SET_CACHE_PATH):
#         train_set = pd.read_csv(TRAIN_SET_CACHE_PATH)
#     else:
#         train_set = pd.DataFrame()
#         train_ranges, test_ranges = get_date_ranges()
#         min_train_date = train_ranges[0].timestamp()
#         max_train_date = test_ranges[0].timestamp()
#         event_pairs = [
#             (ie, oe) for ie in EVENTS for oe in EVENTS if not (ie == oe == "Liquidated")
#         ]
#         for index_event, outcome_event in event_pairs:
#             # log progress for building the train set
#             logger.info(
#                 "Building train_set: processing %s->%s", index_event, outcome_event
#             )
#             df = get_event_df(index_event, outcome_event)
#             if df is None:
#                 continue
#             # logger = logging.getLogger(__name__)
#             logger.info(f"Processing {index_event}->{outcome_event}")
#             logger.debug(f"Data has {len(df)} rows")
#             logger.debug(f"Min timestamp: {df['timestamp'].min()}")
#             logger.debug(f"Max timestamp: {df['timestamp'].max()}")
#             subsetInRange = df[
#                 (df["timestamp"] >= min_train_date) & (df["timestamp"] < max_train_date)
#             ]
#             logger.debug(
#                 f"Loaded {len(subsetInRange)} rows for {index_event}->{outcome_event}"
#             )
#             train_set = pd.concat(
#                 [
#                     train_set,
#                     subsetInRange.sample(n=300, random_state=seed),
#                 ],
#                 ignore_index=True,
#             )
#         for index_event in EVENTS:
#             if index_event == "Liquidated":
#                 continue
#             outcome_event = "Liquidated"
#             logger.info(
#                 "Building train_set: processing %s->%s", index_event, outcome_event
#             )
#             df = get_event_df(index_event, outcome_event)
#             if df is None:
#                 continue
#             logger.info(f"Processing {index_event}->{outcome_event}")
#             logger.debug(f"Data has {len(df)} rows")
#             logger.debug(f"Min timestamp: {df['timestamp'].min()}")
#             logger.debug(f"Max timestamp: {df['timestamp'].max()}")
#             subsetInRange = df[
#                 (df["timestamp"] >= min_train_date) & (df["timestamp"] < max_train_date)
#             ]
#             logger.debug(
#                 f"Loaded {len(subsetInRange)} rows for {index_event}->{outcome_event}"
#             )
#             train_set = pd.concat(
#                 [
#                     train_set,
#                     subsetInRange.sort_values(by="timeDiff").head(300),
#                 ],
#                 ignore_index=True,
#             )
#         with open(TRAIN_SET_CACHE_PATH, "w") as f:
#             train_set.to_csv(f, index=False)

#     return train_set
