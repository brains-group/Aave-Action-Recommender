import pandas as pd
from typing import Optional
import os
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
