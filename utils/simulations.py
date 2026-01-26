from pathlib import Path
import pickle as pkl
import os
import sys

from utils.logger import logger
from utils.constants import SIMULATION_RESULTS_CACHE_DIR

# Make the bundled Aave-Simulator directory importable (it's next to this file)
this_dir = os.path.dirname(os.path.realpath(__file__))
aave_sim_path = os.path.join(this_dir, "Aave-Simulator")
if aave_sim_path not in sys.path:
    sys.path.insert(0, aave_sim_path)

from tools.run_single_simulation import run_simulation


def get_simulation_outcome(recommendation, suffix, **passed_args):
    """Load cached simulation results or compute and cache them."""
    key = f"{recommendation['user']}_{int(recommendation.get('timestamp', 0))}_{suffix}"
    results_cache_file = Path(SIMULATION_RESULTS_CACHE_DIR) / f"{key}.pkl"
    # logger.debug("Checkpoint 7.6")

    # Try to load from cache
    if results_cache_file.exists():
        try:
            with open(results_cache_file, "rb") as f:
                return pkl.load(f)
        except Exception as e:
            logger.debug(f"Failed to load cache {results_cache_file}: {e}")
    # logger.debug("Checkpoint 7.7")

    # Cache miss - compute results
    results = run_simulation(**passed_args)
    # logger.debug("Checkpoint 7.8")

    # Save to cache (best effort)
    try:
        Path(SIMULATION_RESULTS_CACHE_DIR).mkdir(parents=True, exist_ok=True)
        with open(results_cache_file, "wb") as f:
            pkl.dump(results, f)
    except Exception as e:
        logger.debug(f"Failed to save cache {results_cache_file}: {e}")
    # logger.debug("Checkpoint 7.9")

    return results
