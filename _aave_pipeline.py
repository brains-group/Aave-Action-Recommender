"""Resolve the pinned local pipeline submodule before globally installed copies."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
PIPELINE = ROOT / "Aave-Data-Pipeline"
if not (PIPELINE / "aave_data_pipeline").is_dir():
    raise ImportError("Initialize Aave-Data-Pipeline: git submodule update --init --recursive")
sys.path.insert(0, str(PIPELINE))
