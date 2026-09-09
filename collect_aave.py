from pathlib import Path
import sys
from _aave_pipeline import ROOT
from aave_data_pipeline.collect import main
if __name__ == "__main__":
    main(default_config=ROOT / "pipeline.config.json", default_env=ROOT / ".env", default_log=ROOT / "log")
