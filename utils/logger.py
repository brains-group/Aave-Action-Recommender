import logging

outputFile = "output.log"
# Module logger
logger = logging.getLogger(__name__)
# File handler: capture all log levels to file
_file_handler = logging.FileHandler(outputFile)
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(_file_handler)
# # Console handler: show INFO+ by default
# _console_handler = logging.StreamHandler()
# _console_handler.setLevel(logging.INFO)
# _console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
# logger.addHandler(_console_handler)
# Ensure logger forwards all levels to handlers (file will receive DEBUG+)
logger.setLevel(logging.DEBUG)
logger.propagate = False


def set_log_file(path: str, file_level: str | int = "DEBUG"):
    """Change the log file path and level. File will receive all levels at or above file_level."""
    # remove old file handler
    global _file_handler
    try:
        logger.removeHandler(_file_handler)
    except Exception:
        pass
    _file_handler = logging.FileHandler(path)
    if isinstance(file_level, str):
        lvl = getattr(logging, file_level.upper(), logging.DEBUG)
    else:
        lvl = int(file_level)
    _file_handler.setLevel(lvl)
    _file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    logger.addHandler(_file_handler)
