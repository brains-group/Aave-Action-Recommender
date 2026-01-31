import logging
import os


def set_log_file(
    path: str,
    increment_name: bool = True,
    output_folder: str = "./outputs/",
    file_level: str | int = "DEBUG",
):
    """Change the log file path and level. File will receive all levels at or above file_level."""
    # remove old file handler
    global _file_handler
    try:
        logger.removeHandler(_file_handler)
    except Exception:
        pass

    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
    path = os.path.join(output_folder, path)

    if increment_name:
        period_index = path.rindex(".")
        path_name = path[:period_index]
        path_suffix = path[period_index:]
        count = 1
        while os.path.exists(path) and os.path.getsize(path) > 0:
            path = path_name + str(count) + path_suffix
            count += 1

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


# Module logger
logger = logging.getLogger(__name__)
set_log_file("output.log")
logger.setLevel(logging.DEBUG)
logger.propagate = False
