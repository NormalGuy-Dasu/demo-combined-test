from __future__ import annotations

import logging
import sys
from datetime import date
from pathlib import Path

_LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
_initialized: set[str] = set()


def get_logger(name: str) -> logging.Logger:
    """
    Return a configured logger that writes to both console and a daily log file.

    Console → INFO and above, compact format.
    File    → DEBUG and above, verbose format, at logs/<top_package>_YYYY-MM-DD.log.

    Call get_logger(__name__) once at the top of each module.
    """
    logger = logging.getLogger(name)

    if name in _initialized:
        return logger

    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] %(levelname)-5s %(name)-30s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    top_package = name.split(".")[0]
    log_file = _LOG_DIR / f"{top_package}_{date.today().isoformat()}.log"

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] %(levelname)-5s %(name)-30s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.propagate = False

    _initialized.add(name)
    return logger
