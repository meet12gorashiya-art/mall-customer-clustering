"""Centralized logging configuration for the project.

Every module calls ``get_logger(__name__)`` to obtain a configured
logger instead of setting up handlers itself. Logs are written both
to stdout and to a rotating file under ``logs/app.log`` so failures
during a Streamlit Cloud run can be traced after the fact.
"""

import logging
import os
from logging.handlers import RotatingFileHandler

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs")
LOG_FILE = os.path.join(LOG_DIR, "app.log")


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger.

    Parameters
    ----------
    name : str
        Usually ``__name__`` of the calling module.
    level : int
        Logging level, defaults to ``logging.INFO``.

    Returns
    -------
    logging.Logger
        A logger with a stream handler and a rotating file handler
        attached exactly once (safe to call repeatedly, e.g. on every
        Streamlit rerun, without duplicating log lines).
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        # Already configured (e.g. Streamlit reruns the script on every
        # interaction) - avoid attaching duplicate handlers.
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        file_handler = RotatingFileHandler(
            LOG_FILE, maxBytes=1_000_000, backupCount=3
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError:
        # Streamlit Cloud's filesystem is ephemeral/read-only in some
        # contexts - fall back to stream-only logging rather than crash.
        logger.warning("Could not attach file handler; logging to stdout only.")

    logger.propagate = False
    return logger
