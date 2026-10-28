"""
logger.py — Logging setup for the AI Property Video Generator.

File handler: DEBUG and above -> logs/app.log
Console handler: WARNING and above (so normal runs stay quiet)
"""

import logging
from pathlib import Path


def setup_logger(name: str = "property_video") -> logging.Logger:
    """
    Configure and return the application logger.
    Creates logs/ next to this project's root if it doesn't exist.
    Safe to call multiple times (handlers are added only once).
    """
    # Place logs/ at the project root (two levels up from src/)
    logs_dir = Path(__file__).parent.parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "app.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Guard against duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler — full detail
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console handler — only warnings and above
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(ch)

    return logger
