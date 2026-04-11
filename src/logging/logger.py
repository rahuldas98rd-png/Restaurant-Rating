"""
src/logging/logger.py
Centralized logging setup for the entire project.
"""

import logging
import os
from datetime import datetime
from pathlib import Path


def get_logger(name: str, log_dir: str = "logs", level: int = logging.INFO) -> logging.Logger:
    """
    Create and return a logger that writes to both console and a rotating log file.

    Args:
        name:    Logger name (typically __name__ of the calling module).
        log_dir: Directory to write log files.
        level:   Logging level (default INFO).

    Returns:
        Configured logging.Logger instance.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger  # Avoid duplicate handlers on re-import

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(lineno)d | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Console handler ──────────────────────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ── File handler ─────────────────────────────────────────
    log_filename = os.path.join(log_dir, f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log")
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger