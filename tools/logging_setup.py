"""
Reusable logging configuration module.
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(log_dir, script_name, file_level=logging.DEBUG, console_level=logging.INFO):
    """
    Configure logging to file and console.

    Args:
        log_dir: Directory to store log files
        script_name: Name of the script (used in log filename)
        file_level: Logging level for file handler (default: DEBUG)
        console_level: Logging level for console handler (default: INFO)

    Returns:
        Path to the log file
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{script_name}.log"

    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Remove existing handlers
    root_logger.handlers.clear()

    # File handler
    file_handler = RotatingFileHandler(
        log_file, maxBytes=100 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    root_logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    root_logger.addHandler(console_handler)

    logging.info(f"Logging to: {log_file}")
