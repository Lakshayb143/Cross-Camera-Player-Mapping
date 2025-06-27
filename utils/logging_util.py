import logging
from logging.handlers import RotatingFileHandler
import sys
from pathlib import Path

def initialize_logging():
    """
    Configures the root logger for the entire application.

    This setup provides a centralized, modern logging system with:
    - Dual output: to the console and a rotating file.
    - Structured formatting for clarity and parsing.
    """

    log_directory :Path = Path("logs")
    log_directory.mkdir(exist_ok=True)
    log_file :Path = Path("logs/application.log")

    format = logging.Formatter(
        "%(asctime)s - %(levelname)s - [%(name)s] - %(message)s"
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    """Implementing Console Handler"""
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(format)


    """Implementing File Handler for persistent logs"""
    file_handler = RotatingFileHandler(
        log_file, maxBytes=15*1024*1024, backupCount=3
    )
    file_handler.setFormatter(format)
    file_handler.setLevel(logging.INFO)

    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)


    logging.info("Logging system configured successfully")