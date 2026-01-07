# Author: Jun Lv
"""
Unified logging configuration using loguru.
Provides colored console output and file logging with rotation.
"""

import sys
from pathlib import Path
from typing import Any
from loguru import logger


def setup_logger() -> Any:
    """
    Setup loguru logger with colored console output and file logging.
    """
    # TODO: fix logger issues in sub-processes, currently it only works in main process.
    # Remove default handler
    logger.remove()
    
    # Console handler with colors
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>",
        # TODO: set the level based on configs
        level="INFO",
        colorize=True,
        backtrace=True,
        diagnose=True
    )
    
    # File handler for all logs
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logger.add(
        log_dir / "app.log",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
        level="DEBUG",
        rotation="1 day",
        retention="7 days",
        compression="zip",
        backtrace=True,
        diagnose=True
    )
    
    # Separate error log file
    logger.add(
        log_dir / "error.log",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
        level="ERROR",
        rotation="1 day",
        retention="30 days",
        compression="zip",
        backtrace=True,
        diagnose=True
    )
    
    return logger


# Initialize logger
setup_logger()

# Export logger for use in other modules
__all__ = ["logger"] 