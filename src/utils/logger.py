"""Logging utilities for the TB Drug Discovery pipeline.

This module provides consistent logging configuration across the project.
"""

import sys
from pathlib import Path
from typing import Optional

from loguru import logger


def setup_logger(
    log_file: Optional[str] = None,
    level: str = "INFO",
    rotation: str = "10 MB",
    retention: str = "1 week",
) -> None:
    """Configure the logger for the pipeline.
    
    Sets up loguru with consistent formatting for both console
    and file output.
    
    Args:
        log_file: Optional path to log file. If None, logs only to console.
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        rotation: When to rotate log files (e.g., "10 MB", "1 day").
        retention: How long to keep old log files.
        
    Example:
        >>> setup_logger("logs/pipeline.log", level="DEBUG")
        >>> logger.info("Pipeline started")
    """
    # Remove default handler
    logger.remove()
    
    # Console handler with colors
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    
    logger.add(
        sys.stderr,
        format=console_format,
        level=level,
        colorize=True,
    )
    
    # File handler if path provided
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        
        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message}"
        )
        
        logger.add(
            log_file,
            format=file_format,
            level=level,
            rotation=rotation,
            retention=retention,
            compression="zip",
        )
    
    logger.info(f"Logger initialized with level={level}")


def get_logger(name: str):
    """Get a logger instance with the given name.
    
    Args:
        name: Logger name (typically __name__).
        
    Returns:
        Logger instance bound to the given name.
    """
    return logger.bind(name=name)
