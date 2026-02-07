"""Logging utilities."""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "perception",
    level: str = "INFO",
    log_file: Optional[str] = None,
    console: bool = True,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Set up and configure a logger.

    Args:
        name: Logger name.
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional file path for logging.
        console: Whether to log to console.
        format_string: Custom format string.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    logger.handlers = []

    # Default format
    if format_string is None:
        format_string = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "perception") -> logging.Logger:
    """
    Get an existing logger or create a basic one.

    Args:
        name: Logger name.

    Returns:
        Logger instance.
    """
    logger = logging.getLogger(name)

    # If no handlers, set up basic logging
    if not logger.handlers:
        logger = setup_logger(name)

    return logger


class LoggerMixin:
    """Mixin class to add logging to any class."""

    @property
    def logger(self) -> logging.Logger:
        """Get class-specific logger."""
        if not hasattr(self, "_logger"):
            self._logger = get_logger(self.__class__.__name__)
        return self._logger


def log_function_call(logger: Optional[logging.Logger] = None):
    """
    Decorator to log function calls.

    Args:
        logger: Logger to use (uses default if None).
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            log = logger or get_logger()
            log.debug(f"Calling {func.__name__}")
            try:
                result = func(*args, **kwargs)
                log.debug(f"{func.__name__} completed successfully")
                return result
            except Exception as e:
                log.error(f"{func.__name__} failed: {e}")
                raise
        return wrapper
    return decorator


class ProgressLogger:
    """Log progress for long-running operations."""

    def __init__(
        self,
        total: int,
        logger: Optional[logging.Logger] = None,
        description: str = "Processing",
        log_interval: int = 10,
    ):
        """
        Initialize progress logger.

        Args:
            total: Total number of items.
            logger: Logger to use.
            description: Progress description.
            log_interval: Percentage interval for logging.
        """
        self.total = total
        self.logger = logger or get_logger()
        self.description = description
        self.log_interval = log_interval

        self.current = 0
        self.last_logged_pct = -1

    def update(self, n: int = 1) -> None:
        """
        Update progress.

        Args:
            n: Number of items processed.
        """
        self.current += n
        pct = int(100 * self.current / self.total)

        if pct >= self.last_logged_pct + self.log_interval:
            self.logger.info(
                f"{self.description}: {self.current}/{self.total} ({pct}%)"
            )
            self.last_logged_pct = pct

    def __enter__(self):
        self.logger.info(f"{self.description}: Starting ({self.total} items)")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.logger.info(f"{self.description}: Completed")
        else:
            self.logger.error(f"{self.description}: Failed - {exc_val}")
