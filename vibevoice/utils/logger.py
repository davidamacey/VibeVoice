"""
Logging configuration for VibeVoice.

Usage:
    from vibevoice.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Hello world")
"""

import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Union
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler


DEFAULT_LOG_LEVEL = logging.INFO
DEFAULT_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] %(funcName)s - %(message)s'
DEFAULT_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
DEFAULT_LOG_DIR = Path('./logs')
DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10MB
DEFAULT_BACKUP_COUNT = 5

LOG_LEVEL_ENV = os.environ.get('LOG_LEVEL', '').upper()
LOG_TO_FILE_ENV = os.environ.get('LOG_TO_FILE', 'false').lower() == 'true'
LOG_DIR_ENV = os.environ.get('LOG_DIR', str(DEFAULT_LOG_DIR))

_logger_registry = {}


def _get_log_level(level: Optional[Union[int, str]] = None) -> int:
    if level is None:
        if LOG_LEVEL_ENV:
            level = LOG_LEVEL_ENV
        else:
            return DEFAULT_LOG_LEVEL

    if isinstance(level, str):
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL,
        }
        return level_map.get(level.upper(), DEFAULT_LOG_LEVEL)

    return level


def _create_console_handler(level: int, formatter: logging.Formatter, stream=None) -> logging.StreamHandler:
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(formatter)
    return handler


def _create_file_handler(
    name: str,
    level: int,
    formatter: logging.Formatter,
    log_dir: Path,
    max_bytes: int = DEFAULT_MAX_BYTES,
    backup_count: int = DEFAULT_BACKUP_COUNT,
    rotation_type: str = 'size'
) -> Union[RotatingFileHandler, TimedRotatingFileHandler]:
    log_dir.mkdir(parents=True, exist_ok=True)
    safe_name = name.replace('.', '_').replace('/', '_')
    log_file = log_dir / f"{safe_name}.log"

    if rotation_type == 'time':
        handler = TimedRotatingFileHandler(
            log_file, when='midnight', interval=1, backupCount=backup_count, encoding='utf-8'
        )
    else:
        handler = RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count, encoding='utf-8'
        )

    handler.setLevel(level)
    handler.setFormatter(formatter)
    return handler


def get_logger(
    name: str,
    level: Optional[Union[int, str]] = None,
    format: Optional[str] = None,
    date_format: Optional[str] = None,
    handlers: Optional[List[logging.Handler]] = None,
    propagate: bool = True,
    log_to_file: Optional[bool] = None,
    log_dir: Optional[Union[str, Path]] = None,
    file_rotation: str = 'size',
    max_bytes: int = DEFAULT_MAX_BYTES,
    backup_count: int = DEFAULT_BACKUP_COUNT,
    force_reconfigure: bool = False
) -> logging.Logger:
    if name in _logger_registry and not force_reconfigure:
        return _logger_registry[name]

    logger = logging.getLogger(name)

    if logger.handlers and not force_reconfigure:
        _logger_registry[name] = logger
        return logger

    if force_reconfigure:
        logger.handlers.clear()

    log_level = _get_log_level(level)
    logger.setLevel(log_level)
    logger.propagate = propagate

    log_format = format or DEFAULT_FORMAT
    log_date_format = date_format or DEFAULT_DATE_FORMAT
    formatter = logging.Formatter(log_format, log_date_format)

    if handlers is not None:
        for handler in handlers:
            logger.addHandler(handler)
    else:
        console_handler = _create_console_handler(log_level, formatter)
        logger.addHandler(console_handler)

        should_log_to_file = log_to_file if log_to_file is not None else LOG_TO_FILE_ENV
        if should_log_to_file:
            file_log_dir = Path(log_dir) if log_dir else Path(LOG_DIR_ENV)
            file_handler = _create_file_handler(
                name=name, level=log_level, formatter=formatter,
                log_dir=file_log_dir, max_bytes=max_bytes,
                backup_count=backup_count, rotation_type=file_rotation
            )
            logger.addHandler(file_handler)

    _logger_registry[name] = logger
    return logger
