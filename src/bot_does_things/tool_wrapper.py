"""
Includes a decorating wrapper for tools that adds minimal logging and error 
handling.
"""

import logging
from functools import wraps

from bot_does_things.config import LOGGING_LEVEL, RETURN_EXCEPTION_AS_STR

logger = logging.getLogger(__name__)
logger.handlers = []
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
logger.addHandler(_handler)
logger.setLevel(LOGGING_LEVEL)


def tool_wrapper(func):
    """
    Decorator that adds minimal logging and error handling to tools.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            logger.info(f"Running {func.__name__}")
            logger.debug(f"Arguments: {args}, {kwargs}")
            result = func(*args, **kwargs)
            logger.info(f"Finished {func.__name__}")
            logger.debug(f"Result: {result}")
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            if RETURN_EXCEPTION_AS_STR:
                return str(e)
            raise e

    return wraps(func)(wrapper)
