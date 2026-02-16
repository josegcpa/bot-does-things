"""
Includes a decorating wrapper for tools that adds minimal logging and error 
handling.
"""

from functools import wraps

from bot_does_things.config import RETURN_EXCEPTION_AS_STR
from bot_does_things.logger import get_logger

logger = get_logger(__name__)


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
