import logging

from bot_does_things.config import LOGGING_LEVEL


def get_logger(logger_name: str):
    logger = logging.getLogger(logger_name)
    logger.handlers = []
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(levelname)s:%(name)s:%(message)s")
    )
    logger.addHandler(_handler)
    logger.setLevel(LOGGING_LEVEL)

    return logger
