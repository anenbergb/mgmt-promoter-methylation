import logging

from fvcore.common.config import CfgNode
from loguru import logger


def setup_logger(cfg: CfgNode):
    """
    https://loguru.readthedocs.io/en/stable/resources/migration.html
    """
    lit_logger = logging.getLogger("lightning.pytorch")
    lit_logger.handlers = [InterceptHandler()]
    lit_logger.setLevel(logging.INFO)
    logger.add(f"{cfg.OUTPUT_DIR}/train.log", level="INFO")


class InterceptHandler(logging.Handler):
    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())
