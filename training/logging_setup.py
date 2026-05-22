from loguru import logger
import sys
import os


def setup_logger(log_dir: str = "logs") -> None:
    logger.remove()

    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
        colorize=True,
    )

    os.makedirs(log_dir, exist_ok=True)
    logger.add(
        os.path.join(log_dir, "training_{time:YYYY-MM-DD}.log"),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="10 MB",
        retention="30 days",
    }
