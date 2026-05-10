import logging
import sys

from loguru import logger


class _InterceptHandler(logging.Handler):
    """Route stdlib logging through loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = str(record.levelno)

        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back  # type: ignore[assignment]
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_logging(log_level: str = "INFO", app_env: str = "development") -> None:
    logger.remove()

    if app_env == "production":
        logger.add(
            sys.stdout,
            level=log_level.upper(),
            format=(
                '{{"time":"{time:YYYY-MM-DDTHH:mm:ss.SSSZ}",'
                '"level":"{level}",'
                '"message":"{message}",'
                '"name":"{name}",'
                '"function":"{function}",'
                '"line":{line}}}'
            ),
            serialize=False,
        )
    else:
        logger.add(
            sys.stdout,
            level=log_level.upper(),
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                "<level>{message}</level>"
            ),
            colorize=True,
        )

    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)
    for name in logging.root.manager.loggerDict:
        logging.getLogger(name).handlers = [_InterceptHandler()]
        logging.getLogger(name).propagate = False
