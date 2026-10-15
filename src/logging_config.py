import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def configure_logging(
    log_dir: str = "logs",
    level: int = logging.INFO,
) -> None:
    """Configure console and rotating-file logging once."""
    root = logging.getLogger()

    if getattr(root, "_alignment_logging_configured", False):
        return

    Path(log_dir).mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    console = logging.StreamHandler()
    console.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        Path(log_dir) / "pipeline.log",
        maxBytes=1_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    root.setLevel(level)
    root.addHandler(console)
    root.addHandler(file_handler)
    root._alignment_logging_configured = True