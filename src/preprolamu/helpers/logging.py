from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

from rich.console import Console
from rich.logging import RichHandler
from rich.style import Style
from rich.theme import Theme


# --- Default rich theme -------------------------------------------------------

_CUSTOM_THEME = Theme(
    {
        "logging.level.info": Style(color="#00f0f0"),
        "logging.level.debug": Style(color="#32ff32"),
        "logging.level.warning": Style(color="#feff32"),
        "logging.level.error": Style(color="#ff3232", bold=True),
        "logging.level.critical": Style(
            color="#ff3232", bold=True, reverse=True
        ),
    }
)


def setup_logging(
    log_dir: Path | str | None = None,
    level: int = logging.INFO,
    *,
    suppress_loggers: Iterable[str] | None = None,
    reset_handlers: bool = True,
) -> None:
    """
    Configure logging with rich console output and file logging.

    This function is safe to call multiple times (will only configure once).

    Args:
        log_dir: Directory where logs are written (default: ./logs)
        level: Console logging level (default: INFO)
        suppress_loggers: Iterable of logger names to silence (set to WARNING)
        reset_handlers: Whether to remove existing handlers first

    Example:
        setup_logging(
            suppress_loggers=[
                "PIL",
                "matplotlib.font_manager",
                "matplotlib.texmanager",
            ]
    )
    """
    root_logger = logging.getLogger()

    # Prevent duplicate configuration
    if getattr(root_logger, "_utils_logging_configured", False):
        return

    # --- Optional noise suppression ------------------------------------------

    if suppress_loggers:
        for name in suppress_loggers:
            logging.getLogger(name).setLevel(logging.WARNING)

    # --- Reset handlers (important for notebooks / repeated runs) -------------

    if reset_handlers:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    # --- Log directory --------------------------------------------------------

    if log_dir is None:
        log_dir = Path("logs")
    else:
        log_dir = Path(log_dir)

    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "debug.log"

    # --- Console (rich) handler ----------------------------------------------

    console = Console(theme=_CUSTOM_THEME)

    console_handler = RichHandler(
        level=level,
        rich_tracebacks=True,
        console=console,
        markup=True,
        show_time=False,
        show_path=False,
    )
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    # --- File handler (full debug info) ---------------------------------------

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            "%(levelname)s\t%(asctime)s [%(filename)s:%(funcName)s:%(lineno)s] %(message)s"
        )
    )

    # --- Attach handlers ------------------------------------------------------

    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Mark as configured
    root_logger._utils_logging_configured = True

    # --- Startup messages -----------------------------------------------------

    root_logger.info(
        "------------------------------------------ NEW RUN ------------------------------------------"
    )
    root_logger.info(f"Logging configured. Log file: {log_file}")
