import logging
from pathlib import Path

from rich.logging import RichHandler
from rich.theme import Theme
from rich.console import Console
from rich.style import Style

# logger = logging.getLogger(__name__)

custom_theme = Theme({
    "logging.level.info": Style(color="#00f0f0"),
    "logging.level.debug": Style(color="#32ff32"),
    "logging.level.warning": Style(color="#feff32"),
    "logging.level.error": Style(color="#ff3232", bold=True),
    "logging.level.critical": Style(color="#ff3232", bold=True, reverse=True)
})

console = Console(theme=custom_theme)

def setup_logging(log_dir: Path | None = None, level: int = logging.INFO) -> None:
    root_logger = logging.getLogger()

    # Avoid adding duplicate handlers if called multiple times
    if getattr(root_logger, "_acb_logging_configured", False):
        return

    # Remove any existing handlers associated with the root logger (important for notebook use)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    if log_dir is None:
        log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "debug.log"

    # for logging to console
    shell_handler = RichHandler(
        level=level, 
        rich_tracebacks=True, 
        console=console, 
        markup=True) 
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)

    fmt_shell = '%(message)s'
    fmt_file = '%(levelname)s %(asctime)s [%(filename)s:%(funcName)s:%(lineno)s] %(message)s'

    shell_formatter = logging.Formatter(fmt_shell)
    file_formatter = logging.Formatter(fmt_file)

    # here we hook everything together
    shell_handler.setFormatter(shell_formatter)
    file_handler.setFormatter(file_formatter)

    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(shell_handler)
    root_logger.addHandler(file_handler)

    root_logger._acb_logging_configured = True
    root_logger.debug(f"Logging configured. Log file: {log_file}")