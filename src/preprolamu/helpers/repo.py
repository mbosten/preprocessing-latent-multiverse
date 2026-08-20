from __future__ import annotations

from pathlib import Path
from typing import Iterable


def tree(
    path: str | Path = ".",
    prefix: str = "",
    allowed_extensions: Iterable[str] | None = None,
    ignore_folders: Iterable[str] | None = None,
    _is_root: bool = True,
) -> str:
    """
    Return a directory tree as a string.

    Args:
        path: Root directory to display.
        prefix: Used internally for recursive indentation.
        allowed_extensions: File extensions to include, e.g. {".py", ".md"}.
            If None, defaults to DEFAULT_ALLOWED_EXTENSIONS.
            If empty, no files will be included.
        ignore_folders: Folder names to skip entirely.
            If None, defaults to DEFAULT_IGNORE_FOLDERS.

    Returns:
        A string containing the formatted directory tree.
    """

    DEFAULT_IGNORE_FOLDERS = {
        "__pycache__",
        ".git",
        ".github",
        ".ruff_cache",
        ".venv",
    }

    DEFAULT_ALLOWED_EXTENSIONS = {".py", ".yml", ".md"}
    
    path = Path(path)
    ignore_folders = set(ignore_folders or DEFAULT_IGNORE_FOLDERS)
    allowed_extensions = set(
        DEFAULT_ALLOWED_EXTENSIONS if allowed_extensions is None else allowed_extensions
    )
    lines: list[str] = []

    if _is_root:
        lines.append(path.resolve().name)

    items: list[Path] = []
    for item in path.iterdir():
        if item.is_dir() and item.name in ignore_folders:
            continue

        if item.is_file() and allowed_extensions is not None:
            if item.suffix not in allowed_extensions:
                continue

        items.append(item)

    items.sort(key=lambda p: (not p.is_dir(), p.name.lower()))

    pointers = ["├── "] * (len(items) - 1) + ["└── "] if items else []

    for pointer, item in zip(pointers, items):
        lines.append(f"{prefix}{pointer}{item.name}")

        if item.is_dir():
            extension = "│   " if pointer == "├── " else "    "
            subtree = tree(
                item,
                prefix=prefix + extension,
                allowed_extensions=allowed_extensions,
                ignore_folders=ignore_folders,
                _is_root=False,  # 👈 important
            )
            if subtree:
                lines.append(subtree)

    return "\n".join(lines)