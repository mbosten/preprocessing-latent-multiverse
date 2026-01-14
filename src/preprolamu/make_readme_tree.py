import os

# config
IGNORE_FOLDERS = {
    "__pycache__",
    ".git",
    ".github",
    ".ruff_cache",
    ".venv",
}

# file extensions we want to show (change accordingly)
ALLOWED_EXTENSIONS = {".py", ".yml", ".md"}


# tree function
def tree(path=".", prefix=""):
    items = []
    for name in os.listdir(path):
        full_path = os.path.join(path, name)

        # Ignore folders we dont want
        if os.path.isdir(full_path) and name in IGNORE_FOLDERS:
            continue

        # Only show folders + allowed file types
        if os.path.isfile(full_path):
            ext = os.path.splitext(name)[1]
            if ext not in ALLOWED_EXTENSIONS:
                continue  # Skip data files, images, etc.

        items.append(name)

    # Sort suc that folders appear before files
    items.sort(key=lambda x: (not os.path.isdir(os.path.join(path, x)), x))

    # Tree drawing
    pointers = ["├── "] * (len(items) - 1) + ["└── "] if items else []
    for pointer, name in zip(pointers, items):
        full_path = os.path.join(path, name)
        print(prefix + pointer + name)

        # Recurse into directories
        if os.path.isdir(full_path):
            extension = "│   " if pointer == "├── " else "    "
            tree(full_path, prefix + extension)


# RUN FOREST, RUN
if __name__ == "__main__":
    tree(".")
