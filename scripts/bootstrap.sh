#!/usr/bin/env bash
set -euo pipefail


# Usage:
# bash scripts/bootstrap.sh "Maarten Bosten" "m.n.m.l.bosten@tilburguniversity.edu" my-project my_project
# Args:
# 1: Author name
# 2: Author email
# 3: Distribution name (CLI/Project) e.g. my-project
# 4: Package name (import path) e.g. my_project
# Notes:
# - Works on Windows (Git Bash), macOS, and Linux.
# - No admin rights required.


AUTHOR_NAME=${1:-"Maarten Bosten"}
AUTHOR_EMAIL=${2:-"m.n.m.l.bosten@tilburguniversity.edu"}
DIST_NAME=${3:-"alphacomplexbenchmarking"}
PKG_NAME=${4:-"alphacomplexbenchmarking"}


# Validate PKG_NAME is a valid Python package name (letters, digits, underscore)
if ! [[ "$PKG_NAME" =~ ^[a-zA-Z_][a-zA-Z0-9_]*$ ]]; then
echo "❌ Package name '$PKG_NAME' is not a valid Python identifier (use letters, digits, underscore; no dashes)." >&2
exit 1
fi


# Helper: in-place sed that's portable (GNU/BSD)
replace_in_file() {
local pattern="$1"; shift
local repl="$1"; shift
local file="$1"
local tmp
tmp=$(mktemp)
# shellcheck disable=SC2016
sed "s/${pattern}/${repl}/g" "$file" > "$tmp" && mv "$tmp" "$file"
}


# 1) Rename package folder first (fixes 'file not found' issues on sed)
if [ -d "src/alphacomplexbenchmarking" ] && [ "$PKG_NAME" != "alphacomplexbenchmarking" ]; then
mkdir -p "src"
mv "src/alphacomplexbenchmarking" "src/${PKG_NAME}"
fi


# 2) Bulk replace placeholders across common text/code files
# (skip .git, venvs, data, and binary files)
while IFS= read -r -d '' file; do
# Author
replace_in_file 'Maarten Bosten' "$AUTHOR_NAME" "$file"
replace_in_file 'm.n.m.l.bosten@tilburguniversity.edu' "$AUTHOR_EMAIL" "$file"
# Dist/project (hyphenated)
replace_in_file 'alphacomplexbenchmarking' "$DIST_NAME" "$file"
# Package/import (underscored)
replace_in_file 'alphacomplexbenchmarking' "$PKG_NAME" "$file"
Done=false


done < <(find . \
-type f \
\( -name '*.py' -o -name '*.toml' -o -name '*.md' -o -name '*.yaml' -o -name '*.yml' -o -name 'Makefile' -o -name '*.sh' -o -name 'LICENSE' -o -name '*.cfg' -o -name '*.txt' \) \
-not -path './.git/*' \
-not -path './.venv/*' \
-not -path './.uv/*' \
-not -path './data/*' \
-print0)


# 3) Create env & install deps
uv venv --seed --python 3.12
uv sync --all-extras --group dev


# 4) Install pre-commit hooks
uv run pre-commit install || true


# 5) Initialize git if needed
if [ ! -d .git ]; then
git init
git add .
git commit -m "Initial commit: scaffolded template"
fi


echo "
✅ Bootstrap complete. Next steps:"
echo " uv run python -m ${PKG_NAME}.cli --help"
echo " uv run pytest"