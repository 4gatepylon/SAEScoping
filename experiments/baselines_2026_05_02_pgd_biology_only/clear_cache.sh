#!/usr/bin/env bash
set -euo pipefail

SUBDIR="baselines_2026_05_02_pgd_biology_only"
TARGET="${SAESCOPING_ARTIFACTS_LOCATION:?SAESCOPING_ARTIFACTS_LOCATION not set}/${SUBDIR}"

if [ ! -d "$TARGET" ]; then
    echo "Nothing to clear: $TARGET does not exist."
    exit 0
fi

echo "Will remove: $TARGET"
du -sh "$TARGET"
read -p "Proceed? [y/N] " confirm
if [[ "$confirm" =~ ^[Yy]$ ]]; then
    rm -rf "$TARGET"
    echo "Removed $TARGET"
else
    echo "Aborted."
fi
