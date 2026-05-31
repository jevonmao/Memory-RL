#!/bin/bash
# Rsync the local robomme_benchmark repo to the SVL cluster.
# Excludes runs/, .git/, .venv/, __pycache__/, and other heavy artifacts.
#
# Usage:
#   scripts/cluster/sync_to_cluster.sh                    # default host
#   REMOTE=jevon@scdt.stanford.edu scripts/cluster/sync_to_cluster.sh
#   DRY=1 scripts/cluster/sync_to_cluster.sh              # dry-run

set -euo pipefail

REMOTE=${REMOTE:-jevon@scdt.stanford.edu}
DEST=${DEST:-/vision/u/jevon/robomme_benchmark/}
SRC=${SRC:-/home/jevon/projects/robomme_benchmark/}

RSYNC_FLAGS=(-avzh --delete --info=progress2
    --exclude='.git/'
    --exclude='.venv/'
    --exclude='runs/'
    --exclude='wandb/'
    --exclude='logs/'
    --exclude='__pycache__/'
    --exclude='*.pyc'
    --exclude='.cache/'
    --exclude='.pytest_cache/'
    --exclude='*.zip'
    --exclude='*.pkl'
    --exclude='*.pt'
)

if [[ "${DRY:-0}" == "1" ]]; then
    RSYNC_FLAGS+=(--dry-run)
fi

echo "syncing $SRC  -->  ${REMOTE}:${DEST}"
rsync "${RSYNC_FLAGS[@]}" "$SRC" "${REMOTE}:${DEST}"
