#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python floor_pipeline/process_full_floor.py "$@"

