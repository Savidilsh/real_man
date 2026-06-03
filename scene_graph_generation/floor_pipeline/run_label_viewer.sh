#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python floor_pipeline/label_viewer/server.py "$@"
