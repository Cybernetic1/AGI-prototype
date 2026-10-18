#!/usr/bin/env bash
# Run the minimal ILP Metagol-like demo from repository root
set -euo pipefail
PY=python3
cd "$(dirname "$0")"
$PY metagol_nds.py
