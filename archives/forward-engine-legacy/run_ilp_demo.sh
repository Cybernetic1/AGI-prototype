#!/usr/bin/env bash
# Run the minimal ILP Metagol-like demo
set -euo pipefail
PY=python3
cd "$(dirname "$0")"
$PY ilp_metagol_nds.py
