#!/usr/bin/env bash
set -euo pipefail

# Install dev requirements (optional)
if command -v pip3 >/dev/null 2>&1; then
  pip3 install -r requirements-dev.txt || true
fi

pytest -q
