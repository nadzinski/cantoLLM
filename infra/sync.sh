#!/usr/bin/env bash
# Rsync the working tree to ~/cantoLLM on the node.
# Explicit excludes rather than a .gitignore filter: macOS's default rsync is
# openrsync, which lacks the filter syntax; the list below works on both it
# and real rsync (Homebrew's is preferred when present).
set -euo pipefail
cd "$(dirname "$0")"

RSYNC="/opt/homebrew/bin/rsync"
[[ -x "$RSYNC" ]] || RSYNC="rsync"

IP=$(terraform output -raw public_ip)
"$RSYNC" -az --delete \
  --exclude '.git' --exclude '.venv' --exclude '__pycache__' \
  --exclude 'viz/data' --exclude 'model_data' \
  --exclude '.terraform' --exclude '*.tfstate*' --exclude '*.tfvars' \
  --exclude '.pytest_cache' --exclude '.ruff_cache' \
  ../ "ubuntu@${IP}:cantoLLM/"
echo "Synced to ubuntu@${IP}:cantoLLM/"
