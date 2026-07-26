#!/usr/bin/env bash
# SSH to the node (extra args are passed through, e.g. ./ssh.sh nvidia-smi).
set -euo pipefail
cd "$(dirname "$0")"

IP=$(terraform output -raw public_ip)
exec ssh -o StrictHostKeyChecking=accept-new "ubuntu@${IP}" "$@"
