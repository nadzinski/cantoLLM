#!/usr/bin/env bash
# Destroy just the instance (the meter). Key pair + security group persist.
set -euo pipefail
cd "$(dirname "$0")"

terraform destroy -auto-approve -target=aws_instance.cantollm
echo "Instance destroyed. Key pair and security group kept; ./up.sh recreates the node."
