#!/usr/bin/env bash
# Destroy just the instance (the meter). Key pair + security group persist.
set -euo pipefail
cd "$(dirname "$0")"

# REGION must match the region the node was created in (e.g. the Tokyo H100
# profile): with the default var, terraform refreshes against us-west-2,
# concludes the instance is already gone, drops it from state, and destroys
# nothing while the real instance keeps billing. Discovered 2026-08-15.
REGION="${REGION:-us-west-2}"
terraform destroy -auto-approve -target=aws_instance.cantollm -var "region=${REGION}"
echo "Instance destroyed. Key pair and security group kept; ./up.sh recreates the node."
