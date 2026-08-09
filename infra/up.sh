#!/usr/bin/env bash
# Spin up the cantollm GPU node: price check, SSH locked to your current IP, apply.
set -euo pipefail
cd "$(dirname "$0")"

REGION="${REGION:-us-west-2}"
TYPE="${INSTANCE_TYPE:-g6.xlarge}"
ROOT_GB="${ROOT_VOLUME_GB:-100}"
AZ="${AZ:-}" # optional AZ pin, e.g. AZ=us-west-2a for capacity dodging

echo "== Pricing check: ${TYPE} in ${REGION} =="
ONDEMAND=$(aws pricing get-products --region us-east-1 --service-code AmazonEC2 \
  --filters "Type=TERM_MATCH,Field=instanceType,Value=${TYPE}" \
            "Type=TERM_MATCH,Field=regionCode,Value=${REGION}" \
            "Type=TERM_MATCH,Field=operatingSystem,Value=Linux" \
            "Type=TERM_MATCH,Field=preInstalledSw,Value=NA" \
            "Type=TERM_MATCH,Field=tenancy,Value=Shared" \
            "Type=TERM_MATCH,Field=capacitystatus,Value=Used" \
  --query 'PriceList[0]' --output text 2>/dev/null \
  | python3 -c "import json,sys; p=json.load(sys.stdin); od=list(p['terms']['OnDemand'].values())[0]; d=list(od['priceDimensions'].values())[0]; print(d['pricePerUnit']['USD'])" \
  2>/dev/null || echo "lookup-failed")
echo "on-demand (what we launch): \$${ONDEMAND}/hr"
echo "spot (reference only):"
aws ec2 describe-spot-price-history --region "${REGION}" \
  --instance-types "${TYPE}" --product-descriptions "Linux/UNIX" \
  --start-time "$(date -u +%Y-%m-%dT%H:%M:%S)" \
  --query 'SpotPriceHistory[].[AvailabilityZone,SpotPrice]' --output text | sort

MYIP=$(curl -s https://checkip.amazonaws.com)
echo
echo "SSH will be locked to ${MYIP}/32"
read -r -p "Launch ${TYPE} on-demand in ${REGION}? [y/N] " ok
[[ "${ok}" == "y" || "${ok}" == "Y" ]] || { echo "aborted"; exit 1; }

[[ -d .terraform ]] || terraform init -input=false
EXTRA_VARS=(-var "root_volume_gb=${ROOT_GB}")
[[ -n "${AZ}" ]] && EXTRA_VARS+=(-var "availability_zone=${AZ}")
terraform apply -auto-approve -var "ssh_cidr=${MYIP}/32" -var "instance_type=${TYPE}" -var "region=${REGION}" "${EXTRA_VARS[@]}"

IP=$(terraform output -raw public_ip)
echo
echo "Node is up. Give cloud-init a minute, then:"
echo "  ssh ubuntu@${IP}        (or ./ssh.sh)"
echo "  ./sync.sh               to push the working tree"
echo "  ./down.sh               to stop paying"
