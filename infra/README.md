# infra/: on-demand GPU node for CUDA work

Terraform for a single GPU EC2 node in the personal AWS account, used when the
5090 isn't reachable (travel) or when a task wants a throwaway CUDA box: the
CUDA-graphs capture (`viz/capture_cudagraphs.py`) was its first job.

Defaults: **g6.xlarge** (1x L4 24GB, bf16-capable, ~$0.80/hr on-demand) in
**us-west-2**, running the AWS Deep Learning Base GPU AMI (Ubuntu 24.04,
NVIDIA driver preinstalled; `nvidia-smi` works at first boot). The repo's own
uv environment provides CUDA torch; the AMI brings only the driver.

## The loop

```
./up.sh      # price check, then apply; SSH is locked to your current public IP (/32)
./sync.sh    # rsync the working tree to ubuntu@node:cantoLLM/ (gitignore-aware)
./ssh.sh     # ssh in (args pass through: ./ssh.sh nvidia-smi)
             # on the node: cd cantoLLM && uv sync && ...   (use tmux for long runs)
             # copy results home with scp, or run sync in reverse by hand
./down.sh    # destroy the instance only; key pair + security group persist
```

`up.sh` is safe to re-run: if the node is already up and your IP changed
(new cafe), it just updates the SSH rule in place.

## Notes

- **State is local** (`terraform.tfstate`, gitignored), so up/down works only
  from the machine that applied. Same convention as the personal infra repo.
- `terraform.tfvars` (gitignored) holds `ssh_public_key`; see
  `terraform.tfvars.example`.
- **Cost control is `./down.sh`**: the instance is the only metered resource.
  There is no auto-shutdown timer; do not leave the node up overnight.
- If apply fails with `InsufficientInstanceCapacity`, retry, or set
  `INSTANCE_TYPE=g5.xlarge ./up.sh` (A10G) as the fallback type.
- The instance's public IP changes on every up; the scripts always read it
  from `terraform output`.

## The H100 profile

For big-model sessions (h100-plan.md was the first):

```
terraform workspace new tokyo   # once; see the workspace note below
REGION=ap-northeast-1 AZ=ap-northeast-1c INSTANCE_TYPE=p5.4xlarge ROOT_VOLUME_GB=150 ./up.sh
```

- **p5.4xlarge** = 1x H100 80GB, 16 vCPU, 256 GiB RAM, 3.8 TB local NVMe,
  $8.60/hr on-demand in Tokyo. At that rate an overnight forget is a
  ~$140 mistake: `./down.sh` the moment the session ends, then verify
  with `aws ec2 describe-instances --region ap-northeast-1`.
- **Region (learned 2026-08-11, the hard way)**: on-demand p5.4xlarge is
  sold only in London, Mumbai, Jakarta, Tokyo, and São Paulo. us-west-2
  carries the type for **Capacity Blocks for ML only**: a launch there
  returns `InsufficientInstanceCapacity` in every AZ, forever, no matter
  what quota you hold (the spot feed you can see in 2a/2c is resold
  unused capacity-block inventory). Tokyo is the pick (closest on-demand
  region, ~110 ms SSH), and **ap-northeast-1c is the only AZ with the
  type**, so pin it; an AZ walk is meaningless there.
- **Workspace**: terraform state is local and per-workspace; the default
  workspace holds the us-west-2 key pair + security group. Launching in
  another region from the same state would orphan those and later
  collide on `cantollm-key`. `terraform workspace new tokyo` (later:
  `terraform workspace select tokyo|default`) keeps the two regions'
  states separate; ssh.sh/sync.sh read `terraform output`, which is
  workspace-aware.
- **Quota**: "Running On-Demand P instances" (`L-417A185B`) is
  vCPU-denominated, region-scoped, and 0 on a fresh account; p5.4xlarge
  needs 16. us-west-2 was granted 2026-08-11 (useless, see above);
  ap-northeast-1 requested 2026-08-11 (`d690a3ee…`). Both G- and
  P-quota tickets have taken about a day.
- **Weights go on the local NVMe**, not the root EBS volume: gp3 baseline
  throughput is 125 MB/s and a 32B is ~65 GB. `weights.py` downloads into
  the repo tree (`src/cantollm/models/model_data/`), so after the first
  `./sync.sh`, symlink that dir onto the NVMe on the node **before**
  anything downloads:
  `mkdir -p /opt/dlami/nvme/model_data && rm -rf ~/cantoLLM/src/cantollm/models/model_data && ln -s /opt/dlami/nvme/model_data ~/cantoLLM/src/cantollm/models/model_data`
  (also `export HF_HOME=/opt/dlami/nvme/hf` for the hub cache). The NVMe
  is wiped on stop/terminate, which is fine: the node is a throwaway.
- Before walking AZs on a capacity failure, check where the type is
  actually offered: `aws ec2 describe-instance-type-offerings
  --location-type availability-zone --filters
  Name=instance-type,Values=<type>`. Scarce types often live in one or
  two AZs (p5.4xlarge in Tokyo: only 1c), and the offerings list doesn't
  distinguish on-demand from capacity-block-only fleets, so a listed AZ
  can still ICE forever (all of us-west-2, see above). Note the AWS
  terraform provider retries `InsufficientInstanceCapacity` internally
  with backoff: an apply that hangs at "Still creating..." with no
  instance visible in `describe-instances` is that retry loop, not
  progress.
