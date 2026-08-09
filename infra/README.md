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
INSTANCE_TYPE=p5.4xlarge ROOT_VOLUME_GB=150 ./up.sh
```

- **p5.4xlarge** = 1x H100 80GB, 16 vCPU, 256 GiB RAM, 3.8 TB local NVMe,
  ~$6.88/hr on-demand. At that rate an overnight forget is a ~$110 mistake:
  `./down.sh` the moment the session ends, then verify with
  `aws ec2 describe-instances`.
- **Quota**: "Running On-Demand P instances" (`L-417A185B`, us-west-2) is
  vCPU-denominated and 0 on a fresh account; p5.4xlarge needs 16. Request
  raised 2026-08-09; approval took about a day for the G-quota equivalent.
- **Weights go on the local NVMe**, not the root EBS volume: gp3 baseline
  throughput is 125 MB/s and a 32B is ~65 GB. `weights.py` downloads into
  the repo tree (`src/cantollm/models/model_data/`), so after the first
  `./sync.sh`, symlink that dir onto the NVMe on the node **before**
  anything downloads:
  `mkdir -p /opt/dlami/nvme/model_data && rm -rf ~/cantoLLM/src/cantollm/models/model_data && ln -s /opt/dlami/nvme/model_data ~/cantoLLM/src/cantollm/models/model_data`
  (also `export HF_HOME=/opt/dlami/nvme/hf` for the hub cache). The NVMe
  is wiped on stop/terminate, which is fine: the node is a throwaway.
- If capacity fails in one AZ, pin another with `AZ=us-west-2b ./up.sh`
  (any of a/b/c/d) rather than retrying blind.
